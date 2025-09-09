from collections import OrderedDict
from pathlib import Path


import numpy as np
import fastplotlib as fpl
import pynapple as nap
from tqdm import tqdm
import cv2
import pygfx


VALID_DEMIXING_VIDS = ["raw", "ac", "fbg", "baseline", "pmd", "residuals"]

DEMIXING_MAP = {
    "ac": "ac_array",
    "colors": "colorful_ac_array",
    "fbg": "fluctuating_background_array",
    "baseline": "baseline",
    "pmd": "pmd_array",
    "residuals": "residual_array",
}


def area_to_vertices(a: np.ndarray) -> np.ndarray:
    """
    Get the vertices of a convex hull generated from an area represented by a binary mask
    :param a: binary mask
    :return: 2D array of x-y coordinates for the hull vertices
    """
    a = a > 0.1
    contours, hierarchy = cv2.findContours(a.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    sizes = [c.shape[0] for c in contours]
    if len(sizes) == 0:
        # for now just return a degenerate contour
        return np.array([[np.nan, np.nan, np.nan]])

    biggest_ix = np.argmax(sizes)

    c = contours[biggest_ix][:, 0, :]

    c = np.vstack([c, c[0]])  # connect last point and first point

    return c


class OphysViz:
    def __init__(
            self,
            raw_vid_paths: list[str | Path],  # each item in list is for 1 plane
            demixed_paths: list[str | Path],  # each item in list is for 1 plane, must correspond to raw_vid_paths!!
            demixing_display_selection: list[str] = None,
    ):
        if len(raw_vid_paths) != len(demixed_paths):
            raise ValueError("must supply corresponding planes for `raw_vid_paths` and `demixed_paths`")

        self._n_planes = len(raw_vid_paths)

        self.demixed_data = list()

        # contours and coms (center of mass) data, each item in the list is for one plane
        self._contours_data = list()
        self._contours_coms = list()

        print("loading demixed data")
        for path in tqdm(demixed_paths):
            self.demixed_data.append(np.load(path, allow_pickle=True)["results"][()])
            self.demixed_data[-1].to("cuda")

        if demixing_display_selection is None:
            self._demixing_display_selection = ["raw", "pmd", "ac", "residuals"]
        else:
            self._demixing_display_selection = demixing_display_selection

        # outer list is per-plane, inner list is the various display selections
        self._demixed_arrays = [list() for i in range(self._n_planes)]

        # similarly, outer list is per-plane, inner list is identical graphics, one per display selection
        self.contours_graphics: list[list[fpl.LineGraphic]] = [list() for i in range(self._n_planes)]


        # data arrays for each plane
        for z_index in range(self._n_planes):
            for selection in self.demixing_display_selection:
                if selection == "raw":
                    shape = (13585, 512, 512)
                    array = np.memmap(
                        raw_vid_paths[z_index],
                        dtype=np.int16,
                        mode="r",
                        shape=shape,
                    )
                else:
                    array = getattr(self.demixed_data[z_index], DEMIXING_MAP[selection])
                self._demixed_arrays[z_index].append(array)

        print("creating vertices from spatial components")
        for z_index in tqdm(range(self._n_planes)):
            self._contours_data.append(list())
            self._contours_coms.append(list())
            for comp_index in range(self.demixed_data[z_index].a.shape[1]):
                A_i = self.demixed_data[z_index].a.to_dense()[:, comp_index].cpu().numpy().reshape(512, 512)

                vertices = area_to_vertices(A_i)
                com = vertices.mean(axis=0)

                self._contours_data[z_index].append(vertices)
                self._contours_coms[z_index].append(com)
            self._contours_coms[z_index] = np.array(self._contours_coms[z_index])


        self._calcium_timings = np.load("/home/kushal/amol_data/SP044/2023-06-27/001/alf/FOV_07/mpci.times.npy")

        # make ImageWidget to display the calcium vids
        # TODO: z index slider with imgui
        self._iw_calcium_vids = fpl.ImageWidget(
            data=self._demixed_arrays[0],
            cmap="viridis",
            histogram_widget=False,
            names=self.demixing_display_selection
        )

        print("create contour graphics")
        for subplot in tqdm(self._iw_calcium_vids.figure):
            # add contours
            # TODO: Share data buffers so that we save GPU RAM!!
            for i in range(self._n_planes):
                contours_g = subplot.add_line_collection(data=self._contours_data[i], colors=(1, 1, 1, 0.05), thickness=1.1, name=f"contours-plane-{i}")
                self.contours_graphics[i].append(contours_g)
                if i > 0:
                    contours_g.visible = False

        # when image clicked, highlight nearest contour
        for g in self._iw_calcium_vids.managed_graphics:
            g.add_event_handler(self._image_clicked, "click")

        self._iw_calcium_vids.show()

        self.fig_heatmap = fpl.Figure(names="heatmap")

        heatmap = self.fig_heatmap[0, 0].add_image(self.demixed_data[0].c.T.cpu().numpy(), cmap="viridis")

        self._heatmap_time_sel = heatmap.add_linear_selector()
        self._heatmap_time_sel.add_event_handler(self._current_index_changed, "selection")

        self._heatmap_comp_sel = heatmap.add_linear_selector(axis="y")

        self._heatmap_comp_sel.add_event_handler(self._component_index_changed, "selection")

        self.fig_heatmap.show(maintain_aspect=False)

        self.fig_selected_temporal = fpl.Figure()
        self.component_temporal_graphic = self.fig_selected_temporal[0, 0].add_line(heatmap.data[0], thickness=1.0)
        self.temporal_linear_selector = self.component_temporal_graphic.add_linear_selector()
        self.temporal_linear_selector.add_event_handler(self._current_index_changed, "selection")

        self.fig_selected_temporal.show(maintain_aspect=False)

        self._iw_calcium_vids.add_event_handler(self._current_index_changed)

        # used to display the activity of the hovered pixel in each displayed demixing movie
        # TODO: Should sync x axis scale of this but not y-axis scales across cameras
        self.fig_temporal_pixel = fpl.Figure(
            shape=(len(self.demixing_display_selection), 1),
            names=self.demixing_display_selection
        )
        self._pixel_plot_selectors: list[fpl.LinearSelector] = list()
        self._pixel_clicked_scatters: list[fpl.ScatterGraphic] = list()

        for i, name in enumerate(self.demixing_display_selection):
            lg = self.fig_temporal_pixel[name].add_line(self._demixed_arrays[0][i][:, 0, 0], name="line", thickness=1.0)
            sel = lg.add_linear_selector()
            #
            # # make a linear selector
            self._pixel_plot_selectors.append(sel)
            sel.add_event_handler(self._current_index_changed, "selection")

            #
            # # update line plot when iw graphic is clicked
            self._iw_calcium_vids.figure[name]["image_widget_managed"].add_event_handler(self._update_temporal_pixel_lines, "click")

            # scatter point to indicate current clicked pixel
            self._pixel_clicked_scatters.append(self._iw_calcium_vids.figure[name].add_scatter(np.array([[0, 0, 1]]), colors=[1, 1, 1, 0.5], sizes=10))

        self.fig_temporal_pixel.show(maintain_aspect=False)

        self._z_index = 0

    @property
    def z_index(self) -> int:
        return self._z_index

    @z_index.setter
    def z_index(self, val: int):
        if val < 0 or val > self._n_planes:
            raise IndexError(f"z index value out of bounds, valid range is: (0, {self._n_planes})")

        # make current contours invisible
        for g in self.contours_graphics[self._z_index]:
            g.visible = False

        self._z_index = val

        self._iw_calcium_vids.set_data(self._demixed_arrays[self._z_index])

        for g in self.contours_graphics[self._z_index]:
            g.visible = True

        # delete and add new heatmap graphic
        self.fig_heatmap[0, 0].clear()

        heatmap = self.fig_heatmap[0, 0].add_image(self.demixed_data[self._z_index].c.T.cpu().numpy())

        self._heatmap_time_sel = heatmap.add_linear_selector()
        self._heatmap_time_sel.add_event_handler(self._current_index_changed, "selection")

        self._heatmap_comp_sel = heatmap.add_linear_selector(axis="y")

        self._heatmap_comp_sel.add_event_handler(self._component_index_changed, "selection")

        # reset component index, temporal graphic will automatically chage

    @property
    def demixing_display_selection(self) -> tuple[str]:
        return tuple(self._demixing_display_selection)

    @demixing_display_selection.setter
    def demixing_display_selection(self, *args):
        for arg in args:
            if arg not in VALID_DEMIXING_VIDS:
                raise ValueError(f"Invalid demixing video option, valid options are: {VALID_DEMIXING_VIDS}")

        self._demixing_display_selection = args

    @property
    def iw_calcium_vids(self) -> fpl.ImageWidget:
        return self._iw_calcium_vids

    @property
    def heatmap(self):
        pass

    @property
    def fig_behavior_vids(self) -> fpl.Figure:
        pass

    def set_component_index(self, index: int, clear: bool = True):
        if clear:
            for i in range(len(self.demixed_data)):
                # make all contours low alpha and thin
                self.contours_graphics[self.z_index][i].colors[:] = [1, 1, 1, 0.05]
                self.contours_graphics[self.z_index][i].thickness[:] = 1

                # set higher alpha and thickness of selected component
                self.contours_graphics[self.z_index][i].graphics[index].colors = "w"
                self.contours_graphics[self.z_index][i].graphics[index].thickness = 3.0

            self._heatmap_comp_sel.selection = index

            if len(self.fig_selected_temporal[0, 0].graphics) > 1:
                for g in self.fig_selected_temporal[0, 0].graphics[1:]:
                    self.fig_selected_temporal[0, 0].delete_graphic(g)

            self.component_temporal_graphic.data[:, 1] = self.fig_heatmap[0, 0].graphics[0].data[index]

        # add an extra graphic and highlight for comparison
        else:
            # set higher alpha and thickness of selected component
            for i in range(len(self.demixed_data)):
                self.contours_graphics[self.z_index][i].graphics[index].colors = "r"
                self.contours_graphics[self.z_index][i].graphics[index].thickness = 3.0

            self.fig_selected_temporal[0, 0].add_line(self.fig_heatmap[0, 0].graphics[0].data[index], thickness=1, colors="r")


    def _current_index_changed(self, ev):
        if isinstance(ev, dict):
            index = ev["t"]

        else:
            index = ev.get_selected_index()

        for sel in self._pixel_plot_selectors:
            sel.selection = index

        self._iw_calcium_vids.current_index = {"t": index}

        self._heatmap_time_sel.selection = index

        self.temporal_linear_selector.selection = index

    def _image_clicked(self, ev):
        if "Control" in ev.modifiers:
            # to not conflict with the single pixel event
            return

        if "Shift" in ev.modifiers:
            clear = False
        else:
            clear = True

        click_point = ev.pick_info["index"]

        index = np.argmin(np.linalg.norm(self._contours_coms[self.z_index] - click_point, ord=2, axis=1))
        self.set_component_index(index, clear=clear)

    def _component_index_changed(self, ev):
        index = ev.get_selected_index()
        # reset colors
        self.set_component_index(index)
        self.fig_selected_temporal[0, 0].auto_scale()

    def _update_temporal_pixel_lines(self, ev: pygfx.PointerEvent):
        if "Control" not in ev.modifiers:
            return

        col, row = ev.pick_info["index"]

        for i, subplot in enumerate(self.fig_temporal_pixel):
             # array that corresponds to this plane and demixing array
            subplot["line"].data[:, 1] = self._demixed_arrays[self.z_index][i][:, row, col]
            self._pixel_clicked_scatters[i].data = np.array([[col, row, 1]])

        for subplot in self.fig_temporal_pixel:
            subplot.auto_scale()


if __name__ == "__main__":
    raw_vid_paths = [
        f"/media/kushal/kushal_ssd/amol_data/SP044/2023-06-27/001/suite2p/plane{i}/imaging.frames_motionRegistered.bin" for i in range(7, 9)
    ]

    demixed_paths = [
        f"/media/kushal/kushal_ssd/amol_data/demixing_plane{i}.npz" for i in range(7, 9)
    ]

    viz = OphysViz(
        raw_vid_paths=raw_vid_paths,
        demixed_paths=demixed_paths,
        demixing_display_selection=["raw", "ac"]
    )

    fpl.loop.run()
