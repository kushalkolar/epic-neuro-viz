from pathlib import Path
from typing import Literal, Generator

import numpy as np
import pygfx
from numpy.typing import ArrayLike
from tqdm import tqdm
import cmap
import fastplotlib as fpl
from fastplotlib.ui import EdgeWindow
from imgui_bundle import imgui
import masknmf
import zarr

from calcium_widget import CalciumWidget, mask_to_contour_points
from temporal_widgets import TemporalComponentWidget, RawTemporalWidget
from utils import get_extent, find_nearest_index


class UIWIndow(fpl.ui.EdgeWindow):
    def __init__(self, figure, size, location, title, z_range: tuple[int, int]):
        super().__init__(figure=figure, size=size, location=location, title=title)
        self._value = z_range[0]
        self._min = z_range[0]
        self._max = z_range[1]

        self._event_handlers = list()

    def update(self):
        # slider for gaussian filter sigma value
        changed, value = imgui.v_slider_int(
            "z",
            size=[20, 200],
            v=self._value,
            v_min=self._min,
            v_max=self._max
        )
        if changed:
            self._value = value
            self._call_event_handlers(self._value)

    def _call_event_handlers(self, z_value):
        for func in self._event_handlers:
            func(z_value)

    def add_event_handler(self, handler):
        if not callable(handler):
            raise TypeError("handler must be callable")

        self._event_handlers.append(handler)


class OphysViz:
    def __init__(
            self,
            raw_paths: list[str | Path],
            demixing_paths: list[str | Path],
            display_selection: list[Literal["pmd", "ac", "colored", "fbg", "baseline", "residuals"]],
            raw_shape: tuple[int, ...] | None = None,
            raw_dtype: np.dtype = np.int16,
            select_mode: Literal["auto", "manual"] = "manual",
            selection_cmap: str = "tab10",
            temporal_lines_separation: float = None,
    ):
        if len(raw_paths) != len(demixing_paths):
            raise IndexError

        self._n_planes = len(raw_paths)

        self._raw_array_planes: list[ArrayLike] = list()
        self._demixing_results_planes: list[masknmf.DemixingResults] = list()

        for i in range(self._n_planes):
            demixed_path = demixing_paths[i]
            demixing_results: masknmf.DemixingResults = np.load(demixed_path, allow_pickle=True)["results"][()]
            demixing_results.to("cuda")

            shape = demixing_results.shape
            raw_array = np.memmap(
                raw_paths[i],
                dtype=np.int16,
                mode="r",
                shape=shape,
            )

            self._raw_array_planes.append(raw_array)
            self._demixing_results_planes.append(demixing_results)

            # TODO: remove this code block once contours is part of the DemixingResults,
            #  something like: torch.argwhere(A > 1e-6).mean(axis=0, dtype=torch.float32)
            sparse_data = demixing_results.a

            contours = list()
            centers = np.zeros((sparse_data.shape[1], 2), dtype=np.float32)

            for comp_index in tqdm(range(sparse_data.shape[1])):
                # TODO: keep this as a torch tensor to compute center, will b 10x faster
                mask = sparse_data.T[comp_index].to_dense().cpu().numpy().reshape(demixing_results.fov_shape) > 1e-6

                center = np.argwhere(mask).mean(axis=0)
                centers[comp_index] = center
                points = mask_to_contour_points(mask, outline_mode="top")
                contours.append(points)

            demixing_results.contours = contours
            demixing_results.contour_centers = centers

        self._calcium_widget = CalciumWidget(
            data=(self._raw_array_planes[0], self._demixing_results_planes[0]),
            display_selection=display_selection,
            fill_contours=False,
            outline_mode="top",
        )
        self._calcium_widget.image_widget.cmap = "gray"

        # initialize pixel markers data buffer with nans to create a degenerate point
        pixel_marker_data = np.zeros((1, 3), dtype=np.float32)
        pixel_marker_data[:] = np.nan
        pixel_markers_1 = fpl.ScatterGraphic(
            data=pixel_marker_data,
            isolated_buffer=False,
            sizes=5,
            uniform_size=True,
            name="pixel-marker"
        )

        for i, subplot in enumerate(self._calcium_widget.image_widget.figure):
            if i == 0:
                subplot.add_graphic(
                    pixel_markers_1,
                    center=False  # don't center on the graphic after adding it
                                  # data are all nan, so no bounding sphere exists for centering
                )
            else:
                # add scatter graphic which shares the data buffer
                subplot.add_scatter(
                    data=pixel_markers_1.data,
                    sizes=5,
                    uniform_size=True,
                    name="pixel-marker",
                    center=False,
                )

        self._n_pixels_added = 0

        for g in self._calcium_widget.image_widget.managed_graphics:
            if select_mode == "manual":
                g.add_event_handler(self._image_clicked_select_component, "double_click")

            g.add_event_handler(self._image_clicked_select_pixel, "click")

        if select_mode == "auto":
            self._calcium_widget.image_widget.figure.add_animations(self._auto_select_component)

        # extent, [xmin, xmax, ymin, ymax]
        self._last_extent = np.array(
            [0, self._demixing_results_planes[0].fov_shape[1], 0, self._demixing_results_planes[0].fov_shape[1]]
        )

        # throttle auto-selection to once every 60 render cycles
        self._auto_select_throttle = 60

        z_gui = UIWIndow(
            self._calcium_widget.image_widget.figure,
            size=50,
            location="right",
            title="Plane",
            z_range=(0, len(self._raw_array_planes) - 1)
        )

        z_gui.add_event_handler(self._z_changed)

        self._calcium_widget.image_widget.figure.add_gui(z_gui)

        self._temporal_component_widget = TemporalComponentWidget(
            data=(self._raw_array_planes[0], self._demixing_results_planes[0]),
            display_selection=display_selection,
            separation=temporal_lines_separation,
        )

        self._raw_temporal_widget = RawTemporalWidget(
            data=(self._raw_array_planes[0], self._demixing_results_planes[0]),
            display_selection=display_selection,
        )

        self._calcium_widget.image_widget.add_event_handler(self._calcium_time_changed)

        # all linear selectors
        for sel in [*self._temporal_component_widget.linear_selectors, *self._raw_temporal_widget.linear_selectors]:
            sel.add_event_handler(self._calcium_time_changed, "selection")

        self._selection_cmap = cmap.Colormap(selection_cmap)
        self._selection_color_generator = self._selection_cmap.iter_colors()
        self._selected_components = list()


        self._calcium_timings = np.load("/home/kushal/amol_data/SP044/2023-06-27/001/alf/FOV_07/mpci.times.npy")


        print("Loading behavior vids")
        self.behavior_vid_l = zarr.open("/home/kushal/repos/epic-neuro-viz/left_vid.zarr")
        self.behavior_vid_r = zarr.open("/home/kushal/repos/epic-neuro-viz/right_vid.zarr")

        self._fig_behavior_vids = fpl.Figure(shape=(1, 2), names=["left", "right"], size=(1200, 500))
        self._fig_behavior_vids.renderer.pixel_ratio = 1.0

        self._fig_behavior_vids["left"].add_image(self.behavior_vid_l[0], name="image", cmap="gray")
        self._fig_behavior_vids["right"].add_image(self.behavior_vid_r[0], name="image", cmap="gray")

        self._fig_behavior_vids.show()

        for subplot in self._fig_behavior_vids:
            subplot.axes.visible = False
            subplot.toolbar = False

        behavior_data_l = np.load("/home/kushal/amol_data/kushal_datashare/behavior_features_leftcam.npz", allow_pickle=True)["data"][()]
        behavior_data_r = np.load("/home/kushal/amol_data/kushal_datashare/behavior_features_rightcam.npz", allow_pickle=True)["data"][()]

        self.behavior_vid_left_timings = behavior_data_l["times"]
        self.behavior_vid_right_timings = behavior_data_r["times"]

        self.dlc_left = behavior_data_l["dlc"]
        self.dlc_right = behavior_data_r["dlc"]

        # every 3rd column is likelihood
        self.lh_ixs_l = list(range(2, len(self.dlc_left.columns), 3))
        self.point_ixs_l = [i for i in range(len(self.dlc_left.columns)) if i not in self.lh_ixs_l]
        self.x_cols_l = self.dlc_left.columns[self.point_ixs_l][::2]
        self.y_cols_l = self.dlc_left.columns[self.point_ixs_l][1::2]
        self.lh_cols_l = self.dlc_left.columns[self.lh_ixs_l]

        # produce [n_frames, n_points, 2] array
        self.keypoints_left = np.dstack(
            [self.dlc_left[self.x_cols_l].to_numpy(), self.dlc_left[self.y_cols_l].to_numpy()]
        )

        self.keypoints_alpha_left = self.dlc_left[self.lh_cols_l].to_numpy()

        self._fig_behavior_vids["left"].add_scatter(
            self.keypoints_left[0], cmap="tab20", sizes=10, name="keypoints", alpha=self.keypoints_alpha_left[0]
        )

        self.lh_ixs_r = list(range(2, len(self.dlc_right.columns), 3))
        self.point_ixs_r = [i for i in range(len(self.dlc_right.columns)) if i not in self.lh_ixs_r]
        self.x_cols_r = self.dlc_right.columns[self.point_ixs_r][::2]
        self.y_cols_r = self.dlc_right.columns[self.point_ixs_r][1::2]
        self.lh_cols_r = self.dlc_right.columns[self.lh_ixs_r]

        # produce [n_frames, n_points, 2] array
        self.keypoints_right = np.dstack(
            [self.dlc_right[self.x_cols_r].to_numpy(), self.dlc_right[self.y_cols_r].to_numpy()]
        )

        self.keypoints_alpha_right = self.dlc_right[self.lh_cols_r].to_numpy()

        self._fig_behavior_vids["right"].add_scatter(
            self.keypoints_right[0], cmap="tab20", sizes=10, name="keypoints", alpha=self.keypoints_alpha_right[0]
        )

        self._fig_kinematics = fpl.Figure(shape=(2, 1), names=["left paw", "pupil-diameter"], size=(800, 500))

        line_paw = self._fig_kinematics["left paw"].add_line(
            self.dlc_left["paw_l_x"].values[::10],
            colors="r",
            uniform_color=True,
            thickness=1.1,
        )
        self._selector_paw = line_paw.add_linear_selector()

        line_pupil = self._fig_kinematics["pupil-diameter"].add_line(
            behavior_data_l["features"]["pupilDiameter_smooth"].values[::10],
            colors="purple",
            uniform_color=True,
            thickness=1.1,
        )

        self._fig_kinematics.show(maintain_aspect=False)

        self._selector_pupil = line_pupil.add_linear_selector()

        for sel in [self._selector_paw, self._selector_pupil]:
            sel.add_event_handler(self._behavior_time_changed, "selection")

        self._block_reentrance_behavior: bool = False
        self._block_reentrance_calcium: bool = False

    @property
    def selected_components(self) -> tuple[int, ...]:
        return tuple(self._selected_components)

    @property
    def selection_cmap(self) -> str:
        return self._selection_cmap.name

    @selection_cmap.setter
    def selection_cmap(self, cmap_name: str):
        self._selection_cmap = cmap.Colormap(cmap_name)
        self.clear_selection()

    @property
    def fig_behavior_vids(self) -> fpl.Figure:
        return self._fig_behavior_vids


    def _auto_select_component(self):
        """
        Runs on each render cycle to auto-select 2 components closest to the center
        and within the bounds of the current extent
        """
        # throttling
        self._auto_select_throttle -= 1
        if self._auto_select_throttle > 0:
            return
        self._auto_select_throttle = 60

        # all subplot share the same controller so we can just get the extent of the first subplot
        extent = [int(val) for val in get_extent(self._calcium_widget.image_widget.figure[0, 0])]

        if (self._last_extent == extent).all():
            # extent hasn't changed
            return

        xmin, xmax, ymin, ymax = np.asarray(extent).clip(0)

        # row col index of center
        row, col = (ymin + ymax) / 2, (xmax + xmin) / 2

        closest = self._calcium_widget.find_closest_components((row, col))

        self.clear_selection()

        # select top 10 closest
        for i in range(10):
            self.select_component(closest[i])

        self._last_extent[:] = extent

    def select_component(self, index: int):
        # check if component is already selected
        if index in self._selected_components:
            return

        try:
            color = next(self._selection_color_generator)
        except StopIteration:
            raise StopIteration(
                f"The current `selection_cmap`: '{self._selection_cmap}' does not have enough colors to select "
                f"more components. Set a colormap that has more colors to select more components."
            )

        self._calcium_widget.highlight_component(index, color)
        self._temporal_component_widget.add(index, color, autoscale=True)

        self._selected_components.append(index)

    def clear_selection(self):
        self._calcium_widget.clear_component_selection()
        self._temporal_component_widget.clear()
        self._selected_components.clear()

        self._selection_color_generator = self._selection_cmap.iter_colors()

    def _image_clicked_select_pixel(self, ev: pygfx.PointerEvent):
        if "Control" not in ev.modifiers:
            return

        col, row = ev.pick_info["index"]

        self._raw_temporal_widget.set_pixel((row, col))

        # change one of the pixel markers, the rest will change since buffer is hared
        self._calcium_widget.image_widget.figure[0, 0]["pixel-marker"].data[self._n_pixels_added] = [col, row, 2]

        # TODO: figure out separate color cycler for this.
        # self._n_pixels_added += 1

    def _image_clicked_select_component(self, ev: pygfx.PointerEvent):
        if "Control" in ev.modifiers:
            # to not conflict with accidental double_click meant for pixel setting
            return

        col, row = ev.pick_info["index"]
        # get the closest component
        index = self._calcium_widget.find_closest_components((row, col))[0]

        if "Shift" in ev.modifiers:
            self.select_component(index)
        else:
            self.clear_selection()
            self.select_component(index)

    def _z_changed(self, new_z: int):
        raw_array = self._raw_array_planes[new_z]
        demixing_results = self._demixing_results_planes[new_z]

        self._calcium_widget.data = (raw_array, demixing_results)
        self._temporal_component_widget.data = (raw_array, demixing_results)
        self._raw_temporal_widget.data = (raw_array, demixing_results)

        self._selection_color_generator = self._selection_cmap.iter_colors()

    def _calcium_time_changed(self, ev: fpl.GraphicFeatureEvent | dict):
        if self._block_reentrance_calcium:
            return

        self._block_reentrance_calcium = True

        if isinstance(ev, dict):
            # from the calcium movie ImageWidget or _behavior_time_changed method
            index = ev["t"]
        else:
            index = int(ev.info["value"])

        self._calcium_widget.image_widget.current_index = {"t": index}
        self._temporal_component_widget.frame_index = index
        self._raw_temporal_widget.frame_index = index

        time_from_calcium = self._calcium_timings[index]

        behavior_left_index = find_nearest_index(self.behavior_vid_left_timings, time_from_calcium)
        behavior_right_index = find_nearest_index(self.behavior_vid_right_timings, time_from_calcium)

        # set left and right images using current frame
        self.fig_behavior_vids["left"]["image"].data = self.behavior_vid_l[behavior_left_index]
        self.fig_behavior_vids["right"]["image"].data = self.behavior_vid_r[behavior_right_index]

        # update keypoints
        self._fig_behavior_vids["left"]["keypoints"].data[:, :-1] = self.keypoints_left[behavior_left_index]
        self._fig_behavior_vids["left"]["keypoints"].colors[:, -1] = self.keypoints_alpha_right[behavior_left_index]


        self._fig_behavior_vids["right"]["keypoints"].data[:, :-1] = self.keypoints_right[behavior_right_index]
        self._fig_behavior_vids["right"]["keypoints"].colors[:, -1] = self.keypoints_alpha_right[behavior_right_index]

        self._selector_paw.selection = behavior_left_index / 10
        self._selector_pupil.selection = behavior_left_index / 10

        self._block_reentrance_calcium = False

    def _behavior_time_changed(self, ev: fpl.GraphicFeatureEvent):
        if self._block_reentrance_behavior:
            return

        self._block_reentrance_behavior = True

        behavior_index = ev.get_selected_index() * 10

        time_from_left_behavior = self.behavior_vid_left_timings[behavior_index]

        calcium_index = find_nearest_index(self._calcium_timings, time_from_left_behavior)

        self._calcium_time_changed({"t": calcium_index})

        self._block_reentrance_behavior = False


if __name__ == "__main__":
    raw_paths = list()
    demixing_paths = list()
    display_selection = ["raw", "pmd", "ac", "fbg"]
    for i in tqdm(range(7, 11)):
        raw_path = f"/home/kushal/amol_data/SP044/2023-06-27/001/suite2p/plane{i}/imaging.frames_motionRegistered.bin"
        demixing_path = f"/home/kushal/amol_data/demixing_plane{i}.npz"
        raw_paths.append(raw_path)
        demixing_paths.append(demixing_path)

    viz = OphysViz(
        raw_paths,
        demixing_paths,
        display_selection=display_selection,
        temporal_lines_separation=None,
        select_mode="manual",
    )

    fpl.loop.run()
