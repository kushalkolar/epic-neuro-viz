import numpy as np
import pygfx
from numpy.typing import ArrayLike
import cmap
import fastplotlib as fpl
from fastplotlib.ui import EdgeWindow
from imgui_bundle import imgui
import torch
import cv2
from tqdm import tqdm

import masknmf

from utils import DEMIXING_MAP


def mask_to_contour_points(mask: np.ndarray, outline_mode) -> np.ndarray:
    # make contour outlines
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    if outline_mode == "all":
        points = np.fliplr(np.vstack(contours).squeeze())
    elif outline_mode == "top":
        sizes = [c.shape[0] for c in contours]
        if len(sizes) < 1:
            points = []
        else:
            biggest_ix = np.argmax(sizes)
            points = np.fliplr(contours[biggest_ix].squeeze())
    else:
        raise ValueError("`outline_mode` must be one of: 'top' | 'all'")

    return points


def generate_contours_texture(
    demixing_results: masknmf.DemixingResults,
    fov_shape: tuple[int, int],
    fill_contours: bool,
    alpha: float,
    outline_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    sparse_data = demixing_results.a
    texture_data = np.zeros((*fov_shape, 4))

    if hasattr(demixing_results, "contours") and hasattr(demixing_results, "contour_centers"):
        for comp_index in range(sparse_data.shape[1]):
            for p in demixing_results.contours[comp_index]:
                texture_data[p[0], p[1]] += [1, 1, 1, alpha]

        return texture_data, demixing_results.contour_centers

    centers = np.zeros(shape=(sparse_data.shape[1], 2), dtype=np.float32)

    for comp_index in tqdm(range(sparse_data.shape[1])):
        # TODO: keep this as a torch tensor to compute center, will b 10x faster
        mask = sparse_data.T[comp_index].to_dense().cpu().numpy().reshape(fov_shape) > 0.1

        center = np.argwhere(mask).mean(axis=0)
        centers[comp_index] = center
        # TODO: Figure out fastest way to upscale, I can actually just implement
        #  this on the GPU by taking advantage of sampling, can make a new type of ImageMaterial or something
        # rescale to make it look nice and not jagged
        # mask = rescale(mask, scale=scale)

        if fill_contours:
            texture_data[mask] += [1, 1, 1, alpha]
        else:
            points = mask_to_contour_points(mask, outline_mode=outline_mode)

            for p in points:
                texture_data[p[0], p[1]] += [1, 1, 1, alpha]

    return texture_data, centers


class CalciumWidget:
    # manages the
    def __init__(
            self,
            data: tuple[ArrayLike, masknmf.DemixingResults],
            display_selection: list[str],
            contours_cmap: str = "tab10",
            contours_alpha: float = 0.05,
            fill_contours: bool = False,
            outline_mode: str = "top",  # one of "top" or "all"
            contour_raster_scale: float = 1.0,  # TODO: not used yet, placeholder when we figure out a fast way to do this on the GPU
    ):
        self._verify_datatypes(data)
        self._raw_array, self._demixing_results = data

        self._demixed_arrays = list()
        self._display_selection = display_selection

        for selection in self._display_selection:
            if selection == "raw":
                self._demixed_arrays.append(self._raw_array)
            else:
                array = getattr(self.demixing_results, DEMIXING_MAP[selection])
                self._demixed_arrays.append(array)

        self._image_widget = fpl.ImageWidget(
            data=self._demixed_arrays,
            cmap="viridis",
            histogram_widget=True,
            names=self._display_selection,
            figure_kwargs={"size": (1000, 1200), "show_tooltips": True}
        )

        self.image_widget.figure.renderer.pixel_ratio = 1.0

        self._sparse_data = self._demixing_results.a

        self._contours_alpha = contours_alpha
        self._fill_contours = fill_contours

        self._outline_mode = outline_mode

        self._original_texture_data, self._contour_centers = generate_contours_texture(
            demixing_results=self._demixing_results,
            fov_shape=self._demixing_results.fov_shape,
            fill_contours=self._fill_contours,
            alpha=self._contours_alpha,
            outline_mode=outline_mode,
        )

        # the first image graphic
        image_graphic_1 = fpl.ImageGraphic(
            self._original_texture_data,
            isolated_buffer=True,  # so we can reset the data using the original texture array to clear highlights
            vmin=0,  # makes it easier to set the colors of the contour highlights using vals between 0 - 1
            vmax=1,
            name="contours",
            offset=(0, 0, 1),  # make sure it's above the calcium video image
        )

        for i, subplot in zip(range(len(self._display_selection)), self._image_widget.figure):
            if i == 0:
                # add the existing ImageGraphic
                subplot.add_graphic(image_graphic_1)
            else:
                # create a new ImageGraphic using the existing TextureArray buffer
                subplot.add_image(
                    data=image_graphic_1.data,  # this will use the same data buffer
                    vmin=0,
                    vmax=1,
                    name="contours",
                    offset=(0, 0, 1)
                )

        self._image_widget.show()

        for subplot in self._image_widget.figure:
            subplot.toolbar = False
            subplot.axes.visible = False
            subplot.camera.zoom = 1.0

        self._contours_cmap = cmap.Colormap(contours_cmap)
        self._contours_color_generator = self._contours_cmap.iter_colors()

        self._highlighted_components: list[int] = list()

    def _verify_datatypes(self, data):
        raw_array, demixing_results = data
        for attr in ["shape", "__getitem__"]:
            if not hasattr(raw_array, attr):
                raise TypeError

        if not isinstance(demixing_results, masknmf.DemixingResults):
            raise TypeError

    @property
    def data(self) -> tuple[ArrayLike, masknmf.DemixingResults]:
        """(raw array, masknmf.DemixingResults)"""
        return self._raw_array, self._demixing_results

    @data.setter
    def data(self, new_data: tuple[ArrayLike, masknmf.DemixingResults]):
        self._verify_datatypes(new_data)
        raw_array, demixing_results = new_data

        self.clear_component_selection()

        self._raw_array = raw_array
        self._demixing_results = demixing_results

        # clear demixed arrays
        self._demixed_arrays.clear()

        # get new demixed arrays
        for selection in self._display_selection:
            if selection == "raw":
                self._demixed_arrays.append(raw_array)
            else:
                array = getattr(demixing_results, DEMIXING_MAP[selection])
                self._demixed_arrays.append(array)

        # set ImageWidget
        self._image_widget.set_data(self._demixed_arrays, reset_indices=False, reset_vmin_vmax=False)

        self._sparse_data = self.demixing_results.a

        # set texture data for image graphics that display the contours
        self._original_texture_data, self._contour_centers = generate_contours_texture(
            demixing_results=demixing_results,
            fov_shape=self.demixing_results.fov_shape,
            fill_contours=self._fill_contours,
            alpha=self._contours_alpha,
            outline_mode=self._outline_mode,
        )

        # set data of only the first ImageGraphic representing contours, and all of them will change since the buffer is shared
        self._image_widget.figure[0, 0]["contours"].data = self._original_texture_data

    @property
    def demixing_results(self) -> masknmf.DemixingResults:
        return self._demixing_results

    @property
    def raw_array(self) -> ArrayLike:
        return self._raw_array

    @property
    def image_widget(self) -> fpl.ImageWidget:
        return self._image_widget

    def highlight_component(self, index: int, color):
        """highlight a component using the given color"""
        if self._fill_contours:
            mask = self._sparse_data.T[index].to_dense().cpu().numpy().reshape(self.demixing_results.fov_shape) > 1e-6
            self._image_widget.figure[0, 0]["contours"].data[mask] = color
        else:
            points = self._demixing_results.contours[index]

            for p in points:
                self._image_widget.figure[0, 0]["contours"].data[p[0], p[1]] = color

    def clear_component_selection(self):
        self._image_widget.figure[0, 0]["contours"].data = self._original_texture_data

    def find_closest_components(self, point: tuple[float, float]):
        """

        Args:
            point (float, float): [row, col] index of the point, NOT x, y coordinates

        Returns:
            Indices of components from closest to farthest
        """
        # need to use nanargmin because some centers will be nan if the contour is degenerate
        indices = np.argsort(np.linalg.norm(self._contour_centers - point, ord=2, axis=1))
        return indices

    def _tooltip_info(self, ev) -> str:
        col, row = ev.pick_info["index"]
        index = self.raster_mask.find_closest((row, col))

        info = f"comp index: {index}"

        # return this string to display it in the tooltip
        return info

if __name__ == "__main__":
    raw_array_planes = list()
    demixing_results_planes = list()

    for i in tqdm(range(7, 11)):
        raw_path = f"/home/kushal/amol_data/SP044/2023-06-27/001/suite2p/plane{i}/imaging.frames_motionRegistered.bin"
        shape = (13585, 512, 512)
        raw_array = np.memmap(
            raw_path,
            dtype=np.int16,
            mode="r",
            shape=shape,
        )

        demixed_path = f"/home/kushal/amol_data/demixing_plane{i}.npz"
        demixing_results: masknmf.DemixingResults = np.load(demixed_path, allow_pickle=True)["results"][()]
        demixing_results.to("cuda")

        sparse_data = demixing_results.a

        contours = list()
        centers = np.zeros((sparse_data.shape[1], 2), dtype=np.float32)

        for comp_index in tqdm(range(sparse_data.shape[1])):
            mask = sparse_data.T[comp_index].to_dense().cpu().numpy().reshape(demixing_results.fov_shape) > 0.1

            center = np.argwhere(mask).mean(axis=0)
            centers[comp_index] = center
            points = mask_to_contour_points(mask, outline_mode="top")
            contours.append(points)

        demixing_results.contours = contours
        demixing_results.contour_centers = centers

        raw_array_planes.append(raw_array)
        demixing_results_planes.append(demixing_results)

    viz = CalciumWidget(
        data=(raw_array_planes[0], demixing_results_planes[0]),
        display_selection=["raw", "pmd", "ac", "residuals"],
        fill_contours=False,
        outline_mode="top",
    )
    viz.image_widget.cmap = "gray"

    def image_clicked(ev: pygfx.PointerEvent):
        col, row = ev.pick_info["index"]
        index = viz.find_closest_components((row, col))

        if "Shift" in ev.modifiers:
            viz.highlight_component(index)

        else:
            viz.clear_component_selection()
            viz.highlight_component(index)

    for g in viz.image_widget.managed_graphics:
        g.add_event_handler(image_clicked, "double_click")

    class ZSliderWindow(fpl.ui.EdgeWindow):
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


    gui = ZSliderWindow(
        viz.image_widget.figure,
        size=200,
        location="right",
        title="Plane",
        z_range=(0, len(raw_array_planes) - 1)
    )

    viz.image_widget.figure.add_gui(gui)

    def z_changed(new_z: int):
        raw_array = raw_array_planes[new_z]
        demixing_results = demixing_results_planes[new_z]

        viz.data = (raw_array, demixing_results)

    gui.add_event_handler(z_changed)

    fpl.loop.run()
