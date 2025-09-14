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

from calcium_widget import CalciumWidget, mask_to_contour_points
from temporal_widgets import TemporalComponentWidget, RawTemporalWidget


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


class OphysViz:
    def __init__(
            self,
            raw_paths: list[str | Path],
            demixing_paths: list[str | Path],
            display_selection: list[Literal["pmd", "ac", "colored", "fbg", "baseline", "residuals"]],
            raw_shape: tuple[int, ...] | None = None,
            raw_dtype: np.dtype = np.int16,
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

            # TODO: remove this code block once contours is part of the DemixingResults
            sparse_data = demixing_results.a

            contours = list()
            centers = np.zeros((sparse_data.shape[1], 2), dtype=np.float32)

            for comp_index in tqdm(range(sparse_data.shape[1])):
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

        for g in self._calcium_widget.image_widget.managed_graphics:
            g.add_event_handler(self._image_clicked, "double_click")

        z_gui = ZSliderWindow(
            self._calcium_widget.image_widget.figure,
            size=30,
            location="right",
            title="Plane",
            z_range=(0, len(self._raw_array_planes) - 1)
        )

        z_gui.add_event_handler(self._z_changed)

        self._calcium_widget.image_widget.figure.add_gui(z_gui)

        self._temporal_component_widget = TemporalComponentWidget(
            data=self._demixing_results_planes[0],
            separation=temporal_lines_separation,
        )

        self._calcium_widget.image_widget.add_event_handler(self._calcium_time_changed)
        self._temporal_component_widget.linear_selector.add_event_handler(self._calcium_time_changed, "selection")

        self._selection_cmap = cmap.Colormap(selection_cmap)
        self._selection_color_generator = self._selection_cmap.iter_colors()
        self._selected_components = list()

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

    def select_component(self, index: int):
        # check if component is already selected
        if index in self._selected_components:
            return

        try:
            color = next(self._selection_color_generator)
            print(type(color))
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

    def _image_clicked(self, ev: pygfx.PointerEvent):
        col, row = ev.pick_info["index"]
        index = self._calcium_widget.find_closest_component((row, col))

        if "Shift" in ev.modifiers:
            self.select_component(index)
        else:
            self.clear_selection()
            self.select_component(index)

    def _z_changed(self, new_z: int):
        raw_array = self._raw_array_planes[new_z]
        demixing_results = self._demixing_results_planes[new_z]

        self._calcium_widget.data = (raw_array, demixing_results)
        self._temporal_component_widget.data = demixing_results

        self._selection_color_generator = self._selection_cmap.iter_colors()

    def _calcium_time_changed(self, ev: fpl.GraphicFeatureEvent | dict):
        if isinstance(ev, dict):
            # from the calcium movie ImageWidget
            index = ev["t"]
        else:
            index = int(ev.info["value"])

        self._calcium_widget.image_widget.current_index = {"t": index}
        self._temporal_component_widget.linear_selector.selection = index


if __name__ == "__main__":
    raw_paths = list()
    demixing_paths = list()
    display_selection = ["raw", "pmd", "ac", "residuals"]
    for i in tqdm(range(7, 11)):
        raw_path = f"/home/kushal/amol_data/SP044/2023-06-27/001/suite2p/plane{i}/imaging.frames_motionRegistered.bin"
        demixing_path = f"/home/kushal/amol_data/demixing_plane{i}.npz"
        raw_paths.append(raw_path)
        demixing_paths.append(demixing_path)

    viz = OphysViz(raw_paths, demixing_paths, display_selection=display_selection)

    fpl.loop.run()
