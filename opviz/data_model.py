from typing import Any

import cmap
import numpy as np
from numpy.typing import ArrayLike


class DataModel:
    event_types = {
        "select_component",
        "clear_selection",
        "selection_cmap",
        "frame_index",
        "time_index",
        "labels_cmap",
        "displayed_label",
    }

    def __init__(
            self,
            movie,
            contours,
            contour_centers,
            traces,
            selection_cmap,
            selected_components: list[int] = None,
            time_index: float = 0.0,
            frame_index: int = 0,
            component_labels: dict[str, np.ndarray] = None,
            displayed_label: str = None,
            labels_cmap: str = "spring",
    ):
        self._movie = movie
        self._contours = contours
        self._contour_centers = contour_centers
        self._traces = traces
        self._selection_cmap = cmap.Colormap(selection_cmap)

        if selected_components is None:
            selected_components = list()

        self._selected_components = selected_components
        self._labels_cmap = cmap.Colormap(labels_cmap)

        self._time_index = time_index
        self._frame_index = frame_index
        self._component_labels = component_labels
        self._displayed_label = None

        self._event_handlers = dict[str, list] = {et: list() for et in self.event_types}

    @property
    def movie(self) -> ArrayLike:
        return self._movie

    @property
    def contours(self) -> tuple[np.ndarray, ...]:
        return self._contours

    @property
    def contour_centers(self) -> np.ndarray:
        return self._contour_centers

    @property
    def traces(self) -> ArrayLike:
        return self._traces

    @property
    def selection_cmap(self) -> str:
        return self._selection_cmap.name

    @selection_cmap.setter
    def selection_cmap(self, cmap_name: str):
        self._selection_cmap = cmap.Colormap(cmap_name)
        self._call_event_handlers("selection_cmap", cmap_name)

    @property
    def selected_components(self) -> tuple[int, ...]:
        return tuple(self._selected_components)

    @property
    def time_index(self) -> float:
        return self._time_index

    @time_index.setter
    def time_index(self, index: float):
        self._time_index = index
        self._call_event_handlers("time_index", index)

    @property
    def frame_index(self) -> int:
        return self._frame_index

    @frame_index.setter
    def frame_index(self, index: float):
        self._frame_index = index
        self._call_event_handlers("frame_index", index)

    @property
    def component_labels(self) -> dict[str, np.ndarray]:
        return self._component_labels

    @property
    def displayed_label(self) -> str | None:
        return self._displayed_label

    @displayed_label.setter
    def displayed_label(self, label_name: str):
        if label_name not in self.component_labels.keys():
            raise KeyError

        self._displayed_label = label_name

        self._call_event_handlers("displayed_label", label_name)

    @property
    def labels_cmap(self) -> str:
        return self._labels_cmap.name

    def select_component(self, index: int):
        self._selected_components.append(index)
        self._call_event_handlers("select_component", index)

    def clear_selection(self):
        self._selected_components.clear()
        self._call_event_handlers("clear_selection", None)

    def _call_event_handlers(self, event_type: str, value: Any):
        handlers = self._event_handlers[event_type]

        for handler in handlers:
            handler(value)

    def add_event_handler(self, handler, event_type):
        if event_type not in self.event_types:
            raise KeyError

        if not callable(handler):
            raise TypeError

        self._event_handlers[event_type].append(handler)

    def find_closest_components(self, point: tuple[float, float]) -> np.ndarray:
        """

        Args:
            point (float, float): [row, col] index of the point, NOT x, y coordinates

        Returns:
            Indices of components from closest to farthest
        """
        # need to use nanargmin because some centers will be nan if the contour is degenerate
        indices = np.argsort(np.linalg.norm(self.contour_centers - point, ord=2, axis=1))
        return indices
