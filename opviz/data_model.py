from typing import Any

import cmap
import numpy as np
from numpy.typing import ArrayLike


EVENT_TYPES = {
        "select_component",
        "clear_selection",
        "selection_cmap",
        "frame_index",
        "time_index",
        "labels_cmap",
        "active_label",
        "set_data",
    }
# TODO: pixel selection


class Model:
    pass


class OphysModel(Model):
    def __init__(
            self,
            movie,
            contours,
            contour_centers,
            traces,
            fov_shape: tuple[int, int] | tuple[int, int, int],  # TODO: decide how to deal with 3D
            n_timepoints: int,
            name: str,
            selected_components: list[int] = None,
            selection_cmap: str = "tab10",
            time_index: float = 0.0,
            frame_index: int = 0,
            component_labels: dict[str, np.ndarray] = None,
            active_label: str = None,
            labels_cmap: str = "spring",
    ):
        self._movie = movie
        self._contours = contours
        self._contour_centers = contour_centers
        self._traces = traces
        self._selection_cmap = cmap.Colormap(selection_cmap)
        self._name = name

        self._n_timepoints = n_timepoints
        self._fov_shape = fov_shape

        if selected_components is None:
            selected_components = list()

        self._selected_components = selected_components
        self._labels_cmap = cmap.Colormap(labels_cmap)

        self._time_index = time_index
        self._frame_index = frame_index
        self._component_labels = component_labels
        self._active_label = None

        self._event_handlers: dict[str, list] = {et: list() for et in EVENT_TYPES}
        self._re_entrance_block: dict[str, bool] = {et: False for et in EVENT_TYPES}

        self._selection_cmap_cycler = self._selection_cmap.iter_colors()
        self._selection_color: cmap.Color | None = None

    def set_data(
            self,
            movie,
            contours,
            contour_centers,
            traces,
            fov_shape: tuple[int, ...],
            n_timepoints: int,
            component_labels: dict[str, np.ndarray],
            active_label: str = None,
            time_index: float = None,
            frame_index: float = None,
    ):
        """set new data, example: different plane, different session"""
        self.clear_selection()

        # block all events
        for event_type in EVENT_TYPES:
            if event_type == "set_data":
                continue
            self._re_entrance_block[event_type] = True

        try:
            self._movie = movie
            self._contours = contours
            self._contour_centers = contour_centers
            self._traces = traces
            self._n_timepoints = n_timepoints
            self._component_labels = component_labels
            self._active_label = active_label

            if frame_index is not None:
                if frame_index > self._n_timepoints:
                    self._frame_index = self._n_timepoints - 1
                else:
                    self._frame_index = frame_index
        except Exception as e:
            raise e from None
        finally:
            for event_type in EVENT_TYPES:
                if event_type == "set_data":
                    continue
                self._re_entrance_block[event_type] = False
            self._call_event_handlers("set_data", None)

    @property
    def name(self) -> str:
        return self._name

    @property
    def fov_shape(self) -> tuple[int, int] | tuple[int, int, int]:
        return self._fov_shape

    @property
    def n_timepoints(self) -> int:
        return self._n_timepoints

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
    def selection_color(self) -> cmap.Color | None:
        return self._selection_color

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
    def frame_index(self, index: int):
        self._frame_index = index
        self._call_event_handlers("frame_index", index)

    @property
    def timings(self) -> np.ndarray:
        """
        array where each index is the time at which the data was captured by the instrument
        """
        pass

    @property
    def component_labels(self) -> dict[str, np.ndarray]:
        return self._component_labels

    @property
    def active_label(self) -> str | None:
        return self._active_label

    @active_label.setter
    def active_label(self, label_name: str):
        if label_name not in self.component_labels.keys():
            raise KeyError

        self._active_label = label_name

        self._call_event_handlers("active_label", label_name)

    @property
    def labels_cmap(self) -> str:
        return self._labels_cmap.name

    def select_component(self, index: int):
        self._selected_components.append(index)

        # iter color cycler
        self._selection_color = next(self._selection_cmap_cycler)

        self._call_event_handlers("select_component", index)

    def clear_selection(self):
        self._selected_components.clear()

        # reset color cycler
        self._selection_color = None
        self._selection_cmap_cycler = self._selection_cmap.iter_colors()

        self._call_event_handlers("clear_selection", None)

    def _call_event_handlers(self, event_type: str, value: Any):
        if self._re_entrance_block[event_type]:
            # force function to be non-reentrant
            return

        self._re_entrance_block[event_type] = True

        try:
            handlers = self._event_handlers[event_type]

            for handler in handlers:
                handler(value)
        except Exception as e:
            raise e from None
        finally:
            self._re_entrance_block[event_type] = False

    def add_event_handler(self, handler, event_type):
        if event_type not in EVENT_TYPES:
            raise KeyError

        if not callable(handler):
            raise TypeError

        self._event_handlers[event_type].append(handler)

    def find_closest_components(self, point: tuple[float, float]) -> np.ndarray[int]:
        """

        Args:
            point (float, float): [row, col] index of the point, NOT x, y coordinates

        Returns:
            Indices of components from closest to farthest
        """
        # need to use nanargmin because some centers will be nan if the contour is degenerate
        indices = np.argsort(np.linalg.norm(self.contour_centers - point, ord=2, axis=1))
        return indices


class BehaviorDataModel(Model):
    @property
    def movie(self):
        pass

    @property
    def keypoints(self):
        pass

    @property
    def frame_index(self):
        pass

    @property
    def time_index(self):
        pass

    @property
    def kinematics(self):
        pass
