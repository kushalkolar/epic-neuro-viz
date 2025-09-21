from functools import partial

import fastplotlib as fpl

from ..data_model import DataModel, EVENT_TYPES


class ModelView:
    def __init__(
            self,
            data_models: list[DataModel],
            sync_time: bool,
            sync_selection: bool,
    ):
        self._data_models = data_models

        self._sync_selection = sync_selection
        self._sync_time = sync_time

        for dm_index, dm in enumerate(self._data_models):
            for event_type in EVENT_TYPES:
                handler = getattr(self, f"_{event_type}_handler")
                dm.add_event_handler(
                    partial(handler, dm_index), event_type
                )

    @property
    def figure(self) -> fpl.Figure:
        pass

    def show(self, **kwargs):
        self.figure.show(**kwargs)

    def _select_component_handler(self, dm_index: int, index: int):
        pass

    def _clear_selection_handler(self, dm_index: int, _: None):
        pass

    def _selection_cmap_handler(self, dm_index: int, cmap_name: str):
        pass

    def _frame_index_handler(self, dm_index: int, index: int):
        pass

    def _time_index_handler(self, dm_index: int, index: float):
        pass

    def _labels_cmap_handler(self, dm_index: int, cmap_name: str | None):
        """set the labels cmap"""
        pass

    def _active_label_handler(self, dm_index: int, label: str):
        """set which label (ex: metric, cluster identify, etc.) is currently displayed"""
        pass

    def _set_data_handler(self, dm_index):
        """reset everything in the view"""
