from functools import partial

import numpy as np
import fastplotlib as fpl
import pygfx

from .data_model import DataModel


class MovieWidget:
    def __init__(
            self,
            data_models: list[DataModel],
            sync_time: bool,
            sync_selection: bool,
    ):
        self._data_models = data_models

        self._sync_selection = sync_selection
        self._sync_time = sync_time

        # TODO: Decide if we should use Figure or modify ImageWidget so
        #  that we can display different timepoints on each subplot for Multi-session displays

        self._image_widget = fpl.ImageWidget(
            data=[dm.movie for dm in data_models]
        )

        self._contour_graphics: list[fpl.ImageGraphic] = list()
        if sync_selection:
            # TODO: Create one original_contour_texture
            self._original_contours_texture = np.random.rand()
            pass
        else:
            self._original_contours_textures = list()
            # TODO: Create multiple original_contour_texture
            for dm_index, dm in enumerate(self._data_models):
                dm.contours
                self._original_contours_textures.append(np.random.rand())

        for dm_index, dm in enumerate(self._data_models):
            dm.add_event_handler(
                partial(self._select_component, dm_index), "select_component"
            )

        for dm_index, g in enumerate(self._image_widget.managed_graphics):
            g.add_event_handler(
                partial(self._image_clicked, dm_index), "double_click"
            )

    def _frame_index_changed(self, dm_index, index):
        pass

    def _time_index_changed(self):
        pass

    def _select_component(self):
        pass

    def _clear_selection(self, dm_index: int):
        if self._sync_selection:
            for dm in self._data_models:
                dm.clear_selection()
            # just change the first contour graphic, the buffer is shared
            self._contour_graphics[0].data = self._original_contours_texture
        else:
            # change the corresponding contours for only this data model and contour graphic
            self._data_models[dm_index].clear_selection()
            self._contour_graphics[dm_index] = self._original_contours_textures[dm_index]

    def _image_clicked(self, dm_index: int, ev: pygfx.PointerEvent):
        if "Shift" not in ev.modifiers:
            self._clear_selection(dm_index)

        if self._sync_selection:
            # TODO
            pass
        else:
            # TODO
            pass
