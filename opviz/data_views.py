from functools import partial

import numpy as np
import fastplotlib as fpl
import pygfx

from .data_model import DataModel, EVENT_TYPES


def texture_from_contours(contours: tuple[np.ndarray, ...], fov_shape: tuple[int, int], alpha: float = 0.05) -> np.ndarray:
    texture_data = np.zeros((*fov_shape, 4), dtype=np.float32)

    for comp_index in range(len(contours)):
        for p in contours[comp_index]:
            texture_data[p[0], p[1]] += [1, 1, 1, alpha]

    return texture_data


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

    def _select_component_handler(self, dm_index: int, index: int):
        pass

    def _clear_selection_handler(self, dm_index: int):
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
        #  that we can display different timepoints on each subplot
        #  for Multi-session displays,
        #  OR create an ImageWidget subclass which allows independent time-selection

        self._image_widget = fpl.ImageWidget(
            data=[dm.movie for dm in data_models],
            names=[dm.name for dm in data_models],
        )

        if sync_selection:
            self._original_contours_textures = None
            self._original_contours_texture = texture_from_contours(
                contours=self._data_models[0].contours,
                fov_shape=self._data_models[0].fov_shape,
            )

            # the first image graphic
            image_graphic_1 = fpl.ImageGraphic(
                self._original_contours_texture,
                isolated_buffer=True,  # so we can reset the data using the original texture array to clear highlights
                vmin=0,  # makes it easier to set the colors of the contour highlights using vals between 0 - 1
                vmax=1,
                name="contours",
                offset=(0, 0, 1),  # make sure it's above the calcium video image
            )

            # make ImageGraphic for the rest of the data models
            # we already have the first ImageGraphic so we just
            # need to make the rest and share the buffer with
            # the first ImageGraphic
            for dm in self._data_models[1:]:
                self._image_widget.figure[dm.name].add_image(
                    data=image_graphic_1.data,  # this will use the same data buffer
                    vmin=0,
                    vmax=1,
                    name="contours",
                    offset=(0, 0, 1)
                )

        else:
            self._original_contours_textures = None
            self._original_contours_textures = list()

            for dm in self._data_models:
                texture = texture_from_contours(
                    contours=dm.contours,
                    fov_shape=dm.fov_shape,
                )
                self._original_contours_textures.append(texture)

        for dm_index, dm in enumerate(self._data_models):
            dm.add_event_handler(
                partial(self._select_component_handler, dm_index), "_select_component_handler"
            )

        for dm_index, g in enumerate(self._image_widget.managed_graphics):
            g.add_event_handler(
                partial(self._image_clicked, dm_index), "double_click"
            )

        self._block_select_component_handler = False
        self._block_clear_selection_handler = False

    def _set_data_handler(self, dm_index):
        # ImageWidget.set_data() will ignore any arrays that are already displayed by the ImageWidget
        # so we can just naively use set_data() and only the array which has changed will be updated!
        self._image_widget.set_data(
            [dm.movie for dm in self._data_models],
            reset_vmin_vmax=True,
            reset_indices=False
        )

    def _frame_index_changed(self, dm_index, index):
        if self._image_widget.current_index["t"] == index:
            return

        self._image_widget.current_index = {"t": index}

    def _time_index_changed(self):
        pass

    def _select_component_handler(self, dm_index: int, index: int):
        if self._block_select_component_handler:
            # when we select the component from all data models,
            # this method will be triggered so we need to block it
            # a shared texture buffer is used when selections are synced
            # so there is no need to set the ImageGraphics representing
            # the contours separately
            return

        dm = self._data_models[dm_index]
        # get the contour that corresponds to this component index
        contour = dm.contours[index]

        # get the current selection color
        color = dm.selection_color

        # if the buffer is shared (synced selection), then this will change ALL contour ImageGraphics
        # if the contours are independent per-subplot, then this will change it for just that subplot
        self._image_widget.figure[dm.name]["contours"].data[contour[:, 0], contour[:, 1]] = color

    def _clear_selection_handler(self, dm_index: int):
        if self._block_clear_selection_handler:
            # when we clear components from all data models,
            # this method will be triggered so we need to block it
            # a shared texture buffer is used when selections are synced
            # so there is no need to set the ImageGraphics representing
            # the contours separately
            return

        name = self._data_models[dm_index].name

        # if the buffer is shared (synced selection), then this will clear ALL contour ImageGraphic textures
        # if the contours are independent per-subplot, then this will change it for just that subplot
        self._image_widget.figure[name]["contours"].data = self._original_contours_textures[dm_index]

    def _image_clicked(self, dm_index: int, ev: pygfx.PointerEvent):
        if "Shift" not in ev.modifiers:
            # clear selection
            if self._sync_selection:
                self._data_models[0].clear_selection()
                self._block_clear_selection_handler = True

                for dm in self._data_models[1:]:
                    dm.clear_selection()

                self._block_clear_selection_handler = False
            else:
                self._data_models[dm_index].clear_selection()

        col, row = ev.pick_info["index"]

        index = self._data_models[dm_index].find_closest_components((row, col))[0]

        if self._sync_selection:
            # set the selection of the first data model, shared buffer sets the visual representation (contour color)
            self._block_select_component_handler = False

            try:
                self._data_models[0].select_component(index)
            except Exception as e:
                raise e from None
            finally:
                # buffers are shared, block calls to self._select_component_handler
                self._block_select_component_handler = True

            # change the selection of remaining data models
            try:
                for dm in self._data_models[1:]:
                    dm.select_component(index)
            except Exception as e:
                raise e from None
            finally:
                # select_compoenent() has been called for all data models, restore
                self._block_select_component_handler = False
        else:
            # just set the selection for this one data model
            self._data_models[dm_index].select_component(index)


class TemporalWidget(ModelView):
    def __init__(
            self,
            data_models: list[DataModel],
            sync_time: bool,
            sync_selection: bool,  # TODO: add alpha property for semi-transparent lines
    ):
        super().__init__(
            data_models=data_models,
            sync_time=sync_time,
            sync_selection=sync_selection,
        )

        self._figure = fpl.Figure(
            shape=(len(data_models), 1),
            names=[dm.name for dm in data_models]
        )

        self._lines: list[fpl.LineGraphic] = list()
        self._linear_selectors: list[fpl.LinearSelector] = list()

    @property
    def figure(self) -> fpl.Figure:
        return self._figure

    def _frame_index_handler(self, dm_index, index: int):
        self._linear_selectors[dm_index].selection = index

    def _select_component_handler(self, dm_index, index):
        dm = self._data_models[dm_index]

        subplot = self.figure[dm.name]
        subplot.add_line(
            dm.traces[index],
            colors=dm.selection_color,
            uniform_color=True,
            thickness=1.1,
        )

    def _clear_selection_handler(self, dm_index):
        dm = self._data_models[dm_index]

        for g in self.figure[dm.name].graphics:
            self.figure[dm.name].delete_graphic(g)
