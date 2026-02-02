from functools import partial

import numpy as np
import fastplotlib as fpl
import pygfx

from ..data_model import OphysModel
from ._base import ModelView


def texture_from_contours(
        contours: tuple[np.ndarray, ...],
        fov_shape: tuple[int, int],
        alpha: float = 0.05
) -> np.ndarray:

    texture_data = np.zeros((*fov_shape, 4), dtype=np.float32)

    for comp_index in range(len(contours)):
        for p in contours[comp_index]:
            texture_data[p[0], p[1]] += [1, 1, 1, alpha]

    return texture_data


def tooltip_info(curr_graphic: fpl.ImageGraphic, pick_info: dict) -> str:
    # get index of the scatter point that is being hovered
    col, row = pick_info["index"]

    data_val = curr_graphic.data[row, col]

    info = (f"{data_val:e}")

    # return this string to display it in the tooltip
    return info


class MovieWidget(ModelView):
    def __init__(
            self,
            data_models: list[OphysModel],
            sync_time: bool,
            sync_selection: bool,
    ):
        self._data_models = data_models

        self._sync_selection = sync_selection
        self._sync_time = sync_time

        super().__init__(data_models=data_models, sync_time=sync_time, sync_selection=sync_selection)

        # TODO: Decide if we should use Figure or modify ImageWidget so
        #  that we can display different timepoints on each subplot
        #  for Multi-session displays,
        #  OR create an ImageWidget subclass which allows independent time-selection

        self._image_widget = fpl.ImageWidget(
            data=[dm.movie for dm in data_models],
            names=[dm.name for dm in data_models],
            figure_kwargs={"size": (1000, 800)}
        )

        for subplot in self.figure:
            subplot.toolbar = False

        self._create_contours()

        # TODO: decide how to do with when time isn't synced
        if self._sync_time:
            self._image_widget.add_event_handler(self._iw_current_index_changed, "current_index")

        self._block_select_component_handler = False
        self._block_clear_selection_handler = False

    @property
    def figure(self) -> fpl.Figure:
        return self._image_widget.figure

    @property
    def original_contours_textures(self) -> np.ndarray:
        return self._original_contours_textures

    def _create_contours(self):
        # first clear any existing contour graphics
        for subplot in self.figure:
            if "contours" in subplot:
                subplot["contours"].clear_event_handlers()
                subplot.delete_graphic(subplot["contours"])

        if self._sync_selection:
            # contours same for all data models, can just use first data model to create them
            self._original_contours_textures = [texture_from_contours(
                contours=self._data_models[0].contours,
                fov_shape=self._data_models[0].fov_shape,
            )]

            # the first image graphic
            contours_graphic = fpl.ImageGraphic(
                self._original_contours_textures[0],
                isolated_buffer=True,  # so we can reset the data using the original texture array to clear highlights
                vmin=0,  # makes it easier to set the colors of the contour highlights using vals between 0 - 1
                vmax=1,
                name="contours",
                offset=(0, 0, -0.1),  # make sure it's above the calcium video image
            )

            contours_graphic.tooltip_format = partial(tooltip_info, self._image_widget.managed_graphics[0])

            self._image_widget.figure[self._data_models[0].name].add_graphic(contours_graphic)

            # make ImageGraphic for the rest of the data models
            # we already have the first ImageGraphic so we just
            # need to make the rest and share the buffer with
            # the first ImageGraphic
            for ig, dm in zip(self._image_widget.managed_graphics[1:], self._data_models[1:]):
                cg = self._image_widget.figure[dm.name].add_image(
                    data=contours_graphic.data,  # this will use the same data buffer
                    vmin=0,
                    vmax=1,
                    name="contours",
                    offset=(0, 0, -0.1)
                )
                cg.tooltip_format = partial(tooltip_info, ig)

        else:
            self._original_contours_textures = list()

            for dm in self._data_models:
                texture = texture_from_contours(
                    contours=dm.contours,
                    fov_shape=dm.fov_shape,
                )

                self._original_contours_textures.append(texture)

        for i, dm in enumerate(self._data_models):
            if "contours" in self.figure[dm.name]:
                self.figure[dm.name]["contours"].add_event_handler(
                    partial(self._image_clicked, i), "double_click"
                )

    def _set_data_handler(self, dm_index: int, _: None):
        for dm in self._data_models:
            # clear all selections
            dm.clear_selection()

        # ImageWidget.set_data() will ignore any arrays that are already displayed by the ImageWidget
        # so we can just naively use set_data() and only the array which has changed will be updated!
        self._image_widget.set_data(
            [dm.movie for dm in self._data_models],
            reset_vmin_vmax=True,
            reset_indices=False
        )

        self._create_contours()

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

    def _clear_selection_handler(self, dm_index: int, _):
        if self._block_clear_selection_handler:
            # when we clear components from all data models,
            # this method will be triggered so we need to block it
            # a shared texture buffer is used when selections are synced
            # so there is no need to set the ImageGraphics representing
            # the contours separately
            return

        name = self._data_models[dm_index].name

        # if the buffer is shared (synced selection), then this will clear ALL contour ImageGraphic textures
        if self._sync_selection:
            self._image_widget.figure[name]["contours"].data = self._original_contours_textures[0]

        else:
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

        # print("image clicked")

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

    def _frame_index_handler(self, dm_index: int, index: int):
        if self._sync_time:
            if index == self._image_widget.current_index["t"]:
                return

            self._image_widget.current_index = {"t": index}
        else:
            # TODO: change frame index of only one supblot
            pass

    def _iw_current_index_changed(self, ev):
        index = ev["t"]

        if self._sync_time:
            for dm in self._data_models:
                dm.frame_index = index
        else:
            # TODO: need to figure this out
            pass
