from functools import partial

import fastplotlib as fpl

from ..data_model import OphysModel
from ._base import ModelView

class TemporalWidget(ModelView):
    def __init__(
            self,
            data_models: list[OphysModel],
            sync_time: bool,
            sync_selection: bool,  # TODO: add alpha property for semi-transparent lines
            separation: float | None = None,
    ):
        super().__init__(
            data_models=data_models,
            sync_time=sync_time,
            sync_selection=sync_selection,
        )

        self._separation = separation

        self._figure = fpl.Figure(
            shape=(len(data_models), 1),
            names=[dm.name for dm in data_models],
            size=(1300, 200 * len(data_models))
        )
        for subplot in self.figure:
            subplot.toolbar = False

        self._lines: list[fpl.LineGraphic] = list()
        self._linear_selectors: list[fpl.LinearSelector] = list()

        for i, dm in enumerate(self._data_models):
            subplot = self.figure[dm.name]

            linear_selector = fpl.LinearSelector(
                selection=0,
                limits=(0, dm.n_timepoints - 1),
                name="index-selector",
            )

            subplot.add_graphic(linear_selector)

            linear_selector.add_event_handler(
                partial(self._linear_selector_handler, dm), "selection"
            )

            self._linear_selectors.append(linear_selector)

        self._block_reentrance = False

    @property
    def figure(self) -> fpl.Figure:
        return self._figure

    def _frame_index_handler(self, dm_index, index: int):
        with fpl.pause_events(self._linear_selectors[dm_index]):
            # prevent setting the selection index here from triggering an event
            self._linear_selectors[dm_index].selection = index

    def _select_component_handler(self, dm_index, index):
        dm = self._data_models[dm_index]

        if self._separation is not None:
            z_offset = 0  # z offset doesn't matter if we're stacking lines vertically
            if len(self.figure[0, 0].graphics) > 0:
                g: fpl.LineGraphic = self.figure[0, 0].graphics[-1]
                y_offset = g.data[:, 1].max() + g.offset[1] + self._separation
            else:
                y_offset = 0
        else:
            y_offset = 0

            # place on top of existing lines
            z_offset = len(self.figure[0, 0].graphics)

        subplot = self.figure[dm.name]
        subplot.add_line(
            dm.traces[index],
            colors=dm.selection_color,
            uniform_color=True,
            thickness=1.1,
            offset=(0, y_offset, z_offset),

        )

        subplot.auto_scale(maintain_aspect=False)

    def _clear_selection_handler(self, dm_index, _):
        dm = self._data_models[dm_index]

        for g in self.figure[dm.name].graphics:
            self.figure[dm.name].delete_graphic(g)

    def _linear_selector_handler(self, data_model: OphysModel, ev: fpl.GraphicFeatureEvent):
        index = int(ev.info["value"])

        if self._sync_time:
            for dm in self._data_models:
                dm.frame_index = index
        else:
            data_model.frame_index = index
