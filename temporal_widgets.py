import numpy as np
from numpy.typing import ArrayLike
import fastplotlib as fpl
import masknmf
import torch

from utils import DEMIXING_MAP, get_roi_avg


class TemporalComponentWidget:
    def __init__(
            self,
            data: tuple[ArrayLike, masknmf.DemixingResults],  # The raw array and demixing results
            display_selection: list[str],
            separation: float | None = None,
    ):
        self._raw_array, self._demixing_results = data

        if "ac" in display_selection:
            display_selection.remove("ac")

        self._figure = fpl.Figure(
            shape=(len(display_selection) + 1, 1),
            names=["temporal component", *display_selection],
            size=(800, 250 * (len(display_selection) + 1))
        )

        self._separation = separation

        self._display_selection = display_selection

        self._linear_selectors = list()
        for subplot in self.figure:
            linear_selector = fpl.LinearSelector(
                selection=0,
                limits=(0, self._demixing_results.shape[0] - 1),
                name="calcium-selector",
                offset=(0, 0, 1_000)
            )
            subplot.add_graphic(linear_selector)
            self._linear_selectors.append(linear_selector)

        self._frame_index = 0

        self.figure.show(maintain_aspect=False)

    @property
    def figure(self) -> fpl.Figure:
        return self._figure

    @property
    def data(self) -> tuple[ArrayLike, masknmf.DemixingResults]:
        """(raw_array, demixing_results)"""
        return (self._raw_array, self._demixing_results)

    @data.setter
    def data(self, new_data: tuple[ArrayLike, masknmf.DemixingResults]):
        self.clear()
        self._raw_array, self._demixing_results = new_data

    @property
    def linear_selectors(self) -> tuple[fpl.LinearSelector, ...]:
        return tuple(self._linear_selectors)

    @property
    def frame_index(self) -> int:
        return self._frame_index

    @frame_index.setter
    def frame_index(self, index: int):
        for s in self._linear_selectors:
            s.selection = index

    def add(self, index, color, autoscale: bool = True):
        c_data = self._demixing_results.c[:, index].cpu().numpy()

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

        self.figure[0, 0].add_line(
            c_data,
            thickness=1.1,
            colors=color,
            uniform_color=True,
            offset=(0, y_offset, z_offset),
            # name=f"line-{len(self.figure[0, 0].graphics)}"
        )

        A = self._demixing_results.a.T[index].to_dense().reshape(self._demixing_results.fov_shape) > 1e-6
        ixs = torch.argwhere(A)

        for selection in self._display_selection:
            if selection == "ac":
                continue

            if selection == "raw":
                line_data = get_roi_avg(
                    self._raw_array,
                    ixs[:, 0].cpu().numpy(),
                    ixs[:, 1].cpu().numpy()
                )
            else:
                line_data = get_roi_avg(
                    getattr(self._demixing_results, DEMIXING_MAP[selection]),
                    ixs[:, 0].cpu().numpy(),
                    ixs[:, 1].cpu().numpy()
                )

            if self._separation is not None:
                z_offset = 0  # z offset doesn't matter if we're stacking lines vertically
                if len(self.figure[selection].graphics) > 0:
                    g: fpl.LineGraphic = self.figure[selection].graphics[-1]
                    y_offset = g.data[:, 1].max() + g.offset[1] + self._separation
                else:
                    y_offset = 0
            else:
                y_offset = 0

                # place on top of existing lines
                z_offset = len(self.figure[selection].graphics)

            self.figure[selection].add_line(
                line_data,
                thickness=1.1,
                colors=color,
                uniform_color=True,
                offset=(0, y_offset, z_offset),
            )

        if autoscale:
            for subplot in self.figure:
                subplot.auto_scale(maintain_aspect=False)

    def clear(self):
        for subplot in self.figure:
            for g in subplot.graphics:
                subplot.delete_graphic(g)


class RawTemporalWidget:
    def __init__(
            self,
            data: tuple[ArrayLike, masknmf.DemixingResults],  # The raw array and demixing results
            display_selection: list[str],
            mode: str = "pixel",
            # separation: float = 0.0
    ):
        self._data = data

        self._raw_array, self._demixing_results = data

        self._display_selection = display_selection

        self._figure = fpl.Figure(
            shape=(len(display_selection), 1),
            names=display_selection,
            size=(800, len(display_selection) * 250)
        )

        self._mode = mode  # one of "pixel" or "component"
        # self._separation = separation

        for subplot in self.figure:
            subplot.add_line(
                data=np.zeros(self._demixing_results.shape[0]),
                thickness=1.1,
                colors="w",
                uniform_color=True,
                name="line"
            )


        self._linear_selectors = list()
        for subplot in self.figure:
            linear_selector = fpl.LinearSelector(
                selection=0,
                limits=(0, self.data[1].shape[0] - 1),
                name="calcium-selector",
                offset=(0, 0, 1_000)
            )
            subplot.add_graphic(linear_selector)
            self._linear_selectors.append(linear_selector)

        self._frame_index = 0

        self.figure.show()

    @property
    def figure(self) -> fpl.Figure:
        return self._figure

    @property
    def data(self) -> tuple[ArrayLike, masknmf.DemixingResults]:
        return self._data

    @data.setter
    def data(self, new_data: tuple[ArrayLike, masknmf.DemixingResults]):
        self._data = new_data

    @property
    def linear_selectors(self) -> tuple[fpl.LinearSelector, ...]:
        return tuple(self._linear_selectors)

    @property
    def frame_index(self) -> int:
        return self._frame_index

    @frame_index.setter
    def frame_index(self, index: int):
        for s in self._linear_selectors:
            s.selection = index

    def set_pixel(self, index: tuple[int, int], autoscale: bool = True):
        """

        Args:
            index (int, int): (row, col) index of pixel

        """
        for i, selection in enumerate(self._display_selection):
            if selection == "raw":
                array = self.data[0]
            else:
                array = getattr(self.data[1], DEMIXING_MAP[selection])

            line_data = array[:, index[0], index[1]]

            self.figure[selection]["line"].data[:, 1] = line_data

            if autoscale:
                self.figure[selection].auto_scale(maintain_aspect=False)
