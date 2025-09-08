from collections import OrderedDict
from pathlib import Path


import numpy as np
import fastplotlib as fpl
import pynapple as nap


VALID_DEMIXING_VIDS = ["raw", "ac", "fbg", "baseline", "pmd", "residuals"]

DEMIXING_MAP = {
    "ac": "ac_array",
    "fbg": "fluctuating_background_array",
    "baseline": "baseline",
    "pmd": "pmd_array",
    "residuals": "residual_array",
}


class OphysViz:
    def __init__(
            self,
            raw_vid_paths: list[str | Path],  # each item in list is for 1 plane
            demixed_paths: list[str | Path],  # each item in list is for 1 plane, must correspond to raw_vid_paths!!
            demixing_display_selection: list[str] = None,
    ):
        if len(raw_vid_paths) != len(demixed_paths):
            raise ValueError("must supply corresponding planes for `raw_vid_paths` and `demixed_paths`")

        self._n_planes = len(raw_vid_paths)

        self.demixed_data = list()

        for path in demixed_paths:
            self.demixed_data.append(np.load(path, allow_pickle=True)["results"][()])

        if demixing_display_selection is None:
            self._demixing_display_selection = ["raw", "pmd", "ac", "residuals"]

        # outer list is per-plane, inner list is the various display selections
        self._demixed_arrays = [list()]

        # data arrays for each plane
        for z_index in range(self._n_planes):
            for selection in self.demixing_display_selection:
                if selection == "raw":
                    shape = (13585, 512, 512)
                    array = np.memmap(
                        raw_vid_paths[z_index],
                        dtype=np.int16,
                        mode="r",
                        shape=shape,
                    )
                else:
                    array = getattr(self.demixed_data[z_index], DEMIXING_MAP[selection])
                self._demixed_arrays[z_index].append(array)

        self._calcium_timings = np.load("/home/kushal/amol_data/SP044/2023-06-27/001/alf/FOV_07/mpci.times.npy")

        # make ImageWidget to display the calcium vids
        self._iw_calcium_vids = fpl.ImageWidget(
            data=self._demixed_arrays[0],
            cmap="viridis",
            histogram_widget=False,
            names=self.demixing_display_selection
        )

        # used to display the activity of the hovered pixel in each displayed demixing movie
        # TODO: Should sync x axis scale of this but not y-axis scales across cameras
        self.fig_temporal_pixel = fpl.Figure(shape=(len(demixing_display_selection), 1))

    @property
    def demixing_display_selection(self) -> tuple[str]:
        return tuple(self._demixing_display_selection)

    @demixing_display_selection.setter
    def demixing_display_selection(self, *args):
        for arg in args:
            if arg not in VALID_DEMIXING_VIDS:
                raise ValueError(f"Invalid demixing video option, valid options are: {VALID_DEMIXING_VIDS}")

        self._demixing_display_selection = args

    @property
    def iw_calcium_vids(self) -> fpl.ImageWidget:
        pass

    @property
    def heatmap(self):
        pass

    @property
    def fig_behavior_vids(self) -> fpl.Figure:
        pass
