import numpy as np
import fastplotlib as fpl
from fastplotlib.ui import EdgeWindow
from imgui_bundle import imgui
import os
import torch
import masknmf
from opviz import OphysModel, MovieWidget, TemporalWidget
from opviz.utils import mask_to_contour_points, get_roi_avg
from ipywidgets import VBox, HBox
from tqdm import tqdm


data = np.memmap(
    "/home/kushal/repos/epic-mcorr-viz/demo_mcorr.mmap",
    shape=(3000, 170, 170),
    dtype="float32"
)

block_sizes = (32, 32)
max_components = 20
device = "cuda"
num_frames_for_spatial_fit = data.shape[0]  # How many frames we use to estimate the spatial basis in PMD

pmd_result = masknmf.compression.pmd_decomposition(
    data,
    block_sizes,
    num_frames_for_spatial_fit,
    max_components=max_components,
    device="cuda",
    frame_batch_size=1024
)

frame_batch_size = 1024
spatial_hp_sigma = 4

spatial_filt_pmd = masknmf.demixing.filters.spatial_filter_pmd(
    pmd_result,
    batch_size=frame_batch_size,
    filter_sigma=spatial_hp_sigma,
    device="cuda"
)

torch.cuda.empty_cache()


class InitUI(EdgeWindow):
    def __init__(self, figure, update_mad_handler, update_nmf_preview_handler):
        super().__init__(figure=figure, size=300, location="right", title="Initialization")

        self._mad_correlation_threshold = 0.8

        self._update_mad_handler = update_mad_handler
        self._update_nmf_preview_handler = update_nmf_preview_handler

    def update(self):
        _, self._mad_correlation_threshold = imgui.slider_float(
            label="mad corr thres", v_min=0.0, v_max=1.0, v=self._mad_correlation_threshold,
        )

        if imgui.button("Update MAD thres"):
            self._update_mad_handler(self._mad_correlation_threshold)

        if imgui.button("Update NMF Preview"):
            self._update_nmf_preview_handler()


class InitViz:
    def __init__(self):
        self._image_widget = fpl.ImageWidget(
            data=[None] * 6,
            names=["pmd", "corr", "above-thres", "mask", "res", "below-thres"],
            figure_shape=(2, 3),
            figure_kwargs={"size": (1000, 1000)},
        )

        self._temporal_widget: TemporalWidget = None

        self._pmd_movie = None
        self._signal_demixer: masknmf.SignalDemixer = None
        self._init_results: masknmf.InitializationResults = None

    @property
    def pmd_movie(self) -> masknmf.PMDArray | None:
        return self._pmd_movie

    @pmd_movie.setter
    def pmd_movie(self, new_movie: masknmf.PMDArray):
        self._pmd_movie = new_movie

        self._image_widget.data["pmd"] = self._pmd_movie

    def _run_mad(self, threshold: float):
        # new signal demixer
        self._signal_demixer = masknmf.demixing.signal_demixer.SignalDemixer(
            self._pmd_movie,
            device=device,
            frame_batch_size=frame_batch_size,
            pixel_batch_size=1024,
        )

        init_kwargs = {
            'mad_correlation_threshold': threshold,

            # Mostly stable
            'mad_threshold': 0,
            'residual_threshold': 0.1,
            'patch_size': (40, 40),
        }

        self._signal_demixer.initialize_signals(**init_kwargs, is_custom=False)
        self._init_results = self._signal_demixer.results

        if not isinstance(self._init_results, masknmf.InitializationResults):
            return

        self._image_widget.data["corr"] = self._init_results.correlation_img

        if "init_pixels" in self._image_widget.figure["corr"]:
            self._image_widget.figure["corr"].delete_graphic(self._image_widget.figure["corr"]["init_pixels"])

        self._image_widget.figure["corr"].add_scatter(
            np.column_stack(np.where(self._init_results.pure_nmf_seed_map)),
            colors="r",
            markers="s",
            sizes=1.0,
            size_space="model",
            edge_width=0.0,
            alpha=0.75,
            name="init_pixels",
        )

        self._image_widget.cmap = "viridis"
        torch.cuda.empty_cache()

    def _run_preview(self):
        num_iters = 5
        localnmf_params = {
            'maxiter': num_iters,
            'support_threshold': np.linspace(0.95, 0.7, num_iters).tolist(),
            'deletion_threshold': 0.2,
            'ring_model_start_pt': num_iters + 1,
            'merge_threshold': 0.9,
            'merge_overlap_threshold': 0.6,
            'update_frequency': 4,
            'c_nonneg': True,
            'denoise': False,
            'plot_en': False,
        }

        with torch.no_grad():
            self._signal_demixer.demix(**localnmf_params)

        res = self._signal_demixer.results.resid_corr_img_normalizer.reshape(170, 170).numpy()
        max_image = self._signal_demixer.results.a.to_dense().max(axis=1).values.reshape(170, 170).numpy()
        mask = max_image.copy()
        mask[mask > 0] = 1

        self._image_widget.data["mask"] = mask
        self._image_widget.data["res"] = res

        self._image_widget.cmap = "viridis"
        torch.cuda.empty_cache()


init_viz = InitViz()
ui = InitUI(
    init_viz._image_widget.figure,
    update_mad_handler=init_viz._run_mad,
    update_nmf_preview_handler=init_viz._run_preview,
)
init_viz._image_widget.figure.add_gui(ui)

init_viz.pmd_movie = spatial_filt_pmd

init_viz._image_widget.show()

fpl.loop.run()
