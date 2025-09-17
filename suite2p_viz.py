import os

import numpy as np
import pygfx
from numpy.typing import ArrayLike
import cmap
import fastplotlib as fpl
from fastplotlib.ui import EdgeWindow
from imgui_bundle import imgui
import torch
import cv2
from tqdm import tqdm

import masknmf


def mask_to_contour_points(mask: np.ndarray, outline_mode) -> np.ndarray:
    # make contour outlines
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    if outline_mode == "all":
        points = np.fliplr(np.vstack(contours).squeeze())
    elif outline_mode == "top":
        sizes = [c.shape[0] for c in contours]
        if len(sizes) < 1:
            points = []
        else:
            biggest_ix = np.argmax(sizes)
            points = np.fliplr(contours[biggest_ix].squeeze())
    else:
        raise ValueError("`outline_mode` must be one of: 'top' | 'all'")

    return points


class CalciumWidgetSuite2p:
    # manages the
    def __init__(
            self,
            raw_array,
            ac,
            # neuropil,
            b,
            residual,
            contours,
            contour_centers,
            contours_cmap: str = "tab10",
            contours_alpha: float = 0.05,
            fill_contours: bool = False,
            outline_mode: str = "top",  # one of "top" or "all"
    ):
        self._raw_array = raw_array
        self._ac = ac
        # self._neuropil = neuropil
        self._b = b
        self._residual = residual


        self._image_widget = fpl.ImageWidget(
            data=[self._raw_array, self._ac, self._b, self._residual],
            names=["raw", "ac", "b", "residual"],
            histogram_widget=True,
            cmap="viridis",
            figure_kwargs={"show_tooltips": True, "size": (1000, 1200)}
        )

        self._contours = contours
        self._contour_centers = contour_centers

        self._original_texture_data = np.zeros((*fov_shape, 4))

        for comp_index in range(len(contours)):
            for p in contours[comp_index]:
                self._original_texture_data[p[0], p[1]] += [1, 1, 1, contours_alpha]

        # the first image graphic
        image_graphic_1 = fpl.ImageGraphic(
            self._original_texture_data,
            isolated_buffer=True,  # so we can reset the data using the original texture array to clear highlights
            vmin=0,  # makes it easier to set the colors of the contour highlights using vals between 0 - 1
            vmax=1,
            name="contours",
            offset=(0, 0, 1),  # make sure it's above the calcium video image
        )

        for i, subplot in enumerate(self._image_widget.figure):
            if i == 0:
                # add the existing ImageGraphic
                subplot.add_graphic(image_graphic_1)
            else:
                # create a new ImageGraphic using the existing TextureArray buffer
                subplot.add_image(
                    data=image_graphic_1.data,  # this will use the same data buffer
                    vmin=0,
                    vmax=1,
                    name="contours",
                    offset=(0, 0, 1)
                )

        self._image_widget.show()

        for subplot in self._image_widget.figure:
            subplot.toolbar = False
            subplot.axes.visible = False
            subplot.camera.zoom = 1.0


    # @property
    # def data(self) -> tuple[ArrayLike, masknmf.DemixingResults]:
    #     """(raw array, masknmf.DemixingResults)"""
    #     return self._raw_array, self._demixing_results
    #
    # @data.setter
    # def data(self, new_data: tuple[ArrayLike, masknmf.DemixingResults]):


    @property
    def raw_array(self) -> ArrayLike:
        return self._raw_array

    @property
    def image_widget(self) -> fpl.ImageWidget:
        return self._image_widget

    def highlight_component(self, index: int, color):
        """highlight a component using the given color"""
        points = self._contours[index]

        for p in points:
            self._image_widget.figure[0, 0]["contours"].data[p[0], p[1]] = color

    def clear_component_selection(self):
        self._image_widget.figure[0, 0]["contours"].data = self._original_texture_data

    def find_closest_components(self, point: tuple[float, float]):
        """

        Args:
            point (float, float): [row, col] index of the point, NOT x, y coordinates

        Returns:
            Indices of components from closest to farthest
        """
        # need to use nanargmin because some centers will be nan if the contour is degenerate
        indices = np.argsort(np.linalg.norm(self._contour_centers - point, ord=2, axis=1))
        return indices


def make_masks_from_suite2p_statfile(stat: dict,
                                     Ly: int,
                                     Lx: int) -> tuple[np.ndarray, np.ndarray]:
    signal_arr = np.zeros((Ly * Lx, len(stat)))
    neuropil_signal_arr = np.zeros((Ly * Lx, len(stat)))

    for k in range(len(stat)):
        cell_mask = np.ravel_multi_index((stat[k]["ypix"], stat[k]["xpix"]), (Ly, Lx))
        lam_val = stat[k]['lam']  # / np.sum(stat[k]['lam'])
        signal_arr[cell_mask, k] = lam_val

        # Let's do the same thing for the neuropil now
        neuropil_pixels = np.ones_like(lam_val)
        neuropil_pixels = neuropil_pixels  # / np.sum(neuropil_pixels)
        neuropil_signal_arr[cell_mask, k] = neuropil_pixels

    return signal_arr, neuropil_signal_arr


if __name__ == "__main__":
    folder = f"/home/kushal/amol_data/SP044/2023-06-27/001/suite2p/plane7/"
    raw_path = f"/home/kushal/amol_data/SP044/2023-06-27/001/suite2p/plane7/imaging.frames_motionRegistered.bin"
    demixed_path = f"/home/kushal/amol_data/demixing_plane7.npz"

    demixing_results: masknmf.DemixingResults = np.load(demixed_path, allow_pickle=True)["results"][()]
    demixing_results.to("cuda")

    pmd_array = demixing_results.pmd_array

    device = "cuda"

    is_cell = np.load(os.path.join(folder, "iscell.npy"))
    decisions = is_cell[:, 0].astype(np.bool)
    c_traces = np.load(os.path.join(folder, "F.npy"), allow_pickle=True)[decisions, :]
    c_neuropil = np.load(os.path.join(folder, "Fneu.npy"), allow_pickle=True)[decisions, :]
    stat = np.load(os.path.join(folder, "stat.npy"), allow_pickle=True)
    ops = np.load(os.path.join(folder,'ops.npy'), allow_pickle=True).item()
    fov_shape = ops['Ly'], ops['Lx']

    # Now let's do neuropil correction
    c_traces = c_traces - ops['neucoeff']*c_neuropil

    #Define the neuropil estimate (at each cell) as a scaled version of the neuropil mask ROI average:
    c_neuropil = ops['neucoeff']*c_neuropil

    c_traces = c_traces.T
    c_neuropil = c_neuropil.T

    c_traces_val = c_traces.shape[1]
    c_agg = torch.from_numpy(np.concatenate([c_traces, c_neuropil], axis=1))

    #Load the spatial data
    signal_spatial, neuropil_spatial = make_masks_from_suite2p_statfile(stat, ops['Ly'], ops['Lx'])
    signal_spatial = signal_spatial.reshape((ops['Ly'], ops['Lx'], -1))[:,:, decisions]
    # neuropil_spatial = neuropil_spatial.reshape((ops['Ly'], ops['Lx'], -1))

    a_agg = np.concatenate([signal_spatial, signal_spatial], axis = 2)
    a_agg = masknmf.ndarray_to_torch_sparse_coo(a_agg.reshape((-1, a_agg.shape[2])))

    print(signal_spatial.shape)

    c_agg_rescale, b_rescale = masknmf.demixing.regression_update.alternating_least_squares_affine_fit(pmd_array.u.to(device),
                                                                                                       pmd_array.v.to(device),
                                                                                                       a_agg.to(device),
                                                                                                       c_agg.to(device),
                                                                                                      scale_nonneg=False,
                                                                                                      num_iters = 150)

    print(c_agg_rescale.shape)

    c_suite2p = c_agg_rescale[:, :c_traces_val]
    c_neuropil = c_agg_rescale[:, c_traces_val:]


    a_suite2p = masknmf.ndarray_to_torch_sparse_coo(signal_spatial.reshape((-1, signal_spatial.shape[2]))).to(device)

    suite2p_ac = masknmf.ACArray(fov_shape,
                                 "C",
                                 a_suite2p,
                                 c_suite2p)

    suite2p_neuropil = masknmf.ACArray(fov_shape,
                                     "C",
                                     a_suite2p,
                                     c_neuropil)

    suite2p_colorful_signal = masknmf.ColorfulACArray(fov_shape,
                                                      "C",
                                                      a_suite2p,
                                                      c_suite2p)

    pmd_array.rescale = True
    pmd_array.to('cuda')



    mean_img_s2p_ac = torch.sparse.mm(a_suite2p, torch.mean(c_suite2p.T, dim = 1, keepdim = True))
    mean_img_s2p_neuropil = torch.sparse.mm(a_suite2p, torch.mean(c_neuropil.T, dim = 1, keepdim = True))
    b_rescale = b_rescale #+ mean_img_s2p_ac + mean_img_s2p_neuropil
    b_rescale = b_rescale.reshape(ops['Ly'], ops['Lx']).to("cuda")

    resid_arr = masknmf.ResidualArray(pmd_array, suite2p_ac, suite2p_neuropil, b_rescale)

    contours = list()
    centers = np.zeros((a_suite2p.shape[1], 2), dtype=np.float32)

    for comp_index in tqdm(range(a_suite2p.shape[1])):
        mask = a_suite2p.T[comp_index].to_dense().cpu().numpy().reshape(512, 512) > 0.1

        center = np.argwhere(mask).mean(axis=0)
        centers[comp_index] = center
        points = mask_to_contour_points(mask, outline_mode="top")
        contours.append(points)


    shape = (13585, 512, 512)
    raw_array = np.memmap(
        raw_path,
        dtype=np.int16,
        mode="r",
        shape=shape,
    )

    viz_s2p = CalciumWidgetSuite2p(
        raw_array=raw_array,
        ac=suite2p_ac,
        b=b_rescale.cpu().numpy(),
        residual=resid_arr,
        fill_contours=False,
        contours=contours,
        contour_centers=centers,
        outline_mode="top",
    )

    viz_s2p.image_widget.cmap = "gray"

    fig_temporal = fpl.Figure(
        shape=(3, 1),
        names=["masknmf", "suite2p", "diff"],
        size=(1000, 800),
        controller_ids=[[0], [0], [1]]
    )

    linear_selector_s2p = fpl.LinearSelector(0, limits=(0, raw_array.shape[0]))
    linear_selector_masknmf = fpl.LinearSelector(0, limits=(0, raw_array.shape[0]))
    linear_selector_diff = fpl.LinearSelector(0, limits=(0, raw_array.shape[0]))

    fig_temporal["masknmf"].add_graphic(linear_selector_masknmf)
    fig_temporal["suite2p"].add_graphic(linear_selector_s2p)
    fig_temporal["diff"].add_graphic(linear_selector_diff)

    cmap_cycler = cmap.Colormap("tab10").iter_colors()

    def image_clicked(ev: pygfx.PointerEvent):
        global cmap_cycler
        if "Shift" not in ev.modifiers:
            for subplot in fig_temporal:
                for g in subplot.graphics:
                    subplot.delete_graphic(g)
            cmap_cycler = cmap.Colormap("tab10").iter_colors()
            viz_s2p.clear_component_selection()
            viz_masknmf.clear_component_selection()

        col, row = ev.pick_info["index"]
        index = viz_s2p.find_closest_components((row, col))[0]

        color = next(cmap_cycler)

        viz_s2p.highlight_component(index, color=color)

        yvals = c_suite2p.T[index].cpu().numpy()
        yvals -= yvals.min()
        yvals /= yvals.max()

        fig_temporal["suite2p"].add_line(yvals, colors=color, uniform_color=True, alpha=0.5, thickness=1.1)

        # for mask-nmf widget
        index = viz_masknmf.find_closest_components((row, col))[0]

        viz_masknmf.highlight_component(index, color=color)

        m_yvals = demixing_results.c.T[index].cpu().numpy()
        m_yvals -= m_yvals.min()
        m_yvals /= m_yvals.max()

        fig_temporal["masknmf"].add_line(m_yvals, colors=color, uniform_color=True, alpha=0.5, thickness=1.1)

        fig_temporal["diff"].add_line(
            m_yvals - yvals,
            colors=color,
            uniform_color=True,
            alpha=0.5,
            thickness=1.1,
        )

        for subplot in fig_temporal:
            subplot.auto_scale(maintain_aspect=False)

    for g in viz_s2p.image_widget.managed_graphics:
        g.add_event_handler(image_clicked, "double_click")


    from calcium_widget import CalciumWidget

    sparse_data = demixing_results.a

    contours = list()
    centers = np.zeros((sparse_data.shape[1], 2), dtype=np.float32)

    for comp_index in tqdm(range(sparse_data.shape[1])):
        mask = sparse_data.T[comp_index].to_dense().cpu().numpy().reshape(demixing_results.fov_shape) > 0.1

        center = np.argwhere(mask).mean(axis=0)
        centers[comp_index] = center
        points = mask_to_contour_points(mask, outline_mode="top")
        contours.append(points)

    demixing_results.contours = contours
    demixing_results.contour_centers = centers

    viz_masknmf = CalciumWidget(
        data=(raw_array, demixing_results),
        display_selection=["raw", "pmd", "ac", "residuals"],
        fill_contours=False,
        outline_mode="top",
    )
    viz_masknmf.image_widget.cmap = "gray"


    for g in viz_masknmf.image_widget.managed_graphics:
        g.add_event_handler(image_clicked, "double_click")

    # sync controllers
    for subplot_masknmf, subplot_s2p in zip(viz_masknmf.image_widget.figure, viz_s2p.image_widget.figure):
        subplot_masknmf.controller = subplot_s2p.controller

    # from rastermap import Rastermap

    # rmap_s2p = Rastermap().fit(c_suite2p.T.cpu().numpy())
    # rmap_masknmf = Rastermap().fit(demixing_results.c.T.cpu().numpy())
    #
    # heatmap_iw_s2p = fpl.ImageWidget(
    #     c_suite2p.T.cpu().numpy()[rmap_s2p.isort],
    #     names=["Suite2p Rastermap"],
    #     histogram_widget=True,
    #     figure_kwargs={"size": (800, 1400)}
    # )
    # heatmap_iw_s2p.show(maintain_aspect=False)
    #
    # heatmap_masknmf = fpl.ImageWidget(
    #     demixing_results.c.T.cpu().numpy()[rmap_masknmf.isort],
    #     names=["masknmf Rastermap"],
    #     histogram_widget=True,
    #     figure_kwargs={"size": (800, 1400)}
    # )
    # heatmap_masknmf.show(maintain_aspect=False)

    def sync_time(ev: dict | fpl.GraphicFeatureEvent):
        if isinstance(ev, dict):
            index = ev["t"]
        else:
            index = int(ev.info["value"])

        viz_s2p.image_widget.current_index = {"t": index}
        viz_masknmf.image_widget.current_index = {"t": index}
        linear_selector_s2p.selection = index
        linear_selector_masknmf.selection = index
        linear_selector_diff.selection = index

    viz_s2p.image_widget.add_event_handler(sync_time, "current_index")
    viz_masknmf.image_widget.add_event_handler(sync_time, "current_index")
    linear_selector_s2p.add_event_handler(sync_time, "selection")
    linear_selector_masknmf.add_event_handler(sync_time, "selection")
    linear_selector_diff.add_event_handler(sync_time, "selection")

    fig_temporal.show(maintain_aspect=False)

    fpl.loop.run()
