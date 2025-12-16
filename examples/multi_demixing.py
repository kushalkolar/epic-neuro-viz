import numpy as np
from tqdm import tqdm
import fastplotlib as fpl
import masknmf

from opviz import OphysModel, MovieWidget, TemporalWidget
from opviz.utils import mask_to_contour_points, get_roi_avg

# Open data
# raw_path = f"/home/kushal/amol_data/SP044/2023-06-27/001/suite2p/plane7/imaging.frames_motionRegistered.bin"
demixing_path = f"/home/kushal/data/ibl/2024_07_18_demix.hdf5"


demixing_results: masknmf.DemixingResults = masknmf.DemixingResults.from_hdf5(demixing_path)
demixing_results.to("cuda")

shape = demixing_results.shape
# raw_array = np.memmap(
#     raw_path,
#     dtype=np.int16,
#     mode="r",
#     shape=shape,
# )

sparse_data = demixing_results.a

contours = list()
centers = np.zeros((sparse_data.shape[1], 2), dtype=np.float32)

# create masks
masks_argwhere = list()
for comp_index in tqdm(range(sparse_data.shape[1])):
    # TODO: keep this as a torch tensor to compute center, will b 10x faster
    mask = sparse_data.T[comp_index].to_dense().cpu().numpy().reshape(demixing_results.fov_shape) > 1e-6

    center = np.argwhere(mask).mean(axis=0)
    centers[comp_index] = center
    points = mask_to_contour_points(mask, outline_mode="top")
    contours.append(points)

    ixs = np.argwhere(mask)
    masks_argwhere.append(ixs)


# traces_raw = np.vstack(
#     Parallel(n_jobs=20)(
#         delayed(get_roi_avg)(raw_array, ixs[:, 0], ixs[:, 1]) for ixs in masks_argwhere
#     )
# )

## Create data models
# raw_model = DataModel(
#     movie=raw_array,
#     contours=contours,
#     contour_centers=centers,
#     traces=traces_raw,
#     fov_shape=demixing_results.fov_shape,
#     n_timepoints=demixing_results.shape[0],
#     name="raw",
# )

demixing_array_names = [
    # "pmd_array",
    "ac_array",
    "residual_array",
    "fluctuating_background_array",
]

# need to get ROI averages for each movie other than AC
demixing_results_models = list()
for name in demixing_array_names:
    movie = getattr(demixing_results, name)

    print(f"getting traces for: {name}")
    if name == "ac_array":
        traces = demixing_results.c.T.cpu().numpy()
    else:
        traces = np.zeros((len(masks_argwhere), demixing_results.shape[0]))

        for i, ixs in enumerate(masks_argwhere):
            traces[i] = get_roi_avg(movie, ixs[:, 0], ixs[:, 1])

    # create data model for this demixing result array
    dm = OphysModel(
        movie=movie,
        contours=contours,
        contour_centers=centers,
        traces=traces,
        fov_shape=demixing_results.fov_shape,
        n_timepoints=demixing_results.shape[0],
        name=name,
    )

    demixing_results_models.append(dm)

# common kwargs for creating all widgets
# basically just give them the data models objects!
model_view_kwargs = dict(
    data_models=[*demixing_results_models],
    sync_selection=True,
    sync_time=False
)

# create the widgets
movie_widget = MovieWidget(**model_view_kwargs)
temporal_widget = TemporalWidget(**model_view_kwargs)

# show widgets
movie_widget.show()
temporal_widget.show()

fpl.loop.run()
