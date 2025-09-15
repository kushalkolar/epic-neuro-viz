from abc import ABC, abstractmethod
from typing import *
from pathlib import Path
from warnings import warn
import math

import numpy as np

from decord import VideoReader


DEMIXING_MAP = {
    "ac": "ac_array",
    "colored": "colorful_ac_array",
    "fbg": "fluctuating_background_array",
    "baseline": "baseline",
    "pmd": "pmd_array",
    "residuals": "residual_array",
}


def find_nearest_index(timepoints: np.ndarray, find_value: float):
    # find the index of the data closest to given timepoint

    # get closest data index to the world space position of the selector
    idx = np.searchsorted(timepoints, find_value, side="left")

    # bisection algo is the fastest way to do this
    # math.fabs is faster than numpy abs for this usecase
    if idx > 0 and (
            idx == len(timepoints)
            or math.fabs(find_value - timepoints[idx - 1])
            < math.fabs(find_value - timepoints[idx])
    ):
        return round(idx - 1)
    else:
        return round(idx)


# from fastplotlib axes update code
def get_extent(subplot) -> tuple[float, float, float, float] | None:
    """Returns the extent of the subplot in world space, [xmin, xmax, ymin, ymax]"""
    xpos, ypos, width, height = subplot.viewport.rect
    # orthographic projection, get ranges using inverse

    # get range of screen space by getting the corners
    xmin, xmax = xpos, xpos + width
    ymin, ymax = ypos + height, ypos

    min_vals = subplot.map_screen_to_world((xmin, ymin))
    max_vals = subplot.map_screen_to_world((xmax, ymax))

    if min_vals is None or max_vals is None:
        return

    world_xmin, world_ymin, _ = min_vals
    world_xmax, world_ymax, _ = max_vals

    return (world_xmin, world_xmax, world_ymin, world_ymax)


# From mask nmf
def pixel_crop_stack(array, p1, p2):
    if array.shape[0] == 1:
        raise ValueError("Need more than 1 frame in data")
    if np.amin(p1) == np.amax(p1):
        term1 = slice(np.amin(p1), np.amin(p1) + 1)
        dim1_flag = True
    else:
        term1 = slice(np.amin(p1), np.amax(p1) + 1)
        dim1_flag = False

    if np.amin(p2) == np.amax(p2):
        term2 = slice(np.amin(p2), np.amin(p2) + 1)
        dim2_flag = True
    else:
        term2 = slice(np.amin(p2), np.amax(p2) + 1)
        dim2_flag = False

    selected_pixels = array[:, term1, term2].squeeze()

    if dim1_flag and dim2_flag:
        data_2d = selected_pixels[:, None]
    elif dim1_flag and not dim2_flag:
        data_2d = selected_pixels[:, None, p2 - np.amin(p2)]
    elif not dim1_flag and dim2_flag:
        data_2d = selected_pixels[:, p1 - np.amin(p1), None]
    else:
        data_2d = selected_pixels[:, p1 - np.amin(p1), p2 - np.amin(p2)]
    return data_2d

# From mask nmf
# For every signal, need to look at the temporal trace and the PMD average, superimposed
def get_roi_avg(array, p1, p2, normalize=True):
    """
    Given nonzero dim1 and dim2 indices p1 and p2, get the ROI average
    """
    data_2d = pixel_crop_stack(array, p1, p2)
    avg_trace = np.mean(data_2d, axis=1)
    if normalize:
        return avg_trace / np.amax(avg_trace)
    else:
        return avg_trace


slice_or_int_or_range = Union[int, slice, range]


class LazyArray(ABC):
    """
    Base class for arrays that exhibit lazy computation upon indexing
    """

    @property
    @abstractmethod
    def dtype(self) -> str:
        """
        str
            data type
        """
        pass

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, int, int]:
        """
        Tuple[int]
            (n_frames, dims_x, dims_y)
        """
        pass

    @property
    @abstractmethod
    def n_frames(self) -> int:
        """
        int
            number of frames
        """
        pass

    @property
    @abstractmethod
    def min(self) -> float:
        """
        float
            min value of the array if it were fully computed
        """
        pass

    @property
    @abstractmethod
    def max(self) -> float:
        """
        float
            max value of the array if it were fully computed
        """
        pass

    @property
    def ndim(self) -> int:
        """
        int
            Number of dimensions
        """
        return len(self.shape)

    @property
    def nbytes(self) -> int:
        """
        int
            number of bytes for the array if it were fully computed
        """
        return np.prod(self.shape + (np.dtype(self.dtype).itemsize,), dtype=np.int64)

    @property
    def nbytes_gb(self) -> float:
        """
        float
            number of gigabytes for the array if it were fully computed
        """
        return self.nbytes / 1e9

    @abstractmethod
    def _compute_at_indices(self, indices: Union[int, slice]) -> np.ndarray:
        """
        Lazy computation logic goes here. Computes the array at the desired indices.

        Parameters
        ----------
        indices: Union[int, slice]
            the user's desired slice, i.e. slice object or int passed from `__getitem__()`

        Returns
        -------
        np.ndarray
            array at the indexed slice
        """
        pass

    def as_numpy(self):
        """
        NOT RECOMMENDED, THIS COULD BE EXTREMELY LARGE. Converts to a standard numpy array in RAM.

        Returns
        -------
        np.ndarray
        """
        warn(
            f"\nYou are trying to create a numpy.ndarray from a LazyArray, "
            f"this is not recommended and could take a while.\n\n"
            f"Estimated size of final numpy array: "
            f"{self.nbytes_gb:.2f} GB"
        )
        a = np.zeros(shape=self.shape, dtype=self.dtype)

        for i in range(self.n_frames):
            a[i] = self[i]

        return a

    def save_hdf5(self, filename: Union[str, Path]):
        pass

    def __getitem__(self, item: Union[int, Tuple[slice_or_int_or_range]]):
        if isinstance(item, int):
            indexer = item

        # numpy int scaler
        elif isinstance(item, np.integer):
            indexer = item.item()

        # treat slice and range the same
        elif isinstance(item, (slice, range)):
            indexer = item

        elif isinstance(item, tuple):
            if len(item) > len(self.shape):
                raise IndexError(
                    f"Cannot index more dimensions than exist in the array. "
                    f"You have tried to index with <{len(item)}> dimensions, "
                    f"only <{len(self.shape)}> dimensions exist in the array"
                )

            indexer = item[0]

        else:
            raise IndexError(
                f"You can index LazyArrays only using slice, int, or tuple of slice and int, "
                f"you have passed a: <{type(item)}>"
            )

        # treat slice and range the same
        if isinstance(indexer, (slice, range)):
            start = indexer.start
            stop = indexer.stop
            step = indexer.step

            if start is not None:
                if start > self.n_frames:
                    raise IndexError(
                        f"Cannot index beyond `n_frames`.\n"
                        f"Desired frame start index of <{start}> "
                        f"lies beyond `n_frames` <{self.n_frames}>"
                    )
            if stop is not None:
                if stop > self.n_frames:
                    raise IndexError(
                        f"Cannot index beyond `n_frames`.\n"
                        f"Desired frame stop index of <{stop}> "
                        f"lies beyond `n_frames` <{self.n_frames}>"
                    )

            if step is None:
                step = 1

            # convert indexer to slice if it was a range, allows things like decord.VideoReader slicing
            indexer = slice(start, stop, step)  # in case it was a range object

            # dimension_0 is always time
            frames = self._compute_at_indices(indexer)

            # index the remaining dims after lazy computing the frame(s)
            if isinstance(item, tuple):
                if len(item) == 2:
                    return frames[:, item[1]]
                elif len(item) == 3:
                    return frames[:, item[1], item[2]]

            else:
                return frames

        elif isinstance(indexer, int):
            return self._compute_at_indices(indexer)

    def __repr__(self):
        return (
            f"{self.__class__.__name__} @{hex(id(self))}\n"
            f"{self.__class__.__doc__}\n"
            f"Frames are computed only upon indexing\n"
            f"shape [frames, x, y]: {self.shape}\n"
        )


class LazyVideo(LazyArray):
    def __init__(
        self,
        path: Union[Path, str],
        min_max: Tuple[int, int] = None,
        as_grayscale: bool = False,
        rgb_weights: Tuple[float, float, float] = (0.299, 0.587, 0.114),
        **kwargs,
    ):
        """
        LazyVideo reader, basically just a wrapper for ``decord.VideoReader``.
        Should support opening anything that decord can open.

        **Important:** requires ``decord`` to be installed: https://github.com/dmlc/decord

        Parameters
        ----------
        path: Path or str
            path to video file

        min_max: Tuple[int, int], optional
            min and max vals of the entire video, uses min and max of 10th frame if not provided

        as_grayscale: bool, optional
            return grayscale frames upon slicing

        rgb_weights: Tuple[float, float, float], optional
            (r, g, b) weights used for grayscale conversion if ``as_graycale`` is ``True``.
            default is (0.299, 0.587, 0.114)

        kwargs
            passed to ``decord.VideoReader``

        Examples
        --------

        Lazy loading with CPU

        .. code-block:: python

            from mesmerize_core.arrays import LazyVideo

            vid = LazyVideo("path/to/video.mp4")

            # use fpl to visualize

            import fastplotlib as fpl

            iw = fpl.ImageWidget(vid)
            iw.show()


        Lazy loading with GPU, decord must be compiled with CUDA options to use this

        .. code-block:: python

            from decord import gpu
            from mesmerize_core.arrays import LazyVideo

            gpu_context = gpu(0)

            vid = LazyVideo("path/to/video.mp4", ctx=gpu_context)

        """
        self._video_reader = VideoReader(str(path), **kwargs)

        try:
            frame0 = self._video_reader[10].asnumpy()
        except IndexError:
            frame0 = self._video_reader[0].asnumpy()

        self._shape = (self._video_reader._num_frame, *frame0.shape[:-1])

        if len(frame0.shape) > 2:
            # we assume the shape of a frame is [x, y, RGB]
            self._is_color = True
        else:
            # we assume is already grayscale
            self._is_color = False

        self._dtype = frame0.dtype

        if min_max is not None:
            self._min, self._max = min_max
        else:
            self._min = frame0.min()
            self._max = frame0.max()

        self.as_grayscale = as_grayscale
        self.rgb_weights = rgb_weights

    @property
    def dtype(self) -> str:
        return self._dtype

    @property
    def shape(self) -> Tuple[int, int, int]:
        """[n_frames, x, y], RGB color dim not included in shape"""
        return self._shape

    @property
    def n_frames(self) -> int:
        return self.shape[0]

    @property
    def min(self) -> float:
        warn("min not implemented for LazyTiff, returning min of 0th index")
        return self._min

    @property
    def max(self) -> float:
        warn("max not implemented for LazyTiff, returning min of 0th index")
        return self._max

    def _compute_at_indices(self, indices: Union[int, slice]) -> np.ndarray:
        if not self.as_grayscale:
            return self._video_reader[indices].asnumpy()

        if self._is_color:
            a = self._video_reader[indices].asnumpy()

            # R + G + B -> grayscale
            gray = (
                a[..., 0] * self.rgb_weights[0]
                + a[..., 1] * self.rgb_weights[1]
                + a[..., 2] * self.rgb_weights[2]
            )

            return gray

        warn("Video is already grayscale, just returning")
        return self._video_reader[indices].asnumpy()
