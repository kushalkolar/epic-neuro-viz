import cv2
import numpy as np


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
            contour_biggest = contours[biggest_ix].squeeze()
            if contour_biggest.ndim < 2: # single point
                # force to be 2d
                contour_biggest = contour_biggest[None]
            points = np.fliplr(contour_biggest)
    else:
        raise ValueError("`outline_mode` must be one of: 'top' | 'all'")

    return points


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