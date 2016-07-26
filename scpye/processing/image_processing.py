import cv2
import numpy as np


def morph_closing(bw, ksize=3, iters=1):
    """
    :param bw: binary image
    :param ksize: kernel size
    :param iters: number of iterations
    :return: binary image after closing
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    bw_closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=iters)
    return bw_closed


def morph_opening(bw, ksize=3, iters=1):
    """
    :param bw: binary image
    :param ksize: kernel size
    :param iters: number of iterations
    :return: binary image after opening
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    bw_opened = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel=kernel,
                                 iterations=iters)
    return bw_opened


def clean_bw(bw, ksize=3, iters=1):
    """
    Clean binary image by doing a opening followed by a closing
    :param bw: binary image
    :param ksize: kernel size
    :param iters: number of iterations
    :return: cleaned binary image
    """
    bw = morph_opening(bw, ksize=ksize, iters=iters)
    bw = morph_closing(bw, ksize=ksize, iters=iters)
    return bw


def u8_from_bw(bw, val=255):
    """
    Convert bw image from bool to uint8 if possible
    :param bw: binary image
    :param val: max_val of image
    :return: greyscale image
    """
    assert np.ndim(bw) == 2, 'Image dimension wrong'
    return np.array(bw > 0, dtype=np.uint8) * val


def fill_bw(bw, cntrs, in_place=False):
    """
    Redraw contours of binary image
    :param bw:
    :param cntrs:
    :param in_place: draw in place
    :return: filled image
    """
    if in_place:
        bw_filled = bw
    else:
        bw_filled = np.zeros_like(bw)

    cv2.drawContours(bw_filled, cntrs, -1, color=255, thickness=-1)

    return bw_filled


def scale_array(data, val=100):
    """
    Scale array to value
    :param data:
    :param val:
    :return:
    """
    max_data = np.max(data)
    scale = float(val) / max_data
    return np.multiply(data, scale)
