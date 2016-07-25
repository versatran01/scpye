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


def uint8_from_bw(bw, val=255):
    """
    Convert bw image from bool to uint8 if possible
    :param bw: binary image
    :param val: max_val of image
    :return: greyscale image
    """
    assert np.ndim(bw) == 2, 'Image dimension wrong'
    return np.array(bw > 0, dtype=np.uint8) * val


def fill_bw(bw, contours):
    """
    Redraw contours of binary image
    :param bw:
    :param contours:
    :return: filled image
    """
    bw_filled = np.zeros_like(bw)
    cv2.drawContours(bw_filled, contours, -1, color=255, thickness=-1)

    return bw_filled
