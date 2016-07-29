from __future__ import (print_function, division, absolute_import)

import cv2
import numpy as np
import matplotlib.pyplot as plt
from itertools import izip


class Colors:
    """
    Collection of colors
    """

    predict = (0, 0, 255)  # blue
    detect = (0, 0, 255)  # red
    counted = (0, 255, 0)  # green
    match = (0, 255, 255)  # yellow
    flow = (255, 0, 255)  # magenta
    text = (0, 255, 255)

    def __init__(self):
        pass


def imshow(*images, **options):
    """
    A helper function to show multiple images in one row
    :param images:
    :param options:
    :return: fig, axarr
    """
    figsize = options.pop('figsize', (10, 10))
    interp = options.pop('interp', None)
    cmap = options.pop('cmap', None)

    fig = plt.figure(figsize=figsize)
    naxes = len(images)
    axarr = np.empty(naxes, dtype=object)

    for i, image in enumerate(images):
        axarr[i] = fig.add_subplot(1, naxes, i + 1)
        axarr[i].imshow(image, interpolation=interp, cmap=cmap)

    return fig, axarr


def draw_bboxes(image, bboxes, color=(255, 0, 0), thickness=1):
    bboxes = np.atleast_2d(bboxes)
    for bbox in bboxes:
        x, y, w, h = np.array(bbox, dtype=int)
        cv2.rectangle(image, (x, y), (x + w, y + h), color=color,
                      thickness=thickness)


def draw_circles(image, circles, color=(255, 0, 0), thickness=1):
    circles = np.atleast_2d(circles)
    for circle in circles:
        x, y, r = np.array(circle, dtype=int)
        cv2.circle(image, (x, y), r, color=color, thickness=thickness)


def draw_points(image, points, color=(255, 0, 0), radius=1):
    points = np.atleast_2d(points)
    for point in points:
        x, y = np.array(point, dtype=int)
        cv2.circle(image, (x, y), radius, color=color, thickness=-1)


def draw_ellipses(image, ellipses, color=(255, 0, 0), thickness=1):
    ellipses = np.atleast_2d(ellipses)
    for ellipse in ellipses:
        x, y, ax1, ax2, ang = np.array(ellipse, dtype=int)
        cv2.ellipse(image, (x, y), (ax1, ax2), ang, 0, 360, color=color,
                    thickness=thickness)


def draw_contour(image, cntr, color=(255, 0, 0), thickness=1):
    cv2.drawContours(image, [cntr], 0, color, thickness)


def draw_contours(image, cntrs, color=(255, 0, 0), thickness=1):
    cv2.drawContours(image, cntrs, -1, color, thickness)


def draw_text(image, text, point, color=(255, 0, 0), scale=0.5, thickness=1):
    if type(text) is not str:
        text = str(int(text))

    x, y = np.array(point, dtype=int)
    cv2.putText(image, text, (x, y), 0, scale, color=color,
                thickness=thickness)


def draw_optical_flows(image, points1, points2, status=None, color=(255, 0, 0),
                       thickness=1, draw_invalid=False):
    points1 = np.atleast_2d(points1)
    points2 = np.atleast_2d(points2)

    if status is None:
        status = np.ones(len(points1))

    for pt1, pt2, st in izip(points1, points2, status):
        if st or draw_invalid:
            a, b = np.array(pt1.ravel(), int)
            c, d = np.array(pt2.ravel(), int)

            cv2.line(image, (a, b), (c, d), color=color, thickness=thickness)
            cv2.circle(image, (c, d), 1, color=color, thickness=-1)


def draw_bboxes_matches(image, matches, bboxes1, bboxes2, color, thickness=1):
    matches = np.atleast_2d(matches)
    bboxes1 = np.atleast_2d(bboxes1)
    bboxes2 = np.atleast_2d(bboxes2)

    for pair in matches:
        i1, i2 = pair
        bbox1 = bboxes1[i1]
        bbox2 = bboxes2[i2]
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        a = int(x1 + w1 / 2)
        b = int(y1 + h1 / 2)
        c = int(x2 + w2 / 2)
        d = int(y2 + h2 / 2)
        draw_bboxes(image, bbox2, color=color, thickness=thickness)
        cv2.line(image, (a, b), (c, d), color=color, thickness=thickness)
