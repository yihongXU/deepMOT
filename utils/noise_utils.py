# ==========================================================================
#
# This file is a part of implementation for paper:
# DeepMOT: A Differentiable Framework for Training Multiple Object Trackers.
# This contribution is headed by Perception research team, INRIA.
#
# Contributor(s) : Yihong Xu
# INRIA contact  : yihong.xu@inria.fr
#
# ===========================================================================

import numpy as np
import cv2

# training data augmentation #


def noisy(noise_typ, image):
    """
    add random var=U[10, 50] gaussian or S&P noise to img
    :param noise_typ: string "gauss" or "s&p"
    :param image: numpy input image
    :return: numpy float32 same shape of input image
    """
    if noise_typ == "gauss":
      row,col,ch= image.shape
      mean = 0
      # [10, 30]
      var = 20.0*np.random.rand() + 10.0
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      noisy = image.astype(np.float32) + gauss
      noisy = np.clip(noisy, 0.0, 255.0)
      return noisy.astype(np.uint8)
    elif noise_typ == "s&p":
      s_vs_p = 0.5
      amount = 0.004
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
      out[tuple(coords)] = 1

      # Pepper mode
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
      out[tuple(coords)] = 0
      return out.astype(np.uint8)


def shift_box(bbox, img_h, img_w):
    """
    random (vertical and horizontal shift of a given bbox
    :param bbox: bbox for cropping [x1, y1, x2, y2]
    :param img_h: image height
    :param img_w: image width
    :return: shifted bbox
    """
    tl_x = max(0, bbox[0])
    tl_y = max(0, bbox[1])
    br_x = min(img_w, bbox[2])
    br_y = min(img_h, bbox[3])
    h = br_y - tl_y
    w = br_x - tl_x

    c_x, c_y = 0.5 * (tl_x + br_x), 0.5 * (tl_y + br_y)

    # half w or half h * [0.05, 0.5]
    shift_w = (0.45*np.random.rand() + 0.05) * 0.5 * w
    shift_h = (0.45*np.random.rand() + 0.05) * 0.5 * h

    c_x += shift_w * np.random.choice([-1.0, 1.0])
    c_y += shift_h * np.random.choice([-1.0, 1.0])

    new_box = [int(c_x - 0.5 * w), int(c_y - 0.5 * h), int(c_x + 0.5 * w), int(c_y + 0.5 * h)]
    new_w = min(img_w, new_box[2]) - max(0, new_box[0])
    new_h = min(img_h, new_box[3]) - max(0, new_box[1])

    if new_h <= 10 or new_w <= 10:
        return bbox

    return new_box


def scale_box(bbox, img_h, img_w):
    """
    random scale of a given bbox
    :param bbox: bbox for cropping [x1, y1, x2, y2]
    :param img_h: image height
    :param img_w: image width
    :return: scaled bbox
    """
    tl_x = max(0, bbox[0])
    tl_y = max(0, bbox[1])
    br_x = min(img_w, bbox[2])
    br_y = min(img_h, bbox[3])
    h = br_y - tl_y
    w = br_x - tl_x

    c_x, c_y = 0.5 * (tl_x + br_x), 0.5 * (tl_y + br_y)

    # [0.8, 1.2]
    new_h = (0.4*np.random.rand() + 0.8) * h
    new_w = (0.4*np.random.rand() + 0.8) * w

    if new_h <= 10 or new_w <= 10:
        return bbox

    return [int(c_x-0.5*new_w), int(c_y-0.5*new_h), int(c_x+0.5*new_w), int(c_y+0.5*new_h)]


def blur_img(img):
    """
    gaussian blurring image
    :param img: image: numpy input image
    :return: blurred image
    """
    # [1, 3]
    radius = np.random.randint(1, 3)
    diameter = 2*radius + 1
    blur = cv2.GaussianBlur(img, (diameter, diameter), 0)
    return blur


def constant_change_luminance(img):
    """
    luminance noise added to image
    :param img: image: numpy input image
    :return: blurred image
    """
    # constant [-25, 25]
    constant = np.random.randint(-25, 25, size=(img.shape[0], img.shape[1], 1))
    new_img = np.clip(img.astype(np.float32) - constant, 0.0, 255.0).astype(np.uint8)
    return new_img