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
import copy
import torch
import cv2
# numpy version #


def xywh2xyxy(bbox):
    """
    convert bbox from [x,y,w,h] to [x1, y1, x2, y2]
    :param bbox: bbox in string [x, y, w, h], list
    :return: bbox in float [x1, y1, x2, y2], list
    """
    copy.deepcopy(bbox)
    bbox[0] = float(bbox[0])
    bbox[1] = float(bbox[1])
    bbox[2] = float(bbox[2]) + bbox[0]
    bbox[3] = float(bbox[3]) + bbox[1]

    return bbox


def bb_fast_IOU_v1(boxA, boxB):
    """
    Calculation of IOU, version numpy
    :param boxA: numpy array [top left x, top left y, x2, y2]
    :param boxB: numpy array of [top left x, top left y, x2, y2], shape = [num_bboxes, 4]
    :return: IOU of two bounding boxes of shape [num_bboxes]
    """
    if type(boxA) is type([]):
        boxA = np.array(copy.deepcopy(boxA), dtype=np.float32)[-4:]
        boxB = np.array(copy.deepcopy(boxB), dtype=np.float32)[:, -4:]
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = np.maximum(boxA[0], boxB[:, 0])
    yA = np.maximum(boxA[1], boxB[:, 1])
    xB = np.minimum(boxA[2], boxB[:, 2])
    yB = np.minimum(boxA[3], boxB[:, 3])

    # compute the area of intersection rectangle
    interArea = np.maximum(0.0, xB - xA + 1) * np.maximum(0.0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[:, 2] - boxB[:, 0] + 1) * (boxB[:, 3] - boxB[:, 1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / (boxAArea + boxBArea - interArea)
    return iou


def warpcoordinates(coordinates, warp_matrix):
    """
    camera motion compensations
    :param coordinates: np.darray of shape [num_bbox, 4], 4=[x,y,x,y] or [num_bbox, 2], 2=[x,y]
    :param warp_matrix: numpy.darray of shape [2,3]
    :return: warped coordinates: np.darray of shape [num_bbox, 4], 4=[x,y,x,y]
    """
    if coordinates.shape[1] == 4:
        split_tl = coordinates[:, 0:2].copy()
        split_br = coordinates[:, 2:4].copy()
        pad_ones = np.ones((split_tl.shape[0], 1))
        split_tl = np.transpose(np.hstack([split_tl, pad_ones]))
        split_br = np.transpose(np.hstack([split_br, pad_ones]))
        warped_tl = np.transpose(np.dot(warp_matrix, split_tl))
        warped_br = np.transpose(np.dot(warp_matrix, split_br))
        return np.hstack([warped_tl, warped_br])
    else:
        pad_ones = np.ones((coordinates.shape[0], 1))
        coordinates = np.transpose(np.hstack([coordinates, pad_ones]))
        return np.transpose(np.dot(warp_matrix, coordinates))


# def getWarpMatrix(im1, im2):
#     """
#     get warp matrix
#     :param im1: curr image, numpy array, (h, w, c)
#     :param im2: prev image, numpy array, (h, w, c)
#     :return: affine transformation matrix, numpy array, (h, w)
#     """
#     im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
#     im2Gray_ref = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

#     warp_mode = cv2.MOTION_AFFINE
#     warp_matrix = np.eye(2, 3, dtype=np.float32)

#     cc, warp_matrix = cv2.findTransformECC(im2Gray_ref, im1Gray, warp_matrix, warp_mode)

#     return warp_matrix

def getWarpMatrix(im1, im2):
    """
    get warp matrix
    :param im1: curr image, numpy array, (h, w, c)
    :param im2: prev image, numpy array, (h, w, c)
    :return: affine transformation matrix, numpy array, (h, w)
    """
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray_ref = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    warp_mode = cv2.MOTION_EUCLIDEAN
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    number_of_iterations = 100
    termination_eps = 0.00001
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TermCriteria_COUNT, number_of_iterations, termination_eps)

    cc, warp_matrix = cv2.findTransformECC(im2Gray_ref, im1Gray, warp_matrix, warp_mode, criteria)
    return warp_matrix

# torch version #


def IOUmask_fast(boxA, boxesB):
    """
    get iou among boxA and many others boxes
    :param boxA: [top left x, top left y, bottom right x, bottom right y], float torch tensor requiring gradients of shape (4,)!! not shape (1,4) !!
    :param boxesB: [top left x, top left y, bottom right x, bottom right y], float torch tensor requiring gradients of shape (4,)!! not shape (1,4) !!
    :return: iou
    """
    boxesB = torch.FloatTensor(boxesB).cuda()  # gt box
    # determine the (x, y)-coor dinates of the intersection rectangle
    xA = torch.max(boxA[0], boxesB[:, 0])

    yA = torch.max(boxA[1], boxesB[:, 1])
    xB = torch.min(boxA[2], boxesB[:, 2])
    yB = torch.min(boxA[3], boxesB[:, 3])

    # compute the area of intersection rectangle
    interArea = torch.max(torch.zeros(1).cuda(), xB - xA + 1)\
                * torch.max(torch.zeros(1).cuda(), yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxesBArea = (boxesB[:, 2] - boxesB[:, 0] + 1) * (boxesB[:, 3] - boxesB[:, 1] + 1)

    iou = interArea / (boxAArea + boxesBArea - interArea)
    return iou.unsqueeze(0)


def calculate_dist_fastV4_torch(bbox_det, bbox_gt, im_h, im_w):
    """
    differentiable Version, normalized L2 distance
    :param bbox_det: one detection bbox [x1, y1, x2, y2]
    :param bbox_gt: list of ground truth bboxes [[x1, y1, x2, y2], ... ]
    :param im_h: image height
    :param im_w: image width
    :return: normalized euclidean distance between detection and ground truth
    """
    gt_box = torch.FloatTensor(bbox_gt).cuda()  # gt box
    D = (float(im_h)**2 + im_w**2)**0.5
    c_gt_x, c_gt_y = 0.5 * (gt_box[:, 0] + gt_box[:, 2]), 0.5 * (gt_box[:, 1] + gt_box[:, 3])
    c_det_x, c_det_y = 0.5 * (bbox_det[0] + bbox_det[2]), 0.5 * (bbox_det[1] + bbox_det[3])
    # add eps=1e-12 for gradient numerical stability
    return (1.0 - torch.exp(-5.0*torch.sqrt(1e-12+((c_gt_x-c_det_x)/D)**2 + ((c_gt_y-c_det_y)/D)**2))).unsqueeze(0)


def make_single_matrix_torchV2_fast(gt_bboxes, track_bboxes, img_h, img_w):
    """
    Version torch, differentiable
    :param gt_bboxes: list of ground truth bboxes !! [x1, y1, x2, y2] !! from img of frameID
    :param det_bboxes: detection/hypothesis bboxes !! [x1, y1, x2, y2] !! from img of frameID, torch tensor Variable, [num_dets,4]
    :param frameID: ID of this frame
    :param img_h: height of the image
    :param img_w: width of the image
    :param T: threshold
    :return: matrix
    """
    # number of detections = N = height of matrix
    N = track_bboxes.size(0)
    gt = np.array(gt_bboxes, dtype=np.float32)

    tmp = []
    for i in range(N):
        iou = IOUmask_fast(track_bboxes[i], gt[:, -4:])
        l2_distance = calculate_dist_fastV4_torch(track_bboxes[i], gt[:, -4:], img_h, img_w)
        tmp.append(0.5*(l2_distance + (1.0 - iou)))
    dist_mat = torch.cat(tmp, dim=0)

    if gt.shape[1] == 5:
        gt_ids = gt[:, 0].astype(np.int32).tolist()
    else:
        gt_ids = []
    return gt_ids, dist_mat.unsqueeze(0)


def mix_track_detV2(iou_mat, det, track):
    """
    refine track bounding boxes by detections
    :param iou_mat: iou between dets and tracks, numpy array, [num_track, num_det]
    :param det: detection bbox matrix, numpy array, [num_detections, 4]
    :param track: prediction from trackers, numpy array,  [num_tracks, 4]
    :return: refined new tracks, numpy array, [num_tracks, 4]
    """
    values, idx = torch.max(iou_mat, dim=1)
    mask = torch.ones_like(values)

    for i in range(iou_mat.shape[0]):
        # iou < 0.6, no refinement
        if float(iou_mat[i, idx[i]]) <= 0.6:
            mask[i] = 0.0

    values = mask*values

    return (1.0-values).view(-1, 1)*track + values.view(-1, 1)*torch.index_select(det, 0, idx)


