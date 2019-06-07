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

from utils.box_utils import *
import torch.nn.functional as F

# weighted classification loss #


def weighted_binary_focal_entropy(output, target, weights=None, gamma=2):
    output = torch.clamp(output, min=1e-12, max=(1 - 1e-12))
    if weights is not None:
        assert weights.size(1) == 2

        # weight is of shape [batch,2, 1, 1]
        # weight[:,1] is for positive class, label = 1
        # weight[:,0] is for negative class, label = 0

        loss = torch.pow(output[0, :], gamma)*target[1, :]*weights[:, 1].item() * torch.log(output[1, :]) + \
               target[0, :]*weights[:, 0].item() * torch.log(output[0, :])*torch.pow(output[1, :], gamma)
    else:
        loss = target[1, :] * torch.log(output[1, :]) + target[0, :] * torch.log(output[0, :])

    return torch.neg(torch.mean(loss))


def focaLoss(score_tensor, ancrs, state_curr, gt_ids, gt_boxes, args):
    # classification loss

    # 1)construct gt target label
    gt_label = np.zeros((2, score_tensor.shape[1]))

    # a) find gt box by gt_id
    searched_id = state_curr['gt_id']

    negative_indexes = list(range(gt_label.shape[1]))
    positive_indexes = list()
    if int(searched_id) in gt_ids:
        searched_index = gt_ids.index(int(searched_id))
        boxA = gt_boxes[searched_index, -4:]
        ious = bb_fast_IOU_v1(boxA, ancrs)

        # NEGATIVE EXAMPLES
        negative_indexes = np.where(ious < 0.3)[0].tolist()
        gt_label[1, negative_indexes] = 0.0
        # POSITIVE EXAMPLES
        positive_indexes = np.where(ious > 0.6)[0].tolist()
        gt_label[1, positive_indexes] = 1.0
        gt_label[0, :] = 1.0 - gt_label[1, :]

        # remove undecided class
        gt_label = gt_label[:, negative_indexes + positive_indexes]

    # CALCULATE FOCAL LOSS
    # num_positive = how many labels = 1
    num_positive = len(positive_indexes)
    weight2negative = float(num_positive) / gt_label.shape[1]
    # case all zeros, then weight2negative = 1.0
    if weight2negative <= 0.0:
        weight2negative = 1.0
    # case all ones, then weight2negative = 0.0
    if num_positive == gt_label.shape[1]:
        weight2negative = 0.0
    weight = torch.tensor([weight2negative, 1.0 - weight2negative], dtype=torch.float32).unsqueeze(0).contiguous()
    # weight = weight.view(-1, 2, 1, 1).contiguous()
    if args.is_cuda:
        weight = weight.cuda()
    # weight = Variable(weight, requires_grad=False)
    f_loss = 10.0 * weighted_binary_focal_entropy(
        score_tensor[:, negative_indexes + positive_indexes],
        torch.tensor(gt_label, dtype=torch.float32).cuda(), weights=weight)

    return f_loss


# calculate  MOTA #

# basic operations

def rowSoftMax(MunkresOut, scale=100.0, threshold=0.7):
    """
    row wise softmax function
    :param MunkresOut: MunkresNet Output Variable Matrix of shape [batch, h, w]
    :return: row wise Softmax matrix
    """

    clutter = torch.ones(MunkresOut.size(0), MunkresOut.size(1), 1).cuda() * threshold
    return F.softmax(torch.cat([MunkresOut, clutter], dim=2)*scale,  dim=2)


def colSoftMax(MunkresOut, scale=100.0, threshold=0.7):
    """
    column wise softmax
    :param MunkresOut: MunkresNet Output Variable Matrix of shape [batch, h, w]
    :return: column wise Softmax matrix
    """
    # threshold 0.5
    # if not is_cuda:
    #     missed = Variable(torch.ones(MunkresOut.size(0), 1, MunkresOut.size(2)) * threshold, requires_grad=False)
    # else:
    missed = torch.ones(MunkresOut.size(0), 1, MunkresOut.size(2)).cuda() * threshold
    # requires grad ?
    return F.softmax(torch.cat([MunkresOut, missed], dim=1)*scale,  dim=1)


def updateCurrentListV3(softmaxed, gt_ids):
    """
    find missed objects from matrix_id, mark as -1
    :param softmaxed: column wise softmaxed munkres matrix of shape [batch, h, w]
    :param gt_ids: list of shape [number of objects] of current frame
    :return: updated gt_ids
    """
    # [batch, w]
    _, idx = torch.max(softmaxed, dim=1)
    out = gt_ids + []
    for j in range(idx.size(1)-1, -1, -1):
        if int(idx[0, j]) == (softmaxed.size(1)-1):
            out[j] = -1  # missed object

    return out


def createMatrix(index_h, index_w, size_h, size_w):
    """
    create a tensor variable fo size [size_h, size_w] having all zeros except for [index_h, index_w]
    :param index_h: list of indexes
    :param index_w:  list of indexes
    :return: a matrix, tensor variable
    """
    matrix = torch.zeros(1, size_h, size_w).cuda()

    matrix[0, index_h, index_w] = 1.0
    return matrix

# identity switching


def missedMatchErrorV3(prev_id, gt_ids, hypo_ids, colsoftmaxed, states, toUpdate):
    """
    ID switching error
    :param input: MunkresNet Output Variable Matrix of shape [batch, h, w]
    :param prev_id: dictionary of [object_id (INT): hypothesis_id(INT)] of previous frame
    :param gt_ids: torch tensor of shape [batch size, number of objects] of current frame
    :param hypo_ids: torch tensor of shape [batch size, number of hypothesis] of current frame
    :return: #id_switching, mask for motp
    """
    # softmaxed = colsoftmaxed
    pre_id = copy.deepcopy(prev_id)
    updated_gt_ids = updateCurrentListV3(colsoftmaxed, gt_ids)
    id_switching = torch.zeros(1).float().cuda()
    # remove the row of missed class
    softmaxed = colsoftmaxed[:, :-1, :]

    toputOne_h = []
    toputOne_w = []

    # to record hypo ids needed to switch target images ex. [1,2, 3,4, 5,6....]
    toswitch = []

    for w in range(len(updated_gt_ids)):
        _, idx = torch.max(softmaxed[0, :, w], dim=0)
        if updated_gt_ids[w] == -1 or (gt_ids[w] not in pre_id.keys()):  # lost object or new object or both
            # print("gt id is lost:", gt_ids[w])
            if gt_ids[w] in pre_id.keys():  # not new object but lost

                if pre_id[gt_ids[w]] in hypo_ids:
                    tmp = list(range(len(hypo_ids)))
                    tmp.pop(hypo_ids.index(pre_id[gt_ids[w]]))
                    id_switching = id_switching + torch.sum(softmaxed[0, tmp, w])
                    # print("mm is here")
                    # print(w)
                    # print(tmp)

                else:
                    # print("i am here")
                    id_switching = id_switching + torch.sum(softmaxed[0, :, w])

            elif updated_gt_ids[w] != -1:  # new object but not lost
                # add object id to prev_id, update prev id to current
                toputOne_w.append(w)
                toputOne_h.append(int(idx))
                pre_id[updated_gt_ids[w]] = hypo_ids[int(idx)] + 0

            else:  # new object and lost
                id_switching = id_switching + torch.sum(softmaxed[0, :, w])

        # if object w is not assigned to an target int(hypo_ids[idx])
        # same as previous target pre_id[int(updated_gt_ids[w])]
        elif pre_id[updated_gt_ids[w]] != hypo_ids[int(idx)]:
            if pre_id[updated_gt_ids[w]] in hypo_ids:
                tmp_idx = hypo_ids.index(pre_id[updated_gt_ids[w]])  # index of previous hypo id to this gt_id
                toputOne_w.append(w)
                toputOne_h.append(tmp_idx)  # we minimize old hypo track
                tmp = list(range(len(hypo_ids)))
                tmp.pop(tmp_idx)
                id_switching = id_switching + torch.sum(softmaxed[0, tmp, w])
                # switch target templates
                if toUpdate and pre_id[updated_gt_ids[w]] not in toswitch:  # if pair not yet switched

                    toswitch.append(pre_id[updated_gt_ids[w]])
                    toswitch.append(hypo_ids[int(idx)])
                    state_to_switch = states[pre_id[updated_gt_ids[w]]]
                    states[pre_id[updated_gt_ids[w]]] = states[hypo_ids[int(idx)]]
                    states[hypo_ids[int(idx)]] = state_to_switch
                    del state_to_switch

            else:
                id_switching = id_switching + torch.sum(softmaxed[0, :, w])  # todo wrong
                toputOne_w.append(w)
                toputOne_h.append(int(idx))  # todo

            # update prev id to current
            pre_id[updated_gt_ids[w]] = hypo_ids[int(idx)] + 0

        else:  # no id switching, no missed, prev_id[updated_gt_ids[w]] == hypo_ids[int(idx)]
            tmp = list(range(len(hypo_ids)))
            tmp.pop(int(idx))
            id_switching = torch.sum(softmaxed[0, tmp, w]) + id_switching
            toputOne_w.append(w)
            toputOne_h.append(int(idx))

    mask_for_matrix = createMatrix(toputOne_h, toputOne_w, softmaxed.size(1), softmaxed.size(2))

    return [id_switching, mask_for_matrix, pre_id]


# false negatives

def missedObjectPerframe(colsoftmaxed):
    """
    Frame wise FN != average FN of the sequence
    :param input:  MunkresNet Output Variable Matrix of shape [batch, h, w]
    :return: [number of false negatives(missed), normalized_fn]
    """
    fn = torch.sum(colsoftmaxed[:, -1, :])
    return fn


# false positives


def falsePositivePerFrame(rowsoftmax):
    """
    Frame wise FP != average FP of the sequence
    :param input:  MunkresNet Output Variable Matrix of shape [batch, h, w]
    :return: [number of false positives(alarms), normalized_fp]
    """
    fp = torch.sum(rowsoftmax[:, :, -1])
    return fp


# calculate MOTP #
def deepMOTPperFrame(D, rowsoftmaxed):
    """
    Frame wise MOTP != average MOTP of the sequence
    :param input:  MunkresNet Output Variable Matrix of shape [batch, h, w]
    :param D: distance matrix before inputting to MunkresNet of shape [batch, h, w]
    :return: [sum of distance, matched_objects, approximation of mean metric MOTP for THIS FRAME]
    """
    distance = D*rowsoftmaxed
    sum_distance = torch.sum(distance.view(1, -1), dim=1)
    # +eps preventing zero division
    matched_objects = torch.sum(rowsoftmaxed.view(1, -1), dim=1) + 1e-8
    return [sum_distance, matched_objects]


