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

import torch.nn.functional as F
import torch
from utils.noise_utils import *
from utils.box_utils import warpcoordinates
import copy

# tracking config; generate anchors; crop reference/search images #


def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = to_torch(img)
    return img


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        # return torch.from_numpy(ndarray)
        return torch.FloatTensor(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


class TrackerConfig(object):
    """
    siamrpn tracking configurations
    """

    # These are the default hyper-params for DaSiamRPN 0.3827
    windowing = 'cosine'  # to penalize large displacements [cosine/uniform]
    # Params from the network architecture, have to be consistent with the training
    exemplar_size = 127  # input z size
    instance_size = 271  # input x size (search region)
    total_stride = 8
    score_size = (instance_size-exemplar_size)/total_stride+1
    context_amount = 0.5  # context amount for the exemplar
    ratios = [0.33, 0.5, 1, 2, 3]
    # ratios = [1.0, 1.6, 2.2, 2.8, 3.4]

    scales = [8, ]
    anchor_num = len(ratios) * len(scales)
    anchor = []
    penalty_k = 0.055
    window_influence = 0.42
    lr = 0.295
    # adaptive change search region #
    adaptive = True

    def update(self, cfg):
        for k, v in cfg.items():
            setattr(self, k, v)
        self.score_size = (self.instance_size - self.exemplar_size) / self.total_stride + 1


def generate_anchor(total_stride, scales, ratios, score_size):
    anchor_num = len(ratios) * len(scales)
    anchor = np.zeros((anchor_num, 4),  dtype=np.float32)
    size = total_stride * total_stride
    count = 0
    for ratio in ratios:
        # ws = int(np.sqrt(size * 1.0 / ratio))
        ws = int(np.sqrt(size / ratio))
        hs = int(ws * ratio)
        for scale in scales:
            wws = ws * scale
            hhs = hs * scale
            anchor[count, 0] = 0
            anchor[count, 1] = 0
            anchor[count, 2] = wws
            anchor[count, 3] = hhs
            count += 1

    anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
    ori = - (score_size / 2) * total_stride
    xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                         [ori + total_stride * dy for dy in range(score_size)])
    xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
             np.tile(yy.flatten(), (anchor_num, 1)).flatten()
    anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
    return anchor


def get_subwindow_tracking(im, pos, model_sz, original_sz, avg_chans, out_mode='torch', noisy_bool=False):
    if isinstance(pos, float):
        pos = [pos, pos]
    sz = original_sz
    im_sz = im.shape
    c = (original_sz + 1) / 2
    context_xmin = round(pos[0] - c)  # floor(pos(2) - sz(2) / 2);
    context_xmax = context_xmin + sz - 1
    context_ymin = round(pos[1] - c)  # floor(pos(1) - sz(1) / 2);
    context_ymax = context_ymin + sz - 1
    left_pad = int(max(0., -context_xmin))
    top_pad = int(max(0., -context_ymin))
    right_pad = int(max(0., context_xmax - im_sz[1] + 1))
    bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))
    # make the crop region within the image
    context_xmin = context_xmin + left_pad
    context_xmax = context_xmax + left_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + top_pad

    # zzp: a more easy speed version
    r, c, k = im.shape
    if any([top_pad, bottom_pad, left_pad, right_pad]):
        te_im = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad, k),
                         np.uint8)  # 0 is better than 1 initialization
        te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
        if top_pad:
            te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
        if bottom_pad:
            te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
        if left_pad:
            te_im[:, 0:left_pad, :] = avg_chans
        if right_pad:
            te_im[:, c + left_pad:, :] = avg_chans
        im_patch_original = te_im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]
    else:
        im_patch_original = im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]

    if not np.array_equal(model_sz, original_sz):
        im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))  # zzp: use cv to get a better speed
    else:
        im_patch = im_patch_original

    if noisy_bool:
        #  todo image noise& blur
        if np.random.rand() <= 0.25:
            im_patch = constant_change_luminance(im_patch)

        if np.random.rand() <= 0.25:
            im_patch = blur_img(im_patch)

        if np.random.rand() <= 0.25:
            noise_type = np.random.choice(["gauss", "s&p"])
            # print(type(im_patch))
            im_patch = noisy(noise_type, im_patch)

    return im_to_torch(im_patch) if out_mode in 'torch' else im_patch


# init a track and save it's appearance and current position in a dict #


def SiamRPN_init(im, target_pos, target_sz, net, gt_id, train_bool=False, outputmode='torch', add_noise=False):
    """
    init a track and save its reference image
    :param im: current frame, numpy array, [h, w, c]
    :param target_pos: target center position at previous time step t-1, numpy array, [c_x, c_y]
    :param target_sz: target bounding box size at previous time step t-1, numpy array, [w, h]
    :param net: SOT network, torch model
    :param gt_id: identity for the track to be initialized, int or String
    :param train_bool: train mode or not, bool
    :param outputmode: 'torch', String
    :param add_noise: add noise flag, for training data augmentation, bool
    :return: state dict saving track information, dict
    """
    state = dict()
    p = TrackerConfig()
    p.update(net.cfg)
    state['im_h'] = im.shape[0]
    state['im_w'] = im.shape[1]

    if p.adaptive:
        if ((target_sz[0] * target_sz[1]) / float(state['im_h'] * state['im_w'])) < 0.004:
            p.instance_size = 287  # small object big search region
        else:
            p.instance_size = 271

        p.score_size = (p.instance_size - p.exemplar_size) / p.total_stride + 1

    p.anchor = generate_anchor(p.total_stride, p.scales, p.ratios, int(p.score_size))

    avg_chans = np.mean(im, axis=(0, 1))

    wc_z = target_sz[0] + p.context_amount * sum(target_sz)
    hc_z = target_sz[1] + p.context_amount * sum(target_sz)
    s_z = round(np.sqrt(wc_z * hc_z))
    # initialize the exemplar
    if outputmode=='torch':
        z_crop = get_subwindow_tracking(im, target_pos, p.exemplar_size, s_z, avg_chans, noisy_bool=add_noise)
        z = z_crop.unsqueeze(0).cuda()
    else:
        z = get_subwindow_tracking(im, target_pos, p.exemplar_size, s_z, avg_chans, out_mode=outputmode, noisy_bool=add_noise)

    if p.windowing == 'cosine':
        window = np.outer(np.hanning(p.score_size), np.hanning(p.score_size))
    elif p.windowing == 'uniform':
        window = np.ones((p.score_size, p.score_size))
    window = np.tile(window.flatten(), p.anchor_num)

    # state['track_id'] = track_id
    assert (isinstance(gt_id, str) and train_bool==True) or (isinstance(gt_id, int) and train_bool==False)
    state['gt_id'] = gt_id
    if train_bool:
        state['temple'] = z
    else:
        # state['temple'] = z
        state['temple'] = net.temple(z)
    state['p'] = p
    state['avg_chans'] = avg_chans
    state['window'] = window
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    return state


# predict current position of the track by previous information in the state #


def SiamRPN_track(state, im, net, train=False, noisy_bool=False, CMC=False, prev_xyxy=None, w_matrix=None):
    """
    track target object
    :param state: information of the track at t-1; dict
    :param im: current frame, numpy array, [h, w, c]
    :param net: SOT network, torch model
    :param train: train mode or not, bool
    :param noisy_bool: add noise flag, for training data augmentation, bool
    :param cmc: camera motion compensation flag, for moving camera videos, bool
    :param prev_xyxy: previous position of the track, before warping, numpy array
    :param w_matrix: afine transformation matrix between frame t and t-1, numpy array
    :return: updated state (predicted position and size, etc.) of the track
    """

    p = state['p']
    avg_chans = state['avg_chans']
    window = state['window']
    target_pos = state['target_pos']
    target_sz = state['target_sz']
    im_h, im_w, _ = im.shape

    if CMC:
        prev_xyxy_warp = warpcoordinates(np.array([prev_xyxy], dtype=np.float32), w_matrix)

        assert prev_xyxy_warp.shape[1] == 4
        prev_pos_warp = np.array([0.5 * (prev_xyxy_warp[0, 0] + prev_xyxy_warp[0, 2]),
                                  0.5 * (prev_xyxy_warp[0, 1] + prev_xyxy_warp[0, 3])])

        w_warp = prev_xyxy_warp[0, 2] - prev_xyxy_warp[0, 0]
        h_warp = prev_xyxy_warp[0, 3] - prev_xyxy_warp[0, 1]

        prev_pos_warp[0] = max(0, min(im_w, prev_pos_warp[0]))
        prev_pos_warp[1] = max(0, min(im_h, prev_pos_warp[1]))
        w_warp = max(10, min(im_w, w_warp))
        h_warp = max(10, min(im_h, h_warp))

        target_sz = np.array([w_warp, h_warp])
        target_pos = prev_pos_warp

    wc_z = target_sz[1] + p.context_amount * sum(target_sz)
    hc_z = target_sz[0] + p.context_amount * sum(target_sz)
    s_z = np.sqrt(wc_z * hc_z)
    # scale_z = real image -> 127 scale
    scale_z = p.exemplar_size / s_z
    d_search = (p.instance_size - p.exemplar_size) / 2  #under the scale of exemplar_size
    pad = d_search / scale_z
    s_x = s_z + 2 * pad

    x_crop = get_subwindow_tracking(im, target_pos, p.instance_size, round(s_x), avg_chans, noisy_bool=noisy_bool).unsqueeze(0)

    # print(x_crop.shape)
    if train:

        target_position, target_size, score_single, score_tensor, ancrs = tracker_train(net, x_crop.cuda(), target_pos,
                                                                                 target_sz * scale_z, window, scale_z,
                                                                                 p, state['temple'], im)
        del x_crop
        torch.cuda.empty_cache()

        # variables inside state, they are numpy not differentiable
        target_pos_numpy = target_position.detach().cpu().numpy().copy()
        target_sz_numpy = target_size.detach().cpu().numpy().copy()
        target_pos_numpy[0] = max(0, min(state['im_w'], target_pos_numpy[0]))
        target_pos_numpy[1] = max(0, min(state['im_h'], target_pos_numpy[1]))
        target_sz_numpy[0] = max(10, min(state['im_w'], target_sz_numpy[0]))
        target_sz_numpy[1] = max(10, min(state['im_h'], target_sz_numpy[1]))
        state['target_pos'] = target_pos_numpy
        state['target_sz'] = target_sz_numpy

        state['score'] = score_single
        return target_position, target_size, state, [score_tensor, ancrs]

    else:
        target_pos, target_sz, score = tracker_eval(net, x_crop.cuda(), target_pos, target_sz * scale_z, window, scale_z, p, state['temple'])
        target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
        target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
        target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
        target_sz[1] = max(10, min(state['im_h'], target_sz[1]))
        # inside state, it is not differentiable
        state['target_pos'] = target_pos
        state['target_sz'] = target_sz
        state['score'] = score
        return state


def tracker_train(net, x_crop, target_pos, target_sz, window, scale_z, p, t_plate, im):
    if isinstance(t_plate, list):
        delta, score = net(x_crop, t_plate[0], t_plate[1])
    elif isinstance(t_plate, np.ndarray):
        tmp = net.temple(im_to_torch(t_plate).unsqueeze(0).cuda())
        delta, score = net(x_crop, tmp[0], tmp[1])

    else:
        # print('i am here')
        tmp = net.temple(t_plate.cuda())
        delta, score = net(x_crop, tmp[0], tmp[1])
    # window = torch.FloatTensor(window.tolist()).cuda()
    anchor = torch.tensor(p.anchor, dtype=torch.float32).cuda()

    delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
    score = F.softmax(score.permute(1, 2, 3, 0).contiguous().view(2, -1), dim=0)
    score_numpy = score.detach()[1, :].cpu().numpy().copy()
    proposals = torch.zeros_like(delta, requires_grad=False).cuda()
    proposals[0, :] += delta[0, :] * anchor[:, 2] + anchor[:, 0]
    proposals[1, :] += delta[1, :] * anchor[:, 3] + anchor[:, 1]
    proposals[2, :] += torch.exp(delta[2, :]) * anchor[:, 2]
    proposals[3, :] += torch.exp(delta[3, :]) * anchor[:, 3]

    # print(proposals.requires_grad)

    proposals_numpy = proposals.detach().cpu().numpy().copy()

    def change(r):
        return np.maximum(r, 1./r)
        # return r if r.item() >= 1./r.item() else 1.0/r

    def sz(w, h):
        pad = (w + h) * 0.5
        sz2 = (w + pad) * (h + pad)
        return np.sqrt(sz2+1e-12)

    def sz_wh(wh):
        pad = (wh[0] + wh[1]) * 0.5
        sz2 = (wh[0] + pad) * (wh[1] + pad)
        return np.sqrt(sz2+1e-12)

    # size penalty todo for the moment it has not grad, see as a constant
    s_c = change(sz(proposals_numpy[2, :], proposals_numpy[3, :]) / (sz_wh(target_sz)))  # scale penalty
    r_c = change((target_sz[0] / target_sz[1]) / (proposals_numpy[2, :] / proposals_numpy[3, :]))  # ratio penalty

    penalty = np.exp(-(r_c * s_c - 1.) * p.penalty_k)
    pscore = penalty * score_numpy

    pscore = pscore * (1 - p.window_influence) + window * p.window_influence
    best_pscore_id = np.argmax(pscore)

    target = proposals[:, best_pscore_id] / scale_z
    target_sz = target_sz / scale_z
    #TODO lr NOW IS A CONSTANT, NO GRADS
    lr = float(penalty[best_pscore_id] * score_numpy[best_pscore_id] * p.lr)

    # print(target)
    # print(type(lr))
    res_x = target[0] + float(target_pos[0])
    res_y = target[1] + float(target_pos[1])

    res_w = float(target_sz[0]) * (1 - lr) + target[2] * lr
    res_h = float(target_sz[1]) * (1 - lr) + target[3] * lr

    # anchors reprojected to original image
    ancrs = p.anchor.copy()
    ancrs /= scale_z
    ancrs[:, 0] += target_pos[0]
    ancrs[:, 1] += target_pos[1]
    # cx, cy, w, h to xyxy
    ancrs[:, 0] -= 0.5 * ancrs[:, 2]
    ancrs[:, 1] -= 0.5 * ancrs[:, 3]
    ancrs[:, 2] += ancrs[:, 0]
    ancrs[:, 3] += ancrs[:, 1]

    target_pos = torch.stack([res_x, res_y], dim=0)
    target_sz = torch.stack([res_w, res_h], dim=0)
    return target_pos, target_sz, score_numpy[best_pscore_id], score, ancrs


def tracker_eval(net, x_crop, target_pos, target_sz, window, scale_z, p, template):
    delta, score = net(x_crop, template[0], template[1])

    delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1).data.cpu().numpy()
    score = F.softmax(score.permute(1, 2, 3, 0).contiguous().view(2, -1), dim=0).data[1, :].cpu().numpy()

    delta[0, :] = delta[0, :] * p.anchor[:, 2] + p.anchor[:, 0]
    delta[1, :] = delta[1, :] * p.anchor[:, 3] + p.anchor[:, 1]
    delta[2, :] = np.exp(delta[2, :]) * p.anchor[:, 2]
    delta[3, :] = np.exp(delta[3, :]) * p.anchor[:, 3]

    def change(r):
        return np.maximum(r, 1./r)

    def sz(w, h):
        pad = (w + h) * 0.5
        sz2 = (w + pad) * (h + pad)
        return np.sqrt(sz2)

    def sz_wh(wh):
        pad = (wh[0] + wh[1]) * 0.5
        sz2 = (wh[0] + pad) * (wh[1] + pad)
        return np.sqrt(sz2)

    # size penalty
    s_c = change(sz(delta[2, :], delta[3, :]) / (sz_wh(target_sz)))  # scale penalty
    r_c = change((target_sz[0] / target_sz[1]) / (delta[2, :] / delta[3, :]))  # ratio penalty

    penalty = np.exp(-(r_c * s_c - 1.) * p.penalty_k)
    pscore = penalty * score

    # window float
    pscore = pscore * (1 - p.window_influence) + window * p.window_influence
    best_pscore_id = np.argmax(pscore)

    target = delta[:, best_pscore_id] / scale_z
    target_sz = target_sz / scale_z
    lr = penalty[best_pscore_id] * score[best_pscore_id] * p.lr

    res_x = target[0] + target_pos[0]
    res_y = target[1] + target_pos[1]

    res_w = target_sz[0] * (1 - lr) + target[2] * lr
    res_h = target_sz[1] * (1 - lr) + target[3] * lr

    target_pos = np.array([res_x, res_y])
    target_sz = np.array([res_w, res_h])
    return target_pos, target_sz, score[best_pscore_id]


# update target reference image #


def update_target_image_train(predicted, id_track, dets, img_curr, states, SOT_tracker, prev_id=None):
    """
    :param predicted: assignment matrix {0，1} for track_detection association, numpy array, [batch, H, W]
    :param id_track: current tracking ids, list
    :param dets: current detection (or gt) boxes [[id, bbox], [id, bbox]], list, [num_boxes, 5]
    :param img_curr: current frame, numpy array, [h,w,c]
    :param states: dict of tracks' states, dict
    :param SOT_tracker: single object tracker, torch network
    :param pre_id: AP, historic tracks-gts associations
    :return: updated states
    """

    h, w, _ = img_curr.shape

    for i in range(predicted.shape[1]):
        if np.max(predicted[0, i, :]) == 0.0:  # lost tracks
            continue
        which_det_id = np.argmax(predicted[0, i, :])

        bbox = dets[which_det_id][-4:]

        #  todo bbox Data Augmentation
        bbox_to_crop = copy.deepcopy(bbox)
        if np.random.rand() <= -1.0:
            bbox_to_crop = shift_box(bbox_to_crop, h, w)

        if np.random.rand() <= 0.2:
            bbox_to_crop = scale_box(bbox_to_crop, h, w)

        # init new object
        cx, cy, target_w, target_h = 0.5 * (bbox_to_crop[0] + bbox_to_crop[2]), 0.5 * (bbox_to_crop[1] + bbox_to_crop[3]), \
                       (bbox_to_crop[2] - bbox_to_crop[0]), (bbox_to_crop[3] - bbox_to_crop[1])

        target_pos, target_sz = np.array([cx, cy]), np.array([target_w, target_h])

        state = SiamRPN_init(img_curr.copy(), target_pos, target_sz, SOT_tracker, dets[which_det_id][0], True)

        states[id_track[i]] = state  # create a new id
        if prev_id is not None:
            prev_id[int(dets[which_det_id][0])] = id_track[i]

    return states