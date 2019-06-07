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

from utils.sot_utils import *
from utils.DAN_utils import TrackUtil
from utils.box_utils import bb_fast_IOU_v1

# MOT functions #


def complete_out_of_view(to_check_box, im_w, im_h):
    """
    check if the bounding box is completely out of view from the image
    :param to_check_box: bounding box, numpy array, [x, y, x, y]
    :param im_w: image width, int
    :param im_h:image height, int
    :return: out of view flag, bool
    """
    complete_OofV_left = (to_check_box[0] < 0) and (to_check_box[2] < 0)
    complete_OofV_top = (to_check_box[1] < 0) and (to_check_box[3] < 0)
    complete_OofV_right = (to_check_box[2] > im_w) and (to_check_box[0] > im_w)
    complete_OofV_bottom = (to_check_box[1] > im_h) and (to_check_box[3] > im_h)
    complete_OofV = complete_OofV_left or complete_OofV_top or complete_OofV_right or complete_OofV_bottom
    return complete_OofV


# Malisiewicz et al.
def nms_fast_apperance_as_score(boxes, scores, overlapThresh):
    """
    non maximum suppression
    :param boxes: bounding boxes [x, y, x, y], numpy array, [num_bboxes,4]
    :param scores: appearance score between previous and current reference image, numpy array, [num_bboxes]
    :param overlapThresh: nms threshold, float
    :return: selected bounding boxes
    """
    # boxes = np.array(copy.deepcopy(boxes), dtype=np.float32)[:, -4:]
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    assert boxes.shape[0] == len(scores)

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        # i = real index of a bbox
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick], pick

# birth and death during training, with gts. #


def easy_birth_deathV4_rpn(munkres, bbox_track, det_boxes, curr_img, id_track, count_ids, states, SOT_tracker,
                           pre_id=None):
    """
    birth and death process during training
    :param munkres: assignment matrix {0，1} for track_detection association, numpy array, [batch, H, W]
    :param bbox_track: current tracks bboxes, torch Float tensor, [num_track, 4]
    :param curr_img: current frame, numpy array, [h,w,c]
    :param det_boxes: current detection (or gt) boxes [[id, bbox], [id, bbox]], list, [num_boxes, 5]
    :param id_track: current tracking ids, list
    :param count_ids: total used identities counter, int
    :param states: dict of tracks' states, dict
    :param SOT_tracker: single object tracker, torch network
    :param pre_id: AP, historic tracks-gts associations

    :return: updated bbox_track, count_ids, no_tracks_flag (all tracks lost flag)
    """
    no_tracks_flag = False

    h, w, _ = curr_img.shape

    # ids to die
    sum_row = np.sum(munkres, axis=2)[0]
    to_die = np.where(sum_row == 0.0)[0].tolist()

    # ids to birth
    sum_col = np.sum(munkres, axis=1)[0]
    to_birth = np.where(sum_col == 0.0)[0].tolist()

    # give birth, use det bbox to init track
    for idx in to_birth:
        # init bbox

        bbox = det_boxes[idx][-4:]

        # #  todo bbox Data Augmentation
        bbox_to_crop = copy.deepcopy(bbox)
        if np.random.rand() <= -1.0:
            bbox_to_crop = shift_box(bbox_to_crop, h, w)

        if np.random.rand() <= 0.4:
            bbox_to_crop = scale_box(bbox_to_crop, h, w)

        # init new object
        cx, cy, target_w, target_h = 0.5 * (bbox_to_crop[0] + bbox_to_crop[2]), 0.5 * (bbox_to_crop[1] + bbox_to_crop[3]), \
                       (bbox_to_crop[2] - bbox_to_crop[0]), (bbox_to_crop[3] - bbox_to_crop[1])

        target_pos, target_sz = np.array([cx, cy]), np.array([target_w, target_h])

        curr_img_tosave = curr_img.copy()

        # todo image noise& blur
        if np.random.rand() <= 0.2:
            curr_img_tosave = constant_change_luminance(curr_img_tosave)

        if np.random.rand() <= 0.2:
            curr_img_tosave = blur_img(curr_img_tosave)

        if np.random.rand() <= 0.2:
            noise_type = np.random.choice(["gauss", "s&p"])
            curr_img_tosave = noisy(noise_type, curr_img_tosave)

        state = SiamRPN_init(curr_img_tosave, target_pos, target_sz, SOT_tracker, det_boxes[idx][0], True)
        states[count_ids] = state

        bbox_track = torch.cat([bbox_track, torch.FloatTensor(bbox).cuda().unsqueeze(0)], dim=0)
        id_track.append(count_ids)
        if pre_id is not None:
            pre_id[int(det_boxes[idx][0])] = count_ids
        count_ids += 1

    # put to death
    if len(to_die) > 0:
        to_die = sorted(to_die, reverse=True)

    for idx in to_die:
        id = id_track[idx]
        del states[id]
        id_track.pop(idx)
    torch.cuda.empty_cache()
    res = list(set(to_die) ^ set(list(range(bbox_track.size(0)))))
    bbox_track = torch.index_select(bbox_track, 0, torch.LongTensor(res).cuda())

    return bbox_track, count_ids, no_tracks_flag

# birth and death during tracking, with dets. #


def tracking_birth_death(distance,  bbox_track, det_boxes, curr_img, id_track,
                       count_ids, frameid, birth_candidates, toinit_tracks, death_candidates,
                       states, sot_tracker, collect_prev_pos, sst, th,
                        birth_iou=0.4, death_count=30, iou_th_overlap = 0.4, to_refine=False, DAN_th=0.5,birth_wait=3,
                        to_interpolate=None, interpolate_flag=-1.0, loose_assignment=False, case1_interpolate = True):
    """
    :param distance: IOU between [track, det]  [num_track, num_dets]
    :param velocity_track: {person_id: velocity...}, velocity_track at time t, keep track of current target velocity
    :param bbox_track: torch tensor of shape [num_track, 4], keep track of current target bboxes
    :param det_boxes: current detection [gt] boxes [[(id,) bbox], [(id,) bbox]]
    :param curr_img: numpy array current img to get target crop to give birth
    :param id_track: track ids for current frame
    :param count_ids: total num track ids counter
    :param frameid: current frameid id (int, starts with 0)
    :param birth_candidates: dict for recording candidates to birth, {frameid:[det_id, ...],...}
    :param toinit_tracks: record to birth for previous frames (near online process) [[frameid, det_box_id, track_id], ...]
    :param th: threshold to do
    :return: updated bbox_track, count_ids, no_tracks_flag
    """
    # print("birth and death frameid", frameid)

    # for CURRENT birth process
    det = det_boxes[str(frameid + 1)]
    max_frameid = np.max(np.array(list(det_boxes.keys()), dtype=np.float32))
    im_h, im_w, _ = curr_img.shape

    im_curr_features = TrackUtil.convert_image(curr_img.copy())

    # track ids to die
    to_die = []

    # iou>0.5 and dan >0.7
    to_recover_track_idxes = []
    # det id associated with not_to_die id will not be born
    not_a_birth_candidate = []
    if distance.shape[0] != 0:
        # assigned each detection to a track {track_index:[det_id, iou]}
        track_dets = dict()
        det_ids_iou_lower_th = list()
        for detection_id in range(distance.shape[1]):
            track_indexes = np.where(distance[:, detection_id] >= th)[0]  # == i
            # print(track_indexes)
            if len(track_indexes) > 0:
                # det having some th >= 0.5
                if len(track_indexes) > 1:
                    # records states
                    s = list()
                    for track_index in track_indexes:
                        s.append(collect_prev_pos[id_track[track_index]][5])
                    # print(s)
                    # we have both inactive and active tracks inside,
                    # we prefer assign to an active track
                    if 'inactive' in s and 'active' in s:
                        # print(track_indexes)
                        for idxx in range(len(s)):
                            if s[idxx] == 'inactive':
                                # print('entering here')
                                # print(idxx)
                                track_indexes = np.delete(track_indexes, [idxx])
                    else:
                        pass
                else:
                    pass

                if not loose_assignment:
                    rest_IOUs = [distance[track_index, detection_id] for track_index in track_indexes]
                    # print(rest_IOUs)
                    max_iou_idx = track_indexes[np.argmax(rest_IOUs)]
                    if max_iou_idx not in track_dets.keys():
                        track_dets[max_iou_idx] = [[detection_id, np.max(rest_IOUs)]]

                    else:
                        track_dets[max_iou_idx].append([detection_id, np.max(rest_IOUs)])
                else:
                    for max_iou_idx in track_indexes:
                        if max_iou_idx not in track_dets.keys():
                            track_dets[max_iou_idx] = [[detection_id, distance[max_iou_idx, detection_id]]]

                        else:
                            track_dets[max_iou_idx].append([detection_id, distance[max_iou_idx, detection_id]])


            else:
                # det with all ious <0.5
                det_ids_iou_lower_th.append(detection_id)
                to_allocate_index = np.argmax(distance[:, detection_id])
                iou_max = distance[to_allocate_index, detection_id]

                if to_allocate_index not in track_dets.keys():
                    track_dets[to_allocate_index] = [[detection_id, iou_max]]
                else:
                    track_dets[to_allocate_index].append([detection_id, iou_max])

        # for track with no det assigned, iou = None
        for track_idx in range(distance.shape[0]):
            if track_idx not in track_dets.keys():
                track_dets[track_idx] = [[None, None]]

    for i in range(distance.shape[0]):
        ious_for_i = track_dets[i]
        if ious_for_i[0][1] is None:
            # no det assigned to i
            max_iou = None
            associated_det_id = None
        else:
            # some dets assigned to i( iou >= 0.5 or <0.5)
            where_max_iou = np.argmax(np.array(ious_for_i)[:, 1])
            associated_det_id, max_iou = ious_for_i[where_max_iou]

        if max_iou is None or max_iou < th:  # lost(inactive) tracks

            if interpolate_flag > 0:

                # out of view compensation
                # (1) check velocity
                if collect_prev_pos[id_track[i]][6][-1] != -1 and (id_track[i] not in death_candidates
                                                                   or death_candidates[id_track[i]] < interpolate_flag)\
                        and id_track[i] not in to_interpolate:

                    center_velo, avg_h, avg_w = collect_prev_pos[id_track[i]][6]
                    # (2) check out of view
                    # xyxy
                    to_check_box = bbox_track[i, :].cpu().numpy().copy()
                    # left, top, right, bottom out of view
                    condition_left = (to_check_box[0] < 0) and (to_check_box[2]> 0)
                    condition_top = (to_check_box[1] < 0) and (to_check_box[3] > 0)
                    condition_right = (to_check_box[2] > im_w) and (to_check_box[0] < im_w)
                    condition_bottom = (to_check_box[1] < im_h) and (to_check_box[3] > im_h)

                    if condition_left or condition_top or condition_bottom or condition_right:
                        # check velocity direction going left ?
                        if id_track[i] not in death_candidates:
                            max_comp = interpolate_flag
                            begin_frameid = frameid
                        else:
                            max_comp = interpolate_flag + death_candidates[id_track[i]]
                            begin_frameid = frameid - death_candidates[id_track[i]]
                        com_counter = 0
                        complete_OofV = complete_out_of_view(to_check_box, im_w, im_h)

                        # toinit_tracks = [[frameid, det_box_id, track_id], ...]
                        # interpolate
                        # for lst_to_copy in collect_prev_pos[id_track[to_recover]][4]:
                        #     toinit_tracks.append([copy.deepcopy(lst_to_copy) + [id_track[to_recover]]])

                        pre_pos = collect_prev_pos[id_track[i]][0][-1][-1].copy()
                        to_comp = list()
                        while com_counter<max_comp and not complete_OofV and begin_frameid+com_counter < max_frameid:
                            # print("prev pos", pre_pos)
                            # print(toinit_tracks)
                            cx, cy = 0.5 * (pre_pos[0] + pre_pos[2])+com_counter*center_velo[0], \
                                     0.5 * (pre_pos[1] + pre_pos[3])+com_counter*center_velo[1]
                            comp_box = np.array([cx-avg_w*0.5, cy-avg_h*0.5, cx+avg_w*0.5, cy+avg_h*0.5])
                            complete_OofV = complete_out_of_view(comp_box, im_w, im_h)
                            to_comp.append([[begin_frameid+com_counter, comp_box, id_track[i]]])
                            com_counter += 1

                        if com_counter <= max_comp and (begin_frameid+com_counter) < max_frameid:
                            to_interpolate[id_track[i]] = copy.deepcopy(to_comp)

            # normal process
            collect_prev_pos[id_track[i]][5] = 'inactive'

            if id_track[i] not in death_candidates.keys():
                death_candidates[id_track[i]] = 1
            else:
                death_candidates[id_track[i]] += 1
            # wait 10 times before death
            if death_candidates[id_track[i]] >= death_count:
                to_die.append(i)
                del death_candidates[id_track[i]]
                del collect_prev_pos[id_track[i]]
            else:
                # death times < 30
                # enter inactive mode, keep only the last position
                # collect_prev_pos[id_track[i]][0] = [collect_prev_pos[id_track[i]][0][-1]]

                # get velocity, moving forward with a constant velocity
                if collect_prev_pos[id_track[i]][6][-1] != -1:
                    # if velocity is available
                    # print(collect_prev_pos[id_track[i]][6])
                    center_velo, avg_h, avg_w = collect_prev_pos[id_track[i]][6]
                    # if just entered inactive mode
                    if collect_prev_pos[id_track[i]][7][-1] == -1:
                        #RECORD FIRST LOST POS
                        pre_pos = collect_prev_pos[id_track[i]][0][-1][-1].copy()
                        # print("pre pos", pre_pos)
                    else:
                        pre_pos = collect_prev_pos[id_track[i]][7].copy()
                    cx, cy = 0.5 * (pre_pos[0] + pre_pos[2]), 0.5 * (pre_pos[1] + pre_pos[3])
                    # todo check ordering of states == ordering of box
                    # force inactive object to move in a constant velocity as before, with a fixed bbox
                    states[id_track[i]]['target_pos'] = np.array([cx, cy])+center_velo
                    states[id_track[i]]['target_sz'] = np.array([avg_w, avg_h])
                    # update position in inactive mode
                    # print(center_velo)
                    inactive_pos = [(cx+center_velo[0])-avg_w*0.5, (cy+center_velo[1])-avg_h*0.5,\
                                    (cx+center_velo[0])+avg_w*0.5, (cy+center_velo[1])+avg_h*0.5]
                    collect_prev_pos[id_track[i]][7] = np.array(inactive_pos).copy()

                # collect track trajectory during inactive mode
                collect_prev_pos[id_track[i]][4].append([frameid, bbox_track[i, :].detach().cpu().numpy().copy()])
                # check case 2
                if max_iou is not None and 0.3 < max_iou < 0.5:
                    # compare appearance with track appearance at time before his death
                    # current closest detection
                    if DAN_th > 0.0:
                        detection = np.array([det[associated_det_id][-4:]], dtype=np.float32)
                        #todo check shape

                        detection[:, 2] = detection[:, 2] - detection[:, 0]
                        detection[:, 3] = detection[:, 3] - detection[:, 1]
                        detection[:, [0, 2]] /= float(im_w)
                        detection[:, [1, 3]] /= float(im_h)
                        detection_center = TrackUtil.convert_detection(detection)
                        det_features = sst.forward_feature_extracter(im_curr_features, detection_center).detach_()
                        # detection in column, track in row
                        affinity_mat_after_softmax = 0
                        penalty = 0
                        for old_frame, i_features in collect_prev_pos[id_track[i]][1]:
                            penalty += 0.995**((frameid-old_frame)/3.0)

                            affinity_mat_after_softmax += ((0.995**((frameid-old_frame)/3.0))*sst.forward_stacker_features\
                                (i_features, det_features, False, toNumpy=True)[:, :-1])
                        affinity_mat_after_softmax /= penalty
                    else:
                        affinity_mat_after_softmax = 1.0

                    if float(affinity_mat_after_softmax) >= DAN_th:

                        collect_prev_pos[id_track[i]][2] += 1
                        # collect this associated det box_index and frameid
                        collect_prev_pos[id_track[i]][3].append([frameid, associated_det_id])

                        if collect_prev_pos[id_track[i]][2] == 3:
                            # case 2 condition satisfied
                            # not a birth candidate
                            # correct track with det
                            collected_dets = collect_prev_pos[id_track[i]][3]
                            avg_box = None
                            for f_id, det_index in collected_dets:
                                if avg_box is None:
                                    avg_box = np.array(det_boxes[str(f_id+1)][det_index][-4:]).astype(np.float32)
                                else:
                                    avg_box += np.array(det_boxes[str(f_id+1)][det_index][-4:]).astype(np.float32)

                                # remove this detection from birth candidates
                                if f_id in birth_candidates.keys() and det_index in birth_candidates[f_id]:
                                    del birth_candidates[f_id][birth_candidates[f_id].index(det_index)]
                                if f_id == frameid and det_index not in not_a_birth_candidate:
                                    #not a birth candidate
                                    not_a_birth_candidate.append(det_index)



                            # reinit this track with mean corresponding bboxes of three frames
                            avg_box /=3.0
                            #todo check avg_box shape
                            # print(avg_box.shape)
                            cx, cy, w, h = 0.5 * (avg_box[0] + avg_box[2]), 0.5 * (avg_box[1] + avg_box[3]), \
                                           (avg_box[2] - avg_box[0]), (avg_box[3] - avg_box[1])

                            # todo check ordering of states == ordering of box
                            states[id_track[i]] = SiamRPN_init(curr_img, np.array([cx, cy]),
                                                     np.array([w, h]), sot_tracker, states[id_track[i]]['gt_id'])

                            # correct the collected track trajectory during inactive mode
                            collect_prev_pos[id_track[i]][4][-1] = [frameid, avg_box]
                            collect_prev_pos[id_track[i]][7] = avg_box.copy()


                            # update bbox_track
                            bbox_track[i][0] = float(avg_box[0])
                            bbox_track[i][1] = float(avg_box[1])
                            bbox_track[i][2] = float(avg_box[2])
                            bbox_track[i][3] = float(avg_box[3])

                        else:
                            # still need to wait until 3 frames
                            pass
                    else:
                        # dan < 0.7
                        # clean up matched_counter and det boxes
                        collect_prev_pos[id_track[i]][2] = 0
                        collect_prev_pos[id_track[i]][3] = list()


                else:
                    # iou <0.3
                    # clean up matched_counter and det boxes
                    collect_prev_pos[id_track[i]][2] = 0
                    collect_prev_pos[id_track[i]][3] = list()

        elif id_track[i] in death_candidates:
            if DAN_th > 0.0:
                detection = np.array([det[associated_det_id][-4:]], dtype=np.float32)
                # todo check shape

                detection[:, 2] = detection[:, 2] - detection[:, 0]
                detection[:, 3] = detection[:, 3] - detection[:, 1]
                detection[:, [0, 2]] /= float(im_w)
                detection[:, [1, 3]] /= float(im_h)
                detection_center = TrackUtil.convert_detection(detection)
                det_features = sst.forward_feature_extracter(im_curr_features, detection_center).detach_()
                # detection in column, track in row
                affinity_mat_after_softmax = 0
                penalty = 0
                for old_frame, i_features in collect_prev_pos[id_track[i]][1]:
                    penalty += 0.995 ** ((frameid - old_frame) / 3.0)
                    affinity_mat_after_softmax += ((0.995 ** ((frameid - old_frame) / 3.0)) *
                                                   sst.forward_stacker_features(i_features, det_features, False,
                                                                                toNumpy=True)[:,:-1])
                affinity_mat_after_softmax/= penalty
            else:
                affinity_mat_after_softmax = 1.0

            if affinity_mat_after_softmax >= DAN_th:
                # case 1
                # track becomes active
                collect_prev_pos[id_track[i]][5] = 'active'
                collected_dets = collect_prev_pos[id_track[i]][3]
                for f_id, det_index in collected_dets:
                    if f_id in birth_candidates.keys() and det_index in birth_candidates[f_id]:
                        del birth_candidates[f_id][birth_candidates[f_id].index(det_index)]

                not_a_birth_candidate.append(associated_det_id)
                if id_track[i] in to_interpolate.keys():
                    del to_interpolate[id_track[i]]

                #TODO CHECKPOINT
                # recover tracks later
                to_recover_track_idxes.append(i)

            else:
                # not case 1 because of DAN < 0.7
                # still in inactive mode
                collect_prev_pos[id_track[i]][5] = 'inactive'
                if id_track[i] not in death_candidates.keys():
                    death_candidates[id_track[i]] = 1
                else:
                    death_candidates[id_track[i]] += 1
                # wait 10 times before death
                if death_candidates[id_track[i]] >= death_count:
                    to_die.append(i)
                    del death_candidates[id_track[i]]
                    del collect_prev_pos[id_track[i]]
                else:
                    # death times < 10
                    # collect track trajectory during inactive mode
                    collect_prev_pos[id_track[i]][4].append([frameid, bbox_track[i, :].detach().cpu().numpy().copy()])

                    # DAN < 0.7
                    # clean up matched_counter and det boxes
                    collect_prev_pos[id_track[i]][2] = 0
                    collect_prev_pos[id_track[i]][3] = list()
                    # iou_ok_DAN_not_ok.append(np.argmax(distance[i, :]))
        else:
            # never died before and iou>0.5
            pass

    if bbox_track is not None:
        # nms for those boxes in active mode
        now_track = bbox_track.detach().cpu().numpy().copy()

        # find all active mode candidates
        candidates_index = []
        for indx in range(now_track.shape[0]):
            if indx in to_die or collect_prev_pos[id_track[indx]][5] == 'inactive':
                continue
            else:
                candidates_index.append(indx)
        # print('before', candidates_index)

        if len(candidates_index) > 0:
            self_apperance_scores = [-1.0*id_track[corres_index] for corres_index in candidates_index]
            _, pick = nms_fast_apperance_as_score(now_track[candidates_index, :], self_apperance_scores, 0.5)
            to_deactivate_index = list(set(pick) ^ set(list(range(len(candidates_index)))))

            # index in id_track or box_track
            to_deactivate = [candidates_index[to_pick] for to_pick in to_deactivate_index]

            for inactive_idx in to_deactivate:
                if inactive_idx in to_recover_track_idxes:
                    del to_recover_track_idxes[to_recover_track_idxes.index(inactive_idx)]
                collect_prev_pos[id_track[inactive_idx]][5] = 'inactive'
                if id_track[inactive_idx] not in death_candidates.keys():
                    death_candidates[id_track[inactive_idx]] = 1
                else:
                    death_candidates[id_track[inactive_idx]] += 1
                # wait 10 times before death
                if death_candidates[id_track[inactive_idx]] >= death_count:
                    to_die.append(inactive_idx)
                    del death_candidates[id_track[inactive_idx]]
                    del collect_prev_pos[id_track[inactive_idx]]
                else:
                    # death times < 30
                    # collect track trajectory during inactive mode
                    # todo get velocity, moving forward with a constant velocity
                    if collect_prev_pos[id_track[inactive_idx]][6][-1] != -1:
                        # if velocity is available
                        center_velo, avg_h, avg_w = collect_prev_pos[id_track[inactive_idx]][6]
                        # if just entered inactive mode
                        if collect_prev_pos[id_track[inactive_idx]][7][-1] == -1:
                            # RECORD FIRST LOST POS
                            pre_pos = collect_prev_pos[id_track[inactive_idx]][0][-1][-1].copy()
                        else:
                            pre_pos = collect_prev_pos[id_track[inactive_idx]][7].copy()
                        cx, cy = 0.5 * (pre_pos[0] + pre_pos[2]), 0.5 * (pre_pos[1] + pre_pos[3])
                        # todo check ordering of states == ordering of box
                        # force inactive object to move in a constant velocity as before, with a fixed bbox
                        states[id_track[inactive_idx]]['target_pos'] = np.array([cx, cy]) + center_velo
                        states[id_track[inactive_idx]]['target_sz'] = np.array([avg_w, avg_h])
                        # update position in inactive mode
                        inactive_pos = [(cx + center_velo[0]) - avg_w * 0.5, (cy + center_velo[1]) - avg_h * 0.5,
                                        (cx + center_velo[0]) + avg_w * 0.5, (cy + center_velo[1]) + avg_h * 0.5]
                        collect_prev_pos[id_track[inactive_idx]][7] = np.array(inactive_pos).copy()

                    collect_prev_pos[id_track[inactive_idx]][4].append([frameid, bbox_track[inactive_idx, :].detach().cpu().numpy().copy()])

                    # clean up matched_counter and det boxes
                    collect_prev_pos[id_track[inactive_idx]][2] = 0
                    collect_prev_pos[id_track[inactive_idx]][3] = list()
                    # iou_ok_DAN_not_ok.append(np.argmax(distance[inactive_idx, :]))# todo ???

            for to_recover in to_recover_track_idxes:
                first_frame = collect_prev_pos[id_track[to_recover]][4][0][0]-1
                current_frame = frameid + 0
                first_pos = collect_prev_pos[id_track[to_recover]][0][-1][-1].copy()
                if to_refine:
                    ious_for_to_recover = track_dets[to_recover]
                    # print("refiine track")
                    where_max_iou = np.argmax(np.array(ious_for_to_recover)[:, 1])
                    associated_det_id, _ = ious_for_to_recover[where_max_iou]
                    curr_pos = np.array(det[associated_det_id])

                    cx, cy, w, h = 0.5 * (curr_pos[0] + curr_pos[2]), 0.5 * (curr_pos[1] + curr_pos[3]), \
                                   (curr_pos[2] - curr_pos[0]), (curr_pos[3] - curr_pos[1])

                    # todo check ordering of states == ordering of box
                    states[id_track[to_recover]] = SiamRPN_init(curr_img, np.array([cx, cy]),
                                                       np.array([w, h]), sot_tracker,
                                                                states[id_track[to_recover]]['gt_id'])

                    # update bbox_track
                    bbox_track[to_recover][0] = float(curr_pos[0])
                    bbox_track[to_recover][1] = float(curr_pos[1])
                    bbox_track[to_recover][2] = float(curr_pos[2])
                    bbox_track[to_recover][3] = float(curr_pos[3])
                else:
                    curr_pos = bbox_track[to_recover, :].detach().cpu().numpy()

                velocity = (curr_pos - first_pos)/(current_frame-first_frame)
                if case1_interpolate:
                    for framerate in range(frameid-first_frame-1):
                        toinit_tracks.append([[framerate+first_frame+1, first_pos+(framerate+1)*velocity, id_track[to_recover]]])
                # track becomes active
                collect_prev_pos[id_track[to_recover]][5] = 'active'
                # clean up matched_counter and det boxes
                collect_prev_pos[id_track[to_recover]][2] = 0
                collect_prev_pos[id_track[to_recover]][3] = list()
                collect_prev_pos[id_track[to_recover]][4] = list()
                del death_candidates[id_track[to_recover]]

    # print("**********************")
    # print("todie id:", [id_track[i] for i in to_die])

    # clean all tracks that are out of view

    for i in range(distance.shape[0]):
        if i not in to_die:
            to_check_sz = states[id_track[i]]['target_sz']
            if to_check_sz[0] > im_w and to_check_sz[1] > im_h:
                to_die.append(i)
                del death_candidates[id_track[i]]
                del collect_prev_pos[id_track[i]]


    # det ids to birth
    to_birth = []
    if distance.shape[0] != 0:  # distance matrix is empty

        # update candidates_index
        # find all active mode candidates
        candidates_index = []
        for indx in range(distance.shape[0]):
            if indx in to_die or collect_prev_pos[id_track[indx]][5] == 'inactive':
                continue
            else:
                candidates_index.append(indx)

        if len(candidates_index) > 0:
            # print('after', candidates_index)
            distance_pick = distance[candidates_index, :].copy()
            for j in range(distance_pick.shape[1]):
                if np.max(distance_pick[:, j]) < th and j not in not_a_birth_candidate:  # lost dets
                    to_birth.append(j)
    else:
        to_birth = list(range(len(det)))

    # check birth candidates before give birth
    new_tobirth = []

    # print("to_birth", to_birth)
    # if frameid < 2: #todo verify
    if len(birth_candidates.keys()) < birth_wait:
        birth_candidates[frameid] = to_birth + []
    else:
        not_assigned = []
        # track overlapped bbox det_ids {idx1: [[frameid1, histo_detid1], [frameid2, histo_detid2]], ...}
        associated_bbox_detid = {}

        for idx in to_birth:
            associated_bbox_detid[idx] = []
            bbox = [] + det[idx][-4:]  # current det bbox for det_id = idx

            # for all three continuous frames, check IOU with current frame to_birth candidate
            for frame, compare_lst in birth_candidates.items():
                # for all dets in one previous frame
                tmp = np.array([])  # tmp variable to record IOU
                if len(compare_lst):  # not an empty list
                    old_bbox = [det_boxes[str(frame + 1)][histo_det_idx] for histo_det_idx in compare_lst]

                    tmp = bb_fast_IOU_v1(bbox, old_bbox)

                if tmp.shape[0] == 0 or np.max(tmp) < birth_iou:

                    not_assigned.append(idx)
                    del associated_bbox_detid[idx]
                    break
                else:
                    associated_bbox_detid[idx].append([frame, compare_lst[np.argmax(tmp)]])

        # overlaps
        to_remove = []
        for current_id, v in associated_bbox_detid.items():
            for frame, histo_detid in v:
                if histo_detid in birth_candidates[frame]:
                    birth_candidates[frame].remove(histo_detid)
                else:
                    to_remove.append(current_id)
                    break

        for x in to_remove:
            # print("toremove", to_remove)
            # print("associated_bbox_detid", associated_bbox_detid)
            # print("birth_candidates", birth_candidates)
            del associated_bbox_detid[x]
        not_assigned += to_remove
        new_tobirth = list(set(not_assigned) ^ set(to_birth))
        # record detections that are not assigned
        birth_candidates[frameid] = not_assigned + []
        # only keep 2 frames' records = compare 3 frames (plus current frame).
        if len(list(birth_candidates.keys())) != 0:
            del birth_candidates[list(birth_candidates.keys())[0]]

    # print("after", new_tobirth)
    # print("**********************")

    # give birth, use det bbox to init track
    for idx in new_tobirth:
        bbox = det[idx][-4:]
        # init new object
        cx, cy, w, h = 0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3]), (bbox[2] - bbox[0]), (bbox[3] - bbox[1])

        target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
        state = SiamRPN_init(curr_img, target_pos, target_sz, sot_tracker, count_ids)
        # todo check
        states[count_ids] = state
        if bbox_track is not None:
            bbox_track = torch.cat([bbox_track, torch.FloatTensor(bbox).cuda().unsqueeze(0)], dim=0)
        else:
            bbox_track = torch.FloatTensor(bbox).cuda().unsqueeze(0)

        id_track.append(count_ids)
        # give birth for previous frames, toinit_tracks = [[frameid, det_box_id, track_id], ...]
        for element in associated_bbox_detid[idx]:
            toinit_tracks.append([element+[count_ids]])


        count_ids += 1

    # put to death
    if len(to_die) > 0:
        to_die = sorted(to_die, reverse=True)

    for idx in to_die:
        # to interpolate before being killed

        id = id_track[idx]
        if id in to_interpolate:
            toinit_tracks += to_interpolate[id]
        del states[id]
        id_track.pop(idx)

    if bbox_track is not None:
        res = list(set(to_die) ^ set(list(range(bbox_track.size(0)))))
    else:
        res = []
    if len(res) != 0:
        bbox_track = torch.index_select(bbox_track, 0, torch.LongTensor(res).cuda())
    else:
        print('all tracks died.')
        bbox_track = None
    # print(birth_candidates.keys())
    return bbox_track, count_ids

