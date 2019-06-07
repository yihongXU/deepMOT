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

import os
import csv
import argparse
from utils.sot_utils import *
from models.DAN import build_sst
import torch.backends.cudnn as cudnn
from os.path import realpath, dirname
from utils.DAN_utils import TrackUtil
from models.siamrpn import SiamRPNvot
from utils.io_utils import read_txt_detV2
from utils.mot_utils import tracking_birth_death
from utils.tracking_config import tracking_config
from utils.box_utils import getWarpMatrix, bb_fast_IOU_v1, mix_track_detV2

# tracking mode, no gradient is needed. #
torch.set_grad_enabled(False)
cudnn.benchmark = False
cudnn.deterministic = True


def main(args, sot_tracker, sst):
    train_test = ["/train/", "/test/"]
    for t_t in train_test:

        pth = args.data_root+args.dataset+t_t
        nmspth = args.dets_path+args.dataset+t_t
        videos = os.listdir(pth)

        for vname in videos:

            # if result exists, skip #
            if os.path.exists(args.save_path + args.save_dir + '/' + vname + '.txt'):
                continue

            # load MOT configuration #
            to_refine, to_combine, DAN_th, death_count, birth_wait, loose_assignment, \
            case1_interpolate, interpolate_flag, CMC, birth_iou = tracking_config(vname, args.dataset)

            print("tracking video: ")
            print(vname)

            csv_towrite = []  # list to create final tracking results
            track_init = []  # near online track init

            # load image paths #
            imgs_path = pth + vname + '/img1/'
            imgs = sorted(os.listdir(imgs_path))

            # load clean detections #
            if os.path.exists(nmspth + vname + '/det/det.npy'):
                frames_det = np.load(nmspth + vname + '/det/det.npy').item()
            else:
                frames_det = read_txt_detV2(nmspth + vname + '/det/det.txt')

            if len(frames_det.keys()) == 0:
                print("cannot load detections")
                break

            # tracking #

            # previous numpy frame
            img_prev = None

            # track id counter
            count_ids = 0

            # bbox_track = {frame_id: [[bbox], [bbox], [bbox]..]} dict of torch tensor with shape
            # [num_tracks, 4=(x1,y1,x2,y2)]
            bbox_track = dict()

            # id track = ordered [hypo_id1, hypo_id2, hypo_id3...] corresponding to bbox_track
            # of current frame, torch tensor [num_tracks]
            id_track = list()

            # states = {track_id: state, ...}
            states = dict()

            # previous frame id
            prev_frame_id = 0

            # birth candidates, {frames_id:[to birth id(=det_index)...], ...}
            birth_candidates = dict()

            # death candidates, {track_id:num_times_track_lost,lost_track ...}
            death_candidates = dict()

            # collect_prev_pos = {trackId:[postion_before_lost=[x1,y1,x2,y2], track_appearance_features,
            # matched_count, matched_det_collector(frame, det_id),
            # track_box_collector=[[frameid,[x,y,x,y]],...],'active' or 'inactive', velocity, inactive_pre_pos]}
            collect_prev_pos = dict()

            bbox_track[prev_frame_id] = None

            to_interpolate = dict()

            pre_warp_matrix = None
            w_matrix = None

            for frameid, im_pth in enumerate(imgs):
                print("frameid: ", frameid+1)

                img_curr = cv2.imread(os.path.join(imgs_path, im_pth))
                h, w, _ = img_curr.shape

                # tracking for current frame #

                # having active tracks
                if len(states) > 0:
                    tmp = []
                    im_prev_features = TrackUtil.convert_image(img_prev.copy())

                    # calculate affine transformation for current frame and previous frame
                    if img_prev is not None and CMC:
                        w_matrix = getWarpMatrix(img_curr, img_prev)

                    # FOR every track in PREVIOUS frame
                    for key, state_curr in states.items():
                        # center position at frame t-1
                        prev_pos = state_curr['target_pos'].copy()
                        prev_size = state_curr['target_sz'].copy()

                        prev_xyxy = [prev_pos[0] - 0.5 * prev_size[0],
                                     prev_pos[1] - 0.5 * prev_size[1],
                                     prev_pos[0] + 0.5 * prev_size[0],
                                     prev_pos[1] + 0.5 * prev_size[1]]

                        if state_curr['gt_id'] not in collect_prev_pos.keys():

                            # extract image features by DAN
                            prev_xywh = [prev_pos[0] - 0.5 * prev_size[0], prev_pos[1] - 0.5 * prev_size[1],
                                         prev_size[0], prev_size[1]]
                            prev_xywh = np.array([prev_xywh], dtype=np.float32)
                            prev_xywh[:, [0, 2]] /= float(w)
                            prev_xywh[:, [1, 3]] /= float(h)
                            track_norm_center = TrackUtil.convert_detection(prev_xywh)

                            tracks_features = sst.forward_feature_extracter(im_prev_features,
                                                                            track_norm_center).detach_()

                            collect_prev_pos[state_curr['gt_id']] = [[[frameid-1, np.array(prev_xyxy)]],
                                                                     [[frameid-1, tracks_features.clone()]], 0,
                                                                     list(), list(), 'active', [0.0, -1.0, -1.0],
                                                                     np.zeros((4))-1]
                            del tracks_features

                        elif collect_prev_pos[state_curr['gt_id']][5] == 'active':

                            # extract image features by DAN
                            prev_xywh = [prev_pos[0] - 0.5 * prev_size[0], prev_pos[1] - 0.5 * prev_size[1],
                                         prev_size[0], prev_size[1]]
                            prev_xywh = np.array([prev_xywh], dtype=np.float32)
                            prev_xywh[:, [0, 2]] /= float(w)
                            prev_xywh[:, [1, 3]] /= float(h)
                            track_norm_center = TrackUtil.convert_detection(prev_xywh)
                            tracks_features = sst.forward_feature_extracter(im_prev_features,
                                                                            track_norm_center).detach_()

                            # update positions and appearance features of active track
                            collect_prev_pos[state_curr['gt_id']][0].append([frameid-1, np.array(prev_xyxy)])

                            # only keep the latest 10 active positions (used for estimating velocity for interpolations)
                            if len(collect_prev_pos[state_curr['gt_id']][0]) > 10:
                                collect_prev_pos[state_curr['gt_id']][0].pop(0)

                            # only keep the latest 3 appearance features (used for recovering invisible tracks)
                            collect_prev_pos[state_curr['gt_id']][1].append([frameid-1, tracks_features.clone()])
                            if len(collect_prev_pos[state_curr['gt_id']][1]) > 3:
                                collect_prev_pos[state_curr['gt_id']][1].pop(0)
                            del tracks_features

                            # remove pre_lost_pos when a track is recovered
                            collect_prev_pos[state_curr['gt_id']][7] = np.zeros((4))-1

                            # update velocity during active mode if we have 10 (might be not consecutive) positions
                            if len(collect_prev_pos[state_curr['gt_id']][0]) == 10:
                                avg_h = 0.0
                                avg_w = 0.0
                                for f, pos in collect_prev_pos[state_curr['gt_id']][0]:
                                    avg_h += (pos[3] - pos[1])
                                    avg_w += (pos[2] - pos[0])
                                avg_h /= len(collect_prev_pos[state_curr['gt_id']][0])
                                avg_w /= len(collect_prev_pos[state_curr['gt_id']][0])
                                last_t, last_pos = collect_prev_pos[state_curr['gt_id']][0][-1]
                                first_t, first_pos = collect_prev_pos[state_curr['gt_id']][0][0]
                                # center point
                                first_pos_center = np.array([0.5 * (first_pos[0] + first_pos[2]),
                                                             0.5 * (first_pos[1] + first_pos[3])])
                                last_pos_center = np.array([0.5 * (last_pos[0]+last_pos[2]),
                                                            0.5 * (last_pos[1]+last_pos[3])])
                                velocity = (last_pos_center - first_pos_center)/(last_t-first_t)
                                collect_prev_pos[state_curr['gt_id']][6] = [velocity, avg_h, avg_w]
                                collect_prev_pos[state_curr['gt_id']][0] = [collect_prev_pos[state_curr['gt_id']][0][-1]]

                        else:
                            # inactive mode, do nothing
                            pass

                        target_pos, target_sz, state_curr, _ = SiamRPN_track(state_curr, img_curr.copy(), sot_tracker,
                                                                            train=True, CMC=(img_prev is not None and CMC),
                                                                             prev_xyxy=prev_xyxy, w_matrix=w_matrix)
                        tmp.append(torch.stack([target_pos[0] - target_sz[0]*0.5,
                                                target_pos[1] - target_sz[1]*0.5,
                                                target_pos[0] + target_sz[0]*0.5,
                                                target_pos[1] + target_sz[1]*0.5], dim=0).detach_().unsqueeze(0))
                        del _
                        del target_pos
                        del target_sz
                        torch.cuda.empty_cache()

                    bbox_track[frameid] = torch.cat(tmp, dim=0).detach_()

                    del bbox_track[prev_frame_id]
                    del tmp
                    torch.cuda.empty_cache()

                else:
                    # having no tracks
                    bbox_track[frameid] = None

                # refine and calculate "distance" (actually, iou) matrix #
                if str(frameid + 1) in frames_det.keys():
                    distance = []
                    if bbox_track[frameid] is not None:
                        bboxes = bbox_track[frameid].detach().cpu().numpy().tolist()
                        for bbox in bboxes:
                            IOU = bb_fast_IOU_v1(bbox, frames_det[str(frameid + 1)])
                            distance.append(IOU.tolist())
                        distance = np.vstack(distance)

                        # refine tracks bboxes with dets if iou > 0.6
                        if to_combine:
                            del bboxes
                            # mix dets and tracks boxes
                            bbox_track[frameid] = mix_track_detV2(
                                torch.FloatTensor(distance).cuda(),
                                torch.FloatTensor(frames_det[str(frameid + 1)]).cuda(), bbox_track[frameid])

                            boxes = bbox_track[frameid].detach().cpu().numpy().tolist()
                            for idx, [key, state] in enumerate(states.items()):
                                # print(idx, key, state['gt_id'])
                                box = boxes[idx]
                                state['target_pos'] = np.array([0.5*(box[2] + box[0]), 0.5*(box[3] + box[1])])
                                state['target_sz'] = np.array([box[2] - box[0], box[3] - box[1]])
                                states[key] = state
                            distance = []
                            bboxes = bbox_track[frameid].detach().cpu().numpy().tolist()
                            for bbox in bboxes:
                                IOU = bb_fast_IOU_v1(bbox, frames_det[str(frameid + 1)])
                                distance.append(IOU.tolist())
                            distance = np.vstack(distance)

                    # no tracks
                    else:
                        distance = np.array(distance)

                    # birth and death process, no need to be differentiable #

                    # print([DAN_th, death_count, birth_iou, birth_wait, to_combine, to_refine, interpolate_flag, case1_interpolate, loose_assignment, interpolate_flag])

                    bbox_track[frameid], count_ids = \
                        tracking_birth_death(distance, bbox_track[frameid], frames_det, img_curr,
                                             id_track, count_ids, frameid, birth_candidates, track_init,
                                             death_candidates, states, sot_tracker, collect_prev_pos, sst, th=0.5,
                                             birth_iou=birth_iou, to_refine=to_refine, DAN_th=DAN_th,
                                             death_count=death_count, birth_wait=birth_wait,
                                             to_interpolate=to_interpolate, interpolate_flag=interpolate_flag,
                                             loose_assignment=loose_assignment, case1_interpolate=case1_interpolate)

                    del distance

                    # save current frame results into a txt files for evaluation #
                    if bbox_track[frameid] is not None:
                        bbox_torecord = bbox_track[frameid].detach().cpu().numpy().tolist()
                        for j in range(len(bbox_torecord)):
                            if id_track[j] not in death_candidates.keys():
                                # x1,y1,x2,y2 to x1,y1,w,h
                                torecord = copy.deepcopy(bbox_torecord[j])
                                torecord[2] = torecord[2] - torecord[0]
                                torecord[3] = torecord[3] - torecord[1]
                                towrite = [str(frameid + 1)]
                                towrite += [str(elem) for elem in ([id_track[j]+1] + torecord)]
                                towrite += ['-1', '-1', '-1', '-1']
                                csv_towrite.append(towrite)

                else:
                    print("no detections! all tracks killed.")
                    bbox_track[frameid] = None
                    id_track = list()
                    states = dict()
                    death_candidates = dict()
                    collect_prev_pos = dict()

                img_prev = img_curr.copy()
                prev_frame_id = frameid
                torch.cuda.empty_cache()

            # save interpolations into a txt files for evaluation #
            for lst in track_init:
                for frame_id, det_id, track_id in lst:
                    if not isinstance(det_id, np.ndarray):
                        tmp_box = copy.deepcopy(frames_det[str(frame_id+1)][det_id])
                    else:
                        tmp_box = det_id.tolist()
                    # x1,y1,x2,y2 to x1,y1,w,h
                    tmp_box[2] = tmp_box[2] - tmp_box[0]
                    tmp_box[3] = tmp_box[3] - tmp_box[1]
                    towrite = [str(frame_id + 1)]
                    towrite += [str(elem) for elem in ([track_id+1] + tmp_box)]
                    towrite += ['-1', '-1', '-1', '-1']
                    csv_towrite.append(towrite)

            # write txt file for evaluation #
            if not os.path.exists(args.save_path + args.save_dir +'/'):
                os.makedirs(args.save_path + args.save_dir + '/')

            with open(args.save_path + args.save_dir + '/'
                      + vname+'.txt', 'w', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter=',')
                for row in csv_towrite:
                    writer.writerow(row)

            print("tracking for {:s} is done.".format(vname))


if __name__ == '__main__':
    # init parameters #
    print("Loading parameters...")
    curr_path = realpath(dirname(__file__))
    parser = argparse.ArgumentParser(description='PyTorch MOT tracking')

    # data/models configs
    parser.add_argument('--dataset', dest='dataset', default="mot17", help='dataset name')

    parser.add_argument('--data_root', dest='data_root', default=curr_path + '/data/',
                        help='dataset root path')

    parser.add_argument('--models_root', dest='models_root', default=curr_path + '/pretrained/',
                        help='pretrained models root path')

    parser.add_argument('--dets_path', dest='dets_path', default=curr_path + '/clean_detections/',
                        help='detections root path')

    parser.add_argument('--save_path', dest='save_path', default=curr_path + '/saved_results/txts/',
                        help='saving path for txt results')

    parser.add_argument('--save_dir', dest='save_dir', default='test_folder',
                        help='saving dir name for txt results')

    # tracking configs
    parser.add_argument('--is_cuda', dest='is_cuda', type=bool, default=True, help='use GPU?')

    args = parser.parse_args()

    # init sot tracker #
    print("loading trained tracker from: ")
    print(args.models_root + 'trainedSOTtoMOT.pth')
    sot_tracker = SiamRPNvot()
    sot_tracker.load_state_dict(torch.load(args.models_root + 'trainedSOTtoMOT.pth'))

    # init appearance model #
    print("loading appearance model from: ")
    print(args.models_root + 'DAN.pth')
    sst = build_sst('test', 900)
    sst.load_state_dict(torch.load(args.models_root + 'DAN.pth'))

    if args.is_cuda:
        sot_tracker.cuda()
        sst.cuda()

    # evaluation mode #
    sot_tracker.eval()
    sst.eval()

    # run #
    main(args, sot_tracker, sst)