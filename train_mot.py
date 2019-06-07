
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
import random
import shutil
import argparse
import torch.optim as optim
from models.DHN import Munkrs
from models.siamrpn import SiamRPNvot
from os.path import realpath, dirname
from tensorboardX import SummaryWriter
from utils.sot_utils import *
from utils.io_utils import *
from utils.loss import *
from utils.mot_utils import easy_birth_deathV4_rpn


def main(args, sot_tracker, deepMunkres, optimizer, mota_writer, motp_writer, clasf_writer):
    """
    train a sot to perform MOT using DeepMOT Loss
    :param args: parameters, argparse
    :param sot_tracker: single object tracker, torch network
    :param deepMunkres: deep Hungarian Net, torch network
    :param optimizer: training optimizer, torch optim
    :param mota_writer: record MOTA loss, tensorboardX writer
    :param motp_writer: record MOTP loss, tensorboardX writer
    :param clasf_writer: record classification loss, tensorboardX writer
    """

    iterations = 0
    chunks = {}
    old_loss = 100
    for epoch in range(args.epochs):
        pth = args.data_root + args.dataset + '/train/'
        print("training...")
        print("Dataset from: ", pth)
        videos = os.listdir(pth)
        random.shuffle(videos)

        for vname in videos:
            if "flip" in vname or "rot" in vname:
                continue

            print("***************************************************************")
            print(vname)
            print("***************************************************************")

            # load detections and gt bbox of this sequence
            frames_gt = read_txt_gtV2(pth + vname + '/gt/gt.txt')
            if len(frames_gt.keys()) == 0:
                print("cannot load gt")
                break

            # load image paths
            imgs_path = pth + vname + '/img1/'
            imgs = sorted(os.listdir(imgs_path))

            # cut video into small sequences of 100 frames
            if epoch == 0:
                tem = []
                for i in range(0, len(imgs), args.seq_len):
                    tem.append([i, imgs[i:i + args.seq_len]])
                chunks[vname] = tem + []
                del tem

            random.shuffle(chunks[vname])

            # for a small sequence

            for i in range(len(chunks[vname])):
                first_idx, subset = chunks[vname][i]

                # HYPOTHESIS id counter
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

                # deep Metrics
                pre_id = dict()

                no_tracks_flag = False
                no_detections_flag = False

                for frameid, im_pth in enumerate(subset):
                    distance_matrix = 0
                    frameid += first_idx

                    # no objects in this frame, then skip.
                    if str(frameid + 1) not in frames_gt.keys():
                        no_detections_flag = True
                        continue

                    if frameid == first_idx or no_tracks_flag or no_detections_flag:

                        no_detections_flag = False
                        no_tracks_flag = False
                        img_prev = cv2.imread(os.path.join(imgs_path, im_pth))

                        gt_bboxes = np.array(frames_gt[str(frameid + 1)], dtype=np.float32)
                        bbox_track[frameid] = torch.tensor(gt_bboxes[:, 1:], dtype=torch.float32).cuda()

                        # init trackers with first frame #
                        for k, bbox in enumerate(frames_gt[str(frameid + 1)]):
                            # bbox = [gtid, x1, y1, x2, y2]
                            cx, cy, target_w, target_h = 0.5 * (bbox[1] + bbox[3]), 0.5 * (bbox[2] + bbox[4]), \
                                                         (bbox[3] - bbox[1]), (bbox[4] - bbox[2])

                            target_pos, target_sz = np.array([cx, cy]), np.array([target_w, target_h])

                            state = SiamRPN_init(img_prev, target_pos, target_sz, sot_tracker, bbox[0],
                                                 train_bool=True)

                            states[count_ids+k] = state

                        # record hypo ids for the bbox_track
                        id_track = list(range(count_ids, count_ids + len(frames_gt[str(frameid + 1)])))
                        count_ids += len(frames_gt[str(frameid + 1)])
                        prev_frame_id = frameid
                        pre_id = dict(zip(gt_bboxes[:, 0].astype(np.int32).tolist(), id_track))
                        del gt_bboxes
                        continue

                    img_curr = cv2.imread(os.path.join(imgs_path, im_pth))
                    h, w, _ = img_curr.shape

                    # tracking for current frame #
                    for repeats in range(args.num_repeats):
                        tmp = []
                        gt_boxes = np.array(frames_gt[str(frameid + 1)], dtype=np.float32)
                        gt_ids = gt_boxes[:, 0].astype(np.int32).tolist()
                        focal_loss = 0.0
                        for key, state_curr in states.items():  # FOR every track in PREVIOUS frame
                            # score_tensor is of shape 2,num_anchor_boxes
                            # ancrs is of shape num_anchor_boxes,4 numpy array #xyxy
                            target_pos, target_sz, state_curr, [score_tensor, ancrs] = SiamRPN_track(state_curr, img_curr,
                                                                                     sot_tracker, train=True,
                                                                                                     noisy_bool=True)

                            tmp.append(torch.stack([target_pos[0] - target_sz[0] * 0.5,
                                                    target_pos[1] - target_sz[1] * 0.5,
                                                    target_pos[0] + target_sz[0] * 0.5,
                                                    target_pos[1] + target_sz[1] * 0.5], dim=0).unsqueeze(0))

                            # classification loss for SiamRPN class branch
                            focal_loss += focaLoss(score_tensor, ancrs, state_curr, gt_ids, gt_boxes, args)

                        bbox_track[frameid] = torch.cat(tmp, dim=0)
                        focal_loss = focal_loss/len(states.keys())

                        # get distance matrix tracks-gts #
                        gt_ids, distance_matrix = make_single_matrix_torchV2_fast(frames_gt[str(frameid + 1)],
                                                                                  bbox_track[frameid], h, w)

                        # init hidden states of DHN #
                        deepMunkres.hidden_row = deepMunkres.init_hidden(1)
                        deepMunkres.hidden_col = deepMunkres.init_hidden(1)

                        # get output from DHN, i.e. assignment matrix #
                        output_track_gt = deepMunkres(distance_matrix)

                        softmaxed_row = rowSoftMax(output_track_gt, scale=args.smax_scale).contiguous()
                        softmaxed_col = colSoftMax(output_track_gt, scale=args.smax_scale).contiguous()

                        # for MOTP and MOTA #

                        # false positives, false negatives
                        fn = missedObjectPerframe(softmaxed_col)
                        fp = falsePositivePerFrame(softmaxed_row)

                        # id switching
                        if repeats == args.num_repeats - 1:
                            mm, motp_mask, pre_id = missedMatchErrorV3(pre_id, gt_ids, id_track,
                                                                             softmaxed_col, states,
                                                                             toUpdate=True)
                        else:
                            mm, motp_mask, _ = missedMatchErrorV3(pre_id, gt_ids, id_track,
                                                                        softmaxed_col, states,
                                                                        toUpdate=False)

                        total_objects = float(distance_matrix.size(2))

                        # sum of distances among matched objects and gts
                        sum_distance, matched_objects = deepMOTPperFrame(distance_matrix, motp_mask)

                        total_matched_objs = float(matched_objects)

                        motp = sum_distance / total_matched_objs

                        mota = (fn+fp+mm) / total_objects

                        loss = mota+5.0*motp + 10.0*focal_loss

                        # loss backward and update weights
                        sot_tracker.zero_grad()
                        loss.backward()
                        optimizer.step()

                        if repeats < args.num_repeats-1:
                            del mm
                            del fp
                            del fn
                            del sum_distance
                            del output_track_gt
                            del softmaxed_col
                            del softmaxed_row
                            del _
                            torch.cuda.empty_cache()

                    del bbox_track[prev_frame_id]
                    # free gpu memory
                    torch.cuda.empty_cache()

                    prev_frame_id = frameid

                    # update reference images
                    if (frameid + 1) % args.ref_freq == 0:
                        states = update_target_image_train(motp_mask.detach().cpu().numpy().copy(), id_track,
                                                                  frames_gt[str(frameid + 1)], img_curr, states,
                                                                 sot_tracker)

                    bbox_track[frameid], count_ids, no_tracks_flag = \
                        easy_birth_deathV4_rpn(motp_mask.detach().cpu().numpy().copy(), bbox_track[frameid],
                                               frames_gt[str(frameid + 1)], img_curr, id_track, count_ids, states,
                                               sot_tracker, pre_id)

                    # save best model #
                    if (iterations + 1) % args.save_freq == 0 and old_loss > loss.item():
                        old_loss = float(loss.item())
                        print("best model is saved into:", args.save_path +
                              "best_model_" + str(epoch) + ".pth")

                        torch.save(sot_tracker.state_dict(),
                                   args.save_path+"best_model_" + str(epoch) + ".pth")

                    # print results #
                    if (iterations + 1) % args.print_freq == 0:
                        print('Epoch: [{}] Iterations: [{}]\tLoss {:.4f}'.format(epoch, iterations, float(loss.item())))

                        mota_writer.add_scalar('Loss', mota.item(), iterations)
                        motp_writer.add_scalar('Loss', motp.item(), iterations)
                        clasf_writer.add_scalar('Loss', focal_loss.item(), iterations)

                        # save model #
                        if (iterations + 1) % (args.save_freq * 20) == 0:
                            print("model is saved into:", args.save_path +
                                  "model_" + str(epoch) + ".pth")

                            torch.save(sot_tracker.state_dict(), args.save_path + "model_" + str(epoch) + ".pth")

                    iterations += 1

                    # clean up
                    del mm
                    del fp
                    del fn
                    del sum_distance
                    del output_track_gt
                    del softmaxed_col
                    del softmaxed_row
                    del distance_matrix

                    torch.cuda.empty_cache()


if __name__ == '__main__':

    # init parameters #
    print("Loading parameters...")
    curr_path = realpath(dirname(__file__))
    parser = argparse.ArgumentParser(description='PyTorch DeepMOT train')

    # data configs
    parser.add_argument('--dataset', dest='dataset', default="mot17", help='dataset name')

    parser.add_argument('--logs', dest='logs', default=curr_path + '/logs/',
                        help='logs path')

    parser.add_argument('--data_root', dest='data_root', default= curr_path + '/data/',
                        help='dataset root path')

    parser.add_argument('--models_root', dest='models_root',
                        default=curr_path + '/pretrained/',
                        help='pretrained models root path')

    # BiRNN configs

    parser.add_argument('--element_dim', dest='element_dim', default=1, type=int, help='element_dim')
    parser.add_argument('--hidden_dim', dest='hidden_dim', default=256, type=int, help='hidden_dim')
    parser.add_argument('--target_size', dest='target_size', default=1, type=int, help='target_size')
    parser.add_argument('--batch_size', dest='batch_size', default=1, type=int, help='batch_size')
    parser.add_argument('--bidrectional', dest='bidrectional', default=True, type=bool, help='bidrectional')

    # training configs
    parser.add_argument('--is_cuda', dest='is_cuda', default=True, type=bool, help='use cuda?')
    parser.add_argument('--seq_len', dest='seq_len', default=100, type=int, help='small sequence length')
    parser.add_argument('--epochs', dest='epochs', default=30, type=int, help='number of training epochs')
    parser.add_argument('--old_lr', dest='old_lr', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--print_freq', dest='print_freq', default=200, type=int, help='print loss frequency')
    parser.add_argument('--ref_freq', dest='ref_freq', default=10, type=int, help='update reference images frequency')
    parser.add_argument('--smax_scale', dest='smax_scale', default=10.0, type=float, help='softmax scaling factor')
    parser.add_argument('--save_freq', dest='save_freq', default=20, type=int, help='save model weights frequency')
    parser.add_argument('--save_path', dest='save_path', default=curr_path + '/saved_models/', help='save_path')

    parser.add_argument('--num_repeats', dest='num_repeats', default=3, type=int,
                        help='train a frame for how many times')

    args = parser.parse_args()

    # init sot tracker #
    sot_tracker = SiamRPNvot()
    print("loading SOT from: ", args.models_root + 'SiamRPNVOT.model')
    sot_tracker.load_state_dict(torch.load(args.models_root + 'SiamRPNVOT.model'))

    # freeze first three conv layers (feature extraction layers)
    cntr = 0
    to_freeze = [0, 1, 4, 5, 8, 9]
    for child in sot_tracker.children():
        cntr += 1
        if cntr <= 1:
            for i, param in enumerate(child.parameters()):
                # print(i, param.shape)
                if i in to_freeze:
                    param.requires_grad = False

    # init optimizer #
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, sot_tracker.parameters()), lr=args.old_lr)

    # init munkres net #
    print("loading DHN from: ", args.models_root + "DHN.pth")
    deepMunkres = Munkrs(element_dim=args.element_dim, hidden_dim=args.hidden_dim, target_size=args.target_size,
                         biDirenction=args.bidrectional, minibatch=args.batch_size, is_cuda=args.is_cuda,
                         is_train=False)
    model_dict = torch.load(args.models_root + "DHN.pth")
    deepMunkres.load_state_dict(model_dict)

    # use gpu #
    if args.is_cuda:
        deepMunkres = deepMunkres.cuda()
        sot_tracker.cuda()

    # TensorboardX logs #
    print("creating logs files...")
    print("log path: ", args.logs + 'train_log')
    if os.path.exists(args.logs + 'train_log'):
        shutil.rmtree(args.logs + 'train_log')
    mota_writer = SummaryWriter(args.logs + 'train_log/mota')
    motp_writer = SummaryWriter(args.logs + 'train_log/motp')
    clasf_writer = SummaryWriter(args.logs + 'train_log/clasf')

    main(args, sot_tracker, deepMunkres, optimizer, mota_writer, motp_writer, clasf_writer)
