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
import cv2
import argparse
import motmetrics
import numpy as np
from utils.io_utils import *
from os.path import realpath, dirname
from utils.box_utils import bb_fast_IOU_v1

mh = motmetrics.metrics.create()


def main(args):
    txtes = os.listdir(args.txts_path)

    print("##################")
    print(args.txts_path)
    print("##################")

    total_fn = 0
    total_fp = 0
    total_idsw = 0
    total_num_objects = 0
    total_matched = 0
    sum_distance = 0

    for txt in txtes:
        vname = txt[:-4]

        if not os.path.exists(args.data_root + args.dataset + '/train/' + vname + "/gt/gt.txt"):
            continue

        print(vname)

        acc = motmetrics.MOTAccumulator(auto_id=True)
        # load detections and gt bbox of this sequence

        frames_gt = read_txt_gtV2(args.data_root + args.dataset + '/train/' + vname + "/gt/gt.txt")
        if len(frames_gt.keys()) == 0:
            print("cannot load gts")
        imgs = sorted(os.listdir(args.data_root + args.dataset + '/train/' + vname + '/img1'))
        h, w, _ = cv2.imread(args.data_root + args.dataset + '/train/' + vname + '/img1/000001.jpg').shape

        frames_prdt = read_txt_predictionV2(args.txts_path+txt)
        if len(frames_prdt.keys()) == 0:
            print("cannot load detections")

        # evaluations

        for frameid in frames_gt.keys():
            # print("frameid: ", int(frameid)+1)
            # get gt ids
            gt_bboxes = np.array(frames_gt[frameid], dtype=np.float32)
            gt_ids = gt_bboxes[:, 0].astype(np.int32).tolist()

            if frameid in frames_prdt.keys():
                # get id track
                id_track = np.array(frames_prdt[frameid])[:, 0].astype(np.int32).tolist()
                # get a binary mask from IOU, 1.0 if iou < 0.5, else 0.0
                mask_IOU = np.zeros((len(frames_prdt[frameid]), len(frames_gt[frameid])))
                # distance matrix
                distance_matrix = []
                for i, bbox in enumerate(frames_prdt[frameid]):
                    iou = bb_fast_IOU_v1(bbox, frames_gt[frameid])
                    # threshold
                    th = np.zeros_like(iou)
                    th[np.where(iou <= args.threshold)] = 1.0
                    mask_IOU[i, :] = th

                    # distance
                    distance_matrix.append(1.0-iou)

                distance_matrix = np.vstack(distance_matrix)

                distance_matrix[np.where(mask_IOU == 1.0)] = np.nan

                acc.update(
                    gt_ids,  # number of objects = matrix width
                    id_track,  # number of hypothesis = matrix height
                    np.transpose(distance_matrix)
                )

            else:
                acc.update(
                    gt_ids,  # number of objects = matrix width
                    [],      # number of hypothesis = matrix height
                    [[], []]
                )

        summary = mh.compute(acc, metrics=['motp', 'mota', 'num_false_positives', 'num_misses',
                                           'num_switches', 'num_objects', 'num_matches'], name='final')
        total_fp += float(summary['num_false_positives'].iloc[0])
        total_fn += float(summary['num_misses'].iloc[0])
        total_idsw += float(summary['num_switches'].iloc[0])
        total_num_objects += float(summary['num_objects'].iloc[0])
        total_matched += float(summary['num_matches'].iloc[0])
        sum_distance += float(summary['motp'].iloc[0]) * float(summary['num_matches'].iloc[0])
        strsummary = motmetrics.io.render_summary(
            summary,
            formatters={'mota': '{:.2%}'.format},
            namemap={'motp': 'MOTP', 'mota': 'MOTA', 'num_false_positives': 'FP', 'num_misses': 'FN',
                     'num_switches': "ID_SW", 'num_objects': 'num_objects'}
        )
        print(strsummary)

    print("avg mota: {:.3f} %".format(100.0*(1.0-(total_idsw+total_fn+total_fp)/total_num_objects)))
    print("avg motp: {:.3f} %".format(100.0 * (1.0 - sum_distance / total_matched)))
    print("total fn: ", total_fn)
    print("total fp: ", total_fp)
    print("total idsw: ", total_idsw)
    print("total_num_objects: ", total_num_objects)


if __name__ == '__main__':
    print("Loading parameters...")
    curr_path = realpath(dirname(__file__))
    parser = argparse.ArgumentParser(description='Pytorch Evaluation')
    parser.add_argument('--data_root', dest='data_root', default=curr_path + '/data/',
                        help='dataset root path')

    parser.add_argument('--dataset', dest='dataset', default='mot17',
                        help='dataset')

    parser.add_argument('--txts_path', dest='txts_path', default=curr_path + '/saved_results/txts/test/',
                        help='txt files path')

    parser.add_argument('--threshold', dest='threshold', default=0.5, type=float,
                        help='distance matrix threshold')

    args = parser.parse_args()

    main(args)



