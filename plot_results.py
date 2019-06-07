# ==========================================================================
#
# This file is a part of implementation for paper:
# DeepMOT: A Differentiable Framework for Training Multiple Object Trackers.
# This contribution is headed by Perception research team, INRIA.
#
# Contributor(s) : Yutong Ban
# INRIA contact  : yutong.ban@inria.fr
#
# ===========================================================================

import os
import cv2
import random
import argparse
import numpy as np
import pandas as pd
from os.path import realpath, dirname


def main(args, colorList):

    seqList = os.listdir(args.results_path)

    for seq_name in seqList:

        if os.path.exists(args.data_root + args.dataset + "/train/" + seq_name[:-4]):
            path_data = args.data_root + args.dataset + "/train/" + seq_name[:-4]+'/'

        elif os.path.exists(args.data_root + args.dataset + "test/" + seq_name[:-4]):
            path_data = args.data_root + args.dataset + "test/" + seq_name[:-4] + '/'
        else:
            continue

        path_res = args.results_path + seq_name
        path_images = path_data + 'img1/'
        save_path = args.save_path + seq_name[:-4] + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        res_raw = pd.read_csv(path_res, sep=',', header=None)
        res_raw = np.array(res_raw).astype(np.float32)
        res_raw[:, 0:6] = np.array(res_raw[:, 0:6]).astype(np.int)

        N_frame = max(res_raw[:, 0])
        print('total number of frames: ', N_frame)
        # N_frame = 100

        for t in range(1, int(N_frame)):
            if os.path.exists(save_path + str(t).zfill(6) + '.jpg'):
                continue
            print('t = ' + str(t))
            img_name = path_images + str(t).zfill(6) + '.jpg'
            print(img_name)
            img = cv2.imread(img_name)
            overlay = img.copy()
            # print(img.shape)
            # cv2.imshow('image',img)
            row_ind = np.where(res_raw[:, 0] == t)[0]
            for i in range(0, row_ind.shape[0]):
                id = int(max(res_raw[row_ind[i], 1], 0))
                color_ind = id%len(colorList)

                # plot the line
                row_ind_line = np.where((res_raw[:, 0] > t-50) & (res_raw[:, 0] < t+1) & (res_raw[:, 1] == id))[0]

                # plot the rectangle
                for j in range(0, row_ind_line.shape[0], 5):

                    line_xc = int(res_raw[row_ind_line[j], 2] + 0.5* res_raw[row_ind_line[j], 4])
                    line_yc = int(res_raw[row_ind_line[j], 3] + res_raw[row_ind_line[j], 5])
                    bb_w = 5
                    line_x1 = line_xc - bb_w
                    line_y1 = line_yc - bb_w
                    line_x2 = line_xc + bb_w
                    line_y2 = line_yc + bb_w
                    cv2.rectangle(overlay, (line_x1, line_y1), (line_x2, line_y2), colorList[color_ind], -1)

                    t_past = res_raw[row_ind_line[j], 0]
                    alpha = 1 - (t - t_past)/80  # Transparency factor.
                    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
                    overlay = img.copy()

            for i in range(0, row_ind.shape[0]):
                id = int(res_raw[row_ind[i], 1])
                bb_x1 = int(res_raw[row_ind[i], 2])
                bb_y1 = int(res_raw[row_ind[i], 3])
                bb_x2 = int(res_raw[row_ind[i], 2] + res_raw[row_ind[i], 4])
                bb_y2 = int(res_raw[row_ind[i], 3] + res_raw[row_ind[i], 5])
                str_tmp = str(i) + ' ' + str("{0:.2f}".format(res_raw[row_ind[i], 6]))
                color_ind = id % len(colorList)
                cv2.rectangle(overlay, (bb_x1, bb_y1), (bb_x2, bb_y2), colorList[color_ind], 3)

            save_name = save_path + str(t).zfill(6) + '.jpg'
            cv2.imwrite(save_name, overlay)


if __name__ == '__main__':
    def get_spaced_colors(n):
        max_value = 16581375  # 255**3
        interval = int(max_value / n)
        colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]

        return [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]


    colorList = get_spaced_colors(100)

    random.shuffle(colorList)

    print("Loading parameters...")
    curr_path = realpath(dirname(__file__))
    parser = argparse.ArgumentParser(description='Plot Images from Results')
    parser.add_argument('--data_root', dest='data_root', default=curr_path + '/data/mot17/train/',
                        help='dataset root path')

    parser.add_argument('--results_path', dest='results_path', default=curr_path + '/saved_results/txts/test_folder/',
                        help='txt files path')

    parser.add_argument('--save_path', dest='save_path', default=curr_path + '/saved_results/imgs/test_folder/',
                        help='images save path')

    parser.add_argument('--dataset', dest='dataset', default='mot17', help='dataset')

    args = parser.parse_args()

    main(args, colorList)
