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

import csv
from utils.box_utils import xywh2xyxy

persons_class = ["1"]


def reorder_frameID(frame_dict):
    """
    reorder the frames dictionary in a ascending manner
    :param frame_dict: a dict with key = frameid and value is a list of lists [object id, x, y, w, h] in the frame, dict
    :return: ordered dict by frameid
    """
    keys_int = sorted([int(i) for i in frame_dict.keys()])

    new_dict = {}
    for key in keys_int:
        new_dict[str(key)] = frame_dict[str(key)]
    return new_dict


def read_txt_gtV2(textpath):
    """
    read gt.txt to a dict
    :param textpath: text path, string
    :return: a dict with key = frameid and value is a list of lists [object id, x1, y1, x2, y2] in the frame, dict
    """
    # line format : [frameid, personid, x, y, w, h ...]
    with open(textpath) as f:
        f_csv = csv.reader(f)
        frames = {}
        for line in f_csv:
            if len(line) == 1:
                line = line[0].split(' ')
            # we only consider "pedestrian" class #
            if len(line) < 7 or (line[7] not in persons_class and "MOT2015" not in textpath) or int(float(line[6]))==0:
                continue
            if not (line[0]) in frames:
                frames[line[0]] = []
            bbox = xywh2xyxy(line[2:6])
            frames[line[0]].append([line[1]]+bbox)
    ordered = reorder_frameID(frames)
    return ordered


def read_txt_detV2(textpath):
    """
    read det.txt to a dict
    :param textpath: text path, String
    :return: a dict with key = frameid and value is a list of lists [x1, y1, x2, y2] in the frame, dict
    """
    # line format : [frameid, personid, x, y, w, h ...]
    with open(textpath) as f:
        f_csv = csv.reader(f)
        frames = {}
        for line in f_csv:
            if len(line) == 1:
                line = line[0].split(' ')
            if len(line) <= 5:
                continue
            if not (line[0]) in frames:
                frames[line[0]] = []
            bbox = xywh2xyxy(line[2:6])
            frames[line[0]].append(bbox)
    ordered = reorder_frameID(frames)
    return ordered


def read_txt_predictionV2(textpath):
    """
    read prediction text file to a dict
    :param textpath: text path, String
    :return: a dict with key = frameid and value is a list of lists [track_id, x1, y1, x2, y2] in the frame, dict
    """
    # line format : [frameid, personid, x, y, w, h ...]
    with open(textpath) as f:
        f_csv = csv.reader(f)
        # headers = next(f_csv)
        frames = {}
        for line in f_csv:
            if len(line) == 1:
                line = line[0].split(' ')
            if len(line) <= 5:
                continue
            if not (line[0]) in frames:
                frames[line[0]] = []
            bbox = xywh2xyxy(line[2:6])
            frames[line[0]].append([int(line[1])] + bbox)
    ordered = reorder_frameID(frames)
    return ordered


