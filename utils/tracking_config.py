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


def tracking_config(vname, dataset):
    """
    tracking configuration for MOT challenge
    :param vname: video name, String
    :param dataset: dataset name, String
    :return: tracking configurations
    """

    die_later = ['MOT17-11-DPM', 'MOT17-11-FRCNN', 'MOT17-13-DPM', 'MOT17-04-SDP',
                 'MOT17-04-FRCNN', 'MOT17-02-FRCNN', 'MOT17-04-DPM', 'MOT17-09-FRCNN',

                 'MOT17-12-DPM', 'MOT17-12-FRCNN', 'MOT17-14-DPM', 'MOT17-03-SDP',
                 'MOT17-03-FRCNN', 'MOT17-01-FRCNN', 'MOT17-03-DPM', 'MOT17-08-FRCNN',
                 'MOT17-07-FRCNN']

    if "19" in dataset:
        birth_iou = 0.4
        CMC = False
        interpolate_flag = 20
        case1_interpolate = True
        loose_assignment = True
        birth_wait = 3
        DAN_th = 0.5
        death_count = 60
        to_refine = True
        to_combine = True

    else:

        # to_refine: update reference images after the track is recovered.
        # to_combine: combine tracks with good detections if iou > 0.6
        if "SDP" in vname or "FRCNN" in vname:
            to_refine = True
            to_combine = True

        elif "03" in vname or "04" in vname or "11" in vname or "13" in vname \
                or "12" in vname or "14" in vname:
            to_refine = True
            to_combine = True

        elif "09" in vname or "07" in vname or "08" in vname:
            to_refine = False
            to_combine = False
        else:
            to_refine = True
            to_combine = False

        # DAN_th: appearance model threshold for a track that reappears
        DAN_th = 0.5

        # death_count: max_frames for invisible tracks

        if vname in die_later:
            death_count = 60
        else:
            death_count = 30

        # birth_wait: nb. consecutive frames before giving birth to a track
        birth_wait = 3

        if '13-FRCNN' in vname or '13-SDP' in vname \
                or '14-FRCNN' in vname or '14-SDP' in vname \
                or '14-DPM' in vname or '13-DPM' in vname:
            birth_wait = 2

        # loose_assignment: a track can be assigned to several active tracks
        if "11-FRCNN" in vname or "12-FRCNN" in vname or "02-SDP" in vname or \
                "01-SDP" in vname or "10-SDP" in vname or "04-SDP" in vname or \
                "03-SDP" in vname or "04-FRCNN" in vname or "03-FRCNN" in vname or \
                "02-FRCNN" in vname or "01-FRCNN" in vname or "05-FRCNN" in vname or \
                "06-FRCNN" in vname:

            loose_assignment = True
        else:
            loose_assignment = False

        # case1_interpolate: interpolate occluded positions for a reappearing track

        if "07-FRCNN" in vname or "14-FRCNN" in vname or "13-FRCNN" in vname or \
                "14-SDP" in vname or "13-SDP" in vname:
            case1_interpolate = False
        else:
            case1_interpolate = True

        # interpolate_flag: near out-of-field interpolation
        if '01-DPM' in vname or '02-DPM' in vname or '10-DPM' in vname:

            interpolate_flag = 10
        else:
            interpolate_flag = -1.0

        if "04-DPM" in vname or "03-DPM" in vname:
            interpolate_flag = 20

        # CMC: camera motion compensation for moving camera videos
        if "05" in vname or "06" in vname or "10" in vname \
                or "11" in vname or "13" in vname:
            CMC = True
        else:
            CMC = False

        # birth_iou: birth candidates overlap threshold

        birth_iou = 0.4

        if "06-SDP" in vname or "05-DPM" in vname or "05-SDP" in vname or "06-FRCNN" in vname or "05-FRCNN" in vname:
            birth_iou = 0.1

        if '13-FRCNN' in vname or '13-SDP' in vname or '14-FRCNN' in vname or '14-SDP' in vname or '14-DPM' in vname \
                or '13-DPM' in vname:
            birth_iou = 0.1

    return to_refine, to_combine, DAN_th, death_count, birth_wait, loose_assignment, \
           case1_interpolate, interpolate_flag, CMC, birth_iou