
import cv2
import numpy as np
import torch

class TrackUtil:
    @staticmethod
    def convert_detection(detection):
        '''
        transform the current detection center to [-1, 1]
        :param detection: detection
        :return: translated detection
        '''
        # get the center, and format it in (-1, 1)
        center = (2 * detection[:, 0:2] + detection[:, 2:4]) - 1.0
        # center = torch.from_numpy(center.astype(float)).float()
        center = torch.FloatTensor(center)
        center.unsqueeze_(0)
        center.unsqueeze_(2)
        center.unsqueeze_(3)

        if TrackerConfig.cuda:
            return center.cuda()
        return center

    @staticmethod
    def convert_image(image):
        '''
        transform image to the FloatTensor (1, 3,size, size)
        :param image: same as update parameter
        :return: the transformed image FloatTensor (i.e. 1x3x900x900)
        '''
        image = cv2.resize(image, TrackerConfig.image_size).astype(np.float32)
        # print(image.shape)
        image -= TrackerConfig.mean_pixel
        image = torch.FloatTensor(image)
        image = image.permute(2, 0, 1)
        image.unsqueeze_(dim=0)
        if TrackerConfig.cuda:
            return image.cuda()
        return image


class TrackerConfig:
    cuda = True
    mean_pixel = (104, 117, 123)
    image_size = (900, 900)
