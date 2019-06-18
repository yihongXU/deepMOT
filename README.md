# DeepMOT
[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0) [![HitCount](http://hits.dwyl.io/yihongxu/deepmot.svg)](http://hits.dwyl.io/yihongxu/deepmot)

This is the official implementation with *training* code for the paper:

**DeepMOT: A Differentiable Framework for Training Multiple Object Trackers** <br />
[Yihong Xu](https://team.inria.fr/perception/team-members/yihong-xu/), [Yutong Ban](https://team.inria.fr/perception/team-members/yutong-ban/), [Xavier Alameda-Pineda](https://team.inria.fr/perception/team-members/xavier-alameda-pineda/), [Radu Horaud](https://team.inria.fr/perception/team-members/radu-patrice-horaud/) <br />
**[[Paper](https://arxiv.org/abs/1906.06618)]** <br />


<div align="center">
  <img src="https://gitlab.inria.fr/yixu/deepmot/raw/master/pipelineV3.png" width="900px" />
</div>

### Bibtex
If you find this code useful, please star the project and consider citing:

```
@inproceedings{Xu2019DeepMOT,
    title={DeepMOT: A Differentiable Framework for Training Multiple Object Trackers},
    author={Yihong,Xu and Yutong,Ban and Xavier,Alameda-Pineda and Radu,Horaud},
    booktitle={arXiv preprint arXiv:1906.06618},
    year={2019}
}
```


## Contents
1. [Environment Setup](#environment-setup)
2. [Testing](#testing-models)
3. [Training](#training-models)
4. [Demo](#demo)
5. [Acknowledgement](#acknowledgement)

## Environment setup <a name="environment-setup">
This code has been tested on Ubuntu 16.04, Python 3.6, Pytorch> 0.4.1, CUDA 9.0 and Cuda 10.0, GTX 1080Ti, Titan X and RTX Titan GPUs.

**warning: the results can be slightly different due to Pytorch version and CUDA version.**

- Clone the repository 
```
git clone https://github.com/yihongXU/deepMOT.git && cd deepmot
```
**Option 1:**
- Setup python environment
```
conda create -n deepmot python=3.6
source activate deepmot
pip install -r requirements.txt
```
**Option 2:**
we offer Singularity images (similar to Docker) for training and testing.
- Open a terminal
- Install singularity
```
sudo apt-get install -y singularity-container
```
- Download a Singularity image and put it to *deepmot/SingularityImages* <br />
[pytorch_cuda90_cudnn7.simg(google drive)](https://drive.google.com/file/d/1wh5dcb_Z3wusl5yn_-0dWl0fTgYYfSAY/view?usp=sharing) <br />
[pytorch1-1-cuda100-cudnn75.simg(google drive)](https://drive.google.com/file/d/1zvQ03pw8hqm6rU_w6lMrOjNcvcrjzRQ4/view?usp=sharing)<br />
[pytorch_cuda90_cudnn7.simg(tencent cloud)](https://share.weiyun.com/5G3Y1FK) <br />
[pytorch1-1-cuda100-cudnn75.simg(tencent cloud)](https://share.weiyun.com/5sTjg6J)
- Open a new terminal
- Launch a Singularity image
```shell
cd deepmot
singularity shell --nv --bind yourLocalPath:yourPathInsideImage ./SingularityImages/pytorch1-1-cuda100-cudnn75.simg
```
**- -bind: to link a singularity path with a local path. By doing this, you can find data from local PC inside Singularity image;** <br />
**- -nv: use local Nvidia driver.**

## Testing <a name="testing-models">
We provide code for performing tracking with our pre-trained models on MOT Challenge dataset. The code outputs txt files for MOT Challenge submissions, they can also be used for plotting bounding boxes and visualization. 
- [Setup](#environment-setup) your environment

- Download MOT data
Dataset can be downloaded here: e.g. [MOT17](https://motchallenge.net/data/MOT17/) 

- Put *MOT* dataset into *deepmot/data/* and it should have the following structure:
```
            mot
            |-------train
            |    |
            |    |---video_folder1
            |    |   |---det
            |    |   |---gt
            |    |   |---img1
            |    |
            |    |---video_folder2
            ...
            |-------test
            |    |
            |    |---video_folder1
            |    |   |---det
            |    |   |---img1
            ...
```
- Download pretrained models
all the pretrained models can be downloaded here: <br />
[pretrained models(google drive)](https://drive.google.com/drive/folders/1HPreiyWbOhgAxhCtvYvoB8wzt_reKzdW?usp=sharing) or <br />
[pretrained models(tencent cloud)](https://share.weiyun.com/5rVqDmu)

-Put all pre-trained models to *deepmot/pretrained/*
- run tracking code
```
python tracking_on_mot.py
```
for more details about parameters, do:
```
python tracking_on_mot.py -h
```
The results are save by default under *deepmot/saved_results/txts/test_folder/*.

- Visualization
After finishing tracking, you can visualize your results by plotting bounding box to images.
```
python plot_results.py
```
the results are save by default under *deepmot/saved_results/imgs/test_folder*

**Note:** 
- we clean the detections with nms and threshold of detection scores. They are saved into numpy array in the folder *deepmot/clean_detections*, if you have trouble opening them, try to add *allow_pickle=True* to *np.load()* function.

### Results
We provide codes for evaluting tracking results in terms of MOTP and MOTA:
```
python evaluation.py --txts_path=yourTxTfilesFolder
```
MOT17:

|  dataset  | MOTA     | MOTP     | FN     | FP    | IDsW | Total Nb. Objs |
|-----------|----------|----------|--------|-------|------|----------------|
|   train   |  49.249% |  82.812% | 149575 | 19807 | 1592 |     336891     |
|   test    |  48.500% |  76.900% | 262765 | 24544 | 3160 |     564228     |

**Note:** 
- the results are better than reported in the paper because we add Camera Motion Compensation to deal with moving camera videos.
- the results can be slightly different depending on the running environment.


## Training <a name="training-models">

- [Setup](#environment-setup) your environment

- Download MOT data
Dataset can be downloaded here: e.g. [MOT17](https://motchallenge.net/data/MOT17/) 

- Put *MOT* dataset into *deepmot/data* and it should have the following structure:
```
            mot
            |-------train
            |    |
            |    |---video_folder1
            |    |   |---det
            |    |   |---gt
            |    |   |---img1
            |    |
            |    |---video_folder2
            ...
            |-------test
            |    |
            |    |---video_folder1
            |    |   |---det
            |    |   |---img1
            ...
```

- Download pretrained SOT model *SiamRPNVOT.model*
SiamRPNVOT.model (from SiamRPN, Li et al., see [Acknowledgement](#acknowledgement)): <br />
[SiamRPNVOT.model(google drive)](https://drive.google.com/drive/folders/1HPreiyWbOhgAxhCtvYvoB8wzt_reKzdW?usp=sharing) or <br />
[SiamRPNVOT.model(tencent cloud)](https://share.weiyun.com/5Fxw6ke)

-Put *SiamRPNVOT.model*  to  *deepmot/pretrained/* folder

- run training code
```
python train_mot.py
```
for more details about parameters, do:
```
python train_mot.py -h
```
The trained models are save by default under *deepmot/saved_models/* folder. <br />
The tensorboard logs are saved by default under *deepmot/logs/train_log/* folder and you can visualize your training process by:
```
tensorboard --logdir=/mnt/beegfs/perception/yixu/opensource/deepMOT/logs/train_log
```
**Note:** 
- you should install *tensorflow* (see [tensorflow installation](https://www.tensorflow.org/install/pip)) in order to visualize your training process.
```
pip install --upgrade tensorflow
```

## Demo <a name="demo">
<div align="center">
  <img src="https://gitlab.inria.fr/yixu/deepmot/raw/master/demo.gif" width="800px" />
</div>

## Acknowledgement <a name="Acknowledgement">
Some codes are modified and network pretrained weights are obtained from the following repositories: <br />
**Single Object Tracker**: [**SiamRPN**](https://github.com/foolwood/DaSiamRPN)
```
@inproceedings{Zhu_2018_ECCV,
  title={Distractor-aware Siamese Networks for Visual Object Tracking},
  author={Zhu, Zheng and Wang, Qiang and Bo, Li and Wu, Wei and Yan, Junjie and Hu, Weiming},
  booktitle={European Conference on Computer Vision},
  year={2018}
}

@InProceedings{Li_2018_CVPR,
  title = {High Performance Visual Tracking With Siamese Region Proposal Network},
  author = {Li, Bo and Yan, Junjie and Wu, Wei and Zhu, Zheng and Hu, Xiaolin},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2018}
}
```
**MOT Metrics in Python**: [**py-motmetrics**](https://github.com/cheind/py-motmetrics)<br />
**Appearance Features Extractor**: [**DAN**](https://github.com/shijieS/SST)<br />
```
@article{sun2018deep,
  title={Deep Affinity Network for Multiple Object Tracking},
  author={Sun, ShiJie and Akhtar, Naveed and Song, HuanSheng and Mian, Ajmal and Shah, Mubarak},
  journal={arXiv preprint arXiv:1810.11780},
  year={2018}
}
```
Training and testing Data from: <br />
**MOT Challenge**: [**motchallenge**](https://motchallenge.net/data)
```
@article{MOT16,
	title = {{MOT}16: {A} Benchmark for Multi-Object Tracking},
	shorttitle = {MOT16},
	url = {http://arxiv.org/abs/1603.00831},
	journal = {arXiv:1603.00831 [cs]},
	author = {Milan, A. and Leal-Taix\'{e}, L. and Reid, I. and Roth, S. and Schindler, K.},
	month = mar,
	year = {2016},
	note = {arXiv: 1603.00831},
	keywords = {Computer Science - Computer Vision and Pattern Recognition}
}
```
