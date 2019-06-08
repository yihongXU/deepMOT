
# DeepMOT

This is the official implementation with *training* code for **DeepMOT**:

**DeepMOT: A Differentiable Framework for Training Multiple Object Trackers** <br />
[Yihong Xu](https://team.inria.fr/perception/team-members/yihong-xu/), [Yutong Ban](https://team.inria.fr/perception/team-members/yutong-ban/), [Xavier Alameda-Pineda](https://team.inria.fr/perception/team-members/xavier-alameda-pineda/), [Radu Horaud](https://team.inria.fr/perception/team-members/radu-patrice-horaud/) <br />
**[[Paper](https://arxiv.org)]** <br />


<div align="center">
  <img src="https://gitlab.inria.fr/yixu/deepmot/raw/master/pipelineV3.png" width="600px" />
</div>

### Bibtex
If you find this code useful, please consider citing:

```
@inproceedings{Xu2019DeepMOT,
    title={DeepMOT: A Differentiable Framework for Training Multiple Object Trackers},
    author={Yihong,Xu and Yutong,Ban and Xavier,Alameda-Pineda and Radu,Horaud},
    booktitle={arxiv:preprint},
    year={2019}
}
```


## Contents
1. [Environment Setup](#environment-setup)
2. [Testing Models](#testing-models)
3. [Training Models](#training-models)
4. [Demo](#demo)
5.  [Acknowledgement](#acknowledgement)

## Environment setup
This code has been tested on Ubuntu 16.04, Python 3.6, Pytorch> 0.4.1, CUDA 9.0 and Cuda 10.0, GTX 1080Ti, Titan X and RTX Titan GPUs.

**warning: the results can be slightly different due to Pytorch Version and CUDA Version.**

- Clone the repository 
```
git clone git@gitlab.inria.fr:yixu/deepmot.git && cd deepmot
```
**Option 1:**
- Setup python environment
```
conda create -n deepmot python=3.6
source activate deepmot
pip install -r requirements.txt
```
**Option 2:**
we offer a Singularity image (similar to Docker) for training and testing.
- Open a terminal
- Install singularity
```
sudo apt-get install -y singularity-container
```
- Open a new terminal
- Download a Singularity image
[pytorch_cuda90_cudnn7.simg](https://drive.google.com/file/d/1wh5dcb_Z3wusl5yn_-0dWl0fTgYYfSAY/view?usp=sharing), [pytorch1-1-cuda100-cudnn75.simg](https://drive.google.com/file/d/1zvQ03pw8hqm6rU_w6lMrOjNcvcrjzRQ4/view?usp=sharing)
- Launch a Singularity image
```shell
cd deepmot
singularity shell --nv --bind yourLocalPath:yourPathInsideImage ./SingularityImages/pytorch1-1-cuda100-cudnn75.simg
```
**- -bind: to link a singularity path with a local path. By doing this, you can find data from local PC inside Singularity image;**
**- -nv: use local Nvidia driver.**

## Testing
We provide code for performing tracking with our pre-trained models on MOT Challenge 17 dataset. The code outputs txt files for MOT Challenge submission, the txt files can also be used for plotting bounding boxes and visualization. 
- [Setup](#environment-setup) your environment

- Download MOT17 data
Dataset can be downloaded here: [MOT17](https://motchallenge.net/data/MOT17/) 

- Put *mot17* dataset into *deepmot/data*, *mot17* should have the following structure:
```
            mot17
            ©¸©¤©¤©¤train
            ©¦   ©¦
            ©¦   ©¸©¤©¤©¤video_folder1
            |   ©¦   ©¸©¤©¤©¤det
            |   ©¦   ©¸©¤©¤©¤gt
            |   ©¦   ©¸©¤©¤©¤img1
            ©¦   ©¸©¤©¤©¤video_folder2
            ...
            ©¦   
            ©¸©¤©¤©¤test
            ©¦   ©¦
            ©¦   ©¸©¤©¤©¤video_folder1
            |   ©¦   ©¸©¤©¤©¤det
            |   ©¦   ©¸©¤©¤©¤img1
            ...
```
- Download pretrained models
all the pretrained models can be downloaded here: [google driver](https://drive.google.com/drive/folders/1HPreiyWbOhgAxhCtvYvoB8wzt_reKzdW?usp=sharing)

-Put all pre-trained models to *deepmot/pretrained*
- run tracking code
```
python tracking_on_mot.py
```
for more details about parameters, do:
```
python tracking_on_mot.py -h
```
the results are save by default under *deepmot/saved_results/txts/test_folder*

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

| MOTA     | MOTP     | FN     | FP    | IDsW | Total Nb. Objs |
|----------|----------|--------|-------|------|-------------------|
|  |  |  |  | |          |

**Note:** 
- the results are better than reported in the paper because we add Camera Motion Compensation to deal with moving camera videos.
- the results can be slightly different depending on the running environment.


## Training

- [Setup](#environment-setup) your environment

- Download MOT17 data
Dataset can be downloaded here: [MOT17](https://motchallenge.net/data/MOT17/) 

- Put *mot17* dataset into *deepmot/data*, *mot17* should have the following structure:
```
            mot17
            ©¸©¤©¤©¤train
            ©¦   ©¦
            ©¦   ©¸©¤©¤©¤video_folder1
            |   ©¦   ©¸©¤©¤©¤det
            |   ©¦   ©¸©¤©¤©¤gt
            |   ©¦   ©¸©¤©¤©¤img1
            ©¦   ©¸©¤©¤©¤video_folder2
            ...
            ©¦   
            ©¸©¤©¤©¤test
            ©¦   ©¦
            ©¦   ©¸©¤©¤©¤video_folder1
            |   ©¦   ©¸©¤©¤©¤det
            |   ©¦   ©¸©¤©¤©¤img1
            ...
```

- Download pretrained SOT model *SiamRPNVOT.model*
SiamRPNVOT.model (from Li et al.): [SiamRPNVOT.model](https://drive.google.com/drive/folders/1HPreiyWbOhgAxhCtvYvoB8wzt_reKzdW?usp=sharing)

-Put *SiamRPNVOT.model*  to  *deepmot/pretrained* folder

- run training code
```
python train_mot.py
```
for more details about parameters, do:
```
python train_mot.py -h
```
the trained models are save by default under *deepmot/saved_models/* folder.
the tensorboard logs are saved by default under *deepmot/logs/train_log/* folder, you can visualize your training process by:
```
tensorboard --logdir=/mnt/beegfs/perception/yixu/opensource/deepMOT/logs/train_log
```
**Note:** 
- you should install *tensorflow* (see [tensorflow installation](https://www.tensorflow.org/install/pip)) in order to visualize your training process.
```
pip install --upgrade tensorflow
```

## Demo

## Acknowledgement
Some codes are modified from the following repositories:
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
**MOT Metrics in Python**: [**py-motmetrics**](https://github.com/cheind/py-motmetrics)
**Appearance Features Extractor**: [**DAN**](https://github.com/shijieS/SST)
```
@article{sun2018deep,
  title={Deep Affinity Network for Multiple Object Tracking},
  author={Sun, ShiJie and Akhtar, Naveed and Song, HuanSheng and Mian, Ajmal and Shah, Mubarak},
  journal={arXiv preprint arXiv:1810.11780},
  year={2018}
}
```
Training and testing Data from:
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