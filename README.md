# DeepMOT
[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0) [![HitCount](http://hits.dwyl.io/yihongxu/deepmot.svg)](http://hits.dwyl.io/yihongxu/deepmot)

**News: We release the code for training and testing DeepMOT-Tracktor and the code for training DHN. Please visit: https://gitlab.inria.fr/yixu/deepmot/-/tree/master**

**How To Train Your Deep Multi-Object Tracker** <br />
[Yihong Xu](https://team.inria.fr/perception/team-members/yihong-xu/), [Aljosa Osep](https://dvl.in.tum.de/team/osep/), [Yutong Ban](https://team.inria.fr/perception/team-members/yutong-ban/), [Radu Horaud](https://team.inria.fr/perception/team-members/radu-patrice-horaud/),[Laura Leal-Taixé](https://dvl.in.tum.de/team/lealtaixe/), [Xavier Alameda-Pineda](https://team.inria.fr/perception/team-members/xavier-alameda-pineda/) <br />
**[[Paper](https://arxiv.org/abs/1906.06618)]** <br />


<div align="center">
  <img src="https://gitlab.inria.fr/yixu/deepmot/-/raw/master/teaser.pdf" width="900px" />
</div>

## Environment setup <a name="environment-setup">
This code has been tested on Ubuntu 16.04, Python 3.6, Pytorch=0.4.1, CUDA 9.2, GTX 1080Ti, Titan X, and RTX Titan GPUs.

**Warning: the results can be slightly different due to Pytorch version and CUDA version.**

- Clone the repository 
```
git clone git@gitlab.inria.fr:yixu/deepmot.git && cd deepmot
```
**Option 1:**
- Follow the installation instructions in [**Tracktor**](https://github.com/phil-bergmann/tracking_wo_bnw/tree/iccv_19).<br/>

**Option 2 (recommended):**

we provide a Singularity image with all packages pre-installed (similar to Docker) for training and testing.
- Open a terminal
- Install Singularity package:
```
sudo apt-get install -y singularity-container
```
- Download the Singularity image: <br />
[tracker.sif (google drive)](https://drive.google.com/file/d/1sR-tTtprbkQ1NAIpY2oiQjSrbuxoYTCc/view?usp=sharing)  or <br />
[tracker.sif (tencent cloud)](https://share.weiyun.com/5RK76iq) <br />
- Open a new terminal
- Launch a Singularity image
```shell
singularity shell --nv --bind yourLocalPath:yourPathInsideImage tracker.sif
```
**- -bind: to link a singularity path with a local path. By doing this, you can find data from local PC inside Singularity image;** <br />
**- -nv: use the local Nvidia driver.**

## Testing <a name="testing-models">
- [Setup](#environment-setup) your environment
- Go to the test_tracktor folder
- Download MOT data
Dataset can be downloaded here: [MOT17Det](https://motchallenge.net/data/MOT17Det.zip), [MOT16Labels](https://motchallenge.net/data/MOT16Labels.zip), [MOT16-det-dpm-raw](https://motchallenge.net/data/MOT16-det-dpm-raw.zip) and [MOT17Labels](https://motchallenge.net/data/MOT17Labels.zip) .
    2. Unzip all the data by executing:
    ```
    unzip -d MOT17Det MOT17Det.zip
    unzip -d MOT16Labels MOT16Labels.zip
    unzip -d 2DMOT2015 2DMOT2015.zip
    unzip -d MOT16-det-dpm-raw MOT16-det-dpm-raw.zip
    unzip -d MOT17Labels MOT17Labels.zip
    ```
- Enter the data path to *data_pth* in the *test_tracktor/experiments/cfgs/tracktor_pub_reid.yaml* and *test_tracktor/experiments/cfgs/tracktor_private.yaml* <br />

- Download pretrained models
all the pretrained models can be downloaded here: <br />
[deepMOT-Tracktor.pth (google drive)](https://drive.google.com/file/d/181JzMrK5YyGecEZkKj-MPuAx2QLqSryO/view?usp=sharing) or <br />
[deepMOT-Tracktor.pth (tencent cloud)](https://share.weiyun.com/5ZXIUL6)

- Enter the model path to parameter *obj_detect_weights* in the *test_tracktor/experiments/cfgs/tracktor_pub_reid.yaml* and *test_tracktor/experiments/cfgs/tracktor_private.yaml* <br />
- Set the dataset name in the test_tracktor/experiments/cfgs/tracktor_pub_reid.yaml and test_tracktor/experiments/cfgs/tracktor_private.yaml: <br />
   For MOT17 (by default):  
```
dataset: mot17_train_17
```

   For MOT16 (images as the same as MOT17):  
```
dataset: mot17_all_DPM_RAW16
```

- run tracking code
```
python test_tracktor/experiments/scripts/tst_tracktor_private.pytst_tracktor_pub_reid.py (public detections) or test_tracktor/experiments/scripts/tst_tracktor_private.py (private detections)
```

The results are saved by default under *test_tracktor/output/log/*, you can modify it by changing *output_dir* in the *test_tracktor/experiments/cfgs/tracktor_pub_reid.yaml* and *test_tracktor/experiments/cfgs/tracktor_private.yaml*.

- Visualization: <br/>
You can set write_images: True in the test_tracktor/experiments/cfgs/tracktor_pub_reid.yaml and test_tracktor/experiments/cfgs/tracktor_private.yaml to plot and save images.
 By default, they will be saved inside *test_tracktor/output/log/* if *write_images: True*.


## Training <a name="training-models">

- [Setup](#environment-setup) your environment
- Go to the train_tracktor folder
- Download MOT Dataset can be downloaded here: [MOT17Det](https://motchallenge.net/data/MOT17Det.zip), [MOT16Labels](https://motchallenge.net/data/MOT16Labels.zip), [MOT16-det-dpm-raw](https://motchallenge.net/data/MOT16-det-dpm-raw.zip) and [MOT17Labels](https://motchallenge.net/data/MOT17Labels.zip).
- Unzip all the data by executing:
    ```
    unzip -d MOT17Det MOT17Det.zip
    unzip -d MOT16Labels MOT16Labels.zip
    unzip -d 2DMOT2015 2DMOT2015.zip
    unzip -d MOT16-det-dpm-raw MOT16-det-dpm-raw.zip
    unzip -d MOT17Labels MOT17Labels.zip
    ```
- Enter the data path to *data_pth* in the *train_tracktor/experiments/cfgs/tracktor_full.yaml*

- Download the output folder containing the configurations and the model to be fine-tuned and DHN pre-trained model:<br/>
[output.zip (google drive)](https://drive.google.com/file/d/11Vu0bL-JaPQUWqHWv1VO89F0WJZWotUm/view?usp=sharing) or <br />
[output.zip (tencent cloud)](https://share.weiyun.com/5nLyD1I)

- unzip the "output" folder and put it to *train_tracktor*.

- run training code
```
python train_tracktor/experiments/scripts/train_tracktor_full.py
```

The trained models are saved by default under *train_tracktor/output/log_full/* folder. <br />
The tensorboard logs are saved by default under *deepmot/logs/train_log/* folder and you can visualize your training process by:
```
tensorboard --logdir=YourGitFolder/train_tracktor/output/log_full/
```
**Note:** 
- you should install *tensorflow* (see [tensorflow installation](https://www.tensorflow.org/install/pip)) in order to visualize your training process.
```
pip install --upgrade tensorflow
```
### Train DHN
- Download the traindata (distance and ground-truth matrices calculated from MOT datasets): <br/>
[DHN data (google drive)](https://drive.google.com/file/d/1ICCm6tH_AgPSLzD3qac-6sYOvTIwwTNW/view?usp=sharing) or <br />
[DHN data (tencent cloud)](https://share.weiyun.com/5OKPHxJ)

- unzip DHN_data and put the *DHN_data* folder to *train_DHN/*
- Run:
```
python train_DHN/train_DHN.py --is_cuda --bidirectional
```

for more parameter details please run:
```
python train_DHN/train_DHN.py -h
```
By default the trained models are saved into *train_DHN/output/DHN/* and log files in *train_DHN/log/*

your can visualize the training via tensorboard:
```
tensorboard --logdir=YourGitFolder/train_DHN/log/
```
**Note:** 
- you should install *tensorflow* (see [tensorflow installation](https://www.tensorflow.org/install/pip)) in order to visualize your training process.
```
pip install --upgrade tensorflow
```

### Evaluation
You can run *test_tracktor/experiments/scripts/evaluate.py* to evaluate your tracker's performance.
- fill the list *predt_pth* in the code with the folder where the results (.txt files) are saved.
- make sure the data path is correctly set.
- then run
```
python test_tracktor/experiments/scripts/evaluate.py
```

### Results
MOT17 public detections:

|  dataset  | MOTA     | MOTP     | FN     | FP    | IDsW | Total Nb. Objs |
|-----------|----------|----------|--------|-------|------|----------------|
|   train   |  62.5%   |  91.7%   | 124786 | 887   | 798  |     336891     |
|   test    |  53.7%   |  77.2%   | 247447 | 11731 | 1947 |     564228     |

MOT16 public detections:

|  dataset  | MOTA     | MOTP     | FN     | FP    | IDsW | Total Nb. Objs |
|-----------|----------|----------|--------|-------|------|----------------|
|   train   |  58.8%   |  92.2%   | 44711  | 538   | 229  |     110407     |
|   test    |  54.8%   |  77.5%   | 78765  | 2955  | 645  |     182326     |


MOT16/17 private detections:

|  dataset  | MOTA     | MOTP     | FN     | FP    | IDsW | Total Nb. Objs |
|-----------|----------|----------|--------|-------|------|----------------|
|   train   |  70.0%   |  91.3%   | 32513  | 552   | 677  |     112297     |


**Note:** 
- the results can be slightly different depending on the running environment.


### Bibtex
If you find this code useful, please star the project and consider citing:

```
@misc{xu2019train,
    title={How To Train Your Deep Multi-Object Tracker},
    author={Yihong Xu and Aljosa Osep and Yutong Ban and Radu Horaud and Laura Leal-Taixe and Xavier Alameda-Pineda},
    year={2019},
    eprint={1906.06618},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## Demo <a name="demo">
<div align="center">
  <img src="https://gitlab.inria.fr/yixu/deepmot/-/raw/obsolete/demo.gif" width="800px" />
</div>

## Acknowledgement <a name="Acknowledgement">
Some code is modified and network pre-trained weights are obtained from the following repositories: <br />


**Single Object Tracker**: [**SiamRPN**](https://github.com/foolwood/DaSiamRPN), [**Tracktor**](https://github.com/phil-bergmann/tracking_wo_bnw/tree/iccv_19), [**Faster-RCNN pytorch implementation**](https://github.com/jwyang/faster-rcnn.pytorch/).
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

@InProceedings{tracktor_2019_ICCV,
  author = {Bergmann, Philipp and Meinhardt, Tim and Leal{-}Taix{\'{e}}}, Laura},
  title = {Tracking Without Bells and Whistles},
  booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
  month = {October},
  year = {2019}}

@inproceedings{10.5555/2969239.2969250,
author = {Ren, Shaoqing and He, Kaiming and Girshick, Ross and Sun, Jian},
title = {Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks},
year = {2015},
publisher = {MIT Press},
address = {Cambridge, MA, USA},
booktitle = {Proceedings of the 28th International Conference on Neural Information Processing Systems - Volume 1},
pages = {91–99},
numpages = {9},
location = {Montreal, Canada},
series = {NIPS’15}
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

