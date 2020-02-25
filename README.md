# DeepMOT
[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0) [![HitCount](http://hits.dwyl.io/yihongxu/deepmot.svg)](http://hits.dwyl.io/yihongxu/deepmot)

### Important Note
**We will make publicly available the new code and trained models used in the newly submitted paper in the following week.**

**How To Train Your Deep Multi-Object Tracker** <br />
[Yihong Xu](https://team.inria.fr/perception/team-members/yihong-xu/), [Aljosa Osep](https://dvl.in.tum.de/team/osep/), [Yutong Ban](https://team.inria.fr/perception/team-members/yutong-ban/), [Radu Horaud](https://team.inria.fr/perception/team-members/radu-patrice-horaud/),[Laura Leal-Taixé](https://dvl.in.tum.de/team/lealtaixe/), [Xavier Alameda-Pineda](https://team.inria.fr/perception/team-members/xavier-alameda-pineda/) <br />
**[[Paper](https://arxiv.org/abs/1906.06618)]** <br />


<div align="center">
  <img src="https://gitlab.inria.fr/yixu/deepmot/raw/master/pipelineV3.png" width="900px" />
</div>

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
  <img src="https://github.com/yihongXU/deepMOT/raw/master/demo.gif" width="800px" />
</div>

## Acknowledgement <a name="Acknowledgement">
Some codes are modified and network pretrained weights are obtained from the following repositories: <br />
**Single Object Tracker**: [**SiamRPN**](https://github.com/foolwood/DaSiamRPN), [**Tracktor**](https://github.com/phil-bergmann/tracking_wo_bnw/tree/master/src/tracktor).
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
