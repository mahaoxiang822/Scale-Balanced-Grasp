# Scale-Balanced-Grasp

**Towards Scale Balanced 6-DoF Grasp Detection in Cluttered Scenes**<br>

_Haoxiang Ma, Di Huang_<br>
In CoRL'2022
#### [Paper](https://openreview.net/pdf?id=tiPHpS4eA4) [Video](https://youtu.be/EUXYsd5gK8I)

## Introduction
This repository is official PyTorch implementation for our CoRL2022 paper.
The code is based on [GraspNet-baseline](https://github.com/graspnet/graspnet-baseline)

## Environments
- Anaconda3
- Python == 3.7.9
- PyTorch == 1.6.0
- Open3D >= 0.8

## Installation
Follow the installation of graspnet-baseline.

Get the code.
```bash
git clone https://github.com/mahaoxiang822/Scale-Balanced-Grasp.git
cd graspnet-baseline
```
Install packages via Pip.
```bash
pip install -r requirements.txt
```
Compile and install pointnet2 operators (code adapted from [votenet](https://github.com/facebookresearch/votenet)).
```bash
cd pointnet2
python setup.py install
```
Compile and install knn operator (code adapted from [pytorch_knn_cuda](https://github.com/chrischoy/pytorch_knn_cuda)).
```bash
cd knn
python setup.py install
```
Install graspnetAPI for evaluation.
```bash
git clone https://github.com/graspnet/graspnetAPI.git
cd graspnetAPI
pip install .
```


## Prepare Datasets
For GraspNet dataset, you can download from [GraspNet](https://graspnet.net)

#### Clean scene data generation
You can generate clean data for Noisy-clean Mix by yourself.
```bash
cd dataset
sh command_generate_clean_data.sh
```

#### Tolerance Label Generation(Follow graspnet-baseline)
Tolerance labels are not included in the original dataset, and need additional generation. Make sure you have downloaded the orginal dataset from [GraspNet](https://graspnet.net/). The generation code is in [dataset/generate_tolerance_label.py](../Scale-Balanced-Grasp/dataset/generate_tolerance_label.py). You can simply generate tolerance label by running the script: (`--dataset_root` and `--num_workers` should be specified according to your settings)
```bash
cd dataset
sh command_generate_tolerance_label.sh
```

Or you can download the tolerance labels from [Google Drive](https://drive.google.com/file/d/1DcjGGhZIJsxd61719N0iWA7L6vNEK0ci/view?usp=sharing)/[Baidu Pan](https://pan.baidu.com/s/1HN29P-csHavJF-R_wec6SQ) and run:
```bash
mv tolerance.tar dataset/
cd dataset
tar -xvf tolerance.tar
```

## Train&Test

### Train

```bash
sh command_train.sh
```

### Test
 - We offer our checkpoints for inference and evaluation, you can download from [Google Drive]()
```bash
sh command_test.sh
```

If you want to inference with Object Balanced Sampling, download the pretrained segmentation model and run

```bash
sh command_test_obs.sh
```

#### Evaluation

Evaluation in small-, medium- and large-scale:
```
python evaluate_scale.py
```
Top50 evaluation like Graspnet:
```
python evaluate.py
```


### Citation
If any part of our paper and repository is helpful to your work, please generously cite with:
```
@InProceedings{Ma_2021_BMVC,
    author    = {Haoxiang, Ma and Huang, Di},
    title     = {Towards Scale Balanced 6-DoF Grasp Detection in Cluttered Scenes},
    booktitle = {Conference on Robot Learning (CoRL)},
    year      = {2022}
```