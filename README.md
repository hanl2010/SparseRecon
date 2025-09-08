<h1 align="center"> SparseRecon: Neural Implicit Surface Reconstruction from Sparse Views with
Feature and Depth Consistencies </h1>

<p align="center">
    <strong>Liang Han</strong>
    ·
    <strong>Xu Zhang</strong>
    ·
    <strong>Haichuan Song</strong>
    ·
    <strong>Kanle Shi</strong>
    ·
    <a href="https://yushen-liu.github.io/"><strong>Yu-Shen Liu</strong></a>
    ·
    <a href="https://h312h.github.io/"><strong>Zhizhong Han</strong></a>
</p>

<h2 align="center">ICCV 2025</h2>


## [Project page](https://hanl2010.github.io/SparseRecon/) |  [Paper](https://arxiv.org/abs/2508.00366)
This is the official repo for the implementation of **SparseRecon: Neural Implicit Surface Reconstruction from Sparse Views with Feature and Depth Consistencies**.

## Usage

### Setup

1. Clone this repository

```shell
git clone https://github.com/hanl2010/SparseRecon.git
```
2. Setup Environment

```shell
conda create -n sparserecon python=3.9
conda activate sparserecon

pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### Dataset
- Download the processed dataset: [DTU](https://drive.google.com/file/d/1ph5dk7ZAjDhDueB3yzHx2NE2arUGDe6i/view?usp=drive_link) and [BlendedMVS](https://drive.google.com/file/d/1u6xmQicjcOdQVKismwC_5d_WGjND2lLn/view?usp=sharing)

### Obtaining and Calibration of Monocular Depth 

- Get depth by Omnidata

Get pretrained weight from [omnidata_depth_weight](https://drive.google.com/drive/folders/14RV5GdHv6sReFMfEwMrXAfvIAasNlmKg?usp=sharing), and put it in *tools/omnidata/pretrained_models*

```shell
python tools/depth_estimation_omnidata.py --data_root <DATA_PATH> --dataset_name <DTU or BlendedMVS>
```
- Or get depth by Marigold (Optional) 

```shell
python tools/depth_estimation_marigold.py --data_root <DATA_PATH> --dataset_name <DTU or BlendedMVS>
```

- Calibration
```shell
python tools/calibrate_depth.py --data_root <DATA_PATH> --dataset_name <DTU or BlendedMVS> --depth_folder <depths_omnidata or depths_marigold>
```


### Training and Evaluation 
Download the pretrained weight of [VisMVSNet](https://github.com/jzhangbs/Vis-MVSNet) from [here](https://drive.google.com/drive/folders/14RV5GdHv6sReFMfEwMrXAfvIAasNlmKg), and put it in *feat_extractor/vismvsnet*


- **DTU dataset**

```shell
python script/run_dtu.py --conf <confs/CONFIG_FILE> --data_path <TRAINING_DATA_PATH> --GT_data_path <GT_DATA_PATH>
```

- **BlendedMVS dataset**

```shell
python script/run_blendedmvs.py --conf confs/blendedmvs.conf --data_path <TRAINING_DATA_PATH> 
```


## Citation

Cite as below if you find this repository is helpful to your project:

```
@article{han2025sparserecon,
  title={SparseRecon: Neural Implicit Surface Reconstruction from Sparse Views with Feature and Depth Consistencies},
  author={Han, Liang and Zhang, Xu and Song, Haichuan and Shi, Kanle and Liu, Yu-Shen and Han, Zhizhong},
  journal={arXiv preprint arXiv:2508.00366},
  year={2025}
}
```

## Acknowledgement
This project is built upon [NeuS](https://github.com/Totoro97/NeuS). Thanks for the great project.
