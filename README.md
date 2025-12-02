# Segment then Splat: Unified 3D Open-Vocabulary Segmentation via Gaussian Splatting

## [Project page](https://vulab-ai.github.io/Segment-then-Splat/) | [Paper](https://arxiv.org/abs/2503.22204v2)

![Teaser image](assets/teaser.png)

This repository contains the official implementation associated with the paper "Segment then Splat: Unified 3D Open-Vocabulary Segmentation via Gaussian Splatting".


## Dataset

We organize the datasets as follows, take lerf-ovs as an example:

```shell
├── lerf_ovs_langsplat
│   | ramen 
│     ├── images
│     ├── sparse 
│     ├── train.txt
│     ├── test.txt
│     ├── ...
│   | teatime 
│     ├── images
│     ├── sparse 
│     ├── train.txt
│     ├── test.txt
│     ├── ...
│    ...
```
We provide preprocessed lerf-ovs dataset [here](https://drive.google.com/drive/folders/18f9JKqjcgEw-gwBG3z-nk85jqGC7WfcG?usp=sharing).

## Run

### Environment

```shell
git clone https://github.com/ingra14m/Deformable-3D-Gaussians --recursive
cd Segment-then-Splat

conda create -n segment_then_splat python=3.10
conda activate segment_then_splat

pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

### Object Tracking & Object-specific Initialization (To be verified)
Remember to modify the data path inside ```autoseg.sh```.
```shell
# Object Tracking
cd thrid_party/AutoSeg-SAM2/
bash autoseg.sh
# Object-specific Initialization
cd ../..
python ./helpers/preprocess_mask.py
python object_specific_initialization.py
```
We provide preprocessed lerf-ovs dataset with tracking results [here](https://drive.google.com/drive/folders/18f9JKqjcgEw-gwBG3z-nk85jqGC7WfcG?usp=sharing).
### Scene Reconstuction
```shell
python train.py -s <data_dir>/lerf_ovs_langsplat/ramen/ -m output/ramen  --eval --iterations 40000 --num_sample_objects 3 --densify_until_iter 20000 --partial_mask_iou 0.3
```
For dynamic scenes add ```--is_6dof``` and ```--deform```.
****

### CLIP Embedding Association & Evaluation

```shell
# Render object test images
python render_objs.py -m ./output/ramen/ --mode render --skip_train # For dynamic scenes add --is_6dof and --deform.

# Associate CLIP embedding and calculate mIoU
python ./helpers/evaluation.py --scene <data_dir>/lerf_ovs_langsplat/ramen/ --render_dir ./output/ramen/ --label_dir <data_dir>/lerf_ovs_langsplat/label/ramen/gt
```

## Acknowledgments

We sincerely thank the authors of [Deformable-3DGS](https://github.com/ingra14m/Deformable-3D-Gaussians), [AutoSeg-SAM2](https://github.com/zrporz/AutoSeg-SAM2), [LERF](https://github.com/kerrj/lerf), [HyperNeRF](https://hypernerf.github.io/), [3DOVS](https://github.com/Kunhao-Liu/3D-OVS), and [Neu3D](https://github.com/facebookresearch/Neural_3D_Video), whose codes and datasets were used in our work.

## BibTex

```
@article{Lu2025Segment,
  title={Segment then Splat: Unified 3D Open-Vocabulary Segmentation via Gaussian Splatting},
  author={Yiren Lu and Yunlai Zhou and Yiran Qiao and Chaoda Song and Tuo Liang and Jing Ma and Huan Wang and Yu Yin},
  journal={Advances in Neural Information Processing Systems},
  year={2025},
}
```

