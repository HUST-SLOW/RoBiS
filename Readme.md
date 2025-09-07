# ✨RoBiS: Robust Binary Segmentation for High-Resolution Industrial Images (*CVPR2025 VAND3.0 challenge track 1 2rd solution*)

Name(s):
[Xurui Li](https://github.com/xrli-U)<sup>1</sup> | [Zhongsheng Jiang](https://github.com/FoundWind7)<sup>1</sup> | [Tingxuan Ai](https://aitingxuan.github.io/)<sup>1</sup> | [Yu Zhou](https://github.com/zhouyu-hust)<sup>1,2</sup>

Affiliation(s):
<sup>1</sup>Huazhong University of Science and Technology | <sup>2</sup>Wuhan JingCe Electronic Group Co.,LTD

Contact Information:
**xrli\_plus@hust.edu.cn** | zsjiang@hust.edu.cn | tingxuanai@hust.edu.cn | yuzhou@hust.edu.cn

Track: Adapt \& Detect---Robust Anomaly Detection in Real-World Applications

### Technical report: [ResearchGate](https://www.researchgate.net/publication/392124350_RoBiS_Robust_Binary_Segmentation_for_High-Resolution_Industrial_Images) | [arXiv](https://arxiv.org/pdf/2505.21152) | [PDF](assets/RoBiS.pdf) | [PPT](assets/RoBis_PPT.pdf)

### README: English | [Chinese](Readme_CN.md)


## 🧐Overview

This repository is the official implementation of our **winner solution RoBiS** for the CVPR2025 VAND3.0 challenge Track 1.

Our RoBiS combines the traditional *mean+3std* with the MEBin proposed in CVPR2025 [github link](https://github.com/HUST-SLOW/AnomalyNCD) to achieve adaptive binarization. This strategy enables our method not to manually determine different thresholds for each product.

MVTec Benchmark Server: [https://benchmark.mvtec.com/](https://benchmark.mvtec.com/).

Challenge Website: [https://sites.google.com/view/vand30cvpr2025/challenge](https://sites.google.com/view/vand30cvpr2025/challenge)

![winner](.\assets\winner.png)


## 🎯Setup

### Environment:

- Python 3.8
- CUDA 11.7
- PyTorch 2.0.1

Clone the repository locally:

```
git clone https://github.com/HUST-SLOW/RoBiS.git
```

Create virtual environment:

```
conda create --name RoBiS python=3.8
conda activate RoBiS
```

Install the required packages:

```
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install -r requirements.txt
```
## 👇Prepare Datasets

Download the [MVTec AD 2](https://arxiv.org/pdf/2503.21622) dataset through the official link ([web](https://www.mvtec.com/company/research/datasets/mvtec-ad-2))

Put the datasets in `./data` folder.

```
data
|---mvtec_ad_2
|-----|-- can
|-----|-----|----- test_private
|-----|-----|----- test_private_mixed
|-----|-----|----- test_public
|-----|-----|----- train
|-----|-----|----- validation
|-----|-- fabric
|-----|--- ...
```

## 💎Run RoBiS
Before starting to run our RoBiS, execute the `download_weights.sh` script to download the pre-training weights.
```
bash scripts/download_weights.sh
```
We provide two ways to run our code.

### python

```
python examples/robis_main.py
```
Follow the configuration in `./configs/robis.yaml`

### script

```
bash scripts/robis.sh
```
The configuration in the script `robis.sh` takes precedence.

The key arguments of the script are as follows:

- `--device`: GPU_id.
- `--dataset_name`: Dataset name.
- `--original_data_path`: The directory of datasets.
- `--data_path`: The directory of processed datasets.
- `--mvtecad2_class_list`: Category to be tested. If the parameter is set to `all`, all the categories are tested.
- `--test_type`: Test type. Choose from `test_public`, `test_private`, `test_private_mixed` and `validation`.
- `--window_size`: The size of the cropped images.
- `--desired_overlap`: Overlap of the cropped images.
- `--save_dir`: The directory that saves the weights. This directory will be automatically created.
- `--amap_savedir`: The directory that saves the anomaly prediction maps. This directory will be automatically created. 
- `--bin_savedir`: The directory that saves the binary maps. This directory will be automatically created.
- `--encoder`: Feature exractor name.
- `--input_size`: Size of the image after resizing.
- `--crop_size`: Size of the image after center cropping.
- `--INP_num`: Number of INP.
- `--total_epochs`: Number of training epochs.
- `--batch_size`: Batch size.
- `--train_state`: Whether to train.
- `--test_state`: Whether to test.
- `--eval_pub`: Whether to evaluate the test_public metrics. 

You can transfer the continuous anomaly maps and thresholded binary masks for `test_private` and `test_private_mixed` to `./submission_folder` for compression and evaluation.
```
mkdir -p ./submission_folder
cp -r ${amap_savedir}/anomaly_images ./submission_folder/
cp -r ${bin_savedir}/anomaly_images_thresholded ./submission_folder/
```
The final continuous anomaly maps could be download in [google drive](https://drive.google.com/file/d/1OqejveTgEuYr9obEUV3h3Vzq2HTp29ua/view?usp=sharing).
The final thresholded binary masks could be download in [google drive](https://drive.google.com/file/d/1ilMnxisuQOYnvllu1kUHaibkzHiHN_R-/view?usp=sharing).
You can also download the trained checkpoints by this link [(google drive)](https://drive.google.com/drive/folders/1JvbEru6W1RxThjjiPJSONbO97j9_I6dN?usp=drive_link).


### Step by step explanation
**1. Pre-processing**
```
python models/modules/_swin_cropping.py \
       --original_data_path ./data/mvtec_ad_2 \
       --data_path ./mvtec_ad_2_processed \
       --window_size 1024 \
       --desired_overlap 0.1 \
       --mvtecad2_class_list all
```
Use the `_swin_cropping` module to pre-process the data.
Please leave about 50GB for the pre-processed data.

Key arguments:
- `--original_data_path`: The directory of the original dataset.
- `--data_path`: The directory that saves the pre-processed dataset. This directory will be automatically created.
- `--mvtecad2_class_list`: The product categories of MVTec AD 2 dataset.
- `--window_size`: The size of the cropped images.
- `--desired_overlap`: Overlap of the cropped images.

**2. Model training**
```
python models/robis.py \
  --device 0 \
  --original_data_path ./data/mvtec_ad_2 \
  --data_path ./mvtec_ad_2_processed \
  --mvtecad2_class_list all \
  --encoder dinov2reg_vit_base_14 \
  --input_size 518 \
  --crop_size 518 \
  --INP_num 6 \
  --total_epochs 200 \
  --batch_size 16 \
  --save_dir ./saved_weights \
  --amap_savedir ./anomaly_map_results \
  --test_type test_public \
  --train_state
```
We use ViT-B-14 initialized with DINOv2-R pre-trained weights as the encoder.
The pre-trained weights will be download automatically as `.models/backbones/weights/dinov2_vitb14_reg4_pretrain.pth`

To train the AD model under the default settings, please reserve at least 17GB of GPU memory.
You can use different GPUs to train different categories to reduce time consumption.
You can also download the trained checkpoints by this link [(google drive)](https://drive.google.com/drive/folders/1JvbEru6W1RxThjjiPJSONbO97j9_I6dN?usp=drive_link).

Key arguments:
- `--data_path`: The directory of the pre-processed dataset.
- `--save_dir`: The directory that saves model weights. This directory will be automatically created.
- `--mvtecad2_class_list`: The product categories of MVTec AD 2 dataset. Since our method trains one model for each category, different GPUs could be used to train different categories.
- `--encoder`: Feature exractor name.
- `--input_size`: Size of the image after resizing.
- `--crop_size`: Size of the image after center cropping.
- `--INP_num`: Number of INP.
- `--total_epochs`: Number of training epochs.
- `--batch_size`: Batch size.
- `--train_state`: Whether to train. 

**3. Model testing**
```
python models/robis.py \
  --device 0 \
  --original_data_path ./data/mvtec_ad_2 \
  --data_path ./mvtec_ad_2_processed \
  --mvtecad2_class_list all \
  --encoder dinov2reg_vit_base_14 \
  --input_size 518 \
  --crop_size 518 \
  --INP_num 6 \
  --batch_size 16 \
  --save_dir ./saved_weights \
  --amap_savedir ./anomaly_map_results \
  --test_type test_public \
  --test_state
```
Testing on the selected test dataset.

Key arguments:
- `--data_path`: The directory of the pre-processed dataset.
- `--save_dir`: The directory that saves model weights.
- `--amap_savedir`: The directory that saves anomaly maps *(.tiff)* of all sub-images. This directory will be automatically created.
- `--test_type`: The test set of MVTec AD 2 dataset, setting *test_private* or *test_private_mixed* or *test_public* or *validation*.
- `mvtecad2_class_list`: The product categories of MVTec AD 2 dataset.
- `--test_state`: Whether to test.

**4. Post-processing**
```
python models/modules/_merging.py \
    --amap_savedir ./anomaly_map_results \
    --mvtecad2_class_list all \
    --window_size 1024 \
    --desired_overlap 0.1 \
    --test_type test_public
```
Use the `_merging` module to reconstruct the anomaly maps of sub-images into the corresponding original anomaly map.

Key arguments:
- `--amap_savedir`: The directory that saves anomaly maps *(.tiff)* of all sub-images. After the merge, the anomaly maps of sub-images are automatically deleted.
- `--window_size`: The size of the cropped images.Same as pre-processing.
- `--desired_overlap`: Overlap of the cropped images.Same as pre-processing.

**5. Binarization**
```
# Using MEBin and mean+3std to generate coarse binary masks.
python models/modules/_binarization.py \
  --amap_savedir ./anomaly_map_results \
  --bin_savedir  ./binary_map_results \
  --mvtecad2_class_list all \
  --test_type test_public

# Using SAM to generate finer binary masks.
python models/modules/_SAM_Finer.py \
  --original_data_path ./data/mvtec_ad_2 \
  --mvtecad2_class_list all \
  --test_type test_public \
  --bin_savedir ./binary_map_results \
  --device 0
```
Use the `_binarization` module to generate coarse binary masks combining MEBin and mean+3std.
Use the `_SAM_Finer` module to generate finer binary masks.

Before using `_SAM_Finer.py`, make sure that the pre-trained weights *(sam_b and sam_h)* of SAM are downloaded to the current directory (`bash scripts/download_weights.sh`).

Key arguments:
- `--amap_savedir`: The directory that saves anomaly maps *(.tiff)*.
- `--bin_savedir`: The directory that saves thresholded binary masks.
- `--original_data_path`: The directory of the original dataset.

**6. Evaluation**
You can transfer the continuous anomaly maps and thresholded binary masks for `test_private` and `test_private_mixed` to `./submission_folder` for compression and evaluation.
```
mkdir -p ./submission_folder
cp -r ${amap_savedir}/anomaly_images ./submission_folder/
cp -r ${bin_savedir}/anomaly_images_thresholded ./submission_folder/
```
`test_public` dataset allows direct evaluation on the binary masks.

Key arguments:
- `--eval_pub`: Whether to evaluate `test_public` dataset, setting `True` to evaluate the metrics.

## 🎖️Results

All the results are calculated by the official leaderboard.

### MVTec AD 2

|   Object    | AucPro_0.05 |  ClassF1  |   SegF1   |   AucPro_0.05   |     ClassF1     |      SegF1      |
| :---------: | :---------: | :-------: | :-------: | :-------------: | :-------------: | :-------------: |
|             |  (private)  | (private) | (private) | (private_mixed) | (private_mixed) | (private_mixed) |
|     Can     |    30.28    |   60.93   |   1.86    |      20.03      |      65.04      |      0.84       |
|   Fabric    |    79.45    |   83.79   |   87.46   |      79.27      |      83.80      |      73.37      |
| Fruit Jelly |    74.46    |   87.35   |   53.63   |      74.11      |      87.55      |      52.62      |
|    Rice     |    62.27    |   72.00   |   63.86   |      63.89      |      73.45      |      63.23      |
| Sheet Metal |    75.51    |   87.68   |   70.98   |      73.54      |      86.69      |      70.92      |
|    Vial     |    76.81    |   84.61   |   48.73   |      69.59      |      85.77      |      48.83      |
| Wall Plugs  |    62.20    |   75.20   |   14.38   |      24.77      |      72.66      |      3.40       |
|   Walnuts   |    77.05    |   85.42   |   67.13   |      72.00      |      83.95      |      58.94      |
|    Mean     |    67.25    |   79.62   |   51.00   |      59.65      |      79.86      |      46.52      |


## Thanks

Our repo is built on [INP-Former](https://github.com/luow23/INP-Former), thanks their clear and elegant code !

## License
RoBiS is released under the **MIT Licence**, and is fully open for academic research and also allow free commercial usage. To apply for a commercial license, please contact yuzhou@hust.edu.cn.