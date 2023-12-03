# ANSAC

This repository contains the source code for "Classification of whole-slide images based on tumor infiltrating lymphocyte grades using image-level labels" paper. Kindly refer to our main paper for more details on ANSAC pipeline.

ANSAC pipeline consists of two stages: a) Pre-processing slides, b) Model training

![](./ansac_framework.png)

## Pre-requisites

- Linux (Tested on Red Hat Enterprise Linux Server release 7.9)
- NVIDIA GPU (Tested on Nvidia v100 GPUs on [Spartan](https://dashboard.hpc.unimelb.edu.au/))
- Python version 3.7.4
- [Openslide](https://openslide.org/) package was used to open, read and extract information from WSIs. Refer instructions [here](https://openslide.org/download/) to download OpenSlide into your OS.
- Python packages can be found in [requirements.txt](./requirements.txt) and can be installed via following command.

```
pip install -r requirements.txt
```

## Pre-processing slides

The scripts used for preprocessing are provided under `preprocessing_slides` folder.

1. Convert WSI into a PIL package friendly format(.ome.tif, .png, .tiff)
2. Extract patches and save patches: 

```
WSI_IMAGE_FOLDER=Path to folder containing converted WSIs from step 1.
VECTOR_PATH=Path to folder to store vector data
EXTENSION=Extension obtained from step 1

python vectorise_wsi.py ${WSI_IMAGE_FOLDER} ${VECTOR_PATH} {EXTENSION}
```


3. Extract Feature information for WSIs:
```
FEATURE_PATH = Path to folder to store feature data
MODEL_NAME = 'moco' # set this to 'moco' or 'imagenet' based on which encoder you want to use
MOCO_CKPT_PATH = Path to moco checkpoint if MODEL_NAME='moco', else set None

python -u featurize_slides.py ${VECTOR_PATH} ${FEATURE_PATH} ${MODEL_NAMES} ${MOCO_CKPT_PATH} 
```

4. Extract Segmentation information for WSIs:
```
SEGMENTATION_PATH = Path to folder to store segmentation data
SEGMENTER_CKPT_PATH = Path to segmentation model checkpoint

python -u segment_slides.py ${VECTOR_PATH} ${SEGMENTATION_PATH} ${SEGMENTER_CKPT_PATH}
```

## Model Training

The model specific scripts are provided under the 3 folders: `ansac`, `cnn` and `clam`.

### Train ANSAC
```
RESULTS_DIR = Path to result dir

python -u train_ansac.py ${RESULTS_DIR} ${CSV_FOLDER_CONTAINING_ANNOTATIONS} ${FEATURE_PATH} ${SEGMENTATION_PATH} \
${WSI_IMAGE_EXT} 
```

### Train CNN
```
RESULTS_DIR = Path to result dir

python -u train_cnn.py ${RESULTS_DIR} ${CSV_FOLDER_CONTAINING_ANNOTATIONS} ${FEATURE_PATH} ${WSI_IMAGE_EXT}
```

### Train CLAM (CLAM_MB) 

```
RESULTS_DIR = Path to result dir
MODEL_SIZE = set to 'small' for Imagenet features and to 'moco' for moco features

CUDA_VISIBLE_DEVICES=XX python -u main.py --drop_out --early_stopping --lr 2e-4 --k 1 --label_frac 0.8 \
	--exp_code ${RESULTS_DIR} --weighted_sample --bag_loss ce --inst_loss svm \
	--task task_4_tcga_binary --model_type 'clam_mb' --log_data --data_root_dir ${FEATURE_PATH} \
	--csv_dir ${CSV_FOLDER_CONTAINING_ANNOTATIONS} --model_size ${MODEL_SIZE} --drop_out \
	--seeds 42 333 2468 1369 2021 21 121 8642 7654 2010
```

## Model Evaluating

### Evaluating ANSAC
```
RESULTS_DIR = Path to result dir
CKPT_PATH = Path to folder containing trained models for the 10 random seeds (eg: './result_folder/ANSAC/')
TEST_DATASET_NAME = set as 'tcga' for 'TCGA-BRCA' for others set as 'Other'
CSV_FILE_CONTAINING_ANNOTATIONS = Path to csv file with test data

python -u eval_ansac.py ${RESULTS_DIR} ${CSV_FILE_CONTAINING_ANNOTATIONS} ${FEATURE_PATH} ${SEGMENTATION_PATH} \
${WSI_IMAGE_EXT} ${CKPT_PATH} ${TEST_DATASET_NAME}

```

### Evaluating CNN
```
RESULTS_DIR = Path to result dir
CKPT_PATH = Path to folder containing trained models for the 10 random seeds (eg: './result_folder/CNN/')
TEST_DATASET_NAME = set as 'tcga' for 'TCGA-BRCA' for others set as 'Other'
CSV_FILE_CONTAINING_ANNOTATIONS = Path to csv file with test data

python -u eval_cnn.py ${RESULTS_DIR} ${CSV_FILE_CONTAINING_ANNOTATIONS} ${FEATURE_PATH} ${WSI_IMAGE_EXT} \
${CKPT_PATH} ${TEST_DATASET_NAME}
```

### Evaluating CLAM (CLAM_MB) 

```
RESULTS_DIR = Path to result dir
MODEL_SIZE = set to 'small' for Imagenet features and to 'moco' for moco features
CKPT_PATH = Path to folder containing trained models for the 10 random seeds (eg: './result_folder/')
CSV_FILE_CONTAINING_ANNOTATIONS = Path to csv file with test data

CUDA_VISIBLE_DEVICES=XX python eval_clam.py --drop_out --k 1 --models_exp_code ${CKPT_PATH} \
	--save_exp_code ${RESULTS_DIR} --task task_3_cleo_binary --model_type 'clam_mb' \
	--results_dir results --data_root_dir ${FEATURE_PATH} --csv_dir ${CSV_FILE_CONTAINING_ANNOTATIONS} \
	--model_size ${MODEL_SIZE} --drop_out
```


### Data availability

TCGA slide image data can be accessed through the Genomic Data Commons (GDC) portal https://portal.gdc.cancer.gov/. TIL scores for the TCGA slides are available from the authors upon reasonable request. The slide images used in this study from clinical trials are subject to strict control and usage requirements and have been used with direct permission from the data custodians. To obtain the slide images and TIL scores requests should be made to the corresponding authors of the current study. TIGER slide image data along with TIL scores are available at https://tiger.grand-challenge.org/

### Trained model availability

Trained model weights for the MoCo encoder and the two mixed dataset classification models of this paper are available upon request from corresponding authors of this work.
Model weights for segmentation model can be requested from corresponding author of https://doi.org/10.1093/bioinformatics/btz083.

## Acknowledgement

This code thankfully re-uses/ modifies/ extends code shared by following repositories.

- https://github.com/mahmoodlab/CLAM
- https://github.com/davidtellez/neural-image-compression/
- https://github.com/facebookresearch/moco
- https://github.com/DigitalSlideArchive/HistomicsTK

## License
This source code is released under the GNU General Public License v3.0, included [here](./LICENSE).
