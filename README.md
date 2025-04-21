# ASCF-RTDETR: Adaptive Scale Collaborative Feature Learning for Precise Multichannel Fluorescence Epithelial Cell Detection

This is the official implementation of the "ASCF-RTDETR (Adaptive Scale Collaborative Fusion Real-time Detection Transformer)" algorithm. This algorithm is specifically designed for precise detection of epithelial cells in multichannel fluorescence images.

## Model Features

- Innovative hybrid encoder architecture, combining the advantages of CNN and Transformer
- Deformable attention mechanism, improving the efficiency and accuracy of feature extraction
- IoU-aware query selection strategy, optimizing the detection process
- Real-time inference capability, suitable for various practical application scenarios

## Environment Requirements

This code has been tested in the following environment:

- Python 3.9.19
- PyTorch 2.1.0+cu121
- CUDA 12.1
- For other dependencies, please refer to `requirements.txt`

## Installation

Create and activate a new Python environment, then install the required dependencies:

```bash
# Create environment
conda create -n rtdetr python=3.9
conda activate rtdetr

# Install PyTorch
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121

# Install other dependencies
pip install -r requirements.txt
```

## Dataset Preparation

Supports various common dataset formats, including COCO, VOC, and YOLO format.

Example dataset organization structure:

detect/

└── datasets/

├── Epithelial_Cell/

│ ├── train/

│ │ ├── images/

│ │ └── labels/

│ └── val/

│ ├── images/

│ └── labels/

## Training

Basic command for training the model:

```bash
python train.py --data data.yaml --weights '' --epochs 500 --batch-size 16 --device 0,1,2,3
```

You can customize the training process through the following parameters:

- `--img-size`: Set input image size
- `--batch-size`: Training batch size
- `--epochs`: Number of training epochs
- `--workers`: Number of worker processes for the data loader

## Evaluation

Evaluate the model's performance on the validation set:

```bash
python val.py --weights runs/train/exp/weights/best.pt --data data.yaml --img-size 640
```

We trained the ASCF-RTDETR model on our self-constructed multichannel fluorescence-labeled epithelial cell dataset and evaluated it on two public datasets:

1. **BCCD (Blood Cell Count and Detection) Dataset**  
   Source: Shenggan, GitHub (2017)  
   Link: https://github.com/Shenggan/BCCD_Dataset

2. **2018 Data Science Bowl (DSB2018) Dataset**  
   Source: Goodman, A., Carpenter, A., Park, E., et al., Kaggle (2018)  
   Link: https://kaggle.com/competitions/data-science-bowl-2018