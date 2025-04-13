import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR



if __name__ == '__main__':
    model = RTDETR('ultralytics/cfg/models/rt-detr/rtdetr-imp.yaml')
    model.train(data='dataset/data.yaml',
                project='runs/train',
                name='exp',
                )