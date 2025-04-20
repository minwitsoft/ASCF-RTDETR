import warnings
warnings.filterwarnings('ignore')
from codeproject import RTDETR



if __name__ == '__main__':
    model = RTDETR('codeproject/cfg/models/rt-detr/rtdetr-imp.yaml')
    model.train(data='dataset/data.yaml',
                project='runs/images',
                name='exp',
                )