import warnings
warnings.filterwarnings('ignore')
from codeproject import RTDETR



if __name__ == '__main__':
    model = RTDETR('runs/images/exp/weights/best.pt')
    model.val(data='dataset/data.yaml',
              split='test',
              imgsz=640,
              batch=4,

              project='runs/val',
              name='exp',
              )