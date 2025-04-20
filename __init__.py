# Ultralytics YOLO 🚀, AGPL-3.0 license

__version__ = '8.0.201'

from codeproject.models import RTDETR, SAM, YOLO
from codeproject.models.fastsam import FastSAM
from codeproject.models.nas import NAS
from codeproject.utils import SETTINGS as settings
from codeproject.utils.checks import check_yolo as checks
from codeproject.utils.downloads import download

__all__ = '__version__', 'YOLO', 'NAS', 'SAM', 'FastSAM', 'RTDETR', 'checks', 'download', 'settings'
