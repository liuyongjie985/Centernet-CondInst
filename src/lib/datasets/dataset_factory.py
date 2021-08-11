from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .sample.ddd import DddDataset
from .sample.exdet import EXDetDataset
from .sample.ctdet import CTDetDataset
from .sample.multi_pose import MultiPoseDataset
from .sample.ctseg import CTSegDataset

from .dataset.coco import COCO
from .dataset.qieti_tigan import qieti_tigan
from .dataset.oversea import oversea
from .dataset.photo_qieti import photo_qieti
from .dataset.photo_qieti_zhiyun import photo_qieti_zhiyun
from .dataset.bbk_formula import bbk_formula
from .dataset.sogou import sogou
from .dataset.sogou_clear import sogou_clear

dataset_factory = {
    'qieti_tigan': qieti_tigan,
    'oversea': oversea,
    'photo_qieti': photo_qieti,
    'photo_qieti_zhiyun': photo_qieti_zhiyun,
    'bbk_formula': bbk_formula,
    'sogou': sogou,
    'sogou_clear': sogou_clear
}

_sample_factory = {
    'exdet': EXDetDataset,
    'ctdet': CTDetDataset,
    'ddd': DddDataset,
    'multi_pose': MultiPoseDataset,
    'ctseg': CTSegDataset,
}


def get_dataset(dataset, task):
    class Dataset(dataset_factory[dataset], _sample_factory[task]):
        pass

    return Dataset
