from six.moves import cPickle as pickle
import json
import sys
import os
import numpy as np
import tensorflow as tf


# ========================== Read/Write Data ===================================
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def write_json(data, outfile):
    json_dir, _ = os.path.split(outfile)
    if not os.path.exists(json_dir):
        os.makedirs(json_dir)

    with open(outfile, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, cls=MyEncoder)


def read_json(data_dir):
    with open(data_dir, 'r') as f:
        output = json.load(f)
    return output


def read_pickle_from_file(filename):
    with tf.gfile.Open(filename, 'rb') as f:
        if sys.version_info >= (3, 0):
            data_dict = pickle.load(f, encoding='bytes')
        else:
            data_dict = pickle.load(f)
    return data_dict


# ========================== Original Cifar Data ===============================
def check_version(cifar_version):
    if cifar_version not in ['10', '100', '20']:
        raise ValueError('cifar version must be one of 10, 20, 100.')


def img_num(cifar_version):
    check_version(cifar_version)
    dt = {'10': 5000, '100': 500, '20': 2500}
    return dt[cifar_version]


def local_folder(cifar_version):
    check_version(cifar_version)
    dt = {'10': 'cifar-10-batches-py',
          '100': 'cifar-100-python',
          '20': 'cifar-100-python'}
    return dt[cifar_version]


def test_file(cifar_version):
    check_version(cifar_version)
    dt = {'10': 'test_batch',
          '100': 'test',
          '20': 'test'}
    return dt[cifar_version]


def local_label_names(cifar_version):
    check_version(cifar_version)
    dt = {
        '10': 'cifar-10-batches-py/batches.meta',
        '100': 'cifar-100-python/meta',
        '20': 'cifar-100-python/meta'
    }
    return dt[cifar_version]


def label_names(cifar_version):
    check_version(cifar_version)
    label_name = {'10': b'labels',
                  '100': b'fine_labels',
                  '20': b'coarse_labels'}
    return label_name[cifar_version]


# ======================== Imbalanced Cifar Data ===============================
def get_img_num_per_cls(cifar_version, imb_factor=None):
    """
    Get a list of image numbers for each class, given cifar version
    Num of imgs follows emponential distribution
    img max: 5000 / 500 * e^(-lambda * 0);
    img min: 5000 / 500 * e^(-lambda * int(cifar_version - 1))
    exp(-lambda * (int(cifar_version) - 1)) = img_max / img_min
    args:
      cifar_version: str, '10', '100', '20'
      imb_factor: float, imbalance factor: img_min/img_max,
        None if geting default cifar data number
    output:
      img_num_per_cls: a list of number of images per class
    """
    cls_num = int(cifar_version)
    img_max = img_num(cifar_version)
    if imb_factor is None:
        return [img_max] * cls_num
    img_num_per_cls = []
    for cls_idx in range(cls_num):
        num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
        img_num_per_cls.append(int(num))
    return img_num_per_cls
