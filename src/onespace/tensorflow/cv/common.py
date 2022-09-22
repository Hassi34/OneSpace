import time
import tensorflow as tf
import numpy as np 
from PIL import Image
'''
import yaml
def read_config(config_path):
    with open(config_path) as config_file:
        content = yaml.safe_load(config_file)
    return content
'''
def get_unique_filename(filename, is_model_name = False):
    if is_model_name:
        time_stamp = time.strftime("_on_%Y%m%d_at_%H%M%S_.h5")
    else :
        time_stamp = time.strftime("_on_%Y%m%d_at_%H%M%S")
    unique_filename  = f"{filename}_{time_stamp}"
    return unique_filename
def set_memory_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

def img_to_array(path):
  img = Image.open(path)
  img = np.asarray(img)
  return img

def get_classwise_img_count(classlabels: list, classnames: list) -> dict:
    data = dict.fromkeys(classnames, 0)
    for label in classlabels:
        label = classnames[label]
        data[label] += 1
    return data
