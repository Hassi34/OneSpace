import tensorflow as tf
import numpy as np
from .common import get_default_device, DeviceDataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as tt
from pathlib import Path
import os, shutil

def remove_ipynb_checkpoints(path):
    for root, dirnames, fnames in os.walk(path):
        for dirname in dirnames:
            dirpath = os.path.join(root, dirname)
            if dirpath.endswith(".ipynb_checkpoints"):
                shutil.rmtree(dirpath)

def get_data_loaders(VALIDATION_SPLIT, IMAGE_SIZE, BATCH_SIZE, data_dir, DATA_AUGMENTATION, train_dir = None, val_dir = None):

  if DATA_AUGMENTATION:
    augmentation = [#tt.RandomCrop(32, padding=4, padding_mode='reflect'), 
                    tt.RandomHorizontalFlip(),
                    tt.RandomRotation((-45,45))
                    #tt.RandomResizedCrop(32, scale=(0.5,0.9), ratio=(1, 1)), 
                    #tt.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
                    ]
  else:
    augmentation = []
  imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
  data_transform = { "train": tt.Compose(augmentation + [tt.Resize(size =(IMAGE_SIZE[1], IMAGE_SIZE[1])),
                              tt.ToTensor(), tt.Normalize(*imagenet_stats,inplace=True)]),
                      "val" : tt.Compose([tt.Resize(size =(IMAGE_SIZE[1], IMAGE_SIZE[1])), tt.ToTensor(), tt.Normalize(*imagenet_stats)])
                      }

  remove_ipynb_checkpoints(data_dir)

  if not isinstance(train_dir, type(None)):
    train_ds = ImageFolder(root = train_dir, transform = data_transform["train"])
    val_ds = ImageFolder(root = val_dir, transform = data_transform["val"])
  else:
    dataset = ImageFolder(root = data_dir, transforms = data_transform["train"])
    val_size = int(VALIDATION_SPLIT * len(dataset))
    train_size = len(dataset) - val_size
    train_ds , val_ds = random_split(dataset, [train_size, val_size])

  train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle = True)
  val_dl = DataLoader(val_ds , BATCH_SIZE*2, shuffle = True)
  device = get_default_device()

  train_dl = DeviceDataLoader(train_dl, device)
  val_dl = DeviceDataLoader(val_dl, device)
  training_images = len(train_ds)
  val_images = len(val_ds)

  num_classes = len(train_ds.classes)
  
  class_names = train_ds.classes
  return train_dl, val_dl, training_images, val_images, num_classes, class_names


def manage_input_data(input_image, IMAGE_SIZE):
    """converting the input array into desired dimension
    Args:
        input_image (nd array): image nd array
    Returns:
        nd array: resized and updated dim image
    """
    images = input_image
    size = IMAGE_SIZE[:-1]
    resized_input_img = tf.image.resize(images, size)
    final_img = np.expand_dims(resized_input_img, axis=0)
    
    return final_img