import time
import torch
import matplotlib.pyplot as plt
import tensorflow as tf 
from torchvision.utils import save_image

'''
import yaml
def read_config(config_path):
    with open(config_path) as config_file:
        content = yaml.safe_load(config_file)
    return content
'''
def denormalize(images, means, stds):
  if len(images.shape) == 3:
    images = images.unsqueeze(0)
  means = torch.tensor(means).reshape(1, 3, 1, 1)
  stds = torch.tensor(stds).reshape(1, 3, 1, 1)
  return images * stds + means 

def download_grid(dl, imagenet_stats, image_file_path):
  i = 0
  while i < 1:
    for images, labels in dl:
        fig, ax = plt.subplots(figsize = (16,16))
        ax.set_xticks([]); ax.set_yticks([])
        save_image(denormalize(images[:64], *imagenet_stats), image_file_path)
        i+=1


def get_unique_filename(filename, is_model_name = False):
    if is_model_name:
        time_stamp = time.strftime("_on_%Y%m%d_at_%H%M%S_.pth")
    else :
        time_stamp = time.strftime("_on_%Y%m%d_at_%H%M%S")
    unique_filename  = f"{filename}_{time_stamp}"
    return unique_filename

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return round(param_group['lr'],6)

def get_classwise_img_count(dataset: object) -> dict:
    classes = dataset.dl.dataset.classes 
    data = dict.fromkeys( classes, 0)
    for label in dataset.dl.dataset.targets:
        label = classes[label]
        data[label] += 1
    return data