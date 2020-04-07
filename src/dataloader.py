import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import torchvision as tv
from PIL import Image

class CaptchaLoader(Dataset):
  def __init__(self, data, shuffle=True):
    super(CaptchaLoader, self).__init__()
    self.shuffle = shuffle
    x_data = data[0]
    self.y_data = data[1]

    self.image_transformer = tv.transforms.Compose(tv.transforms.ToTensor())
    
    self.x_data = []
    for path in x_data:
      img_pil = Image.open(path)
      self.x_data.append(self.image_transformer.transforms(img_pil))

  def __len__(self):
    return self.y_data.shape[0]

  def __getitem__(self, index):
    actualIndex = index % self.y_data.shape[0] # avoid out of bound

    return self.x_data[actualIndex], torch.tensor(self.y_data[actualIndex], dtype=torch.int64)
