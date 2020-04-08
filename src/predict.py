import torch

from model import CaptchaModel

from PIL import Image
from torchvision.transforms import Compose, ToTensor

from data import get_dict

def predict(captcha, model_dir='./model/model-latest.pkl', use_gpu=True):
  gpu_available = torch.cuda.is_available()

  model = CaptchaModel()

  if use_gpu and gpu_available:
    model_state = torch.load(model_dir)
  else:
    model_state = torch.load(model_dir, map_location=lambda storage, loc: storage)

  model.load_state_dict(model_state['network'])

  if use_gpu and gpu_available:
    model = model.cuda()
  else:
    model = model.cpu()

  transformer = Compose(ToTensor())

  img_pil = Image.open(captcha)
  img_tensor = transformer.transforms(img_pil)

  model.eval()
  x = torch.stack([img_tensor])
  if use_gpu and gpu_available:
    x = x.cuda()
  pred1, pred2, pred3, pred4 = model(x)

  pred_seq = [torch.argmax(pred1).item(), torch.argmax(pred2).item(), torch.argmax(pred3).item(), torch.argmax(pred4).item()]
  pred_seq = [item + 1 for item in pred_seq]

  _, id2label = get_dict()

  res = ''.join([id2label[i] for i in pred_seq])

  return res

import matplotlib.pyplot as plt

plt.imshow