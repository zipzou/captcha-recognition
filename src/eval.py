import torch

from data import get_data_split

from dataloader import CaptchaLoader

from train import acc

from model import CaptchaModel

from tqdm import tqdm

import numpy as np

from torch.utils.data.dataloader import DataLoader

import json

import os 

# from tqdm import tqdm_notebook as tqdm

def eval(model_dir, data_dir, batch_size=64, log_dir='./logs', use_gpu=True):
  """
  :param model_dir: 
  :return: 
  """
  x_test, y_test = get_data_split(data_dir, modes=['test'])

  model = CaptchaModel()

  gpu_available = torch.cuda.is_available()

  if use_gpu and gpu_available:
    model = model.cuda()
    model_state = torch.load(model_dir)
  else:
    model_state = torch.load(model_dir, map_location=lambda storage, loc: storage)

  model.load_state_dict(model_state['network'])

  test_ds = CaptchaLoader((x_test, y_test), shuffle=True)

  test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

  model.eval()

  acc_history = []
  with tqdm(total=int(np.ceil(len(test_loader.dataset) / batch_size)), desc='Eval') as eval_bar:
    for _, (x, y) in enumerate(test_loader):
      x = torch.tensor(x, requires_grad=False)
      y = torch.tensor(y, requires_grad=False)

      if use_gpu and gpu_available:
        x = x.cuda()
        y = y.cuda()

      pred1, pred2, pred3, pred4 = model(x)
      acc_mean = np.mean(
        [acc(pred1, y[:,0]), acc(pred2, y[:,1]), acc(pred3, y[:,2]), acc(pred4, y[:,3])]
      )

      acc_history.append(acc_mean.item())

      eval_bar.update()
      eval_bar.set_postfix(acc=acc_mean)

  if not os.path.exists(log_dir):
    os.mkdir(log_dir)
  with open(os.path.join(log_dir, 'eval.json'), mode=r'w') as out_fp:
    json.dump(acc_history, out_fp)
