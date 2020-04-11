import torch

from data import get_data_split

from dataloader import CaptchaLoader

from metrics import acc, multi_acc

# from tqdm import tqdm
from tqdm.notebook import tqdm

import numpy as np

from torch.utils.data.dataloader import DataLoader

import json

import os 

# from tqdm import tqdm_notebook as tqdm

def eval(model_dir, data_dir, batch_size=64, log_dir='./logs', use_gpu=True, mode='captcha'):
  """
  :param model_dir: 
  :param data_dir:
  :param batch_size:
  :param log_dir:
  :param use_gpu:
  :param mode:
  :return: 
  """
  x_test, y_test = get_data_split(data_dir, modes=['test'])
  if mode == 'captcha':
    from model import CaptchaModel
  elif mode =='kaptcha':
    from kaptcha_model import CaptchaModel
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

      pred = torch.stack((pred1, pred2, pred3, pred4), dim=-1)
      multi_acc_mean = multi_acc(torch.argmax(pred, dim=1), y)

      acc_history.append([acc_mean.item(), multi_acc_mean])

      eval_bar.update()
      eval_bar.set_postfix(acc=acc_mean, multi_acc=multi_acc_mean)

  if not os.path.exists(log_dir):
    os.mkdir(log_dir)
  with open(os.path.join(log_dir, 'eval.json'), mode=r'w') as out_fp:
    json.dump(acc_history, out_fp)


import click

@click.command()
@click.help_option('-h', '--help')
@click.option('-i', '--data_dir', default='./captchas', type=click.Path(), help='The path of train data', required=False)
@click.option('-m', '--mode', default='captcha', help='The model type to train, could be captcha or kaptcha', type=click.Choice(['captcha', 'kaptcha']), required=False)
@click.option('-b', '--batch_size', default=128, type=int, help='The batch size of input data', required=False)
@click.option('-o', '--model_dir', default='./captcha_models/model-latest.pkl', type=click.Path(), help='The model dir to save models or load models', required=False)
@click.option('-l', '--log_dir', default='./logs', type=click.Path(), help='The log files path', required=False)
@click.option('-u', '--use_gpu', type=bool, default=False, help='Train by gpu or cpu', required=False)
def read_cli(data_dir, mode, batch_size, model_dir, log_dir, use_gpu):
  """

  :param data_dir:
  :param mode:
  :param batch_size:
  :param model_dir:
  :param log_dir:
  :param use_gpu:
  :return:
  """
  eval(model_dir, data_dir, batch_size, log_dir, use_gpu, mode)



if __name__ == "__main__":
  read_cli()