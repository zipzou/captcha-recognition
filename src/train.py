import torch

import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

import numpy as np

import os

import json

# from tqdm import tqdm
from tqdm.notebook import tqdm

from data import get_data_split
from dataloader import CaptchaLoader

from metrics import acc, multi_acc


def save_history(filename, history, history_path):
  """

  :param filename:
  :param history:
  :param history_path:
  :return:
  """
  if not os.path.exists(history_path):
    os.mkdir(history_path)
  out_file = os.path.join(history_path, filename)
  with open(out_file, mode=r'w', encoding='utf-8') as out_fp:
    json.dump(history, out_fp)

def load_history(filename, history_path):
  """

  :param filename:
  :param history_path:
  :return:
  """
  in_path = os.path.join(history_path, filename)
  if not os.path.exists(in_path):
    return []
  with open(in_path, mode=r'r') as in_fp:
    history = json.load(in_fp)
  return history

def train(path, split=[6, 1, 1], batch_size=64, epochs=100, learning_rate=0.001, initial_epoch=0, step_saving=2, model_dir='./', log_file='./history', continue_pkl=None, gpu=True, mode='captcha'):
  """

  :param path:
  :param split:
  :param batch_size:
  :param epochs:
  :param learning_rate:
  :param initial_epoch:
  :param step_saving:
  :param model_dir:
  :param log_file:
  :param continue_pkl:
  :param gpu:
  :param mode:
  :return:
  """
  if mode == 'captcha':
    from model import CaptchaModel
    CaptchaModelDynamic = CaptchaModel
  elif mode == 'kaptcha':
    from kaptcha_model import CaptchaModel
    CaptchaModelDynamic = CaptchaModel
  else:
    return
  x_train, y_train, x_dev, y_dev = get_data_split(path, split=split, modes=['train', 'dev'])

  train_ds = CaptchaLoader((x_train, y_train), shuffle=True)
  dev_ds = CaptchaLoader((x_dev, y_dev))

  train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
  dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=True)

  gpu_available = torch.cuda.is_available()

  model = CaptchaModelDynamic()
  optm = torch.optim.Adam(model.parameters(), lr=learning_rate)
  loss_fn = nn.CrossEntropyLoss()

  if gpu and gpu_available:
    model = model.cuda()
    loss_fn = loss_fn.cuda()

  # start from a pickle
  if continue_pkl is not None and os.path.exists(os.path.join(model_dir, continue_pkl)):
    if gpu and gpu_available:
      initial_state = torch.load(os.path.join(model_dir, continue_pkl))
    else:
      initial_state = torch.load(os.path.join(model_dir, continue_pkl), map_location=lambda storage, loc: storage)
    model.load_state_dict(initial_state['network'])
    optm.load_state_dict(initial_state['optimizer'])
    initial_epoch = initial_state['epoch'] + 1

  elif continue_pkl is not None and os.path.exists(os.path.join(model_dir, 'model-latest.pkl')):
    if gpu and gpu_available:
      latest_state = torch.load(os.path.join(model_dir, 'model-latest.pkl'))
    else:
      latest_state = torch.load(os.path.join(model_dir, 'model-latest.pkl'), map_location=lambda storage,loc: storage)
    model.load_state_dict(latest_state['network'])
    optm.load_state_dict(latest_state['optimizer'])
    initial_epoch = latest_state['epoch'] + 1

  elif continue_pkl is not None and initial_epoch is not None and os.path.exists(os.path.join(model_dir, 'model-%d.pkl' % initial_epoch)):
    if gpu and gpu_available:
      initial_state = torch.load(os.path.join(model_dir, 'model-%d.pkl' % initial_epoch))
    else:
      initial_state = torch.load(os.path.join(model_dir, 'model-%d.pkl' % initial_epoch), map_location=lambda storage, _: storage)
    model.load_state_dict(initial_state['network'])
    optm.load_state_dict(initial_state['optimizer'])
    initial_epoch = initial_state['epoch'] + 1
  else:
    initial_epoch = 0

  # load history
  batch_history_train = load_history(filename='history_batch_train.json', history_path=log_file)
  epoch_history_train = load_history(filename='history_epoch_train.json', history_path=log_file)
  epoch_history_dev = load_history(filename='history_epoch_dev.json', history_path=log_file)
  # slice
  batch_history_train = batch_history_train[:initial_epoch]
  epoch_history_train = epoch_history_train[:initial_epoch]
  epoch_history_dev = epoch_history_dev[:initial_epoch]

  with tqdm(total=epochs, desc='Epoch', initial=initial_epoch) as epoch_bar:
    for epoch in range(initial_epoch, epochs):
      model.train()
      loss_batchs = []
      acc_batchs = []
      multi_acc_batchs = []
      with tqdm(total=int(np.ceil(len(train_loader.dataset) / batch_size)), desc='Batch') as batch_bar:
        for batch, (x, y) in enumerate(train_loader):
          optm.zero_grad()
          x = torch.tensor(x, requires_grad=True)
          y = torch.tensor(y)
          if gpu and gpu_available:
            x = x.cuda()
            y = y.cuda()
          pred_1, pred_2, pred_3, pred_4 = model(x)

          loss1, loss2, loss3, loss4 = loss_fn(pred_1, y[:,0]), loss_fn(pred_2, y[:,1]), loss_fn(pred_3, y[:,2]), loss_fn(pred_4, y[:,3])

          loss_count = loss1 + loss2 + loss3 + loss4
          acc_count = acc(pred_1, y[:,0]) + acc(pred_2, y[:,1]) + acc(pred_3, y[:,2]) + acc(pred_4, y[:,3])
          acc_mean = acc_count / 4.

          pred = torch.stack((pred_1, pred_2, pred_3, pred_4), dim=-1)
          multi_acc_mean = multi_acc(torch.argmax(pred, dim=1), y)

          loss_batchs.append(loss_count.item())
          acc_batchs.append(acc_mean)
          multi_acc_batchs.append(multi_acc_mean)

          batch_bar.set_postfix(loss=loss_count.item(), acc=acc_mean, multi_acc=multi_acc_mean)
          batch_bar.update()
          batch_history_train.append([loss_count.item(), acc_mean, multi_acc_mean])
          save_history('history_batch_train.json', batch_history_train, log_file)

          loss_count.backward()
          optm.step()

      epoch_bar.set_postfix(loss_mean=np.mean(loss_batchs), acc_mean=np.mean(acc_batchs), multi_acc_mean=np.mean(multi_acc_batchs))
      epoch_bar.update()
      epoch_history_train.append([np.mean(loss_batchs).item(), np.mean(acc_batchs).item(), np.mean(multi_acc_batchs).item()])
      save_history('history_epoch_train.json', epoch_history_train, log_file)

      # validate
      with tqdm(total=int(np.ceil(len(dev_loader.dataset) / batch_size)), desc='Val Batch') as batch_bar:
        model.eval()
        loss_batchs_dev = []
        acc_batchs_dev = []
        multi_acc_batchs_dev = []
        for batch, (x, y) in enumerate(dev_loader):
          x = torch.tensor(x, requires_grad=False)
          y = torch.tensor(y, requires_grad=False)
          if gpu and gpu_available:
            x = x.cuda()
            y = y.cuda()
          pred_1, pred_2, pred_3, pred_4 = model(x)

          loss1, loss2, loss3, loss4 = loss_fn(pred_1, y[:,0]), loss_fn(pred_2, y[:,1]), loss_fn(pred_3, y[:,2]), loss_fn(pred_4, y[:,3])

          loss_count = loss1 + loss2 + loss3 + loss4
          acc_count = acc(pred_1, y[:,0]) + acc(pred_2, y[:,1]) + acc(pred_3, y[:,2]) + acc(pred_4, y[:,3])
          acc_mean = acc_count / 4.

          pred = torch.stack((pred_1, pred_2, pred_3, pred_4), dim=-1)
          multi_acc_mean = multi_acc(torch.argmax(pred, dim=1), y)

          loss_batchs_dev.append(loss_count.item())
          acc_batchs_dev.append(acc_mean)
          multi_acc_batchs_dev.append(multi_acc_mean)

          batch_bar.set_postfix(loss=loss_count.item(), acc=acc_mean, multi_acc=multi_acc_mean)
          batch_bar.update()
        epoch_history_dev.append([np.mean(loss_batchs_dev).item(), np.mean(acc_batchs_dev).item(), np.mean(multi_acc_batchs_dev).item()])
        save_history('history_epoch_dev.json', epoch_history_dev, log_file)

      # saving
      if not os.path.exists(model_dir):
        os.mkdir(model_dir)
      state_dict = {
        'network': model.state_dict(),
        'optimizer': optm.state_dict(),
        'epoch': epoch
      }
      if epoch % step_saving == 0:
        model_path = os.path.join(model_dir, 'model-%d.pkl' % epoch)
        torch.save(state_dict, model_path)

      torch.save(state_dict, os.path.join(model_dir, 'model-latest.pkl'))


import click

@click.command()
@click.help_option('-h', '--help')
@click.option('-i', '--data_dir', default='./captchas', type=click.Path(), help='The path of train data', required=False)
@click.option('-m', '--mode', default='captcha', help='The model type to train, could be captcha or kaptcha', type=click.Choice(['captcha', 'kaptcha']), required=False)
@click.option('-e', '--epoch', default=120, help='The number of epoch model trained', required=False)
@click.option('-p', '--data_split', default=[6, 1, 1], nargs=3, type=int, help='The split of train data to split', required=False)
@click.option('-c', '--continue_train', default=None, help='If continue after last checkpoint or a specified one', required=False)
@click.option('-t', '--checkpoint', default=0, type=int, help='The initial checkpoint to start, if set, it will load model-[checkpoint].pkl', required=False)
@click.option('-b', '--batch_size', default=128, type=int, help='The batch size of input data', required=False)
@click.option('-o', '--model_dir', default='./captcha_models', type=click.Path(), help='The model dir to save models or load models', required=False)
@click.option('-r', '--lr', default=0.001, type=float, help='The learning rate to train', required=False)
@click.option('-l', '--log_dir', default='./logs', type=click.Path(), help='The log files path', required=False)
@click.option('-u', '--use_gpu', type=bool, default=False, help='Train by gpu or cpu', required=False)
@click.option('-s', '--save_frequency', default=2, type=int, help='The frequence to save the models during training', required=False)
def read_cli(data_dir, mode, epoch, data_split, continue_train, checkpoint, batch_size, model_dir, lr, log_dir, use_gpu, save_frequency):
  """

  :param data_dir:
  :param mode:
  :param epoch:
  :param data_split:
  :param continue_train:
  :param checkpoint:
  :param batch_size:
  :param model_dir:
  :param lr:
  :param log_dir:
  :param use_gpu:
  :param save_frequency:
  :return:
  """
  train(data_dir, data_split, batch_size, epoch, lr, checkpoint, save_frequency, model_dir, log_dir, continue_train, use_gpu, mode)


if __name__ == "__main__":
  read_cli()