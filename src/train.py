import torch

import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

import numpy as np

import os

import json

# from tqdm import tqdm
from tqdm import tqdm_notebook as tqdm

from data import get_data_split
from dataloader import CaptchaLoader
from model import CaptchaModel

def acc(input, target):
  input = torch.argmax(input, dim=-1)
  eq = input == target
  return (eq.sum(dtype=torch.float32) / eq.shape[0]).item()

def recall(input, target):
  pass

def precision(input, target):
  pass

def save_history(filename, history, history_path):
  if not os.path.exists(history_path):
    os.mkdir(history_path)
  out_file = os.path.join(history_path, filename)
  with open(out_file, mode=r'w', encoding='utf-8') as out_fp:
    json.dump(history, out_fp)

def load_history(filename, history_path):
  in_path = os.path.join(history_path, filename)
  if not os.path.exists(in_path):
    return []
  with open(in_path, mode=r'r') as in_fp:
    history = json.load(in_fp)
  return history

def train(path, split=[6, 1, 1], batch_size=64, epochs=100, learning_rate=0.001, initial_epoch=0, step_saving=2, model_dir='./', log_file='./history', continue_pkl=None, gpu=True):

  x_train, y_train, x_dev, y_dev = get_data_split(path, split=split, modes=['train', 'dev'])

  train_ds = CaptchaLoader((x_train, y_train), shuffle=True)
  dev_ds = CaptchaLoader((x_dev, y_dev))

  train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
  dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=True)

  gpu_available = torch.cuda.is_available()

  model = CaptchaModel()
  optm = torch.optim.Adam(model.parameters(), lr=learning_rate)
  loss_fn = nn.CrossEntropyLoss()

  # start from a pickle
  if continue_pkl is None and os.path.exists(os.path.join(model_dir, 'model-latest.pkl')):
    latest_state = torch.load(os.path.join(model_dir, 'model-latest.pkl'))
    model.load_state_dict(latest_state['network'])
    optm.load_state_dict(latest_state['optimizer'])
    initial_epoch = latest_state['epoch'] + 1

  elif continue_pkl is not None and os.path.exists(os.path.join(model_dir, continue_pkl)):
    initial_state = torch.load(os.path.join(model_dir, continue_pkl))
    model.load_state_dict(initial_state['network'])
    optm.load_state_dict(initial_state['optimizer'])
    initial_epoch = initial_state['epoch'] + 1

  elif continue_pkl is None and initial_epoch is not None and os.path.exists(os.path.join(model_dir, 'model-%d.pkl' % initial_epoch)):
    initial_state = torch.load(os.path.join(model_dir, 'model-%d.pkl' % initial_epoch))
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


  if gpu and gpu_available:
    model = model.cuda()
    loss_fn = loss_fn.cuda()

  with tqdm(total=epochs, desc='Epoch', initial=initial_epoch) as epoch_bar:
    for epoch in range(initial_epoch, epochs):
      model.train()
      loss_batchs = []
      acc_batchs = []
      with tqdm(total=np.ceil(len(train_loader.dataset) / batch_size), desc='Batch') as batch_bar:
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

          loss_batchs.append(loss_count.item())
          acc_batchs.append(acc_mean)

          batch_bar.set_postfix(loss=loss_count.item(), acc=acc_mean)
          batch_bar.update()
          batch_history_train.append([loss_count.item(), acc_mean])
          save_history('history_batch.json', batch_history_train, log_file)

          loss_count.backward()
          optm.step()

      epoch_bar.set_postfix(loss_mean=np.mean(loss_batchs), acc_mean=np.mean(acc_batchs))
      epoch_bar.update()
      epoch_history_train.append([np.mean(loss_batchs).item(), np.mean(acc_batchs).mean()])
      save_history('history_epoch.json', epoch_history_train, log_file)

      # validate
      with tqdm(total=np.ceil(len(dev_loader.dataset) / batch_size), desc='Val Batch') as batch_bar:
        model.eval()
        loss_batchs_dev = []
        acc_batchs_dev = []
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

          loss_batchs_dev.append(loss_count.item())
          acc_batchs_dev.append(acc_mean)

          batch_bar.set_postfix(loss=loss_count.item(), acc=acc_mean)
          batch_bar.update()
          epoch_history_dev.append([np.mean(loss_batchs_dev), np.mean(acc_batchs_dev)])
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
