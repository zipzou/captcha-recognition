import torch

def recall(input, target):
  pass

def precision(input, target):
  pass

def acc(pred, y):
  """

  :param pred:
  :param y:
  :return:
  """
  pred = torch.argmax(pred, dim=-1)
  eq = pred == y
  return eq.mean(dtype=torch.float32).item()
  # return (eq.sum(dtype=torch.float32) / eq.shape[0]).item()


def multi_acc(pred, y):
  """

  :param pred:
  :param y:
  :return:
  """
  eq = pred == y
  all_eq = torch.all(eq, dim=-1)
  return torch.mean(all_eq, dtype=torch.float32).item()
  # return (all_eq.sum(dtype=torch.float32) / all.shape[0]).item()


