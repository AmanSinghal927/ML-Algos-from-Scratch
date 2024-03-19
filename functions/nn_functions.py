import torch

def ReLU(x):
  x[x<0] = 0
  return x

def softmax(x):
  x = torch.exp(x - x.max(dim=-1, keepdim = True).values)
  x = x/x.sum(dim=-1).unsqueeze(1)
  return x

# torch.nn.functional.one_hot (num_classes = -1)
def one_hot_encode(x, n_labels):
  one_hot = torch.zeros((x.size()[0], n_labels))
  one_hot[torch.arange(len(x)), x] = 1
  return one_hot

def crossentropy_loss(labels, probs):
  correct_probs = probs[torch.arange(len(labels)), labels]
  log_probs = torch.log(correct_probs)
  return -log_probs.mean()