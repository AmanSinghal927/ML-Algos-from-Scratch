import torch
import numpy as np
from functions.nn_functions import ReLU, softmax, crossentropy_loss, one_hot_encode

class NN():
  def __init__(self, n_input, n_hidden, n_output):
    self.n_output = n_output
    self.m1 = torch.rand(n_input, n_hidden) # input x 512
    self.m2 = torch.rand(n_hidden, n_output) # 512x10
    self.del_W2 = torch.zeros(self.m2.size()[0], self.m2.size()[1])
    self.del_W1 = torch.zeros(self.m1.size()[0], self.m1.size()[1])

  def forward(self, batch):
    self.batch = batch
    self.z1 = torch.matmul(batch, self.m1)
    self.a1 = ReLU(self.z1)
    self.z2 = torch.matmul(self.a1, self.m2) # batch x output
    self.batch_out = softmax(self.z2)
    return self.batch_out

  def loss(self, labels):
    self.labels = labels
    return crossentropy_loss(labels, self.batch_out)

  def backward(self):
    del_L = self.batch_out-one_hot_encode(self.labels, self.n_output)
    self.del_W2 = torch.matmul(self.a1.T, del_L)
    self.del_hidden = torch.matmul(del_L, self.m2.T)
    self.del_W1 = torch.matmul(self.batch.T, (self.z1>0).float()*self.del_hidden) # note that derivative of ReLU is self.z1>0


  def update(self, lr):
    self.m1 = self.m1 - lr* self.del_W1
    self.m2 = self.m2 - lr* self.del_W2
    self.del_W1 = torch.zeros_like(self.m1)
    self.del_W2 = torch.zeros_like(self.m2)

if __name__ == "__main__":
    # create data from scratch
    batch_size = 5
    labels = torch.tensor([0,0,1,1,1])
    data1 = torch.tensor([1,10]).repeat(batch_size//2, 1)
    data2 = torch.tensor([3,10]).repeat(batch_size//2+1, 1)
    data = torch.cat((data1, data2), dim = 0)
    data = torch.tensor(data, dtype = torch.float32)

    # Model initialization
    n_input = 2  
    n_hidden = 512
    n_output = 2
    model = NN(n_input, n_hidden, n_output)

    """
    Training Loop
    """
    n_epochs = 1000
    for i in range(n_epochs):
        preds = model.forward(batch = data)
        model.loss(labels=labels)
        model.backward()
        model.update(lr=0.001)
        
    print (np.argmax(preds, axis = -1), labels)