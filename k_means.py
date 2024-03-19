import numpy as np
import random
from collections import defaultdict

class kMeans():
  def __init__(self, n, ITER_MAX):
    """
    Intialize the clusters with random values, such that all the centers are distinct
    """
    self.ITER_MAX = ITER_MAX
    self.n = n
    self.cluster = []
    self.cluster_map = defaultdict(list)

  def calculate_centers(self):
    """
    Function to calcualte cluster centers
    """
    new_cluster = []
    for i in range(len(self.cluster)):
      if len(self.cluster_map[i]) == 0:
        new_cluster.append(self.cluster[i])
      else:
        new_center = np.mean(self.cluster_map[i])
        new_cluster.append(new_center)
    self.cluster = new_cluster

  def train(self, data):
    iter = 0
    self.cluster = np.linspace(min(data), max(data), num = self.n)
    while(iter<self.ITER_MAX):
      self.cluster_map = defaultdict(list)
      for i in data:
        l2 = [(j-i)**2 for j in self.cluster]
        chosen_cluster = l2.index(min(l2))
        self.cluster_map[chosen_cluster].append(i)
      self.calculate_centers()
      iter = iter+1
    return self.cluster, self.cluster_map
  
if __name__ == "__main__":
    n = 10
    data = [i for i in range(1, 11)]*10
    kmeans = kMeans(10, 3)
    clusters, y = kmeans.train(data)
    print (clusters)
    