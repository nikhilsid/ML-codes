import numpy as np
from pprint import pprint

class DecisionTree:
  def __init__(self):
    pass

  def entropy(self, Y):
    entropy = 0
    vals, counts = np.unique(Y, return_counts=True)
    probabs = counts.astype('float')/Y.size

    for p in probabs:
      if p != 0:
        entropy -= p * np.log2(p)

    return entropy


  def info_gain(self, X, Y):
    gain = self.entropy(Y)

    vals, counts = np.unique(X, return_counts=True)
    probabs = counts.astype('float')/Y.size

    for p, v in zip(probabs, vals):
      gain -= p * self.entropy(Y[X==v])

    return gain

  def create_subsets(self, X):
    return {key: (X==key).nonzero()[0] for key in np.unique(X)}


  def create_tree(self, X, Y):
    if len(set(Y)) <= 1:
      return Y

    #select best attribute
    gains = np.array([self.info_gain(x_attr, Y) for x_attr in X.T])
    selected_attr = np.argmax(gains)

    # can ignore
    if np.all(gains < 1e-6):
      return Y

    #create subrees
    subtree_map = self.create_subsets(X[:, selected_attr])

    subtree = {}
    for key in subtree_map:
      x_subset = X.take(subtree_map[key], axis=0)
      y_subset = Y.take(subtree_map[key], axis=0)
      subtree['attrib='+str(selected_attr)+', val='+str(key)] = self.create_tree(x_subset, y_subset)


    return subtree


def main():
  x1 = [0, 1, 1, 2, 2, 2] # sunny, cloudy, rainy
  x2 = [0, 0, 1, 1, 1, 0] # heat/no heat
  y = np.array([0, 0, 0, 1, 1, 0]) # rain/no rain
  X = np.array([x1, x2]).T #X -> no. of samples(6) X no. of features(2)
  pprint (X)
  pprint(y.shape)


  dt = DecisionTree()
  tree = dt.create_tree(X,y)
  pprint(tree)

if __name__ == "__main__":
  main()
