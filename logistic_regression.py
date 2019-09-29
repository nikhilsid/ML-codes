import numpy as np

class Logistic:
  def __init__(self, epochs=10000, lr=0.001):
    self.epochs = epochs
    self.lr = lr

  def DLoss (self, x, y, y_pred): # TO Learn the formula!!! same as Lin Reg beta1
    return np.dot(x.T, y_pred-y)/y.size

  def sigmoid(self, x):
    return 1.0/(1+np.exp(-x))

  def add_intercept(self, x):
    ones = np.ones((x.shape[0],1))
    return np.hstack((x,ones))

  def predict_proba(self, x):
    x = self.add_intercept(x)
    return self.sigmoid(np.dot(x, self.w))

  def predict(self, x):
    return self.predict_proba(x).round()


  def fit(self, x, y):
    x = self.add_intercept(x)
    self.w = np.zeros((x.shape[1],1))

    for i in range(self.epochs):
      y_pred = self.sigmoid(np.dot(x, self.w))
      DLoss = self.DLoss(x, y, y_pred)

      self.w -= self.lr * DLoss





def main():
  X = np.array([2,3,4,3,4,5,-5,-4,-2,-10,-3,-4])
  X.shape = (4,3)
  Y = np.array([1,1,0,0])
  Y.shape = (np.size(Y),1)
  model = Logistic()
  print ('Data:\n', X)
  print ('Real Labels: ', Y)
  print ('--------------')
  model.fit(X,Y)
  print ('Predictions: ', model.predict(X))

if __name__ == "__main__": 
  main()
