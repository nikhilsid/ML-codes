import numpy as np
import matplotlib.pyplot as plt


class LinearReg:
  def __init__(self, epochs=1000, lr=0.01):
    self.epochs = epochs
    self.lr = lr

  def ols_matrix(self, X, y):
    ones = np.ones((X.shape[0],1))
    X = np.hstack((X, ones))

    inv = np.linalg.inv(np.dot(X.T, X))
    betas = np.dot(np.dot(inv, X.T), y)

    return betas
    

  def grad_descent_by_epochs(self, X, y):
    n = X.shape[0]
    beta1 = 0
    beta0 = 0

    for i in range(self.epochs):
      y_pred = beta1*X + beta0
      Dbeta1 = (-1.0/n) * np.dot(X.T, (y - y_pred))
      Dbeta0 = (-1.0/n) * np.sum((y - y_pred))

      beta1 -= self.lr * Dbeta1
      beta0 -= self.lr * Dbeta0
    
    return beta1, beta0


  def grad_descent_matrix(self, X, y):
    ones = np.ones((X.shape[0],1))
    X = np.hstack([X,ones])

    n = X.shape[0]
    betas = np.zeros((X.shape[1],1))

    for i in range(self.epochs):
      y_pred = np.dot(X, betas)
      DLoss = np.dot(X.T, y_pred-y)/n

      betas -= self.lr*DLoss

    return betas

def plot_regression_line(X, y, b, color_, label_, plot): 
    # plotting the points as per dataset on a graph
    plt.scatter(X, y, color = "m", marker = "o", s = 30) 

    # predicted response vector 
    y_pred = b[0]*X + b[1]
  
    # plotting the regression line
    plt.plot(X, y_pred, color = color_, label=label_)
  
    # putting labels for x and y axis
    plt.xlabel('Size') 
    plt.ylabel('Cost')
  
    # function to show plotted graph
    if plot:
      plt.legend(loc='best')
      plt.show()


def main():
  X = np.array([ 1,   2,   3,   4,   5,   6,   7,   8,    9,   10])
  y = np.array([300, 350, 500, 700, 800, 850, 900, 900, 1000, 1200])
  X.shape = (np.size(X),1)
  y.shape = (np.size(y),1)
  model = LinearReg()
  betas = model.grad_descent_by_epochs(X,y)
  plot_regression_line(X, y, betas, "g", "Grad Descent", False)
  betas = model.grad_descent_matrix(X,y)
  plot_regression_line(X, y, betas, "red", "Grad Descent Matrix", False)
  beta_ols = model.ols_matrix(X,y)
  plot_regression_line(X, y, beta_ols, "blue", "OLS", True)



if __name__ == '__main__':
  main()