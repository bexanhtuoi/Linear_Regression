# Input là mảng 2 chiều, vd: X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
#                            y = np.array([[1], [2], [3])
class LR: # Linear Regression
    import numpy as np
    def __init__(self, learning_rate=0.001, n_iters=1000, val_rate=0, optimizer="GD", lamda=0.9, alpha=0.9, regularization=None, beta_1=0.992, beta_2=0.999, gamma=0.9, epsilon=1e-8, mini_batch=None, decay=None):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.optimizer = optimizer
        self.regularization = regularization
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.gamma = gamma
        self.epsilon = epsilon
        self.cost_train = []
        self.cost_val = []
        self.v_dw = 0
        self.v_db = 0
        self.v_2_dw = 0
        self.v_2_db = 0
        self.val_rate = val_rate
        self.lamda = lamda
        self.alpha = alpha
        self.mini_batch = mini_batch
        self.decay = decay

    def cost_function(self, y, y_predicted):
      if self.regularization == "L1":
        return np.mean((y - y_predicted)**2) + self.lamda * np.sum(np.abs(self.weights))
      elif self.regularization == "L2":
        return np.mean((y - y_predicted)**2) + self.lamda * np.sum(self.weights**2)
      elif self.regularization == "ElasticNet":
        return np.mean((y - y_predicted)**2) + self.alpha * (self.lamda * np.sum(np.abs(self.weights)) + (1 - self.lamda) * np.sum(self.weights**2))
      else:
        return np.mean((y - y_predicted)**2)

    def Derivative(self, X, y, y_predicted):
        n_samples = X.shape[0]
        dw = (-2 / n_samples) * (X.T @ (y - y_predicted))
        db = (-2 / n_samples) * np.sum(y - y_predicted)
        return dw, db

    def mini_batch_data(self, X, y): # Output là 1 mảng 3 chiều
        minibatches = []
        if self.mini_batch == None:
          minibatches.append((X, y))
          return minibatches
        X_batch = []
        y_batch = []
        indices = np.random.permutation(X.shape[0])

        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(0, X.shape[0], self.mini_batch):
          X_batch_temp = X_shuffled[i:i + self.mini_batch]
          y_batch_temp = y_shuffled[i:i + self.mini_batch]
          X_batch.append(X_batch_temp)
          y_batch.append(y_batch_temp)

        if X.shape[0] % self.mini_batch != 0:
          X_batch_temp = X_shuffled[i:]
          y_batch_temp = y_shuffled[i:]
          X_batch.append(X_batch_temp)
          y_batch.append(y_batch_temp)

        minibatches = list(zip(X_batch, y_batch))
        return minibatches


    def Adam_optimizer(self, dw, db):
        self.v_dw = (self.beta_1 * self.v_dw) + (1 - self.beta_1) * dw
        self.v_db = (self.beta_1 * self.v_db) + (1 - self.beta_1) * db

        self.v_2_dw = (self.beta_2 * self.v_2_dw) + (1 - self.beta_2) * dw**2
        self.v_2_db = (self.beta_2 * self.v_2_db) + (1 - self.beta_2) * db**2

        self.weights -= self.lr * (self.v_dw / (1 - self.beta_1)) / ((self.v_2_dw / (1 - self.beta_2)))**0.5 + self.epsilon
        self.bias -= self.lr * (self.v_db / (1 - self.beta_1)) / ((self.v_2_db / (1 - self.beta_2)))**0.5 + self.epsilon

        return self.weights, self.bias

    def Momentum_optimizer(self, dw, db):
        self.v_dw = self.gamma * self.v_dw + (1 - self.gamma) * dw
        self.v_db = self.gamma * self.v_db + (1 - self.gamma) * db

        self.weights -= self.lr * self.v_dw
        self.bias -= self.lr * self.v_db

        return self.weights, self.bias

    def GD_optimizer(self, dw, db):
        self.weights -= self.lr * dw
        self.bias -= self.lr * db

        return self.weights, self.bias

    def fit(self, X, y):
        if self.val_rate != 0:
          X, X_val, y, y_val = train_test_split(X, y, test_size= self.val_rate, random_state=42)

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features).reshape(-1, 1)
        self.bias = 0

        minibatches = self.mini_batch_data(X, y)

        for t in range(self.n_iters):
          for minibatch in minibatches:
            X, y = minibatch
            y_predicted = X @ self.weights + self.bias
            y_predicted = y_predicted.reshape(-1, 1)

            y_val_predicted = X_val @ self.weights + self.bias
            y_val_predicted = y_val_predicted.reshape(-1, 1)

            dw, db = self.Derivative(X, y, y_predicted)
            if self.regularization == "L1":
              dw = dw + self.lamda * np.sign(self.weights)
            if self.regularization == "L2":
              dw = dw + 2 * self.lamda * self.weights
            if self.regularization == "ElasticNet":
              dw = dw + self.alpha * (self.lamda * (np.sign(self.weights) + (1 - self.lamda) * self.weights))

            if self.optimizer == "Adam":
              self.weights, self.bias = self.Adam_optimizer(dw, db)

            elif self.optimizer == "Momentum":
              self.weights, self.bias = self.Momentum_optimizer(dw, db)

            else:
              self.weights, self.bias = self.GD_optimizer(dw, db)
              
            if self.decay != None:
              self.lr = self.lr * (1 / (1 + self.decay * t))

          cost_train = self.cost_function(y, y_predicted)
          self.cost_train.append(cost_train)

          cost_val = self.cost_function(y_val, y_val_predicted)
          self.cost_val.append(cost_val)

    def predict(self, X):
        return X @ self.weights + self.bias


    def plot(self):
      import  matplotlib.pyplot as plt

      plt.plot([i for i in range(self.n_iters)], self.cost_train, label="Train Loss")
      plt.plot([i for i in range(self.n_iters)], self.cost_val, label="Validation Loss")
      plt.xlabel("Epochs")
      plt.ylabel("Loss")
      plt.legend()
      plt.show()