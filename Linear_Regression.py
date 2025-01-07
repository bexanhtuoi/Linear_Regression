class LR: # Linear Regression
    import numpy as np
    def __init__(self, learning_rate=0.001, n_iters=1000, val_rate=0.2, optimizer="GD", lamda=0.9, alpha=0.9, regularization=None, beta_1=0.992, beta_2=0.999, gamma=0.9, epsilon=1e-8, mini_batch=None):
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

    def cost_function(self, y, y_predicted):
      if self.regularization == "L1":
        return np.mean((y - y_predicted)**2) + self.lamda * np.sum(np.abs(self.weights))
      elif self.regularization == "L2":
        return np.mean((y - y_predicted)**2) + self.lamda * np.sum(self.weights**2)
      elif self.regularization == "ElasticNet":
        return np.mean((y - y_predicted)**2) + self.alpha * (self.lamda * np.sum(np.abs(self.weights)) + (1 - self.lamda) * np.sum(self.weights**2))
      else:
        return np.mean((y - y_predicted)**2)

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
          X_batch_temp = X_shuffled[i + self.mini_batch:]
          y_batch_temp = y_shuffled[i + self.mini_batch:]
          X_batch.append(X_batch_temp)
          y_batch.append(y_batch_temp)

        minibatches = list(zip(X_batch, y_batch))
        return minibatches

    def fit(self, X, y):
        X, X_val, y, y_val = train_test_split(X, y, test_size= self.val_rate, random_state=42)

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features).reshape(-1, 1)
        self.bias = 0

        minibatches = self.mini_batch_data(X, y)

        for _ in range(self.n_iters):
          for minibatch in minibatches:
            X, y = minibatch
            y_predicted = X @ self.weights + self.bias
            y_predicted = y_predicted.reshape(-1, 1)

            y_val_predicted = X_val @ self.weights + self.bias
            y_val_predicted = y_val_predicted.reshape(-1, 1)

            if self.regularization == "L1":
              dw = (1 / n_samples) * (X.T @ (y_predicted - y)) + self.lamda * np.sign(self.weights)
            elif self.regularization == "L2":
              dw = (1 / n_samples) * (X.T @ (y_predicted - y)) + 2 * self.lamda * self.weights
            elif self.regularization == "ElasticNet":
              dw = (1 / n_samples) * (X.T @ (y_predicted - y)) + self.alpha * (self.lamda * (np.sign(self.weights) + (1 - self.lamda) * self.weights))
            else:
              dw = (1 / n_samples) * (X.T @ (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            if self.optimizer == "Adam":
              self.v_dw = (self.beta_1 * self.v_dw) + (1 - self.beta_1) * dw
              self.v_db = (self.beta_1 * self.v_db) + (1 - self.beta_1) * db

              self.v_2_dw = (self.beta_2 * self.v_2_dw) + (1 - self.beta_2) * dw**2
              self.v_2_db = (self.beta_2 * self.v_2_db) + (1 - self.beta_2) * db**2

              self.weights -= self.lr * (self.v_dw / (1 - self.beta_1)) / ((self.v_2_dw / (1 - self.beta_2)))**0.5 + self.epsilon
              self.bias -= self.lr * (self.v_db / (1 - self.beta_1)) / ((self.v_2_db / (1 - self.beta_2)))**0.5 + self.epsilon

            elif self.optimizer == "Momentum":
              self.v_dw = self.gamma * self.v_dw + (1 - self.gamma) * dw
              self.v_db = self.gamma * self.v_db + (1 - self.gamma) * db

              self.weights -= self.lr * self.v_dw
              self.bias -= self.lr * self.v_db

            else:
              self.weights -= self.lr * dw
              self.bias -= self.lr * db

          cost_train = self.cost_function(y, y_predicted)
          self.cost_train.append(cost_train)

          cost_val = self.cost_function(y_val, y_val_predicted)
          self.cost_val.append(cost_val)

    def predict(self, X):
        return X @ self.weights + self.bias