import numpy as np

class Linear_Regression:
    def __init__(self, weight_size, bias=True, learning_rate=0.01, reg='l1', reg_strength=0.01):
        self.learning_rate = learning_rate
        self.w = np.random.normal(size=weight_size)
        self.use_bias = bias
        if self.use_bias:
            self.b = np.random.normal()
        
        self.dw = np.zeros_like(self.w)
        if self.use_bias:
            self.db = 0.0
        
        self.reg=reg
        self.reg_strength=reg_strength
    def predict(self, X):

        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        prediction = np.dot(X, self.w)
        if self.use_bias:
            prediction += self.b
        return prediction 

    def get_weights(self):
        return self.w, self.b if self.use_bias else self.w

    def zero_grad(self):
        self.dw = np.zeros_like(self.w)
        if self.use_bias:
            self.db = 0.0
    
    def compute_gradients(self, X_batch, loss_grad_batch):
        

        if loss_grad_batch.ndim > 1 and loss_grad_batch.shape[1] == 1:
            loss_grad_batch = loss_grad_batch.squeeze()

        # Градиент по весам w
        self.dw += np.dot(X_batch.T, loss_grad_batch) / X_batch.shape[0]
        
        if self.use_bias:
            # Градиент по bias b
            self.db += np.mean(loss_grad_batch)
        
        #регуляризация
        if self.reg=='l1' and self.reg_strength > 0:
            self.dw += self.reg_strength * np.sign(self.w)
            if self.use_bias:
                self.db += self.reg_strength * np.sign(self.b)
        
        elif self.reg=='l2' and self.reg_strength > 0:
            self.dw += self.reg_strength * self.w
            if self.use_bias:
                self.db += self.reg_strength * self.b

    def update_weights(self):
        self.w -= self.learning_rate * self.dw
        if self.use_bias:
            self.b -= self.learning_rate * self.db
        self.zero_grad() # Обнуляем градиенты после обновления


class Logistic_Regression:
    def __init__(self, weight_size, bias=True, learning_rate=0.01, reg='l1', reg_strength=0.01):
        self.learning_rate = learning_rate
        self.w = np.random.normal(size=weight_size)
        self.use_bias = bias
        if self.use_bias:
            self.b = np.random.normal()
        
        self.dw = np.zeros_like(self.w)
        if self.use_bias:
            self.db = 0.0
        
        self.reg=reg
        self.reg_strength=reg_strength

    def predict_proba(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        z = np.dot(X, self.w)
        if self.use_bias:
            z += self.b
        return 1 / (1 + np.exp(-z))
    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def get_weights(self):
        return self.w, self.b if self.use_bias else self.w

    def zero_grad(self):
        self.dw = np.zeros_like(self.w)
        if self.use_bias:
            self.db = 0.0
    
    def compute_gradients(self, X_batch, loss_grad_batch):
        # X_batch: (batch_size, num_features)
        # loss_grad_batch: (batch_size, ) - градиент функции потерь по y_pred
        

        if loss_grad_batch.ndim > 1 and loss_grad_batch.shape[1] == 1:
            loss_grad_batch = loss_grad_batch.squeeze()

        # Градиент по весам w
        self.dw += np.dot(X_batch.T, loss_grad_batch) / X_batch.shape[0]
        
        if self.use_bias:
            # Градиент по bias b
            self.db += np.mean(loss_grad_batch)
        
        #регуляризация
        if self.reg=='l1' and self.reg_strength > 0:
            self.dw += self.reg_strength * np.sign(self.w)
            if self.use_bias:
                self.db += self.reg_strength * np.sign(self.b)
        
        elif self.reg=='l2' and self.reg_strength > 0:
            self.dw += self.reg_strength * self.w
            if self.use_bias:
                self.db += self.reg_strength * self.b

    def update_weights(self):
        self.w -= self.learning_rate * self.dw
        if self.use_bias:
            self.b -= self.learning_rate * self.db
        self.zero_grad() # Обнуляем градиенты после обновления
    