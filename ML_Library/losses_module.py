import numpy as np
class MSE_Loss:
    def __init__(self):
        self.grad=[]
    def get_loss(self,y_pred,y_true):
        loss=(y_pred-y_true)**2
        self.grad.append(2*(y_pred-y_true))
        return loss
    def get_grad(self):
        return np.array(self.grad)
    def zero_grad(self):
        self.grad=[]
