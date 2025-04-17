class MSE_Loss:
    def __init__(self):
        self.grad=0
        self.loss=0
    def get_loss(self,y_pred,y_true):
        self.grad=0
        self.loss=0
        self.loss=(y_pred-y_true)**2
        self.grad=2*(y_pred-y_true)
        return self.loss
    def get_grad(self):
        return self.grad
