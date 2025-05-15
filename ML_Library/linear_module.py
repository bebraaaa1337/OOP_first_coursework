import numpy as np
class Linear_Regression:
    def __init__(self, weight_size,bias=True, learning_rate=0.01):
        self.weight_size=weight_size
        self.bias=bias
        self.bias_mean=np.array([1.0])
        self.learning_rate=learning_rate
        self.w=np.random.normal(size=weight_size)
        self.w_grad=[]
        self.b_grad=[]
    def predict(self,x,grad=True):
        ans=x.dot(self.w)

        #считаем градиенты во время train
        if grad:
            self.w_grad.append(x)
            
        if self.bias:
            ans+=np.squeeze(self.bias_mean)
            if grad:
                self.b_grad.append([1])
        return ans
    
    def get_weights(self):
        return self.w
    
    def zero_grad(self):
        self.w_grad=[]
        self.b_grad=[]
    
    def update(self,lossgrad):
        self.w-=(np.array(self.w_grad).T).dot(lossgrad)/(len(self.w_grad))*self.learning_rate
        if self.bias:
            self.bias_mean-=np.mean(lossgrad)*self.learning_rate
            self.bias_mean = np.array([float(self.bias_mean)])
    

