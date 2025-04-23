
from random import randint
class Linear_Regression:
    def __init__(self, weight_size=1,bias=False, learning_rate=0.01):
        self.weight_size=weight_size
        self.bias=bias
        self.learning_rate=0.01
        self.w=[1]*(weight_size+bias)
        for i in range(weight_size+bias):
            self.w[i]=randint(-10,10)*0.01
        self.mean=None
        self.std=None
    def normilize(self,x):
        self.mean=[0]*self.weight.size()
        for i in range (self.weight.size()):
            for j in range (x[i]):
                self.mean[i]+=x[i][j]
            self.mean[i]=self.mean[i]/len(x[i])
        self.std=[0]*self.weight_size

    def predict(self,x):
        if len(x)!=self.weight_size:
            print('PROBLEM')
        x_
        ans=0
        for i in range(self.weight_size):
            ans+=x[i]*self.w[i]
        if self.bias:
            ans+=self.w[self.weight_size]
        return ans
    def get_weights(self):
        return self.w
    def get_bias(self):
        return self.b

            
    

