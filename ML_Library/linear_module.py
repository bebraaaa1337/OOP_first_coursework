from random import randint
class Linear_Regression:
    def __init__(self, weight_size=1,bias=False):
        self.weight_size=weight_size
        self.bias=bias
        self.w=[1]*(weight_size+bias)
        for i in range(weight_size):
            self.w[i]=randint(-10,10)
    def predict(self,x):
        if len(x)!=self.weight_size:
            print('PROBLEM')
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

            
    

