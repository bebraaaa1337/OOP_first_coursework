import numpy as np

class MSE_Loss:
    """"
    Класс реализации MSE
    """
    def get_loss(self, y_pred, y_true):
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        
        if y_pred.shape != y_true.shape:
            raise ValueError(f"Shapes of y_pred {y_pred.shape} and y_true {y_true.shape} must match.")
            
        loss = np.mean((y_pred - y_true)**2)
        return loss

    def get_grad(self, y_pred, y_true):
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)

        if y_pred.shape != y_true.shape:
            raise ValueError(f"Shapes of y_pred {y_pred.shape} and y_true {y_true.shape} must match.")
        
        num_elements = y_pred.size
        if num_elements == 0:
             return np.zeros_like(y_pred)
        
        grad = 2 * (y_pred - y_true) / num_elements 
        return grad
    


class MAE_Loss:
    """"
    Класс реализации MAE
    """
    def get_loss(self, y_pred, y_true):
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        
        if y_pred.shape != y_true.shape:
            raise ValueError(f"Shapes of y_pred {y_pred.shape} and y_true {y_true.shape} must match.")
            
        loss = np.mean(np.abs(y_pred - y_true))
        return loss

    def get_grad(self, y_pred, y_true):
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)

        if y_pred.shape != y_true.shape:
            raise ValueError(f"Shapes of y_pred {y_pred.shape} and y_true {y_true.shape} must match.")
        
        num_elements = y_pred.size
        if num_elements == 0: # Избегаем деления на ноль
             return np.zeros_like(y_pred)
        
        grad = np.sign(y_pred - y_true) / num_elements
        return grad
    
class Log_loss:

    def get_loss(self, y_pred,y_true):
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        
        if y_pred.shape != y_true.shape:
            raise ValueError(f"Shapes of y_pred {y_pred.shape} and y_true {y_true.shape} must match.")
            
        # Защита от логарифма нуля
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        loss = -np.mean(y_true*np.log(y_pred)+(1-y_true)*np.log(1-y_pred))
        return loss


    def get_grad(self, y_pred, y_true):
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)

        if y_pred.shape != y_true.shape:
            raise ValueError(f"Shapes of y_pred {y_pred.shape} and y_true {y_true.shape} must match.")
        
        num_elements = y_pred.size
        if num_elements == 0: # Избегаем деления на ноль
             return np.zeros_like(y_pred)
        
        # Градиент логистической функции потерь
        grad = (y_pred - y_true) / num_elements
        return grad