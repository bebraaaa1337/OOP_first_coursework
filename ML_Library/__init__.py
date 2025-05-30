from .linear_module import Linear_Regression
from .losses_module import MSE_Loss, MAE_Loss
from .layers_module import Layer, Dense, Activation, ReLU, Sigmoid, Tanh
from .fcnn_module import FCNN
from .cnn_module import CNN, Conv2D, MaxPool2D, Flatten

__version__ = '0.1.0'

__all__ = [
    # Линейная регрессия
    'Linear_Regression',
    
    # Функции потерь
    'MSE_Loss',
    'MAE_Loss',
    
    # Базовые слои
    'Layer',
    'Dense',
    'Activation',
    'ReLU',
    'Sigmoid',
    'Tanh',
    
    # Полносвязные нейронные сети
    'FCNN',
    
    # Сверточные нейронные сети
    'CNN',
    'Conv2D',
    'MaxPool2D',
    'Flatten',
]