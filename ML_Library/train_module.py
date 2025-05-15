import numpy as np
def normilize(X):
        """
        Нормализация входных данных методом z-score
        X - список списков, где каждый внутренний список - это один пример
        """
        #среднее арифметическое
        mean=np.mean(X)
        #стандартное отконение
        std=np.std(X)
        #нормализация
        normilized_X=(X-mean)/std
        return normilized_X