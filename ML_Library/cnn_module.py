#Модуль для работы с сверточными нейронными сетями
import numpy as np
from .layers_module import Layer, Dense, Activation, ReLU

class Conv2D(Layer):
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        
        # Инициализация весов по методу Хе
        scale = np.sqrt(2.0 / (input_channels * self.kernel_size[0] * self.kernel_size[1]))
        self.weights = np.random.normal(0, scale, (output_channels, input_channels, self.kernel_size[0], self.kernel_size[1]))
        self.bias = np.zeros((output_channels, 1))
        
        self.dweights = np.zeros_like(self.weights)
        self.dbias = np.zeros_like(self.bias)

    def forward(self, input):
        batch_size, channels, height, width = input.shape
        
        # Добавляем padding
        padded_input = np.pad(input, ((0, 0), (0, 0), 
                                    (self.padding[0], self.padding[0]),
                                    (self.padding[1], self.padding[1])), 
                            mode='constant')
        
        # Вычисляем размеры выходного тензора
        out_height = (height + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        out_width = (width + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
        
        output = np.zeros((batch_size, self.output_channels, out_height, out_width))
        
        # Свертка
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride[0]
                h_end = h_start + self.kernel_size[0]
                w_start = j * self.stride[1]
                w_end = w_start + self.kernel_size[1]
                
                input_slice = padded_input[:, :, h_start:h_end, w_start:w_end]
                
                for k in range(self.output_channels):
                    output[:, k, i, j] = np.sum(
                        input_slice * self.weights[k, :, :, :],
                        axis=(1, 2, 3)
                    ) + self.bias[k]
        
        self.input = input
        self.padded_input = padded_input
        return output

    def backward(self, output_gradient, learning_rate):
        batch_size, channels, height, width = self.input.shape
        _, _, out_height, out_width = output_gradient.shape
        
        # Инициализация градиентов
        input_gradient = np.zeros_like(self.padded_input)
        self.dweights = np.zeros_like(self.weights)
        self.dbias = np.zeros_like(self.bias)
        
        # Обратное распространение
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride[0]
                h_end = h_start + self.kernel_size[0]
                w_start = j * self.stride[1]
                w_end = w_start + self.kernel_size[1]
                
                input_slice = self.padded_input[:, :, h_start:h_end, w_start:w_end]
                
                for k in range(self.output_channels):
                    # Градиент по весам
                    self.dweights[k] += np.sum(
                        input_slice * output_gradient[:, k:k+1, i:i+1, j:j+1],
                        axis=0
                    )
                    
                    # Градиент по смещению
                    self.dbias[k] += np.sum(output_gradient[:, k, i, j])
                    
                    # Градиент по входу
                    input_gradient[:, :, h_start:h_end, w_start:w_end] += \
                        self.weights[k] * output_gradient[:, k:k+1, i:i+1, j:j+1]
        
        # Обновление весов
        self.weights -= learning_rate * self.dweights
        self.bias -= learning_rate * self.dbias
        
        # Убираем padding из градиента входа
        if self.padding[0] > 0 or self.padding[1] > 0:
            input_gradient = input_gradient[:, :, 
                                          self.padding[0]:-self.padding[0],
                                          self.padding[1]:-self.padding[1]]
        
        return input_gradient

class MaxPool2D(Layer):
    def __init__(self, pool_size=2, stride=None):
        super().__init__()
        self.pool_size = pool_size if isinstance(pool_size, tuple) else (pool_size, pool_size)
        self.stride = stride if stride is not None else pool_size
        self.stride = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride)
        
        self.max_indices = None

    def forward(self, input):
        batch_size, channels, height, width = input.shape
        
        out_height = (height - self.pool_size[0]) // self.stride[0] + 1
        out_width = (width - self.pool_size[1]) // self.stride[1] + 1
        
        output = np.zeros((batch_size, channels, out_height, out_width))
        self.max_indices = np.zeros((batch_size, channels, out_height, out_width), dtype=np.int32)
        
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride[0]
                h_end = h_start + self.pool_size[0]
                w_start = j * self.stride[1]
                w_end = w_start + self.pool_size[1]
                
                input_slice = input[:, :, h_start:h_end, w_start:w_end]
                output[:, :, i, j] = np.max(input_slice, axis=(2, 3))
                
                # Сохраняем индексы максимальных значений
                for b in range(batch_size):
                    for c in range(channels):
                        max_idx = np.unravel_index(
                            np.argmax(input_slice[b, c]),
                            (self.pool_size[0], self.pool_size[1])
                        )
                        self.max_indices[b, c, i, j] = max_idx[0] * self.pool_size[1] + max_idx[1]
        
        self.input = input
        return output

    def backward(self, output_gradient, learning_rate):
        batch_size, channels, height, width = self.input.shape
        _, _, out_height, out_width = output_gradient.shape
        
        input_gradient = np.zeros_like(self.input)
        
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride[0]
                h_end = h_start + self.pool_size[0]
                w_start = j * self.stride[1]
                w_end = w_start + self.pool_size[1]
                
                for b in range(batch_size):
                    for c in range(channels):
                        max_idx = np.unravel_index(
                            self.max_indices[b, c, i, j],
                            (self.pool_size[0], self.pool_size[1])
                        )
                        input_gradient[b, c, h_start + max_idx[0], w_start + max_idx[1]] = \
                            output_gradient[b, c, i, j]
        
        return input_gradient

class Flatten(Layer):
    def __init__(self):
        super().__init__()
        self.input_shape = None

    def forward(self, input):
        self.input_shape = input.shape
        return input.reshape(input.shape[0], -1)

    def backward(self, output_gradient, learning_rate):
        return output_gradient.reshape(self.input_shape)

class CNN:
    def __init__(self, layers=None):
        self.layers = layers if layers is not None else []
        self.training = True

    def add(self, layer):
        self.layers.append(layer)

    def set_training(self, training):
        self.training = training
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = training

    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def backward(self, output_gradient, learning_rate):
        for layer in reversed(self.layers):
            output_gradient = layer.backward(output_gradient, learning_rate)

    def predict(self, input):
        self.set_training(False)
        return self.forward(input)

    def fit(self, X, y, epochs, batch_size, learning_rate, loss_fn):
        n_samples = X.shape[0]
        n_batches = n_samples // batch_size

        for epoch in range(epochs):
            epoch_loss = 0
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                
                X_batch = X[start_idx:end_idx]
                y_batch = y[start_idx:end_idx]

                # Forward pass
                output = self.forward(X_batch)
                loss = loss_fn.get_loss(output, y_batch)
                epoch_loss += loss

                # Backward pass
                gradient = loss_fn.get_grad(output, y_batch)
                self.backward(gradient, learning_rate)

            avg_loss = epoch_loss / n_batches
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

    def save_weights(self, filename):
        weights = []
        for layer in self.layers:
            if isinstance(layer, (Conv2D, Dense)):
                weights.append({
                    'weights': layer.weights,
                    'bias': layer.bias
                })
        np.save(filename, weights)

    def load_weights(self, filename):
        weights = np.load(filename, allow_pickle=True)
        weight_idx = 0
        for layer in self.layers:
            if isinstance(layer, (Conv2D, Dense)):
                layer.weights = weights[weight_idx]['weights']
                layer.bias = weights[weight_idx]['bias']
                weight_idx += 1 