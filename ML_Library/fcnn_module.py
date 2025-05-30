#Модуль для работы с полносвязными нейронными сетями
import numpy as np
from .layers_module import Layer, Dense, Activation, ReLU, Sigmoid, Tanh

class FCNN:
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
            if isinstance(layer, Dense):
                weights.append({
                    'weights': layer.weights,
                    'bias': layer.bias
                })
        np.save(filename, weights)

    def load_weights(self, filename):
        weights = np.load(filename, allow_pickle=True)
        weight_idx = 0
        for layer in self.layers:
            if isinstance(layer, Dense):
                layer.weights = weights[weight_idx]['weights']
                layer.bias = weights[weight_idx]['bias']
                weight_idx += 1 