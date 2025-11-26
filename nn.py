import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, layers):
        """Initialize network with layer sizes"""
        self.layers = layers
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases
        for i in range(len(layers) - 1):
            w = np.random.randn(layers[i], layers[i+1]) * 0.01
            b = np.zeros((1, layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def sigmoid_derivative(self, a):
        return a * (1 - a)
    
    def forward(self, X):
        """Forward propagation"""
        self.activations = [X]
        a = X
        
        for w, b in zip(self.weights, self.biases):
            z = np.dot(a, w) + b
            a = self.sigmoid(z)
            self.activations.append(a)
        
        return a
    
    def backward(self, y, learning_rate):
        """Backward propagation"""
        m = y.shape[0]
        delta = self.activations[-1] - y
        
        for i in reversed(range(len(self.weights))):
            dw = np.dot(self.activations[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.sigmoid_derivative(self.activations[i])
            
            self.weights[i] -= learning_rate * dw
            self.biases[i] -= learning_rate * db
    
    def train(self, X, y, epochs, learning_rate):
        """Train the network"""
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(y, learning_rate)
            
            if (epoch + 1) % 100 == 0:
                loss = -np.mean(y * np.log(output + 1e-8) + (1 - y) * np.log(1 - output + 1e-8))
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
    
    def predict(self, X):
        """Make predictions"""
        return self.forward(X)

# Example: XOR Problem
if __name__ == "__main__":
    # Training data (XOR problem)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    # Create and train network
    nn = NeuralNetwork([2, 4, 1])
    nn.train(X, y, epochs=1000, learning_rate=0.5)
    
    # Test
    print("\nPredictions:")
    predictions = nn.predict(X)
    for i, pred in enumerate(predictions):
        print(f"Input: {X[i]} -> Predicted: {pred[0]:.4f}, Actual: {y[i][0]}")
