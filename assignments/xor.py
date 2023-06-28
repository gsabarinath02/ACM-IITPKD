import random
import math

# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Define the derivative of the sigmoid function
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Define the XOR input and target values
x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_data = [[0], [1], [1], [0]]

# Initialize the weights and biases for the network
input_size = 2
hidden_size = 2
output_size = 1

W1 = [[random.uniform(-1, 1) for _ in range(hidden_size)] for _ in range(input_size)]
b1 = [0.0] * hidden_size

W2 = [[random.uniform(-1, 1) for _ in range(hidden_size)] for _ in range(hidden_size)]
b2 = [0.0] * hidden_size

W3 = [[random.uniform(-1, 1) for _ in range(output_size)] for _ in range(hidden_size)]
b3 = [0.0] * output_size

# Training parameters
learning_rate = 0.1
epochs = 1000

# Training loop
for epoch in range(epochs):
    # Forward propagation
    hidden_layer1 = [sigmoid(sum(x * w) + b) for x, w, b in zip(x_data, W1, b1)]
    hidden_layer2 = [sigmoid(sum(x * w) + b) for x, w, b in zip(hidden_layer1, W2, b2)]
    output_layer = [sigmoid(sum(x * w) + b) for x, w, b in zip(hidden_layer2, W3, b3)]
    
    # Calculate the loss (error)
    loss = 0.5 * sum([(y_pred - y_target) ** 2 for y_pred, y_target in zip(output_layer, y_data)]) / len(y_data)
    
    # Backward propagation
    dloss_doutput = [(y_pred - y_target) for y_pred, y_target in zip(output_layer, y_data)]
    doutput_dhidden2 = [sigmoid_derivative(y) for y in hidden_layer2]
    dhidden2_dhidden1 = [sigmoid_derivative(y) for y in hidden_layer1]
    
    dloss_dW3 = [[hidden * dloss * doutput for hidden, doutput in zip(hidden_layer2, doutput_dhidden2)] for dloss in dloss_doutput]
    dloss_db3 = [dloss * doutput for dloss, doutput in zip(dloss_doutput, doutput_dhidden2)]
    
    dloss_dhidden2 = [[dloss * doutput * w for dloss, doutput, w in zip(dloss_doutput, doutput_dhidden2, w_row)] for w_row in W3]
    dloss_dhidden1 = [[dloss * doutput * w for dloss, doutput, w in zip(dloss_row, dhidden2_dhidden1, w_row)] for dloss_row, w_row in zip(dloss_dhidden2, W2)]
    
    dloss_dW2 = [[hidden * dloss * doutput for hidden, dloss, doutput in zip(hidden_layer1, dloss_row, dhidden2_dhidden1)] for dloss_row in dloss_dhidden2]
    dloss_db2 = [sum(dloss * doutput for dloss, doutput in zip(dloss_row, dhidden2_dhidden1)) for dloss_row in dloss_dhidden2]
    
    dloss_dW1 = [[x * dloss for x, dloss in zip(x_row, dloss_row)] for x_row, dloss_row in zip(x_data, dloss_dhidden1)]
    dloss_db1 = [sum(dloss_row) for dloss_row in dloss_dhidden1]
    
    # Update weights and biases
    W3 = [[w - learning_rate * dW for w, dW in zip(w_row, dloss_dW_row)] for w_row, dloss_dW_row in zip(W3, dloss_dW3)]
    b3 = [b - learning_rate * db for b, db in zip(b3, dloss_db3)]
    
    W2 = [[w - learning_rate * dW for w, dW in zip(w_row, dloss_dW_row)] for w_row, dloss_dW_row in zip(W2, dloss_dW2)]
    b2 = [b - learning_rate * db for b, db in zip(b2, dloss_db2)]
    
    W1 = [[w - learning_rate * dW for w, dW in zip(w_row, dloss_dW_row)] for w_row, dloss_dW_row in zip(W1, dloss_dW1)]
    b1 = [b - learning_rate * db for b, db in zip(b1, dloss_db1)]
    
    # Print the loss for every 100 epochs
    if epoch % 100 == 0:
        print("Epoch " + epoch + ": Loss = " + loss)

# Test the model
hidden_layer1 = [sigmoid(sum(x * w) + b) for x, w, b in zip(x_data, W1, b1)]
hidden_layer2 = [sigmoid(sum(x * w) + b) for x, w, b in zip(hidden_layer1, W2, b2)]
output_layer = [sigmoid(sum(x * w) + b) for x, w, b in zip(hidden_layer2, W3, b3)]

print("Predicted Outputs:")
print(output_layer)

