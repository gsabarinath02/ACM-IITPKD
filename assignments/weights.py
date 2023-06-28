import numpy as np

# Define the input, target, and initial weights
x = 2
y_target = 4
w = 0.5

# Perform forward propagation
y_pred = 1 / (1 + np.exp(-w * x))

# Calculate the loss (error)
loss = 0.5 * (y_pred - y_target) ** 2

# Perform backpropagation to calculate gradients
d_loss_d_ypred = y_pred - y_target
d_ypred_d_w = y_pred * (1 - y_pred)
d_loss_d_w = d_loss_d_ypred * d_ypred_d_w

# Update weights using gradient descent
learning_rate = 0.1
w -= learning_rate * d_loss_d_w

# Print the updated weight
print("Updated weight:", w)

