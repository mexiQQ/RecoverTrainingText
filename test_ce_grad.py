import torch
import torch.nn as nn

# Input data and true labels
x = torch.tensor([[0.5, 0.2], [0.3, 0.7]], requires_grad=False)
y_true = torch.tensor([0, 1], dtype=torch.long, requires_grad=False)

# Weights of the two layers
W1 = torch.tensor([
    [0.1, 0.3], 
    [0.4, 0.1]
], requires_grad=True)

W2 = torch.tensor([
    [0.2, 0.4], 
    [0.1, 0.3]
], requires_grad=True)

# Forward pass
h = torch.relu(x @ W1)
output = h @ W2

# Cross-entropy loss
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(output, y_true)

# Backward pass
loss.backward()

# Display gradients
print("Gradients of W2:")
print(W2.grad)
