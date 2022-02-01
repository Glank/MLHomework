import torch
import math

dtype = torch.float
device = torch.device("cpu")

# Create random input and output data
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

# Randomly initialize weights
a = torch.randn((), device=device, dtype=dtype)
b = torch.randn((), device=device, dtype=dtype)
c = torch.randn((), device=device, dtype=dtype)
d = torch.randn((), device=device, dtype=dtype)

learning_rate = 1e-6
for t in range(2000):
  # Forward pass: compute predicted y
  y_pred = a*x**3 + b*x**2 + c*x + d

  # Compute and print loss
  loss = (y_pred - y).pow(2).sum().item()
  if t % 100 == 99:
      print(t, loss)

  # Backprop to compute gradients of a, b, c, d with respect to loss
  grad_y_pred = 2.0 * (y_pred - y)
  grad_a = torch.tensordot(grad_y_pred, x**3, dims=1)
  grad_b = torch.tensordot(grad_y_pred, x**2, dims=1)
  grad_c = torch.tensordot(grad_y_pred, x, dims=1)
  grad_d = grad_y_pred.sum()

  # Update weights using gradient descent
  a -= learning_rate * grad_a
  b -= learning_rate * grad_b
  c -= learning_rate * grad_c
  d -= learning_rate * grad_d

print(f'Result: y = {a.item()}x^3 + {b.item()}x^2 + {c.item()}x + {d.item()}')
