import torch
import matplotlib.pyplot as plt
from torchvision.ops import MLP
from tqdm import tqdm
import numpy as np


def generate_checkerboard_sample(num_samples=10, field_size=0.4, num_fields=2, center=True):
    x = torch.rand(num_samples, 2) * field_size
    offset = torch.randint(0, num_fields, (num_samples, 2)) * field_size * 2
    diagonal_shift = torch.randint(0, num_fields, (num_samples, 1)) * field_size
    x += offset + diagonal_shift

    if center:
        x -= torch.mean(x, dim=0)

    return x


base_distribution_std = 0.15
num_samples = 2000
x = torch.randn(num_samples, 2) * base_distribution_std
y = generate_checkerboard_sample(num_samples=num_samples)

# show points
plt.scatter(x[:, 0], x[:, 1], alpha=0.5, label='base distribution')
plt.scatter(y[:, 0], y[:, 1], alpha=0.5, label='checkerboard distribution')
plt.show()

# define a model


device = "cpu"

model = MLP(in_channels=2 + 1, hidden_channels=[512, 512, 512, 512, 2], activation_layer=torch.nn.SiLU)
model.to(device)

# define a loss function
criterion = torch.nn.MSELoss(reduction="none")

# define an optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# train the model:
num_epochs = 20000  # use fewer epochs if it takes too long
batch_size = 4096
losses = []

for epoch in tqdm(range(num_epochs)):
    x = torch.randn(batch_size, 2) * base_distribution_std
    y = generate_checkerboard_sample(num_samples=batch_size)
    t = torch.rand(batch_size)
    x, y, t = x.to(device), y.to(device), t.to(device)

    # TODO: implement the training loop
    psi_t = (1 - t.unsqueeze(-1)) * x + t.unsqueeze(-1) * y
    model_input = torch.cat([psi_t, t.unsqueeze(-1)], dim=-1)

    v_t = model(model_input)

    # Loss
    v_true = y - x
    loss = criterion(v_t, v_true)  # Shape: (batch_size, 2)
    loss = loss.mean()
    losses.append(loss.item())

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# TODO: run inference with the trained model.
# Visualize the trajectory of the samples and the final samples at t=1.
# Hint: Use a simple Euler integration scheme to integrate the velocity field with 100 steps.


x0 = torch.randn(1500, 2) * base_distribution_std
x = x0.to(device)
t_values = torch.linspace(0, 1, 100 + 1, device=device)
dt = t_values[1] - t_values[0]  # Time step size

# Euler integration loop
for t in t_values:
    t_tensor = t.expand(x.size(0), 1)
    model_input = torch.cat([x, t_tensor], dim=-1)
    v_t = model(model_input)
    x = x + dt * v_t

x = x.cpu().detach().numpy()
# Plot
xt_0 = x0[:, 0]
yt_0 = x0[:, 1]
xt_1 = x[:, 0]
yt_1 = x[:, 1]

plt.figure(figsize=(10, 8))

for i in range(len(xt_0)):
    x_values = [x0[i, 0], x[i, 0]]
    y_values = [x0[i, 1], x[i, 1]]
    if i == 0:
        plt.plot(x_values, y_values, linestyle='-', alpha=0.6, color='gray', linewidth=0.3, label='trajectory')
    else:
        plt.plot(x_values, y_values, linestyle='-', alpha=0.6, color='gray', linewidth=0.3)

plt.scatter(x[:, 0], x[:, 1], alpha=0.5, color='red', label='final position')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# Create a grid
grid_size = 50
x = np.linspace(-0.5, 0.5, grid_size)
y = np.linspace(-0.5, 0.5, grid_size)
X, Y = np.meshgrid(x, y)

grid_points = np.stack([X.flatten(), Y.flatten()], axis=-1)
grid_points_torch = torch.tensor(grid_points, dtype=torch.float32, device=device)

# Calculate velocity field
t = torch.zeros(1, device=device)
t_tensor = t.expand(grid_points_torch.size(0), 1)
model_input = torch.cat([grid_points_torch, t_tensor], dim=-1)
v_t = model(model_input)

# Reshape
U = v_t[:, 0].detach().cpu().numpy().reshape(grid_size, grid_size)
V = v_t[:, 1].detach().cpu().numpy().reshape(grid_size, grid_size)

# Plot
plt.figure(figsize=(8, 8))
plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=10.0, color='blue', alpha=0.7)
plt.title("Velocity Field at t = 0")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.show()