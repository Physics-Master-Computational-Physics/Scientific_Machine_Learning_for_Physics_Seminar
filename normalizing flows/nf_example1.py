import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


# Define a simple affine coupling layer
class AffineCoupling(nn.Module):
    def __init__(self, dim, hidden_dim=64):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, (dim - dim // 2) * 2))

        # Initialize weights to zero for stable training
        self.net[-1].weight.data.zero_()
        self.net[-1].bias.data.zero_()

    def forward(self, x, reverse=False):
        x_a, x_b = x[:, :self.dim // 2], x[:, self.dim // 2:]

        # Compute scale and shift parameters
        params = self.net(x_a)
        log_scale, shift = params.chunk(2, dim=1)
        scale = torch.sigmoid(log_scale + 2.0)  # Constrained to (0,1)

        if not reverse:
            z_b = x_b * scale + shift
            log_det = torch.log(scale).sum(1)
        else:
            z_b = (x_b - shift) / scale
            log_det = -torch.log(scale).sum(1)

        z = torch.cat([x_a, z_b], dim=1)
        return z, log_det


# Define the full normalizing flow
class NormalizingFlow(nn.Module):
    def __init__(self, dim, num_flows=4):
        super().__init__()
        self.dim = dim
        self.flows = nn.ModuleList([AffineCoupling(dim) for _ in range(num_flows)])
        self.base_dist = torch.distributions.MultivariateNormal(
            torch.zeros(dim), torch.eye(dim))

    def forward(self, x):
        log_prob = torch.zeros(x.shape[0]).to(x.device)
        for flow in self.flows:
            x, log_det = flow(x)
            log_prob += log_det

        # Compute final log probability
        log_prob += self.base_dist.log_prob(x)
        return log_prob

    def sample(self, num_samples):
        z = self.base_dist.sample((num_samples,))
        log_prob = self.base_dist.log_prob(z)

        for flow in reversed(self.flows):
            z, log_det = flow(z, reverse=True)
            log_prob -= log_det

        return z, log_prob


# Generate some 2D training data (e.g., a moon shape)
def get_training_data():
    data, _ = datasets.make_moons(128, noise=0.05)
    return torch.tensor(data, dtype=torch.float32)


# Training loop
def train_flow():
    # Create model and optimizer
    flow = NormalizingFlow(dim=2, num_flows=4)
    optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)

    # Training data
    data = get_training_data()

    # Training loop
    for epoch in range(1000):
        optimizer.zero_grad()
        loss = -flow(data).mean()  # Negative log likelihood
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    return data,flow


# Visualize results
def visualize(data,flow):
    # Generate samples from the trained flow
    samples, _ = flow.sample(1000)
    samples = samples.detach().numpy()

    # Plot original data and samples
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(data[:, 0], data[:, 1], alpha=0.5)
    plt.title("Training Data")

    plt.subplot(1, 2, 2)
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5)
    plt.title("Generated Samples")
    plt.show()


# Run everything
if __name__ == "__main__":
    data,flow = train_flow()
    visualize(data,flow)