import torch

# Sample tensors X and Y
X = torch.tensor([1, 2, 3, 4, 5])  # Size [N]
Y = torch.tensor([2, 4, 6])  # Size [D]

# Expand dimensions to make broadcasting possible
expanded_X = X.unsqueeze(1)  # Size [N, 1]
expanded_Y = Y.unsqueeze(0)  # Size [1, D]

# Calculate pairwise distances between X and Y
distances = torch.cdist(expanded_X, expanded_Y)  # Size [N, D]

# Find the index of the nearest item in Y for each item in X
nearest_indices = torch.argmin(distances, dim=1)  # Size [N]

print(nearest_indices)
