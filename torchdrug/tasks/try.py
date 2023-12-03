import torch

# Original tensor
original_tensor = torch.tensor([8, 4, 1, 4, 3, 4, 3, 2, 2, 2, 4, 4, 3, 6, 3, 1])

# Number of unique categories (assuming 0 to max_value in the tensor)
num_categories = original_tensor.max() + 1

# Create the one-hot encoding matrix
one_hot_matrix = torch.eye(num_categories)

# One-hot encode the tensor
one_hot_encoded = one_hot_matrix[original_tensor]

print(one_hot_encoded)
