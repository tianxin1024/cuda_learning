import torch
import torch.nn as nn

def manual_layernorm(x, gamma, beta, eps=1e-5):
    """
    Args:
        x:      Input tensor of shape (B, T, C)
        gamma:  Scale parameter of shape (C,)
        beta:   Shift parameter of shape (C,)
        eps:    Small value to avoid division by zero
    Returns:
        y:      Output tensor of shape (B, T, C)
    """
    # Step 1: Compute mean and variance along the last dimension (C)
    mean = x.mean(dim=-1, keepdim=True)  # shape=(B, T, 1)
    var = x.var(dim=-1, keepdim=True, unbiased=False)  # shape=(B, T, 1)
    
    # Step 2: Normalize
    x_hat = (x - mean) / torch.sqrt(var + eps)  # shape=(B, T, C)
    
    # Step 3: Scale and shift
    y = gamma * x_hat + beta  # gamma/beta broadcast to (B, T, C)
    return y


# example usage
B, T, C = 8, 1024, 768
x = torch.randn(B, T, C)
gamma = torch.ones(C)
beta = torch.zeros(C)

# manual layernorm
y_manual = manual_layernorm(x, gamma, beta)

# pytorch's LayerNorm for validation
layer_norm = nn.LayerNorm(C)
y_pt = layer_norm(x)

# Check if close
print(torch.allclose(y_manual, y_pt, atol=1e-6))
