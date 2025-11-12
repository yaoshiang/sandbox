"""Test fixtures for distributed tests."""

import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    """Simple 2-layer MLP with dimension up and down.

    Architecture:
        input (dim_in) -> Linear -> ReLU -> Linear -> output (dim_out)

    Args:
        dim_in: Input dimension
        dim_hidden: Hidden dimension (expanded dimension)
        dim_out: Output dimension
    """

    def __init__(self, dim_in: int, dim_hidden: int, dim_out: int):
        super().__init__()
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out

        # Dimension up
        self.fc1 = nn.Linear(dim_in, dim_hidden, bias=False)
        # Dimension down
        self.fc2 = nn.Linear(dim_hidden, dim_out, bias=False)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, dim_in)

        Returns:
            Output tensor of shape (batch_size, dim_out)
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class SwiGLUMLP(nn.Module):
    """MLP with SwiGLU activation (used in Llama, GPT-4, etc).

    SwiGLU splits the hidden dimension into two paths:
    - One path goes through Swish/SiLU activation
    - Other path is a gating mechanism
    - Element-wise multiply the two paths

    Architecture:
        input (dim_in) -> gate_proj (dim_hidden) -> SiLU ──┐
                       -> up_proj (dim_hidden) ────────────┤ * -> down_proj (dim_out)

    This is more realistic for modern LLMs than plain ReLU.

    Args:
        dim_in: Input dimension
        dim_hidden: Hidden dimension (expanded dimension)
        dim_out: Output dimension
    """

    def __init__(self, dim_in: int, dim_hidden: int, dim_out: int):
        super().__init__()
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out

        # Gate projection (with SiLU activation)
        self.gate_proj = nn.Linear(dim_in, dim_hidden, bias=False)
        # Up projection (no activation)
        self.up_proj = nn.Linear(dim_in, dim_hidden, bias=False)
        # Down projection
        self.down_proj = nn.Linear(dim_hidden, dim_out, bias=False)

        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with SwiGLU.

        Args:
            x: Input tensor of shape (batch_size, dim_in)

        Returns:
            Output tensor of shape (batch_size, dim_out)
        """
        # SwiGLU: silu(gate_proj(x)) * up_proj(x)
        gate = self.activation(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up

        # Project back down
        output = self.down_proj(hidden)
        return output
