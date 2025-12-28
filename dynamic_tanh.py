import torch
import torch.nn as nn
from timm.layers import LayerNorm2d


class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, channels_last, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        if self.channels_last:
            x = x * self.weight + self.bias
        else:
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, alpha_init_value={self.alpha_init_value}, channels_last={self.channels_last}"


class Weird(nn.Module):
    """
    Weird(x) = x                                   if |x| < 1
              sign(x) * |c * x|^alpha               if |x| >= 1
    where alpha in (0, 1/2) and c = (1/2 - alpha).
    alpha is NOT learnable.
    """

    def __init__(self, normalized_shape, channels_last, alpha: float):
        super().__init__()
        if not (0.0 < float(alpha) < 0.5):
            raise ValueError(f"alpha must be in (0, 0.5), got {alpha}")

        self.normalized_shape = normalized_shape
        self.channels_last = channels_last
        self.alpha = float(alpha)
        # self.c = 0.5 - self.alpha
        self.c = 1.0

        # Match LayerNorm-style affine parameters used by DynamicTanh in this repo
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ax = x.abs()
        y = torch.where(
            ax < 1.0,
            x,
            x.sign() * ax.pow(self.alpha),
        )
        y = y * self.c

        if self.channels_last:
            y = y * self.weight + self.bias
        else:
            y = y * self.weight[:, None, None] + self.bias[:, None, None]
        return y

    def extra_repr(self):
        return (
            f"normalized_shape={self.normalized_shape}, "
            f"alpha={self.alpha}, c={self.c}, "
            f"channels_last={self.channels_last}"
        )


def convert_ln_to_dyt(module, alpha_init_value=0.5):
    module_output = module
    if isinstance(module, nn.LayerNorm):
        module_output = DynamicTanh(
            module.normalized_shape,
            not isinstance(module, LayerNorm2d),
            alpha_init_value=alpha_init_value,
        )
    for name, child in module.named_children():
        module_output.add_module(name, convert_ln_to_dyt(child, alpha_init_value=alpha_init_value))
    del module
    return module_output


def convert_ln_to_weird(module, alpha: float):
    """
    Recursively replace nn.LayerNorm modules with Weird modules.

    alpha is a fixed (non-learnable) scalar in (0, 0.5).
    """
    module_output = module
    if isinstance(module, nn.LayerNorm):
        module_output = Weird(
            module.normalized_shape,
            not isinstance(module, LayerNorm2d),
            alpha=alpha,
        )
    for name, child in module.named_children():
        module_output.add_module(name, convert_ln_to_weird(child, alpha=alpha))
    del module
    return module_output

