import torch


def per_sample(x: torch.Tensor) -> tuple[int, ...]:
    return (x.shape[0],) + (1,) * (x.dim() - 1)


def flip_channels(x: torch.Tensor, y: torch.Tensor, p: float) -> tuple[torch.Tensor, torch.Tensor]:
    m = torch.rand(x.shape[0], device=x.device) < p
    return torch.where(m.view(per_sample(x)), x.flip(-1), x), torch.where(m.view(per_sample(y)), y.flip(-1), y)


def add_gain(x: torch.Tensor, std: float) -> torch.Tensor:
    return x + torch.randn(per_sample(x), device=x.device, dtype=x.dtype) * std


def add_noise(x: torch.Tensor, std: float) -> torch.Tensor:
    return x + torch.randn_like(x) * std


def apply(x: torch.Tensor, y: torch.Tensor, c: dict) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = flip_channels(x, y, c["flip_channels"])
    return add_noise(add_gain(x, c["gain_std"]), c["noise_std"]), y
