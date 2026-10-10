from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from das import augment
from das.cellfeat import normalised_grid
from das.config import Config, data_dir, model_path
from das.data import N_BANDS, N_CLASSES, UNKNOWN, load_all, load_metadata, load_patches, norm_stats
from das.grid import predict_tiled
from das.metrics import macro_f1
from das.patches import class_weights, patch_arrays, sampler_weights, split_rows

MODEL_FILE = "cnn.pt"


def conv3(cin: int, cout: int, d: int) -> nn.Conv2d:
    return nn.Conv2d(cin, cout, 3, padding=d, dilation=d, bias=False)


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, d: int, p: float) -> None:
        super().__init__()
        self.body = nn.Sequential(conv3(cin, cout, d), nn.BatchNorm2d(cout), nn.ReLU(inplace=True), nn.Dropout2d(p), conv3(cout, cout, d), nn.BatchNorm2d(cout))
        self.skip = nn.Identity() if cin == cout else nn.Conv2d(cin, cout, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.body(x) + self.skip(x))


class Net(nn.Module):
    def __init__(self, channels: list[int], dilations: list[int], dropout: float) -> None:
        super().__init__()
        assert len(channels) == len(dilations), f"cnn.channels {channels} and cnn.dilations {dilations} differ in length"
        self.stem = nn.Sequential(conv3(N_BANDS, channels[0], 1), nn.BatchNorm2d(channels[0]), nn.ReLU(inplace=True))
        ins = [channels[0], *channels[:-1]]
        self.blocks = nn.Sequential(*[ResBlock(i, o, d, dropout) for i, o, d in zip(ins, channels, dilations)])
        self.head = nn.Conv2d(channels[-1], N_CLASSES, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.stem(x)))


def build_net(c: dict) -> Net:
    return Net(list(c["channels"]), list(c["dilations"]), float(c["dropout"]))


def cuda_usable() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    archs = torch.cuda.get_arch_list()
    return f"sm_{major}{minor}" in archs or any(a.startswith("compute_") for a in archs)


def assert_device() -> torch.device:
    if cuda_usable():
        return torch.device("cuda")
    print("warning: no usable CUDA device for this torch build, running on CPU")
    return torch.device("cpu")


def amp_ctx(c: dict, dev: torch.device) -> torch.autocast:
    return torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda" and c["amp"] == "bf16")


def to_dev(x: torch.Tensor, y: torch.Tensor, dev: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return x.to(dev, non_blocking=True).contiguous(memory_format=torch.channels_last), y.to(dev, non_blocking=True)


def make_loaders(cfg: Config, recs: dict[str, dict], rows: list[dict], shuffle: bool) -> tuple[DataLoader, np.ndarray]:
    x, y, _ = patch_arrays(recs, rows)
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    c = cfg.cnn
    if not shuffle:
        return DataLoader(ds, batch_size=c["batch"], num_workers=c["workers"]), y
    gen = torch.Generator().manual_seed(c["seed"])
    sampler = WeightedRandomSampler(torch.from_numpy(sampler_weights(rows)), num_samples=2 * len(rows), replacement=True, generator=gen)
    return DataLoader(ds, batch_size=c["batch"], sampler=sampler, num_workers=c["workers"], drop_last=len(rows) > c["batch"]), y


def loss_fn(logits: torch.Tensor, y: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.float(), y, weight=weights, ignore_index=UNKNOWN)


def train_epoch(net: Net, loader: DataLoader, opt: torch.optim.Optimizer, sched: torch.optim.lr_scheduler.LRScheduler, weights: torch.Tensor, cfg: Config, dev: torch.device) -> float:
    net.train()
    losses = []
    for xb, yb in loader:
        xb, yb = augment.apply(*to_dev(xb, yb, dev), cfg.augment)
        with amp_ctx(cfg.cnn, dev):
            loss = loss_fn(net(xb), yb, weights)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def predict_batches(net: Net, loader: DataLoader, cfg: Config, dev: torch.device) -> np.ndarray:
    net.eval()
    out = []
    with torch.inference_mode(), amp_ctx(cfg.cnn, dev):
        for xb, yb in loader:
            out.append(net(to_dev(xb, yb, dev)[0]).argmax(1).cpu().numpy())
    return np.concatenate(out)


def evaluate_cells(net: Net, loader: DataLoader, y: np.ndarray, cfg: Config, dev: torch.device) -> float:
    return macro_f1(y, predict_batches(net, loader, cfg, dev), N_CLASSES)


def make_optim(net: Net, c: dict, steps: int) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    opt = torch.optim.AdamW(net.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
    return opt, torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=c["lr"], total_steps=max(steps, 1))


def update_best(track: dict, f1: float, net: Net) -> bool:
    if f1 > track["f1"]:
        track.update(f1=f1, bad=0, state={k: v.detach().clone() for k, v in net.state_dict().items()})
        return False
    track["bad"] += 1
    return track["bad"] >= track["patience"]


def setup(cfg: Config, recs: dict[str, dict], train_rows: list[dict], epochs: int) -> tuple:
    torch.manual_seed(cfg.cnn["seed"])
    dev = assert_device()
    loader, y = make_loaders(cfg, recs, train_rows, shuffle=True)
    net = build_net(cfg.cnn).to(dev).to(memory_format=torch.channels_last)
    weights = torch.from_numpy(class_weights(y)).to(dev)
    print(f"cnn: {sum(p.numel() for p in net.parameters())} params on {dev}, {len(train_rows)} train patches, {epochs} epochs")
    return dev, loader, net, weights, *make_optim(net, cfg.cnn, epochs * len(loader))


def train(cfg: Config, recs: dict[str, dict], train_rows: list[dict], val_rows: list[dict] | None, epochs: int) -> Net:
    dev, loader, net, weights, opt, sched = setup(cfg, recs, train_rows, epochs)
    val = make_loaders(cfg, recs, val_rows, shuffle=False) if val_rows else None
    track = {"f1": -1.0, "bad": 0, "state": None, "patience": cfg.cnn["patience"]}
    for ep in range(1, epochs + 1):
        loss = train_epoch(net, loader, opt, sched, weights, cfg, dev)
        f1 = evaluate_cells(net, *val, cfg, dev) if val else float("nan")
        print(f"epoch {ep:3d}  loss {loss:.4f}  val macro-F1 {f1:.4f}")
        if val and update_best(track, f1, net):
            print(f"early stop at epoch {ep}, best val macro-F1 {track['f1']:.4f}")
            break
    if track["state"] is not None:
        net.load_state_dict(track["state"])
    return net.eval()


def save(net: Net, cfg: Config) -> None:
    meta = load_metadata(data_dir(cfg))
    mean, std = norm_stats(meta)
    classes = [meta["classes"][str(i)] for i in range(N_CLASSES)]
    blob = {"state_dict": net.state_dict(), "cnn": dict(cfg.cnn), "mean": mean.tolist(), "std": std.tolist(), "classes": classes}
    torch.save(blob, model_path(cfg, MODEL_FILE))
    print(f"saved {model_path(cfg, MODEL_FILE)}")


def fit(cfg: Config, epochs: int | None = None) -> Net:
    rows, recs = load_patches(data_dir(cfg)), load_all(data_dir(cfg))
    net = train(cfg, recs, split_rows(rows, "train"), split_rows(rows, "val"), epochs or cfg.cnn["epochs"])
    save(net, cfg)
    return net


def load(cfg: Config) -> tuple[Net, torch.device]:
    path = model_path(cfg, MODEL_FILE)
    if not path.exists():
        raise SystemExit(f"error: model file {path} not found; run `docker compose run --rm das train cnn` first")
    blob = torch.load(path, map_location="cpu", weights_only=True)
    dev = assert_device()
    net = build_net(blob["cnn"])
    net.load_state_dict(blob["state_dict"])
    return net.to(dev).to(memory_format=torch.channels_last).eval(), dev


def infer_batch(net: Net, dev: torch.device, c: dict, batch: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(batch).to(dev).contiguous(memory_format=torch.channels_last)
    with torch.inference_mode(), amp_ctx(c, dev):
        return torch.softmax(net(x).float(), dim=1).cpu().numpy()


def predict_grid(net: Net, dev: torch.device, rec: dict, cfg: Config) -> np.ndarray:
    grid, coverage = normalised_grid(rec)
    i = cfg.infer
    fn = partial(infer_batch, net, dev, cfg.cnn)
    return predict_tiled(fn, grid, coverage, i["tile_rows"], i["tile_channels"], i["stride_rows"], i["stride_channels"])
