from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Config:
    paths: dict
    data: dict
    coarse_map: dict
    cellfeat: dict
    lgbm: dict
    cnn: dict
    augment: dict
    infer: dict
    alert: dict
    eval: dict
    base: Path


def section_names() -> list[str]:
    return [f.name for f in fields(Config) if f.name != "base"]


def load_config(path: str | Path = "config.yaml") -> Config:
    p = Path(path).resolve()
    assert p.exists(), f"config file not found: {p}"
    raw = yaml.safe_load(p.read_text())
    missing = [n for n in section_names() if n not in raw]
    assert not missing, f"config.yaml is missing sections {missing}"
    return Config(**{n: raw[n] for n in section_names()}, base=p.parent)


def data_dir(cfg: Config) -> Path:
    return cfg.base / cfg.paths["data"]


def model_path(cfg: Config, filename: str) -> Path:
    d = cfg.base / cfg.paths["models"]
    d.mkdir(parents=True, exist_ok=True)
    return d / filename
