from api.catalog import SCENARIOS, class_list
from api.view import positions, restrict, view_window
from das.config import Config, data_dir, load_config
from das.data import row_seconds
from das.predict import class_names, load_model, load_recording_for_inference

MODEL_NAME = "cnn"


def prepare(sc: dict, cfg: Config) -> dict:
    rec = load_recording_for_inference(data_dir(cfg) / f"{sc['recording']}.h5", cfg)
    view = view_window(rec)
    return {"rec": restrict(rec, view), "view": view}


def scenario_card(sc: dict, prepared: dict) -> dict:
    rec, view = prepared["rec"], prepared["view"]
    duration = rec["targets"].shape[0] * row_seconds(rec["attrs"])
    return {**{k: v for k, v in sc.items() if k != "recording"}, "duration_s": duration, **positions(rec, view)}


def build_context(config_path: str = "config.yaml") -> dict:
    cfg = load_config(config_path)
    names = class_names(cfg)
    prepared = {sc["id"]: prepare(sc, cfg) for sc in SCENARIOS}
    cards = [scenario_card(sc, prepared[sc["id"]]) for sc in SCENARIOS]
    return {"cfg": cfg, "fn": load_model(MODEL_NAME, cfg), "names": names, "classes": class_list(names), "prepared": prepared, "cards": cards}
