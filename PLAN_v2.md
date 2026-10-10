# DAS Event Classifier v2: Agent Build Plan (model + CLI only)

Data: colleague's `das_training_v2` (band-power features, 130 MB). Not the raw 50 GB.
Build in one pass. No frontend, no API, no tests.
End state: the user runs only the commands in "Final commands", then tests with the CLI.

## Code rules (hard)
- Small atomic functions, one job each, max ~15 lines.
- No comments, no docstrings.
- All imports at top of file. No lazy or inline imports.
- Type hints on signatures. No classes except `torch.nn.Module` and a frozen `Config` dataclass.
- No global mutable state. Paths and hyperparameters come from `config.yaml` only.
- Fail fast with `assert` + clear message on shape/dtype/label mismatches.
- CLI is `argparse` only.

## Repo state
- Work inside the existing `noDAS/` repo (keep `.git`, `Dockerfile`, `docker-compose.yml`).
- Delete obsolete raw-data modules: `figshare.py`, `h5io.py`, `dsp.py`, `windows.py`, `labels.py`, `features.py`, `build.py`, `split.py`.
- Rewrite the rest from scratch per this plan. Rewrite `config.yaml`, `pyproject.toml`.
- Copy `das_training_v2/das_training_v2/` to `noDAS/data/v2/` (read-only source of truth; never write into it).

## Stack
python 3.11, numpy, scipy, h5py, pyyaml, scikit-learn, lightgbm, joblib, torch (cu128 wheels, >=2.7, installed separately via `requirements-torch.txt`).

## Dataset facts (verified; code must assert them)
- 16 recordings in `data/v2/*.h5`, each with datasets `features` float32 [T, C, 6] and `targets` [T, C] (C is 1663 or 1700).
- Attrs per file: `sample_rate_hz=20000`, `stride_samples=2048`, `support_samples=8192`, `spatial_spacing_m=1.0209523`, `start_locus_index`, `feature_channel_min`, `feature_channel_end`, `recording_id`, `source_class`.
- One feature row = 2048 raw samples = 0.1024 s. Row r covers raw samples [r*2048, r*2048+8192).
- Features: log1p band power in 6 bands (Hz): (0,50], (50,150], (150,500], (500,1500], (1500,5000], (5000,10000].
- NaN features exist outside the stored channel coverage (10-92% of cells per file). Patches never contain NaN.
- Targets: 0 background, 1 car, 2 construction, 3 fence, 4 longboard, 5 manipulation, 6 openclose, 7 running, 8 walk, 255 unknown.
- 255 must never be scored as background and never used as a training target (`ignore_index=255`).
- Background (class 0) exists only in `regularshit_*.h5` (one recording). It is annotated background, not proof of no activity.
- `metadata.json` holds `classes`, `normalization.mean/std` (6 values each, fitted on training cells), `ignore_index`, `patch_shape`.
- `patches.csv` columns: `recording_id, split, class_id, time_row, channel, height, width, labeled_cells`. 486 rows: train 299, val 105, test 82. Height 32 rows, width 128 channels.
- Splits are chronological 60/20/20 per recording with 3-row gaps; patches never cross a boundary. Patches can overlap each other within a split.
- Normalize as `(features - mean) / std` using `metadata.json` values only. Never refit.
- Spatial position of channel c: `(c + start_locus_index) * spatial_spacing_m`. Time of row r: `r * 2048 / 20000` s.
- Multi-recording classes (for leave-one-recording-out): car (4 files: `auto_`, `auto2_`, `auto4_..144655`, `auto4_..145055`), fence (`fence_`, `kopani_plot_`, `plot_`), walk (`walking_`, `walking2_`), longboard (`longboard_`, `longboardroad_`). Single-recording: construction (`reconstruction1_`), manipulation, openclose, running, background.

## Reference results to beat (measured with a quick per-cell LightGBM, 6 bands + 9x9 context mean/std)
- Official test split: macro-F1 0.71, accuracy 0.81, background-vs-activity accuracy 0.97.
- Weak: openclose F1 0.40, running 0.49, construction recall 0.48, walk 0.69.
- Leave-one-recording-out recall: car 0.64, fence 0.73, walk 0.50, longboard 0.50.
- The CNN should match or beat these. Print them next to its own results in `compare`.

## Layout
```
config.yaml
pyproject.toml
requirements-torch.txt
das/
  __main__.py
  cli.py
  config.py        # frozen Config dataclass, load_config
  data.py          # load metadata/patches/recordings into RAM, normalise, coverage masks
  patches.py       # patch table -> arrays, sampler weights, cell masks by split
  augment.py       # channel flip, gain offset, noise
  cellfeat.py      # per-cell tabular features (6 bands + 9x9 context mean/std)
  grid.py          # tile a [T,C,6] grid into windows, stitch probs back
  lgbm.py          # train/load/predict_grid
  cnn.py           # net, train/load/predict_grid
  metrics.py       # cell metrics, coarse map, threat-vs-bg
  events.py        # connected components, event metrics, false alarms per km-day
  postproc.py      # smoothing, k-of-n confirm
  predict.py       # file -> events table
  stream.py        # paced replay with live alerts
  loro.py          # leave-one-recording-out runs
  report.py        # print tables
data/v2/  models/
```

## config.yaml (keys)
- `paths.data: data/v2`, `paths.models: models`
- `data.ignore_index: 255`, `data.n_classes: 9`, `data.unknown: 255`
- `coarse_map`: background->background; walk, running, longboard->person; car, construction->vehicle; fence, manipulation, openclose->intrusion
- `cellfeat.context_rows: 9`, `cellfeat.context_channels: 9`
- `lgbm.*`: n_estimators 150, learning_rate 0.08, num_leaves 31, class_weight balanced, subsample 0.8, colsample 0.8, `max_train_cells: 400000`, `seed: 0`
- `cnn.channels: [32, 48, 64, 64]`, `cnn.dilations: [1, 2, 4, 8]`, `cnn.dropout: 0.1`, `cnn.epochs: 40`, `cnn.batch: 32`, `cnn.lr: 0.002`, `cnn.weight_decay: 0.0001`, `cnn.patience: 8`, `cnn.amp: bf16`, `cnn.workers: 0`, `cnn.seed: 0`
- `augment.flip_channels: 0.5`, `augment.gain_std: 0.15`, `augment.noise_std: 0.05`
- `infer.tile_rows: 32`, `infer.tile_channels: 128`, `infer.stride_rows: 16`, `infer.stride_channels: 64`
- `alert.threshold: 0.6`, `alert.min_cells: 40`, `alert.confirm: [2, 3]`, `alert.merge_gap_channels: 8`, `alert.hop_rows: 4`, `alert.track_radius_m: 20`
- `eval.event_hit_fraction: 0.3`, `eval.min_event_cells: 40`

## Module specs

### data.py
- `load_config`-independent helpers: `load_metadata(root)`, `load_patches(root) -> list[dict]`, `list_recordings(root) -> list[str]`
- `Recording` is a plain `dict` with keys `id, features (normalised, NaN kept), targets, attrs, coverage (bool [T,C])`.
- `load_recording(root, rec_id, mean, std) -> dict`
- `load_all(root) -> dict[str, dict]` (whole set fits in RAM; no per-item HDF5 opens)
- `assert_consistent(recs, patches, meta)` checks: patch bounds, shapes, class ids in 0..8 or 255, stride 2048, support 8192, split names, six normalisation values, every patch inside `coverage`.

### patches.py
- `patch_arrays(recs, rows) -> (x [N,6,32,128] float32, y [N,32,128] int64, meta)` with NaN replaced by 0 only if the cell is outside coverage (never happens for indexed patches; assert).
- `split_rows(rows, split)`, `sampler_weights(rows)` giving equal total mass to background and activity patches (same rule as the colleague's `sampling_weights`).
- `cell_mask(rec, rows) -> bool [T,C]` union of patch regions of a split, intersected with `targets != 255`.
- `class_weights(y) -> np.ndarray[9]` inverse-sqrt frequency on labelled cells.

### augment.py
- `flip_channels(x, y, p)`, `add_gain(x, std)` (adds a per-band-shared scalar offset in normalised space), `add_noise(x, std)`, `apply(x, y, cfg)`.
- Applied to training batches only.

### cellfeat.py
- `normalised_grid(rec) -> [T,C,6]` with NaN->0 and a returned coverage mask.
- `context_mean(x, rows, ch)`, `context_std(x, rows, ch)` via `scipy.ndimage.uniform_filter`.
- `cell_features(grid) -> [T,C,18]` = bands + context mean + context std.
- `gather(rec, mask) -> (X [n,18], y [n])`.

### grid.py
- `tile_starts(n, size, stride) -> list[int]` (last tile aligned to the end).
- `predict_tiled(fn, grid [T,C,6], rows, ch, stride_rows, stride_ch) -> probs [T,C,K]` averaging overlapping tiles, `fn` maps [B,6,rows,ch] to [B,K,rows,ch] probabilities.
- Output cells outside `coverage` are set to NaN so downstream code treats them as unknown.
- Pads when T < tile rows.

### lgbm.py
- `fit(cfg)`: gather training cells from official `train` patches (cap at `max_train_cells`), fit `LGBMClassifier`, early stop on val logloss, save `models/lgbm.joblib` + `lgbm_importance.json` (feature names `band{i}`, `ctxmean{i}`, `ctxstd{i}`).
- `load(cfg)`, `predict_grid(model, rec_grid) -> [T,C,K]` (feature grid computed once over the whole recording, same function as training).

### cnn.py
- `build_net(cfg)`: fully convolutional, resolution preserved. Input [B,6,H,W]. Stem Conv3x3-BN-ReLU, then 4 residual blocks (two 3x3 convs, dilation d from `cnn.dilations` applied on both axes, BN, ReLU, Dropout2d), head Conv1x1 -> 9 logits. About 150k-250k params. Works on any H, W.
- `make_loaders(cfg, recs, rows)` with `WeightedRandomSampler` from `sampler_weights`, `replacement=True`, epoch length = number of train patches x 2.
- `loss_fn(logits, y, weights)` = `cross_entropy(..., ignore_index=255, weight=weights)`.
- `train_epoch`, `evaluate_cells(net, loader) -> macro-F1 over labelled cells`, `fit(cfg)`: AdamW, OneCycleLR, bf16 autocast on CUDA, `channels_last`, early stop on val macro-F1, keep best state.
- `assert_device()` picks CUDA when `sm_120` is in `torch.cuda.get_arch_list()` (or any usable arch), else CPU with a printed one-line warning.
- Saves `models/cnn.pt` with `state_dict`, config snapshot, normalisation, class list.
- `load(cfg)`, `predict_grid(net, rec_grid) -> [T,C,K]` via `grid.predict_tiled`, batched, softmax.

### metrics.py
- `cell_report(y_true, y_pred) -> per-class P/R/F1/support + macro-F1 + accuracy`, scored only where `y_true != 255`.
- `confusion(y_true, y_pred, n)`.
- `coarse(y, cfg)` maps 9 classes to {background, person, vehicle, intrusion}.
- `threat_vs_background(y_true, y_pred) -> {pod, nar, fnr}` using background = class 0.
- `macro_f1(y_true, y_pred)`.

### events.py
- `components(mask, merge_gap_channels) -> label grid, list[dict]` using `scipy.ndimage.label` after a small closing along channels.
- `event_table(labels, probs, attrs, min_cells) -> list[Event]` with fields: `t_start_s, t_end_s, c0, c1, pos_start_m, pos_end_m, class_id, class_name, confidence (mean prob of class), cells`.
- `gt_events(targets, min_cells)`: connected components of each non-background labelled class in ground truth.
- `event_metrics(gt, pred, hit_fraction)`: a GT event is detected if >= `hit_fraction` of its cells are predicted as any non-background class; classified correctly if the mode of predictions over its cells equals its class. Returns detection rate (PoD), classification accuracy given detection, localisation error (m between centroids).
- `false_alarms(pred_events, bg_mask)`: events whose cells overlap annotated background by >50%.
- `km_days(bg_cells, attrs) -> float` = `bg_cells * 0.1024 s * 1.021 m` converted to km-days; `alarms_per_km_day`.
- Prints a one-line warning that false alarms are measured only inside annotated background.

### postproc.py
- `argmax_with_threshold(probs, thr, background=0) -> class grid` (non-background class kept only if max prob >= thr, else background; NaN cells -> 255).
- `smooth_time(probs, k)` uniform smoothing along rows.
- `confirm(history, k, n)` for streaming.
- `to_position_m(c, attrs)`.

### predict.py
- `load_model(name, cfg) -> callable(rec_grid) -> probs [T,C,K]` for `lgbm` and `cnn` (same signature).
- `load_recording_for_inference(path, cfg)`: reads a v2-format `.h5`, normalises with metadata, builds coverage.
- `scan(path, model_name, cfg, t0, t1, thr) -> list[Event]`.
- `events_table(events) -> str`.

### stream.py
- `replay(path, model_name, cfg, speed, t0, t1)`: reads the feature grid, iterates in hops of `alert.hop_rows`, keeps a rolling buffer of `tile_rows` rows, runs the model on the buffer each hop, keeps predictions for the newest hop rows only, applies threshold and `confirm`, merges channels, prints a line when an alert opens or closes (time, position m, class, confidence), pacing with `sleep(hop_rows * 0.1024 / speed)`.
- Ends with a summary: alert count, per-class counts, and, if the file has labels, per-event detection latency (first alert time minus ground-truth onset).
- Note: buffer-end edge effects are expected (no future context); accept.

### loro.py
- `folds(recs) -> list[str]`: recordings of multi-recording classes (car, fence, walk, longboard).
- `run_fold(model_name, hold_id, cfg)`: train on all labelled patches from every other recording (all splits), evaluate on all labelled cells of the held-out recording. For `cnn` use `cnn.epochs_loro` (default 15) and no early stopping on the held-out file.
- `run_all(model_name, cfg, only=None)` prints, per held-out file: true class, recall, predicted-class distribution, and event-level detection rate.

### report.py
- `print_table(rows, header)`, `print_confusion(cm, names)`, `print_cell_report(...)`, `print_events(...)`.

## CLI (`python -m das ...`)
| command | action |
|---|---|
| `inspect [PATH]` | dataset audit: per-recording shape, coverage %, class cell counts, patches per split and class; runs `assert_consistent`; exits non-zero on any mismatch |
| `train lgbm` | train and save LightGBM |
| `train cnn [--epochs N]` | train and save CNN |
| `eval --model {lgbm,cnn} [--coarse]` | official test split: per-class P/R/F1, confusion matrix, threat-vs-background PoD/NAR/FNR, event-level PoD and classification accuracy, localisation error, background false alarms per km-day |
| `loro --model {lgbm,cnn} [--hold REC]` | leave-one-recording-out table |
| `predict FILE --model M [--t0 S --t1 S] [--threshold X]` | scan one v2 `.h5`, print events table (time s, position m, channels, class, confidence) |
| `stream FILE --model M [--speed 5] [--t0 --t1]` | paced replay with live alerts |
| `compare` | `eval` for both models side by side plus the reference numbers above |

## Final commands (all the user runs)
```
pip install -e .
pip install -r requirements-torch.txt
python -m das inspect
python -m das train lgbm
python -m das train cnn
```
Then test:
```
python -m das eval --model cnn
python -m das compare
python -m das loro --model cnn
python -m das predict data/v2/walking2_2023-04-17T122153+0100.h5 --model cnn
python -m das stream data/v2/auto4_2023-06-06T145055+0100.h5 --model cnn --speed 5
```

## Build order for the agent
1. `config.py`, `config.yaml`, `pyproject.toml`, `requirements-torch.txt`; delete obsolete modules.
2. `data.py`, `patches.py`, CLI skeleton with `inspect`. Run `inspect` against the real data and fix any assumption that fails before going on.
3. `cellfeat.py`, `grid.py`, `metrics.py`, `events.py`, `postproc.py`, `report.py`.
4. `lgbm.py`, `eval`, `compare` (lgbm only at first).
5. `augment.py`, `cnn.py`, `train cnn`. Run one epoch on the real data to confirm the loss decreases and shapes match.
6. `predict.py`, `loro.py`, `stream.py`.
7. Wire every subcommand; run `python -m compileall das` and `python -m das --help`. Do not write tests.

## Acceptance
- `python -m das --help` lists every command above.
- No file contains `#` comments, docstrings, or an import below the top block.
- Every function fits ~15 lines.
- `inspect` passes on `data/v2` and prints 16 recordings and 299/105/82 patches.
- `lgbm` official-test macro-F1 is within 0.05 of the 0.71 reference; `cnn` is at least as good, or `compare` prints both honestly.
- Unknown (255) cells are never counted in any metric and never turned into background anywhere.
- `predict` and `stream` use the same normalisation, tiling and post-processing as `eval`.
- A missing model file raises a one-line error naming the training command to run.
- Any shape or label mismatch stops with a message showing both values.

## Known limits to print in `eval` and `loro` output
- Test partitions share recordings with training; scores are within-session. `loro` is the honest check.
- Background comes from a single recording, so false-alarm numbers are indicative only.
- Features are 6 bands at 0.1 s; no phase or velocity information.
