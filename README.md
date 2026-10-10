# noDAS
Repository for the sauliai hackathon 2026

DAS event classifier v2 (LightGBM + CNN) on `data/v2` band-power features. Everything runs in Docker with the GPU.

```
docker compose build
docker compose run --rm das inspect
docker compose run --rm das train lgbm
docker compose run --rm das train cnn
docker compose run --rm das eval --model cnn
docker compose run --rm das compare
docker compose run --rm das loro --model cnn
docker compose run --rm das predict data/v2/walking2_2023-04-17T122153+0100.h5 --model cnn
docker compose run --rm das stream data/v2/auto4_2023-06-06T145055+0100.h5 --model cnn --speed 5
```

`./das`, `./config.yaml`, `./data` and `./models` are mounted, so code and config edits need no rebuild.
