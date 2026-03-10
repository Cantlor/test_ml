# module_net_train

Модуль обучения, инференса и оценки multitask U-Net для сегментации полей по данным из `prep_data`.

Модель предсказывает 2 карты:
- `extent` (маска поля, бинарная),
- `boundary` (границы/скелет границ, бинарная с ignore-зоной).

## Что умеет модуль

- Проверка структуры и содержимого `prep_data`: `scripts/01_check_prep_data.py`
- Обучение модели и сохранение чекпоинтов: `scripts/02_train.py`
- Инференс по AOI GeoTIFF с тайлингом: `scripts/03_predict_aoi.py`
- Оценка на test-сплите: `scripts/04_eval.py`

## Структура модуля

- `configs/train_config.yaml` — датасет, модель, loss, метрики, train/inference.
- `configs/hardware_config.yaml` — device, precision, autotune batch/crop, dataloader.
- `net_train/data/*` — индексация, `PatchDataset`, аугментации, нормализация.
- `net_train/models/*` — `unet_multitask` (2 головы: extent + boundary).
- `net_train/losses/*` — extent `BCE+Dice`, boundary `BCE`.
- `net_train/metrics/*` — extent IoU/F1, boundary F1 с dilation.
- `net_train/train/*` — optimizer/scheduler, train/val loop, checkpoint manager.
- `net_train/infer/*` — генерация окон, blending, запись GeoTIFF.

## Входные данные (контракт)

Корень задается в `paths.prep_data_root` (по умолчанию `../prep_data`).
Ожидаемая структура для каждого сплита (`train`, `validation`, `test`):

```text
prep_data/<split>/
  meta/meta_<patch_id>.json
  img/img_<patch_id>.tif                # >= num_bands каналов
  valid/valid_<patch_id>.tif            # 0/1
  extent/extent_<patch_id>.tif          # 0/1/255 (255 = ignore)
  boundary_bwbl/bwbl_<patch_id>.tif     # 0/1/2   (2 = ignore)
```

`01_check_prep_data.py` валидирует:
- наличие файлов,
- число каналов в `img`,
- допустимые значения в масках,
- базовые мета-статистики (valid_ratio/mask_ratio/edge_ratio).

## Нормализация и каналы

- Нормализуются только спектральные каналы (`dataset.inputs.num_bands`, по умолчанию 8).
- Опционально добавляется `valid` как дополнительный входной канал (`add_valid_channel: true`), то есть модель получает 9 каналов.
- Статистика нормализации сохраняется в `band_stats.npz` и переиспользуется в инференсе/оценке.

## Обучение

В `02_train.py` выполняется пайплайн:
1. Чтение train/hardware-конфигов.
2. Построение runtime-плана (`device`, `precision`, `crop`, `batch`, `workers`).
3. Индексация `train/val`, расчет статистик нормализации.
4. Обучение + валидация по эпохам.
5. Сохранение `last.pt` и `best.pt`.
6. Опциональный инференс AOI (если `inference.enabled=true`, можно отключить флагом `--no_infer`).

Мониторинг лучшего чекпоинта по умолчанию: `val/extent_iou`.

## Инференс AOI

`03_predict_aoi.py`:
- загружает `band_stats.npz` из `run_dir`,
- берет чекпоинт (`best.pt`, fallback `last.pt`),
- читает AOI путь из manifest по `dataset_key`,
- выполняет скользящий тайлинг (`window_size`, `stride`, `blend`),
- пишет:
  - `extent_prob.tif`
  - `boundary_prob.tif`
  - `predict_manifest.json`

Поддерживаются blend-режимы:
- `mean`
- `gaussian`

## Оценка на test

`04_eval.py`:
- грузит `band_stats.npz` и чекпоинт,
- считает метрики на сплите `test`,
- сохраняет `metrics/eval_test.json`.

## Запуск

Команды из корня репозитория:

```bash
# Зависимости проекта (lock-файл в корне репозитория)
./.venv/bin/python -m pip install -r require.txt

# 1) Проверка данных
./.venv/bin/python module_net_train/scripts/01_check_prep_data.py \
  --config module_net_train/configs/train_config.yaml \
  --out_json output_data/module_net_train/runs/prep_data_summary.json

# 2) Обучение (+ инференс AOI по умолчанию)
./.venv/bin/python module_net_train/scripts/02_train.py \
  --config module_net_train/configs/train_config.yaml \
  --hardware module_net_train/configs/hardware_config.yaml

# 3) Ручной инференс для существующего run
./.venv/bin/python module_net_train/scripts/03_predict_aoi.py \
  --config module_net_train/configs/train_config.yaml \
  --hardware module_net_train/configs/hardware_config.yaml \
  --run_dir output_data/module_net_train/runs/<run_id>

# 4) Оценка на test
./.venv/bin/python module_net_train/scripts/04_eval.py \
  --config module_net_train/configs/train_config.yaml \
  --hardware module_net_train/configs/hardware_config.yaml \
  --run_dir output_data/module_net_train/runs/<run_id>
```

## Артефакты запуска

Все результаты в:

```text
output_data/module_net_train/runs/<run_id>/
```

Основные файлы:
- `config_resolved.yaml` — эффективный конфиг после runtime-override.
- `hardware.json` — итоговый runtime-план.
- `band_stats.npz` — статистики нормализации.
- `logs/train.log` — лог обучения.
- `checkpoints/last.pt`, `checkpoints/best.pt` — чекпоинты.
- `metrics/history.csv` — train/val метрики по эпохам.
- `metrics/eval_test.json` — метрики test (после `04_eval.py`).
- `pred/<dataset_key>/extent_prob.tif` — вероятность extent.
- `pred/<dataset_key>/boundary_prob.tif` — вероятность boundary.
- `pred/<dataset_key>/predict_manifest.json` — мета инференса.

## Примечания по конфигам

- Пути в `train_config.yaml` резолвятся относительно папки `module_net_train`, а не текущей директории запуска.
- `model.in_channels` автоматически приводится к контракту данных:
  - `num_bands + 1`, если `add_valid_channel=true`,
  - `num_bands`, если `false`.
- Если CUDA недоступна и `device.mode=auto`, модуль перейдет на CPU (с warning).

## Типичные проблемы

- `missing files` в check/train:
  - запустить `01_check_prep_data.py` и исправить структуру `prep_data`.
- `Missing normalization stats: band_stats.npz`:
  - не был завершен train-run или неверный `--run_dir`.
- `Checkpoint not found`:
  - в `run_dir/checkpoints` нет `best.pt`/`last.pt`.
- Ошибка резолва AOI по `dataset_key`:
  - проверить `inference.aoi.dataset_key` и manifest `aoi_rasters_manifest.json`.
