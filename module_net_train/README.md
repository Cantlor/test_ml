# module_net_train

Модуль обучения, инференса и оценки multitask U-Net.

Состояние описано по фактическим артефактам на **2026-03-16**.

## 1. Назначение

Модель предсказывает 2 probability-map:
- `extent` (маска поля),
- `boundary` (границы полей).

Основные CLI:
- `scripts/01_check_prep_data.py`
- `scripts/02_train.py`
- `scripts/03_predict_aoi.py`
- `scripts/04_eval.py`
- `scripts/05_predict_raster.py` (single direct-raster predict)
- `scripts/06_predict_input_data_batch.py` (batch direct-raster predict from `input_data/`)

## 2. Входной data contract

Ожидается `prep_data/<split>/`:

```text
meta/meta_<patch_id>.json
img/img_<patch_id>.tif
valid/valid_<patch_id>.tif
extent/extent_<patch_id>.tif
boundary_bwbl/bwbl_<patch_id>.tif
```

Семантика:
- `valid`: `0/1`
- `extent`: `0/1/255` (`255` = ignore)
- `bwbl`: `0/1/2` (`2` = ignore)

Фактически используется:
- `dataset.inputs.num_bands=8`
- `add_valid_channel=true`
- итоговый вход в модель: `9` каналов.

## 3. Текущее baseline-конфиг состояние

### Конфиги
- `configs/train_config.yaml`
- `configs/hardware_config.yaml`

Ключевые effective значения (см. `runs/20260316_103300/config_resolved.yaml`):
- `epochs=40`
- `lr=0.0007`
- `boundary loss weight=8.0`
- `sampling.crop_policy.near_invalid.enabled=true`
- `inference.tiling`: `window=512`, `stride=384`, `blend=gaussian`

### Hardware/runtime
(см. `runs/20260316_103300/hardware.json`)
- GPU: `NVIDIA GeForce RTX 3050 Laptop GPU (4GB)`
- precision: `bf16`
- batch size: `1`
- grad accumulation: `4`
- crop size: `256`

## 4. Текущий reliable run

Релевантный run:
- `output_data/module_net_train/runs/20260316_103300`

Ключевые результаты:
- best epoch по `val/boundary_f1_max`: `38`
- best `val/boundary_f1_max`: `0.475046` (`thr=0.70`)
- test (`metrics/eval_test.json`):
  - `val/extent_iou=0.944759`
  - `val/extent_f1=0.971595`
  - `val/boundary_f1_max=0.475501`

Near-invalid диагностика (test):
- `val/extent_f1_near_invalid=0.0`
- `val/boundary_f1_max_near_invalid=0.109235`
- `val/near_invalid_valid_frac=0.0017679`

Интерпретация:
- глобально extent работает хорошо;
- near-invalid зона остается главным слабым местом.

## 5. Инференс и артефакты

`03_predict_aoi.py` создает:
- `pred/<dataset_key>/extent_prob.tif`
- `pred/<dataset_key>/boundary_prob.tif`
- `pred/<dataset_key>/predict_manifest.json`

Фактический manifest для baseline:
- `tiles=644`
- `window_size=512`, `stride=384`
- `blend=gaussian`, `gaussian_sigma=0.3`
- `checkpoint_fallback_used=false`
- `invalid_edge_guard_px=0`

## 6. Оценка

`04_eval.py`:
- использует `run_dir/config_resolved.yaml` как приоритетный source of truth;
- пишет итог в `metrics/eval_test.json`;
- включает near-invalid KPI.

## 7. Команды

Из корня репозитория:

```bash
# Проверка prep_data
./.venv/bin/python module_net_train/scripts/01_check_prep_data.py \
  --config module_net_train/configs/train_config.yaml \
  --out_json output_data/module_net_train/runs/prep_data_summary.json

# Обучение (по умолчанию в конце может сделать infer)
./.venv/bin/python module_net_train/scripts/02_train.py \
  --config module_net_train/configs/train_config.yaml \
  --hardware module_net_train/configs/hardware_config.yaml

# Ручной infer для существующего run
./.venv/bin/python module_net_train/scripts/03_predict_aoi.py \
  --config module_net_train/configs/train_config.yaml \
  --hardware module_net_train/configs/hardware_config.yaml \
  --run_dir output_data/module_net_train/runs/20260316_103300

# Ручной infer по одному заданному raster (без AOI manifest/dataset_key workaround)
./.venv/bin/python module_net_train/scripts/05_predict_raster.py \
  --raster /abs/path/to/target.tif \
  --run_dir output_data/module_net_train/runs/20260316_103300 \
  --output_dir /abs/path/to/pred_out

# Batch infer: автоматически взять *.tif/*.tiff из input_data/
./.venv/bin/python module_net_train/scripts/06_predict_input_data_batch.py \
  --input_dir input_data \
  --run_dir output_data/module_net_train/runs/20260316_103300 \
  --output_root output_data/module_net_train/direct_predict

# Удобный shell-wrapper с дефолтными путями проекта
bash scripts/run_predict_input_data.sh \
  --run_dir output_data/module_net_train/runs/20260316_103300

# Eval на test
./.venv/bin/python module_net_train/scripts/04_eval.py \
  --config module_net_train/configs/train_config.yaml \
  --hardware module_net_train/configs/hardware_config.yaml \
  --run_dir output_data/module_net_train/runs/20260316_103300
```

`05_predict_raster.py` пишет в `--output_dir`:
- `extent_prob.tif`
- `boundary_prob.tif`
- `predict_manifest.json`

`06_predict_input_data_batch.py` пишет в `--output_root`:
- `<sample_id>/extent_prob.tif`
- `<sample_id>/boundary_prob.tif`
- `<sample_id>/predict_manifest.json`
- `batch_predict_manifest.json`

Где `<sample_id>` стабильно выводится из имени входного файла (`.tif/.tiff`) с безопасной нормализацией (lowercase + `_`), а при коллизиях добавляется суффикс `_2`, `_3`, ...

Опционально можно включить постобработку:

```bash
./.venv/bin/python module_net_train/scripts/05_predict_raster.py \
  --raster /abs/path/to/target.tif \
  --run_dir output_data/module_net_train/runs/20260316_103300 \
  --output_dir /abs/path/to/pred_out \
  --with-postprocess \
  --postprocess-config module_postprocess_vectorize/configs/postprocess_config.yaml

# или для batch-flow из input_data/
bash scripts/run_predict_input_data.sh \
  --run_dir output_data/module_net_train/runs/20260316_103300 \
  --with-postprocess
```

## 8. Progress UX в терминале

- Добавлен прогресс для длительных стадий:
  - индексация prep-data и расчет normalization stats;
  - epoch loop + train batch loop + validation batch loop;
  - tiled inference по AOI;
  - eval loop.
- Прогресс не меняет train/eval semantics, чекпоинты и формат метрик.
- Управление через env:
  - `DISABLE_PROGRESS=1`
  - `FORCE_PROGRESS=1`

## 9. Что уже закрыто / что остается

Подтверждено:
- upstream fix в `module_prep_data` восстановил internal boundary semantics в labels.

Остается bottleneck:
- noisier boundary behavior и слабая near-invalid зона;
- часть GT-границ может быть плохо наблюдаема по текущему imagery (observability limit).

Это означает, что после prep-fix новый train-run актуален и нужен как основной baseline для сравнения следующих экспериментов.

## 10. Какие файлы смотреть в первую очередь

1. `output_data/module_net_train/runs/20260316_103300/config_resolved.yaml`
2. `output_data/module_net_train/runs/20260316_103300/metrics/history.csv`
3. `output_data/module_net_train/runs/20260316_103300/metrics/eval_test.json`
4. `output_data/module_net_train/runs/20260316_103300/pred/first_raster/predict_manifest.json`
5. `output_data/module_net_train/runs/20260316_103300/logs/train.log`
