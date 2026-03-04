# module_net_train

Модуль обучения и инференса для `prep_data`.

Реализованы блоки:
- проверка входных данных (`scripts/01_check_prep_data.py`),
- обучение multitask U-Net (`scripts/02_train.py`),
- предикт на AOI (`scripts/03_predict_aoi.py`),
- оценка на test (`scripts/04_eval.py`).

## Структура

- `configs/train_config.yaml` — обучение/данные/инференс.
- `configs/hardware_config.yaml` — автонастройка device/precision/batch/crop/workers.
- `net_train/data/*` — индекс, датасет, аугментации, нормализация.
- `net_train/models/*` — `unet_multitask` (2 головы: extent/boundary).
- `net_train/losses/*` — extent BCE+Dice с ignore=255, boundary BCE с ignore=2.
- `net_train/metrics/*` — extent IoU/F1, boundary F1 с dilation.
- `net_train/train/*` — optimizer/scheduler, loop, checkpoints.
- `net_train/infer/*` — tile inference и запись GeoTIFF вероятностей.

## Артефакты run

`output_data/module_net_train/runs/<run_id>/`

- `config_resolved.yaml`
- `hardware.json`
- `band_stats.npz`
- `logs/train.log`
- `checkpoints/{last.pt,best.pt}`
- `metrics/history.csv`
- `metrics/eval_test.json` (после `04_eval.py`)
- `pred/<dataset_key>/{extent_prob.tif,boundary_prob.tif,predict_manifest.json}`

## Команды

Из корня репозитория:

```bash
./.venv/bin/python -m pip install -r module_net_train/requirements.txt

./.venv/bin/python module_net_train/scripts/01_check_prep_data.py \
  --config module_net_train/configs/train_config.yaml \
  --out_json output_data/module_net_train/runs/prep_data_summary.json

./.venv/bin/python module_net_train/scripts/02_train.py \
  --config module_net_train/configs/train_config.yaml \
  --hardware module_net_train/configs/hardware_config.yaml

./.venv/bin/python module_net_train/scripts/03_predict_aoi.py \
  --config module_net_train/configs/train_config.yaml \
  --hardware module_net_train/configs/hardware_config.yaml \
  --run_dir output_data/module_net_train/runs/<run_id>

./.venv/bin/python module_net_train/scripts/04_eval.py \
  --config module_net_train/configs/train_config.yaml \
  --hardware module_net_train/configs/hardware_config.yaml \
  --run_dir output_data/module_net_train/runs/<run_id>
```
