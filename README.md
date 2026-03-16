# uzcosmos `my_project`

Документ отражает фактическое состояние репозитория на **2026-03-16**.

## 1. Текущий статус проекта

Проект состоит из 3 рабочих модулей:

1. `module_prep_data` — подготовка AOI/patches/labels/split.
2. `module_net_train` — train + infer + eval multitask U-Net.
3. `module_postprocess_vectorize` — постобработка probability raster в полигоны.

Текущее рабочее состояние:
- `prep_data` пересобран и валидирован (947 patch).
- train/infer/eval выполнены в `output_data/module_net_train/runs/20260316_103300`.
- postprocess выполнен для `first_raster` и сохранил manifest/runtime диагностику.

Terminal UX update:
- Во всех 3 модулях добавлены progress-бары для длительных стадий (datasets/patches/batches/tiles/samples/object-loops).
- Единый helper: `project_progress.py`.
- Принудительное отключение: `DISABLE_PROGRESS=1`.
- Принудительное включение: `FORCE_PROGRESS=1`.

## 2. Архитектура пайплайна

```text
initial_data/<dataset>/
  clipped.tif + sardoba.shp
        |
        v
module_prep_data/scripts/
  01_check_inputs.py
  02_clip_to_aoi.py
  03_make_patches.py
  04_split_dataset.py
        |
        v
prep_data/{train,validation,test}/
  img, valid, extent, boundary_bwbl, meta
        |
        v
module_net_train/scripts/
  01_check_prep_data.py
  02_train.py
  03_predict_aoi.py
  04_eval.py
        |
        v
output_data/module_net_train/runs/<run_id>/
  metrics/, pred/, postprocess/, checkpoints/
        |
        v
module_postprocess_vectorize/scripts/
  02_postprocess_single.py
  03_postprocess_run.py
  04_eval_polygons.py
```

## 3. Подтвержденные forensic-факты (accepted technical truths)

1. **Root cause internal boundaries был в label semantics, не в ArcGIS styling.**
   - Подтверждено кодом `module_prep_data/prep/patching/labels.py` и тестом `test_boundary_raw_keeps_shared_internal_boundaries`.
2. **Prepared vector faithful, старая boundary-логика была not faithful к внутренним границам.**
   - Сравнение old/new есть в `/tmp/independent_boundary_patch_validation/independent_validation_summary.json`.
3. **Rebuild prep_data выполнен, post-rebuild validation есть.**
   - `/tmp/post_rebuild_boundary_validation/post_rebuild_summary.json`.
   - Контракты labels подтверждены: `extent ⊂ {0,1,255}`, `bwbl ⊂ {0,1,2}`, `valid ⊂ {0,1}`.
4. **Сильное улучшение retention внутренних границ после фикса labels.**
   - `braw_inner weighted_old=0.0200 -> weighted_new=0.9194`.
   - `bw_any_inner weighted_old=0.1367 -> weighted_new=0.9999`.
5. **Новый train-run после фикса labels актуален и присутствует.**
   - Текущий run: `output_data/module_net_train/runs/20260316_103300`.

## 4. Source of truth (артефакты)

### Prep
- `output_data/module_prep_data_work/check_inputs_manifest.json`
- `output_data/module_prep_data_work/aoi_manifest.json`
- `output_data/module_prep_data_work/patches_manifest.json`
- `output_data/module_prep_data_work/patches_all/first_raster/manifest.json`
- `prep_data/split_manifest.json`
- `output_data/data_check.json`

Ключевые факты:
- `total_patches=947`, split: `train=757`, `validation=95`, `test=95`.
- valid_ratio warning на входном raster: `0.6794 < 0.7`.

### Train / Infer / Eval
- `output_data/module_net_train/runs/20260316_103300/config_resolved.yaml`
- `output_data/module_net_train/runs/20260316_103300/hardware.json`
- `output_data/module_net_train/runs/20260316_103300/metrics/history.csv`
- `output_data/module_net_train/runs/20260316_103300/metrics/eval_test.json`
- `output_data/module_net_train/runs/20260316_103300/pred/first_raster/predict_manifest.json`

Ключевые факты:
- best epoch по `val/boundary_f1_max`: epoch 38, `0.475046` (`thr=0.70`).
- test eval (`best.pt`): `extent_iou=0.9448`, `extent_f1=0.9716`, `boundary_f1_max=0.4755`.
- near-invalid KPI остается слабым: `extent_f1_near_invalid=0.0`, `boundary_f1_max_near_invalid≈0.109`.

### Postprocess
- `output_data/module_net_train/runs/20260316_103300/postprocess/postprocess_run_summary.json`
- `output_data/module_net_train/runs/20260316_103300/postprocess/first_raster/postprocess_manifest.json`
- `output_data/module_net_train/runs/20260316_103300/postprocess/first_raster/params_used.json`

Ключевые факты:
- valid mask восстановлен как `valid_source=footprint_nodata`.
- на текущем AOI runtime pressure высокий (`estimated_pressure≈2.80`), поэтому:
  - `gaussian_sigma_px_effective=0.0`
  - `use_watershed=false`
  - `clean_labels_mode=fast`

### Forensic/validation во временных папках
- `/tmp/independent_boundary_patch_validation/independent_validation_summary.json`
- `/tmp/independent_boundary_patch_validation/independent_validation_per_patch.json`
- `/tmp/post_rebuild_boundary_validation/post_rebuild_summary.json`
- `/tmp/post_rebuild_boundary_validation/post_rebuild_per_patch.json`
- `/tmp/pre_rebuild_quick_stats_1773656501382623004.json`
- `/tmp/prep_rebuild_snapshot_20260316_152141/*`

## 5. Data contract между модулями

`prep_data/<split>/`:

```text
img/img_<patch_id>.tif
valid/valid_<patch_id>.tif           # 0/1
extent/extent_<patch_id>.tif         # 0/1/255
boundary_bwbl/bwbl_<patch_id>.tif    # 0/1/2
meta/meta_<patch_id>.json
```

Фактический train контракт:
- вход модели: `num_bands=8` + `valid` канал (итого `in_channels=9`).
- boundary head учится по `bwbl`, ignore=`2`.
- extent head учится по `extent`, ignore=`255`.

## 6. Принятый baseline (текущее)

### Prep baseline
- `module_prep_data/prep_config.yaml`.
- Labels после фикса semantics для boundary linework.
- split и manifests зафиксированы в `prep_data/split_manifest.json`.

### Train baseline
- `module_net_train/configs/train_config.yaml` + `configs/hardware_config.yaml`.
- run baseline: `runs/20260316_103300`.
- мониторинг best checkpoint: `val/boundary_f1_max`.

### Inference baseline
- tiled predict: `window=512`, `stride=384`, `blend=gaussian`.
- для текущего run invalid-edge guard в predict выключен (`invalid_edge_guard_px=0`).

### Postprocess baseline
- `module_postprocess_vectorize/configs/postprocess_config.yaml`.
- фактический runtime определяется не только config, но и RAM-aware деградацией в manifest (`memory_runtime`).

## 7. Быстрый запуск ключевых стадий

Из корня репозитория:

```bash
# PREP
./.venv/bin/python module_prep_data/scripts/01_check_inputs.py --config module_prep_data/prep_config.yaml
./.venv/bin/python module_prep_data/scripts/02_clip_to_aoi.py --config module_prep_data/prep_config.yaml
./.venv/bin/python module_prep_data/scripts/03_make_patches.py --config module_prep_data/prep_config.yaml
./.venv/bin/python module_prep_data/scripts/04_split_dataset.py --config module_prep_data/prep_config.yaml

# TRAIN / INFER / EVAL
./.venv/bin/python module_net_train/scripts/02_train.py \
  --config module_net_train/configs/train_config.yaml \
  --hardware module_net_train/configs/hardware_config.yaml

./.venv/bin/python module_net_train/scripts/04_eval.py \
  --config module_net_train/configs/train_config.yaml \
  --hardware module_net_train/configs/hardware_config.yaml \
  --run_dir output_data/module_net_train/runs/20260316_103300

# POSTPROCESS
./.venv/bin/python module_postprocess_vectorize/scripts/03_postprocess_run.py \
  --run_dir output_data/module_net_train/runs/20260316_103300 \
  --config module_postprocess_vectorize/configs/postprocess_config.yaml
```

## 8. Ограничения и риски (текущие)

1. После фикса labels качество boundary улучшено, но итоговый bottleneck теперь не только labels.
2. Есть **observability mismatch**: не все GT-границы обязательно визуально читаются в текущем imagery.
3. Near-invalid зона остается слабой по KPI (даже при хорошем global extent F1).
4. Постобработка на большой сцене может автоматически отключать watershed/gaussian из-за RAM pressure.
5. Исторических run-baseline в `output_data/module_net_train/runs` сейчас нет (только `20260316_103300`).

## 9. Что не нужно начинать заново

- Повторное доказательство, что проблема internal boundaries была только “визуализацией в ArcGIS”.
- Повторный forensic root-cause по старой `boundary_raw` логике.
- Ручной пересчет уже сохраненных post-rebuild retention-отчетов, если не менялись labels/patching.

## 10. Что может потребоваться для GT-like quality (гипотезы/рекомендации)

Это **не подтвержденные факты**, а рабочие направления:
- более информативные данные (quality/seasonality/доп. каналы);
- observability-aware label policy;
- priors из внешнего вектора (shape prior) в inference/postprocess;
- multi-temporal данные;
- более сильная специализация модели под near-invalid и boundary ambiguity.
