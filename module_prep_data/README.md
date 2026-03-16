# module_prep_data

Модуль подготовки данных для train/infer пайплайна.

Документ отражает фактическое состояние на **2026-03-16**.

## 1. Назначение

`module_prep_data` выполняет полный prep цикл:
1. QA и подготовка входов (`01_check_inputs.py`).
2. AOI clipping (`02_clip_to_aoi.py`).
3. Генерация patch-датасета и labels (`03_make_patches.py`).
4. Split в `prep_data/{train,validation,test}` (`04_split_dataset.py`).

Ключевой выход: train-ready структура для `module_net_train`.

## 2. Конфиг

Основной конфиг:
- `module_prep_data/prep_config.yaml`

Важные секции:
- `paths.*`
- `datasets.*`
- `nodata_policy.*`
- `patching.*`
- `labels.*`
- `split.*`
- `export.*`

Текущая NoData policy в конфиге:
- `nodata_value=65536`
- `rule=control-band`
- `control_band_1based=1`

## 3. Архитектура стадий и source of truth

### Stage 1: check_inputs
Скрипт:
- `scripts/01_check_inputs.py`

Что делает:
- проверяет raster/vector,
- приводит вектор в CRS растра,
- сохраняет prepared vector в `module_prep_data_work`.

Ключевые артефакты:
- `output_data/data_check.json`
- `output_data/module_prep_data_work/check_inputs_manifest.json`
- `output_data/module_prep_data_work/<dataset>_vector_prepared.gpkg`

### Stage 2: clip_to_aoi
Скрипт:
- `scripts/02_clip_to_aoi.py`

Что делает:
- создает AOI raster (при `aoi_clip.enabled=true`).

Ключевые артефакты:
- `output_data/module_prep_data_work/aoi_manifest.json`
- `output_data/module_prep_data_work/aoi_rasters/<dataset>_aoi.tif`

### Stage 3: make_patches
Скрипт:
- `scripts/03_make_patches.py`

Что делает:
- генерирует `img`, `extent`, `extent_ig`, `boundary_raw`, `boundary_bwbl`, `valid`, `meta`;
- применяет NoData ignore policy на таргеты.

Ключевые артефакты:
- `output_data/module_prep_data_work/patches_manifest.json`
- `output_data/module_prep_data_work/patches_all/<dataset>/manifest.json`
- `output_data/module_prep_data_work/patches_all/<dataset>/{img,extent,extent_ig,boundary_raw,boundary_bwbl,valid,meta}`

### Stage 4: split_dataset
Скрипт:
- `scripts/04_split_dataset.py`

Что делает:
- читает `patches_manifest.json` как строгий source of truth,
- раскладывает патчи по split по `field_id` (fallback: `feat_index`),
- режим по умолчанию: append-safe.

Ключевые артефакты:
- `prep_data/split_manifest.json`
- `prep_data/{train,validation,test}/{img,extent,extent_ig,boundary_raw,boundary_bwbl,valid,meta}`

Важно:
- в `prep_data/<split>/extent/extent_*.tif` копируется именно `extent_ig` (0/1/255),
  это зафиксировано в `split_manifest.json` (`notes.extent_source=extent_ig (0/1/255)`).

## 4. Data contract

### patches_all
`output_data/module_prep_data_work/patches_all/<dataset>/`:
- `img/img_<patch_id>.tif`
- `extent/extent_<patch_id>.tif`
- `extent_ig/extent_ig_<patch_id>.tif`
- `boundary_raw/boundary_raw_<patch_id>.tif`
- `boundary_bwbl/bwbl_<patch_id>.tif`
- `valid/valid_<patch_id>.tif`
- `meta/meta_<patch_id>.json`

### prep_data (после split)
`prep_data/<split>/`:
- `img/img_<patch_id>.tif`
- `extent/extent_<patch_id>.tif` (из `extent_ig`)
- `boundary_bwbl/bwbl_<patch_id>.tif`
- `valid/valid_<patch_id>.tif`
- `meta/meta_<patch_id>.json`
- `extent_ig/extent_ig_<patch_id>.tif` (если `export.folders.extent_ig` отдельная папка)
- `boundary_raw/boundary_raw_<patch_id>.tif` (для диагностики)

## 5. Boundary semantics: подтвержденный статус

Подтвержденные факты:
1. Root cause проблемы internal boundaries был в label semantics, не в ArcGIS styling.
2. Текущая `boundary_raw` логика строится из polygon linework (`boundary_raw_from_linework`, `all_touched=True`) и сохраняет shared internal boundaries.
3. Extent-gradient используется только как fallback, если linework пуст.
4. Rebuild и post-rebuild validation выполнены, retention внутренних границ подтвержден.

Где смотреть в коде:
- `prep/patching/labels.py`
- тест `tests/test_patching_invariants.py::test_boundary_raw_keeps_shared_internal_boundaries`

Где смотреть forensic summary:
- `/tmp/independent_boundary_patch_validation/independent_validation_summary.json`
- `/tmp/post_rebuild_boundary_validation/post_rebuild_summary.json`

## 6. Текущее зафиксированное состояние артефактов

По текущим manifest/JSON:
- dataset: `first_raster`
- total patches: `947`
- split: `train=757`, `validation=95`, `test=95`
- в `check_inputs` есть warning: `valid_ratio_estimate=0.6794 < 0.7`
- в `patches_all` есть shortfall по negatives (`97` вместо target `150`)

Ключевые файлы:
- `output_data/module_prep_data_work/check_inputs_manifest.json`
- `output_data/module_prep_data_work/aoi_manifest.json`
- `output_data/module_prep_data_work/patches_manifest.json`
- `output_data/module_prep_data_work/patches_all/first_raster/manifest.json`
- `prep_data/split_manifest.json`
- `output_data/data_check.json`

## 7. Запуск

Из корня репозитория:

```bash
# По стадиям
./.venv/bin/python module_prep_data/scripts/01_check_inputs.py --config module_prep_data/prep_config.yaml
./.venv/bin/python module_prep_data/scripts/02_clip_to_aoi.py --config module_prep_data/prep_config.yaml
./.venv/bin/python module_prep_data/scripts/03_make_patches.py --config module_prep_data/prep_config.yaml
./.venv/bin/python module_prep_data/scripts/04_split_dataset.py --config module_prep_data/prep_config.yaml

# Быстрый smoke-check по значениям масок
./.venv/bin/python module_prep_data/scripts/smoke_check_patches.py \
  --patches_all output_data/module_prep_data_work/patches_all --k 12

# Оркестратор всего цикла
bash module_prep_data/scripts/run_prep_all.sh --config module_prep_data/prep_config.yaml
```

Опции:
- `03_make_patches.py`: `--n`, `--seed`
- `04_split_dataset.py`: `--patches_manifest`, `--seed`, `--overwrite`
- `run_prep_all.sh`: `--n`, `--seed`, `--overwrite`, `--pytest`

## 8. Progress UX в терминале

- Для длинных операций добавлены progress-бары:
  - проход по datasets (`check_inputs`, `clip_to_aoi`, `make_patches`);
  - генерация patches (`center/boundary/negative`);
  - сбор manifest и копирование в split.
- Авто-режим: бары показываются в interactive terminal (TTY), в non-interactive выводе автоматически отключаются.
- Управление через env:
  - `DISABLE_PROGRESS=1` — отключить progress;
  - `FORCE_PROGRESS=1` — включить принудительно.

## 9. Тесты

Основные:
- `tests/test_patching_invariants.py`
- `tests/test_prep_pipeline.py`
- `tests/test_cli_smoke.py`

Запуск:

```bash
./.venv/bin/python -m pytest module_prep_data/tests -q
```

## 10. Ограничения и caveats

1. В текущих manifest отмечены deferred/неактивные части конфига (например, `patching.sampling.near_nodata.*`, часть `raster_preprocess.*`).
2. `split.unit` реализован только для `by_field`.
3. `patches_all` может не добрать target negatives из-за фильтров (`neg_dist`, `neg_mask`, `valid`).
4. Forensic summaries в `/tmp` не являются постоянным хранилищем и могут быть очищены.
