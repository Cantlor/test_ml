# module_postprocess_vectorize

Модуль постобработки:
`extent_prob/boundary_prob -> labels -> polygons`.

Описание синхронизировано с фактическими артефактами на **2026-03-16**.

## 1. Что делает модуль

Пайплайн (`postprocess/pipeline.py`):
1. Читает probability raster.
2. Восстанавливает/читает valid-mask контекст.
3. Сглаживает вероятности.
4. Строит `field_mask` и boundary barrier.
5. Строит seeds.
6. Выполняет watershed (если не отключен runtime policy).
7. Чистит labels.
8. Векторизует и очищает геометрию.
9. Пишет manifest + параметры + артефакты.

## 2. Входы

Обязательные:
- `extent_prob.tif`
- `boundary_prob.tif`

Опциональные:
- `valid_mask.tif`
- `predict_manifest.json`
- AOI raster (`aoi_raster` через predict manifest)

Если `valid_mask.tif` отсутствует:
- модуль строит valid через `footprint_nodata` (AOI + nodata policy из `config_used`),
- fallback использует nodata метаданные растра.

## 3. Выходы

Основные:
- `fields_pred_raw.gpkg`
- `fields_pred.gpkg`
- `postprocess_manifest.json`
- `params_used.json`

При `save_intermediates=true`:
- `extent_smooth.tif`
- `boundary_smooth.tif`
- `field_mask.tif`
- `boundary_barrier.tif`
- `seeds.tif`
- `labels.tif`

Для batch-запуска:
- `postprocess_run_summary.json`

## 4. Конфиг и runtime policy

Базовый конфиг:
- `configs/postprocess_config.yaml`

Критично:
- фактическое поведение определяется не только config, но и RAM-aware runtime policy (`postprocess/runtime.py`).
- при высокой нагрузке модуль может уменьшать/выключать gaussian, отключать watershed и переключать clean_labels в fast mode.

## 5. Фактическое состояние baseline run

Run:
- `output_data/module_net_train/runs/20260316_103300`

Ключевые postprocess артефакты:
- `postprocess/postprocess_run_summary.json`
- `postprocess/first_raster/postprocess_manifest.json`
- `postprocess/first_raster/params_used.json`

Факты из manifest:
- `valid_source=footprint_nodata`
- `estimated_pressure=2.7987` (существенно выше бюджета)
- `gaussian_sigma_px_effective=0.0`
- `use_watershed=false`
- `clean_labels_mode=fast`
- предупреждения содержат `estimated_peak_exceeds_ram_budget`

Это важно при интерпретации качества: результат зависит не только от `postprocess_config.yaml`, но и от runtime деградации на большой сцене.

## 6. Команды

Из корня репозитория:

```bash
# Single sample (через predict_manifest)
./.venv/bin/python module_postprocess_vectorize/scripts/02_postprocess_single.py \
  --predict_manifest output_data/module_net_train/runs/20260316_103300/pred/first_raster/predict_manifest.json \
  --config module_postprocess_vectorize/configs/postprocess_config.yaml \
  --output_dir output_data/module_postprocess_vectorize/single/first_raster

# Batch по run_dir
./.venv/bin/python module_postprocess_vectorize/scripts/03_postprocess_run.py \
  --run_dir output_data/module_net_train/runs/20260316_103300 \
  --config module_postprocess_vectorize/configs/postprocess_config.yaml

# Polygon eval (если есть GT)
./.venv/bin/python module_postprocess_vectorize/scripts/04_eval_polygons.py \
  --gt <gt_vector_or_gt_raster> \
  --pred output_data/module_net_train/runs/20260316_103300/postprocess/first_raster/fields_pred.gpkg \
  --out_json output_data/module_postprocess_vectorize/eval_20260316.json
```

## 7. Progress UX в терминале

- Добавлен прогресс для:
  - batch postprocess по sample-ам;
  - шагов pipeline в `run_postprocess_pipeline`;
  - grid-search trials;
  - polygon loading/eval loops на больших наборах.
- Для nested loops часть внутренних баров намеренно отключена, чтобы не зашумлять терминал.
- Управление через env:
  - `DISABLE_PROGRESS=1`
  - `FORCE_PROGRESS=1`

## 8. Что смотреть в первую очередь

1. `postprocess/postprocess_run_summary.json`
2. `postprocess/first_raster/postprocess_manifest.json`
3. `postprocess/first_raster/params_used.json`
4. `postprocess/first_raster/labels.tif`
5. `postprocess/first_raster/fields_pred.gpkg`

## 9. Ограничения

1. Даже после исправления labels upstream, качество полигонов ограничено качеством model probabilities.
2. Для текущей сцены runtime policy отключает watershed из-за memory pressure.
3. Есть observability limit: часть GT-границ может быть неявно наблюдаема в imagery, поэтому GT-like разделение не всегда достижимо только постобработкой.
