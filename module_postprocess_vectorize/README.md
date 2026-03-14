# module_postprocess_vectorize

Автономный модуль постобработки предсказаний сегментации полей:

`probability rasters -> smoothing -> threshold/barriers -> seeds -> watershed -> labels -> polygonize -> geometry cleanup -> eval/search`

Цель: получить GIS-friendly полигоны полей с приоритетом на корректное разделение соседних объектов и отсутствие дыр/артефактов.

## Структура

```text
module_postprocess_vectorize/
  configs/
    postprocess_config.yaml
  postprocess/
    __init__.py
    io.py
    raster_ops.py
    seeds.py
    separation.py
    vectorize.py
    geometry_clean.py
    metrics.py
    search.py
    pipeline.py
  scripts/
    01_search_postprocess_params.py
    02_postprocess_single.py
    03_postprocess_run.py
    04_eval_polygons.py
  README.md
```

## Входы

Основной режим:
- `extent_prob.tif` (float32, probability поля)
- `boundary_prob.tif` (float32, probability границы)
- опционально `valid_mask.tif` (1=valid, 0=invalid)
- опционально `footprint` (если нужно ограничить область)

Контроль входов:
- одинаковые `width/height`
- одинаковые `transform`
- одинаковые `CRS`
- проверка/нормализация probability в диапазон `[0, 1]`

Интеграция с `module_net_train`:
- для batch-режима ожидается структура `output_data/module_net_train/runs/<timestamp_run>/pred/<dataset_key>/...`;
- если `valid_mask.tif` отсутствует, модуль пытается взять footprint из `predict_manifest.json` (`aoi_raster`) и не выпускать полигоны за AOI.

## Выходы

При `save_intermediates=true`:
- `extent_smooth.tif`
- `boundary_smooth.tif`
- `field_mask.tif`
- `boundary_barrier.tif`
- `seeds.tif`
- `labels.tif`

Векторы:
- `fields_pred_raw.gpkg`
- `fields_pred.gpkg`
- опционально `fields_pred.shp`

Служебные артефакты:
- `params_used.json`
- `metrics_postproc.json` (если передан GT)
- `search_results.json` + `best_params.yaml` (для search)

## Алгоритм пайплайна

1. Валидация и загрузка co-registered входных raster.
2. Маскирование по `valid_mask` (вне valid всегда фон).
3. Лёгкое Gaussian сглаживание вероятностей.
4. `field_mask` из extent threshold + морфологическая очистка.
5. `boundary_barrier` из boundary threshold + optional dilation.
6. Seeds/markers: distance transform + local maxima/h-maxima.
7. Разделение полей: marker-based watershed внутри `field_mask` с учётом boundary-барьера.
   Для больших AOI автоматически включается memory-safe fallback (watershed выключается по порогу пикселей).
8. Очистка `labels`: fill small holes, merge small regions, drop tiny leftovers, relabel.
9. Векторизация labels в CRS входа.
10. Геометрическая очистка: `make_valid`, remove holes, min area в m², simplify в метрах, optional straighten, clip к valid area.

## Ключевые параметры

В `configs/postprocess_config.yaml` вынесены:
- `extent_thr`
- `boundary_thr`
- `gaussian_sigma_px`
- `boundary_dilate_px`
- `min_area_m2`
- `fill_holes_max_area_m2`
- `small_region_max_area_m2`
- `simplify_m`
- `seed_min_distance_px`
- `seed_hmax`
- `marker_erode_px`
- `use_watershed`
- `sobel_weight`
- `remove_holes`
- `straighten.enabled`
- `straighten.snap_angle_deg`
- `save_intermediates`
- `export_shp`
- `clip_to_valid`
- `scoring.metric_name`
- `memory.prob_dtype`
- `memory.auto_disable_watershed`
- `memory.max_pixels_for_watershed`
- `memory.warn_pixels_threshold`
- `memory.auto_disable_gaussian_large`
- `memory.max_pixels_for_gaussian`
- `memory.sobel_weight_large`

## CLI

### 1) Search параметров на validation

```bash
./.venv/bin/python module_postprocess_vectorize/scripts/01_search_postprocess_params.py \
  --pred_root output_data/module_net_train/runs/20260312_131232/pred \
  --gt_root <path_to_gt_vectors_or_gt_label_rasters> \
  --config module_postprocess_vectorize/configs/postprocess_config.yaml \
  --output_dir output_data/module_postprocess_vectorize/search/<search_id>
```

Результат:
- `best_params.yaml`
- `search_results.json`

### 2) Постобработка одной пары raster

```bash
./.venv/bin/python module_postprocess_vectorize/scripts/02_postprocess_single.py \
  --extent_prob <.../extent_prob.tif> \
  --boundary_prob <.../boundary_prob.tif> \
  --valid_mask <.../valid_mask.tif> \
  --config module_postprocess_vectorize/configs/postprocess_config.yaml \
  --params_override output_data/module_postprocess_vectorize/search/<search_id>/best_params.yaml \
  --output_dir output_data/module_postprocess_vectorize/single/<sample_id>
```

### 3) Пакетная постобработка run_dir

```bash
./.venv/bin/python module_postprocess_vectorize/scripts/03_postprocess_run.py \
  --run_dir output_data/module_net_train/runs/20260312_131232 \
  --config module_postprocess_vectorize/configs/postprocess_config.yaml \
  --params_override output_data/module_postprocess_vectorize/search/<search_id>/best_params.yaml
```

По умолчанию:
- вход: `<run_dir>/pred/**/extent_prob.tif`, `boundary_prob.tif`
- выход: `<run_dir>/postprocess/<sample_id>/...`

Memory-safe поведение:
- для крупных raster модуль автоматически снижает память (`prob_dtype=float16`);
- при превышении `memory.max_pixels_for_gaussian` отключает Gaussian blur;
- при превышении `memory.max_pixels_for_watershed` переключает разделение в safer-mode без watershed;
- это предотвращает OOM на больших AOI ценой более консервативного instance split.

### 5) Один shell-скрипт для запуска модуля целиком

```bash
./module_postprocess_vectorize/scripts/run_postprocess_all.sh \
  --run_dir output_data/module_net_train/runs/20260312_131232 \
  --gt_root <path_to_gt_vectors_or_gt_rasters> \
  --gt_mode vector
```

Поведение:
- `--run_dir` можно не указывать: скрипт сам выберет последний `runs/*`;
- если в выбранном run нет `extent_prob.tif/boundary_prob.tif`, скрипт автоматически вызовет `module_net_train/scripts/03_predict_aoi.py`;
- если передан `--gt_root` и нет `--params_override`, сначала запускается `01_search_postprocess_params.py`;
- затем запускается `03_postprocess_run.py` на весь `run_dir`;
- если `--params_override` передан, search-этап пропускается.

### 4) Оценка полигонов

```bash
./.venv/bin/python module_postprocess_vectorize/scripts/04_eval_polygons.py \
  --gt <gt_vector_or_gt_label_raster> \
  --pred <pred_vector_or_pred_label_raster> \
  --iou_threshold 0.5 \
  --out_json output_data/module_postprocess_vectorize/eval.json
```

Метрики:
- `gt_count`, `pred_count`
- `tp`, `fp`, `fn`
- `precision`, `recall`, `f1`
- `mean_iou_matched`
- `merge_penalty`, `holes_penalty`, `invalid_geometries`
- `area_precision`, `area_recall`, `area_f1`

## Рекомендуемый workflow

1. `01_search_postprocess_params.py` на validation split.
2. Проверить `search_results.json`, выбрать `best_params.yaml`.
3. Прогнать `03_postprocess_run.py` на нужном `run_dir`.
4. Визуально проверить intermediates (`field_mask`, `boundary_barrier`, `seeds`, `labels`).
5. Использовать `fields_pred.gpkg` (и optional `.shp`) для GIS/дальнейшей аналитики.
