# uzcosmos `my_project`

Репозиторий состоит из двух основных модулей:

1. `module_prep_data` — подготовка данных (AOI-клиппинг, QA, генерация патчей, split).
2. `module_net_train` — обучение multitask U-Net, инференс по AOI, оценка на test.

Ниже — подробный технический разбор того, что делает каждый блок, как связаны этапы пайплайна и какие параметры менять, чтобы влиять на результат обучения нейросети (без изменения исходных входных данных).

---

## 1) End-to-End пайплайн

```text
initial_data/
  └── <dataset>/{raster.tif, vector.shp/gpkg}
        |
        v
module_prep_data
  01_check_inputs.py
  02_clip_to_aoi.py
  03_make_patches.py
  04_split_dataset.py
        |
        v
prep_data/{train,validation,test}/
  img/, extent/, boundary_bwbl/, valid/, meta/
        |
        v
module_net_train
  01_check_prep_data.py
  02_train.py
  03_predict_aoi.py
  04_eval.py
        |
        v
output_data/module_net_train/runs/<run_id>/
  checkpoints/, metrics/, pred/, logs/, band_stats.npz
```

---

## 2) Структура репозитория

Ключевые директории:

- `initial_data/` — исходные геоданные.
- `module_prep_data/` — модуль препроцессинга.
- `prep_data/` — готовые train/val/test патчи (вход для обучения).
- `module_net_train/` — модуль обучения/инференса.
- `output_data/` — артефакты этапов (рабочие промежуточные данные, run-логи, чекпоинты, предсказания).

Ключевые конфиги:

- `module_prep_data/prep_config.yaml`
- `module_net_train/configs/train_config.yaml`
- `module_net_train/configs/hardware_config.yaml`

---

## 3) Разбор `module_prep_data` (что делает каждый блок)

### 3.1 Скрипты `module_prep_data/scripts`

1. `01_check_inputs.py`
- Загружает `prep_config.yaml`.
- Проверяет доступность растра/вектора по glob-шаблонам.
- Валидирует базовые параметры растра (CRS, dtype, band_count, оценка valid_ratio).
- Приводит вектор к CRS растра, чистит геометрии в памяти, фильтрует по площади.
- Пишет отчёт `output_data/data_check.json`.
- Сохраняет подготовленный вектор `output_data/module_prep_data_work/<dataset>_vector_prepared.gpkg`.

2. `02_clip_to_aoi.py`
- Опционально клиппит исходный raster по AOI из вектора (`bbox` или `mask` режим).
- Пишет AOI-растры в `output_data/module_prep_data_work/aoi_rasters/`.
- Создаёт манифесты AOI для downstream этапов.

3. `03_make_patches.py`
- Берёт AOI raster (если есть) + prepared vector.
- Генерирует патчи `img`, `extent`, `extent_ig`, `boundary_raw`, `boundary_bwbl`, `valid`, `meta`.
- Применяет политику NoData -> ignore на таргетах.
- Пишет всё в `output_data/module_prep_data_work/patches_all/<dataset>/...`.

4. `04_split_dataset.py`
- Читает `patches_all/*/manifest.json`.
- Делит записи на `train/validation/test` с группировкой по `field_id` (fallback на `feat_index`).
- Работает в append-safe режиме: уже существующие записи в `prep_data` не перераспределяются, новые патчи добавляются.
- Копирует/линкует файлы в `prep_data/{train,validation,test}/...`.
- Источник `extent` для обучения берётся из `extent_ig` (0/1/255).

5. `smoke_check_patches.py`
- Быстрые проверки значений масок:
  - `valid` in {0,1}
  - `extent_ig` in {0,1,255}
  - `bwbl` in {0,1,2}
- Проверка политики: при `valid=0` ожидается `extent_ig=255` и `bwbl=2`.

### 3.2 Пакет `module_prep_data/prep`

1. `config.py`
- Typed-конфиги (dataclasses) для всех разделов `prep_config.yaml`.
- Резолв относительных путей.

2. `utils.py`
- Вспомогательные функции: поиск файлов по glob, запись/чтение JSON, утилиты CRS.

3. `qa_raster.py`
- Чтение метаданных растра.
- Оценка valid_ratio по NoData policy (`control-band`/`all-bands`) через sampling окон.

4. `qa_vector.py`
- Чтение и проверка вектора.
- Приведение к CRS растра.
- Фильтрация non-polygon/empty/small-area.
- Опциональные in-memory операции с invalid геометриями.

5. `clip_raster.py`
- Клиппинг растров по геометриям полей.
- Поддержка `bbox` и `mask`.
- Опциональный `mask_outside`.

6. `patches.py` (центральный блок)
- Сэмплинг центров патчей: `center`, `boundary`, `negative`.
- Растеризация extent и построение boundary карт.
- Формирование `BWBL` (фон/скелет/буфер).
- Формирование valid-mask на основе NoData policy.
- Применение ignore policy на extent/boundary в nodata зоне.
- Экспорт GeoTIFF + per-patch meta JSON + manifest.

---

## 4) Разбор `module_net_train` (что делает каждый блок)

### 4.1 Скрипты `module_net_train/scripts`

1. `01_check_prep_data.py`
- Индексирует `prep_data` по split.
- Проверяет наличие обязательных файлов.
- Проверяет число каналов `img`.
- Проверяет допустимые значения масок.
- Пишет summary JSON.

2. `02_train.py`
- Загружает train/hardware конфиги.
- Строит runtime-план (device, precision, crop, batch, workers).
- Индексирует train/val.
- Считает и сохраняет статистики нормализации (`band_stats.npz`).
- Создаёт датасеты и dataloaders.
- Собирает модель, optimizer, scheduler.
- Запускает train/val loop, пишет history CSV и чекпоинты (`last.pt`, `best.pt`).
- По настройке делает инференс AOI в конце.

3. `03_predict_aoi.py`
- Загружает run (`band_stats.npz`, checkpoint).
- Резолвит AOI raster из manifest по `dataset_key`.
- Делает tiled inference, пишет:
  - `extent_prob.tif`
  - `boundary_prob.tif`
  - `predict_manifest.json`

4. `04_eval.py`
- Загружает test split и checkpoint.
- Считает метрики через `validate_one_epoch`.
- Пишет `metrics/eval_test.json`.

### 4.2 `net_train/*` по подсистемам

1. `config.py`
- Загрузка YAML и резолв путей.
- Вспомогательные функции извлечения nested-полей.

2. `hardware.py`
- Построение runtime-плана:
  - `device` (`cpu/cuda`)
  - `precision` (`fp32/fp16/bf16`)
  - heuristic autotune для crop/batch/grad_accum
  - dataloader workers
- Применение torch runtime flags (`cudnn`, `tf32`).

3. `data/index.py`
- Формирует индекс samples по `meta_*.json` и ожидаемым путям масок/изображений.

4. `data/stats.py`
- Вычисляет нормализацию (`robust_percentile` или `mean_std`) по train или выбранному split.
- Сохраняет/загружает stats в `.npz`.

5. `data/dataset.py`
- Читает `img`, `extent`, `bwbl`, `valid`.
- Делает crop/augment.
- Нормализует спектральные каналы.
- Добавляет `valid` как дополнительный канал (если включено).
- Принудительно проставляет ignore значения в таргетах там, где `valid=0`.

6. `data/transforms.py`
- Простые геометрические аугментации: hflip/vflip/rotate90.

7. `models/unet_multitask.py`
- U-Net encoder-decoder.
- Две головы: `extent_logits` и `boundary_logits`.

8. `losses/extent_loss.py`
- `BCEWithLogits + soft Dice` для extent c ignore-mask.

9. `losses/bwbl_loss.py`
- `BCEWithLogits` для boundary (класс ignore исключается по маске).

10. `metrics/extent_metrics.py`
- IoU/F1/Precision/Recall по extent.

11. `metrics/boundary_metrics.py`
- Boundary F1 с допуском (`dilation_px`) и мульти-threshold.

12. `train/loop.py`
- Train step с AMP, grad accumulation, grad clipping.
- Validation (loss + метрики).
- Сохранение history CSV и checkpoint management.

13. `infer/tiling.py` и `infer/predict_aoi.py`
- Генерация окон, blending overlap, сбор full-size probability raster.

14. `utils/*`
- Логирование, I/O, сиды.

---

## 5) Входной/выходной контракт данных

`prep_data/<split>/` должен содержать:

```text
img/img_<patch_id>.tif
valid/valid_<patch_id>.tif
extent/extent_<patch_id>.tif
boundary_bwbl/bwbl_<patch_id>.tif
meta/meta_<patch_id>.json
```

Семантика масок:

- `valid`: 0/1
- `extent`: 0/1/255 (`255` = ignore)
- `boundary_bwbl`: 0/1/2 (`2` = ignore/buffer)

---

## 6) Какие параметры менять, чтобы влиять на результат обучения

Ниже параметры, которые реально дают наибольший эффект на качество/стабильность, без изменения самих исходных входных данных.

### 6.1 Параметры с самым сильным влиянием (приоритет 1)

1. `module_net_train/configs/train_config.yaml -> train.optimizer.lr`
- Самый чувствительный параметр.
- Слишком высокий `lr` -> нестабильность/прыгающий loss.
- Слишком низкий -> медленное обучение и недообучение.
- Практика: пробовать сетку `3e-4`, `1e-3`, `2e-3`.

2. `train.epochs`
- Больше эпох обычно повышает качество до plateau.
- Контролировать по `val/extent_iou` и `val/boundary_f1@...`.

3. `loss.weights.extent` и `loss.weights.boundary`
- Баланс двух задач multitask.
- Если границы важнее: увеличить `boundary` (например 3.0 -> 4.0/5.0).
- Если переобучается на boundary и теряет extent: снизить `boundary`.

4. `model.base_channels`, `model.depth`
- Ёмкость модели.
- Увеличение улучшает потенциал качества, но растит VRAM и риск переобучения.

5. `sampling.crop_size` (через runtime plan) и `inference.tiling.window_size/stride`
- Влияет на контекст, стабильность границ, VRAM.
- Малый crop лучше для памяти, но теряет контекст.

### 6.2 Сильное влияние (приоритет 2)

1. `normalization.type`, `normalization.p_low/p_high`, `normalization.ignore_nodata`
- Критично для мультиспектральных данных и разных сезонов.
- `robust_percentile` обычно стабильнее на выбросах.

2. `augmentations.{hflip,vflip,rotate90}`
- Улучшает обобщение, особенно на малом датасете.
- Отключать только для диагностики.

3. `loss.extent.{bce_weight,dice_weight}`
- Баланс "локальная точность vs overlap".
- Больше Dice помогает на разреженных масках.

4. `scheduler.{warmup_epochs,min_lr}`
- Влияет на стартовую стабильность и финальную донастройку.

5. `train.grad_clip_norm`
- Стабилизирует обучение при редких всплесках градиента.

### 6.3 Влияние через подготовку таргетов/сэмплинга (приоритет 3)

Это не изменение raw input-данных, но изменение supervision сигнала:

1. `module_prep_data/prep_config.yaml -> patching.negatives.ratio`
- Меняет баланс positive/negative патчей.

2. `patching.filters.{min_valid_ratio,min_mask_ratio,max_mask_ratio,neg_max_mask_ratio}`
- Сильно влияет на "сложность" обучающей выборки.

3. `labels.bwbl.buffer_px`
- Толщина boundary buffer-класса, влияет на boundary head.

4. `labels.ignore_zone.ignore_radius_px`
- Размер ignore-зоны вокруг границы для extent.

5. `nodata_policy` + `labels.nodata_ignore_policy`
- Как учитываются NoData зоны при loss/метриках.

---

## 7) Параметры в конфиге, которые сейчас почти не влияют (или не подключены)

По текущей реализации есть параметры, которые декларированы, но фактически не участвуют в вычислениях или участвуют не полностью:

1. `module_prep_data/prep_config.yaml`
- `patching.sampling.mode`
- `patching.sampling.samples_per_feature`
- `patching.sampling.near_nodata.*`
- `raster_preprocess.*` (convert_dtype/nodata_to_value/compute_band_stats)
- `qa.raster.require_geotransform` (флаг есть, явной проверки нет)

2. `module_net_train/configs/hardware_config.yaml`
- `gpu.max_mem_frac` (не применяется в runtime коде)
- `autotune.test_steps` (runtime probe не реализован, используется heuristic)

---

## 8) Что исправлено после ревью

Внесены следующие изменения в код:

1. `module_net_train/scripts/03_predict_aoi.py` и `module_net_train/scripts/04_eval.py`
- Читают параметры только из модульного конфига `module_net_train/configs/train_config.yaml` (через `--config`).
- Не зависят от временных YAML-файлов внутри `output_data`.

2. `module_prep_data/prep/patches.py` + `module_prep_data/scripts/03_make_patches.py`
- В patch metadata добавлен стабильный `field_id`.
- Источник `field_id`: сначала `vector.id_field` из конфига (если задан и существует), иначе `orig_fid` (если есть), иначе fallback на индекс.

3. `module_prep_data/scripts/04_split_dataset.py`
- Grouping для split теперь использует `field_id` (а не только `feat_index`).
- Это снижает риск утечки частей одного и того же поля между `train/validation/test`.
- Режим по умолчанию: append (новые патчи добавляются к уже существующим в `prep_data`).

4. `module_prep_data/prep/patches.py`
- Для positive-патчей (`center` и `boundary`) добавлен retry-добор до target-квот (с ограничением max attempts).
- В `manifest.summary` добавлены `attempts` и `shortfall`.

5. `module_net_train/net_train/train/loop.py`
- `val/boundary_f1@...` теперь считается из глобально агрегированных `TP/FP/FN` по эпохе, а не усреднением batch-level F1.

6. `module_net_train/net_train/infer/predict_aoi.py`
- Добавлена валидация tiled-inference параметров:
  - `window_size > 0`
  - `stride > 0`
  - `stride <= window_size`
  - `batch_size > 0`

### 8.1 Что остаётся в зоне внимания

1. В `module_prep_data` часть параметров конфига всё ещё объявлена, но не подключена в логику (`near_nodata`, `samples_per_feature`, часть `raster_preprocess`).
2. В `module_net_train` всё ещё нет отдельного набора unit/integration тестов.

---

## 9) Минимальный порядок запуска (справка)

Из корня репозитория:

```bash
# 1) Подготовка данных
bash module_prep_data/scripts/run_prep_all.sh
# По умолчанию: append в prep_data (не пересоздаёт старые сплиты)
# Полный пересбор сплитов:
# bash module_prep_data/scripts/run_prep_all.sh --overwrite

# 2) Проверка prep_data
python module_net_train/scripts/01_check_prep_data.py \
  --config module_net_train/configs/train_config.yaml

# 3) Обучение
python module_net_train/scripts/02_train.py \
  --config module_net_train/configs/train_config.yaml \
  --hardware module_net_train/configs/hardware_config.yaml

# 4) Инференс по run
python module_net_train/scripts/03_predict_aoi.py \
  --config module_net_train/configs/train_config.yaml \
  --hardware module_net_train/configs/hardware_config.yaml \
  --run_dir output_data/module_net_train/runs/<run_id>

# 5) Оценка
python module_net_train/scripts/04_eval.py \
  --config module_net_train/configs/train_config.yaml \
  --hardware module_net_train/configs/hardware_config.yaml \
  --run_dir output_data/module_net_train/runs/<run_id>
```

---

## 10) Куда смотреть в артефактах после обучения

`output_data/module_net_train/runs/<run_id>/`

- `config_resolved.yaml` — фактический конфиг запуска.
- `hardware.json` — итоговый runtime план.
- `band_stats.npz` — нормализация.
- `logs/train.log` — лог эпох/лосса.
- `metrics/history.csv` — динамика train/val.
- `checkpoints/{last.pt,best.pt}` — веса.
- `pred/<dataset_key>/*` — AOI probability карты.
- `metrics/eval_test.json` — итог test оценки.

---

## 11) Рекомендованный цикл тюнинга

1. Зафиксировать data prep конфиг и seed.
2. Подобрать `lr`, `epochs`, `loss.weights`, `base_channels`.
3. Проверить, что `train/val` метрики растут согласованно (без большого gap).
4. Поиграть `boundary thresholds` и `dilation_px` для целевого применения.
5. После каждого run сравнивать `history.csv` + `eval_test.json`.

---

Если нужен следующий шаг, можно сделать отдельный документ `TUNING_GUIDE.md` с матрицей экспериментов (какой параметр, диапазон, ожидаемый эффект, критерий успеха).
