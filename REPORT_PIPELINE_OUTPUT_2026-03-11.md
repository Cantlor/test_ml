# Отчёт по фактическому выходу пайплайна (без перезапуска)

Дата отчёта: 2026-03-11

Основано только на уже существующих артефактах:
- `output_data/module_prep_data_work/patches_all/first_raster/manifest.json`
- `prep_data/split_manifest.json`
- `output_data/data_check.json`
- `output_data/module_net_train/runs/20260311_094426/prep_data_summary.json`
- `output_data/module_net_train/runs/20260311_094426/metrics/history.csv`
- `output_data/module_net_train/runs/20260311_094426/metrics/eval_test.json`
- `output_data/module_net_train/runs/20260311_094426/pred/first_raster/predict_manifest.json`

## 1) Что получилось на выходе

### 1.1 Подготовка данных (`module_prep_data`)
- Проверка входов: `ok=true`, ошибок/предупреждений нет (`output_data/data_check.json`).
- Оценка валидной области AOI: `raster_valid_ratio_estimate = 0.72795`.
- Генерация патчей (`patches_all`):
  - target: `800`
  - записано: `725`
  - центр: `408`
  - граница: `272`
  - негативы: `45`
  - shortfall по негативам: `75` (цель была `120`)
- Финальный split (`prep_data/split_manifest.json`):
  - train: `580`
  - validation: `72`
  - test: `73`
  - режим: `append`

### 1.2 Обучение/инференс/оценка (`module_net_train`, run_id `20260311_094426`)
- Артефакты run присутствуют полностью:
  - checkpoints: `best.pt`, `last.pt`
  - train history: `metrics/history.csv`
  - test eval: `metrics/eval_test.json`
  - AOI predict: `pred/first_raster/{extent_prob.tif,boundary_prob.tif,predict_manifest.json}`
- AOI-предикт:
  - tiles: `644`
  - window/stride: `512/384`

## 2) Ключевые метрики и текущее состояние

### 2.1 По train/val (из `history.csv`)
- Лучшая `val/extent_iou`: `0.879303` на `epoch=6`.
- Финальная `val/extent_iou` (epoch 20): `0.876585`.
- `val/extent_f1` держится около `0.93`.
- Проблема: boundary-голова почти не учится:
  - `val/boundary_f1@0.50`: максимум `0.0` (по всем эпохам)
  - `val/boundary_f1@0.35`: максимум `0.049767` (epoch 1), далее в основном близко к 0

### 2.2 По test (из `eval_test.json`)
- `val/loss`: `0.65682`
- `val/extent_iou`: `0.88472`
- `val/extent_f1`: `0.93883`
- `val/boundary_f1@0.50`: `0.0`
- `val/boundary_f1@0.35`: `0.0`

Вывод: сегментация extent работает хорошо, boundary-направление в текущей постановке практически не даёт полезного сигнала.

## 3) Какие метрики для вас сейчас важны (приоритет)

1. `test extent_iou` и `test extent_f1`.
Это главные метрики качества основной сегментации полей.

2. `boundary_f1` на нескольких порогах.
Сейчас это главная зона деградации. Без роста этой метрики boundary-канал практически не приносит пользы.

3. Стабильность между `val` и `test` по extent.
Сейчас стабильность нормальная (нет явного развала обобщения).

4. Сервисные метрики данных:
- доля валидных пикселей,
- покрытие негативов,
- факт shortfall по негативам.

## 4) Чего не хватает для улучшения результата

1. Не хватает полноценной диагностики boundary-направления.
Сейчас есть F1 на 2 порогах, но нет sweep/PR-кривой и подбора рабочего порога.

2. Негативная выборка недобирается (`45` вместо `120`).
Это ухудшает контраст «граница/не-граница» и затрудняет обучение boundary.

3. Чекпоинт выбирается по `val/extent_iou`.
Boundary-качество не участвует в выборе лучшей модели.

4. Нет AOI-level метрики с GT для выходных `extent_prob.tif` / `boundary_prob.tif`.
Есть patch-level eval на test, но не финальная пространственная метрика по AOI-карте.

5. Датасет фактически из одного источника (`first_raster`).
Для устойчивого boundary-обобщения обычно нужно больше разнообразия сцен.

## 5) Что править в первую очередь (без изменения входных данных)

1. Усилить вклад boundary в обучении:
- `train_config.yaml -> loss.weights.boundary`: поднять с `3.0` до `5.0-8.0` и проверить.

2. Расширить контроль порогов boundary:
- `train_config.yaml -> metrics.boundary.thresholds`: добавить сетку, например `[0.15, 0.25, 0.35, 0.50]`.

3. Добрать негативы в `prep`:
- `prep_config.yaml -> patching.sampling.negatives.min_distance_to_fields_m`: уменьшить (например `15 -> 8..10`),
- `prep_config.yaml -> patching.filters.neg_max_mask_ratio`: чуть ослабить (например `0.01 -> 0.02`).

4. Выбор best-checkpoint учитывать boundary:
- сменить monitor на комбинированную метрику или отдельный boundary-критерий (иначе boundary не оптимизируется при выборе best).

5. Для более стабильной оптимизации boundary:
- увеличить `epochs` и смотреть именно динамику `boundary_f1`,
- при доступной памяти увеличить capacity (`model.base_channels`) или контекст (`sampling.crop_size`).

## 6) Краткий итог

- Пайплайн по артефактам завершился корректно.
- Основная задача extent решается на хорошем уровне (`IoU ~0.88` на test).
- Главный ограничитель качества сейчас — boundary-голова (F1 около нуля).
- Для заметного роста качества в первую очередь нужно улучшить именно boundary-сигнал (данные + лосс + выбор чекпоинта + пороги).
