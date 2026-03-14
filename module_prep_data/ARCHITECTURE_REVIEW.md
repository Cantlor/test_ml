# 1. Executive summary

`module_prep_data` реализует рабочий конвейер подготовки геоданных для сегментации полей: QA входов, опциональный AOI clip, генерация патчей с таргетами и split в структуру `prep_data`.  
Архитектурно это **script-driven pipeline** с библиотечным ядром в `prep/`.

Что хорошо:
- Доменные инварианты вокруг NoData и valid-mask в ядре патчинга реализованы явно и последовательно.
- Векторная подготовка сделана in-memory и не портит GT на диске.
- Есть manifest/meta на уровне патчей и append-safe split с группировкой по полю.

Что ограничивает масштабирование:
- Есть заметный **config drift**: часть конфигурации типизирована, но фактически не используется.
- Есть **hidden coupling** между шагами по путям/именам файлов, а не по явным контрактам.
- Экспортный контракт на финальном шаге расходится с ожиданиями `export`/label-структуры.

Итоговая оценка текущего состояния: архитектура достаточно хороша для 1–2 датасетов и ручного запуска, но без целевого рефакторинга будет сложно масштабировать на много датасетов/экспериментов и поддерживать предсказуемую воспроизводимость.


# 2. Pipeline walkthrough

Ниже восстановленный end-to-end поток по фактическому коду.

1. Запуск оркестратора
- Скрипт [`scripts/run_prep_all.sh`](./scripts/run_prep_all.sh) запускает шаги строго по порядку: `01 -> 02 -> 03 -> 04 -> smoke` (опционально `pytest`).
- Базовые параметры (`n`, `seed`, split ratios) считываются из YAML, но пути для split на этом уровне хардкодятся относительно `PROJ_ROOT`.

2. Шаг `01_check_inputs.py`
- Читает `prep_config.yaml` через typed layer [`prep/config.py`](./prep/config.py).
- Ищет raster/vector через glob-правила датасета.
- Если AOI уже есть и `aoi_clip.out_dir` задан, предпочитает AOI raster.
- Делает raster QA:
  - метаданные растра (`crs`, `dtype`, `band_count`, `nodata`),
  - оценка `valid_ratio` по policy (`nodata_policy.rule`, `control_band_1based`), а не по `ds.nodata`.
- Делает vector QA + подготовку:
  - reprojection в CRS растра,
  - фильтрация геометрий,
  - explode multipolygon,
  - optional in-memory fix invalid,
  - фильтрация по bounds/area.
- Пишет:
  - `data_check.json` (report),
  - `work_dir/<dataset>_vector_prepared.gpkg` (prepared copy, GT исходник не трогается).

3. Шаг `02_clip_to_aoi.py`
- Если `aoi_clip.enabled=true`, клипует исходный raster по vector:
  - режим `bbox` или `mask`,
  - `buffer_m`,
  - `mask_outside`.
- Пишет AOI raster:
  - `<aoi_clip.out_dir>/<dataset>_aoi.tif` (или fallback в `work_dir/aoi_rasters`).
- Пишет manifests:
  - `out_dir/aoi_rasters_manifest.json` (детальный),
  - `work_dir/aoi_rasters_manifest.json` (canonical map `dataset -> path`).

4. Шаг `03_make_patches.py`
- Выбирает входы:
  - raster: AOI (если найден по ожидаемому пути) иначе raw;
  - vector: prepared GPKG из `work_dir`, иначе raw vector.
- Собирает `PatchConfig` из конфига.
- Вызывает ядро [`prep/patches.py`](./prep/patches.py): `make_patches_for_dataset(...)`.
- Внутри ядра:
  - читает chip;
  - строит `valid_mask` строго по `nodata_policy` (control-band/all-bands);
  - фильтрует по `min_valid_ratio`;
  - строит таргеты `extent`, `extent_ig`, `boundary_raw`, `boundary_bwbl`;
  - применяет NoData ignore policy (`valid=0 => extent_ig=255, bwbl=2`);
  - пишет `img`, `extent`, `extent_ig`, `boundary_raw`, `boundary_bwbl`, `valid`, `meta`.
- На датасет пишет `manifest.json` с `summary` и списком всех patch meta.

5. Шаг `04_split_dataset.py`
- Читает все dataset manifests из `patches_all`.
- Группирует патчи по `field_id` (fallback: `feat_index`, negatives отдельно).
- Делает split train/validation/test в append-safe режиме:
  - существующие назначения сохраняются,
  - новые группы назначаются с балансировкой к целевым ratio.
- Копирует/хардлинкует патчи в `prep_data/<split>/...`.
- Пишет `prep_data/split_manifest.json`.

6. Smoke/Tests
- `smoke_check_patches.py` проверяет:
  - словари классов (`valid`, `extent_ig`, `bwbl`),
  - корректность NoData ignore на sampled патчах.
- `tests/test_prep_pipeline.py` проверяет структуру выходов и тот же инвариант на sample.

7. Что уходит в downstream (`module_net_train`)
- Фактически из `prep_data/<split>/...` уходят:
  - `img`, `extent`, `boundary_raw`, `boundary_bwbl`, `valid`, `meta`.
- Важно: на этапе split `extent` формируется копированием `extent_ig`, а не `extent` (см. проблемы ниже).


# 3. File-by-file analysis

| Файл | Ответственность | Входы | Выходы | Связи и оценка ответственности |
|---|---|---|---|---|
| `prep/config.py` | Typed config слой (dataclasses + YAML parsing) | `prep_config.yaml` | `Config` объект | Сильный фундамент типизации. Но валидирует только парсинг, не совместимость полей со stage contracts. |
| `prep/utils.py` | Базовые утилиты путей/json/glob/CRS units | Примитивы пути/JSON | Пути, JSON IO | Хорошо как low-level helper, но часть логики путей дублируется в скриптах. |
| `prep/qa_raster.py` | Raster introspection + sampling-based valid ratio | raster path, nodata policy | `RasterInfo`, `valid_ratio` и meta | Чистая ответственность. Учитывает policy вместо `ds.nodata`, что хорошо для домена. |
| `prep/qa_vector.py` | In-memory vector QA/cleanup/reprojection | vector path + raster bounds/CRS + qa settings | prepared `GeoDataFrame`, `VectorInfo`, extra | Сильная доменная реализация. GT не перезаписывается. Есть granular stats. |
| `prep/clip_raster.py` | AOI clip логика (`bbox`/`mask`) | raster + vector + aoi config | AOI raster + `ClipResult` | Библиотечный слой чистый. Поддержка буфера в метрах и CRS-aware поведение. |
| `prep/patches.py` | Ядро генерации патчей и лейблов | raster + vector + `PatchConfig` | patch files + manifest | Наиболее важный и зрелый доменный модуль. Но в одном файле смешаны sampling, rasterization, IO, manifesting. |
| `scripts/01_check_inputs.py` | Stage orchestration: QA и подготовка vector copy | Config + raw/AOI raster + vector | `data_check.json`, `*_vector_prepared.gpkg` | Есть orchestration leakage: часть правил/контрактов закодирована прямо в скрипте и именах файлов. |
| `scripts/02_clip_to_aoi.py` | Stage orchestration AOI clipping | Config + raw raster/vector | AOI rasters + manifests | Работает как отдельная стадия. Пишет canonical manifest, который downstream не читает. |
| `scripts/03_make_patches.py` | Stage orchestration patching | Config + resolved raster/vector | `patches_all/<ds>/...` + manifest | Сильная интеграция с ядром, но скрыто зависит от артефактов шага 01/02 по naming convention. |
| `scripts/04_split_dataset.py` | Split/export layer | `patches_all/*/manifest.json` + patch files | `prep_data/<split>/...`, split manifest | Логика split хорошая, но экспорт hardcoded и частично расходится с `export` config. |
| `scripts/smoke_check_patches.py` | Быстрый доменный sanity-check | `patches_all` | Проверка инвариантов (stdout/exit code) | Полезно как pipeline guardrail. |
| `scripts/run_prep_all.sh` | High-level pipeline runner | config + CLI overrides | последовательный запуск stage scripts | Удобно для операционного запуска. Но path/config contract частично дублируется и расходится с Python config layer. |
| `tests/test_prep_pipeline.py` | Интеграционные smoke-like проверки структуры/инвариантов | реальные выходные директории | pass/fail | Полезно для smoke, но плохо изолировано от окружения и не покрывает core функции модульно. |


# 4. Architectural assessment

## Логическая карта слоёв

1. Config layer
- Реализован в `prep/config.py`.
- Typed parsing есть, но schema-level и stage-level валидации не хватает.

2. QA/validation layer
- `prep/qa_raster.py`, `prep/qa_vector.py`.
- Неплохо отделён от patching, но orchestration параметров и запись derived artifacts вынесена в скрипт 01.

3. Raster/vector preprocessing layer
- Raster AOI: `prep/clip_raster.py` + script 02.
- Vector prepare: фактически доменный код в `qa_vector.py`, запуск и materialization в script 01.

4. Patch generation layer
- `prep/patches.py` + script 03.
- Доменный центр тяжести модуля.

5. Dataset export/split layer
- `scripts/04_split_dataset.py` (практически весь слой в скрипте).
- Мало переиспользуемого library API.

6. Orchestration layer
- `run_prep_all.sh` + stage scripts.
- Рабочий, но держится на path/naming conventions.

## Насколько структура соответствует целевой архитектуре

Соответствие частичное:
- Да: есть разделение `prep/` (library-ish) vs `scripts/` (orchestration).
- Нет: в `scripts/` осталось много бизнес-логики и неявных контрактов между стадиями.
- Нет: `export` и часть `split`/`reporting`/`raster_preprocess` настроек не материализуются в behavior.


# 5. Problems and risks

Ниже конкретные проблемы из кода, не общие слова.

## High severity

1. Config drift между `export`/`split` и фактическим split/export behavior
- В `config.py` парсятся `export.structure`, `export.folders`, `split.unit`, `split.spatial_blocking_enabled`.
- Но `04_split_dataset.py` использует хардкод:
  - `SPLITS`, `SUBDIRS`, `COPY_RULES`,
  - CLI defaults путей и ratios.
- В итоге конфиг выглядит как API, но stage его игнорирует.
- Код:
  - [`prep/config.py:175`](/home/cantlor/uzcosmos/my_project/module_prep_data/prep/config.py#L175), [`prep/config.py:183`](/home/cantlor/uzcosmos/my_project/module_prep_data/prep/config.py#L183), [`prep/config.py:420`](/home/cantlor/uzcosmos/my_project/module_prep_data/prep/config.py#L420)
  - [`scripts/04_split_dataset.py:17`](/home/cantlor/uzcosmos/my_project/module_prep_data/scripts/04_split_dataset.py#L17), [`scripts/04_split_dataset.py:19`](/home/cantlor/uzcosmos/my_project/module_prep_data/scripts/04_split_dataset.py#L19), [`scripts/04_split_dataset.py:282`](/home/cantlor/uzcosmos/my_project/module_prep_data/scripts/04_split_dataset.py#L282)

2. Hidden coupling AOI stage: canonical manifest создаётся, но downstream не использует
- `02_clip_to_aoi.py` пишет canonical manifest `work_dir/aoi_rasters_manifest.json`.
- `01_check_inputs.py` и `03_make_patches.py` не читают manifest, а реконструируют путь вручную через `aoi_clip.out_dir`.
- При `out_dir=None` AOI реально пишется в fallback, но 01/03 его не используют.
- Код:
  - [`scripts/02_clip_to_aoi.py:38`](/home/cantlor/uzcosmos/my_project/module_prep_data/scripts/02_clip_to_aoi.py#L38), [`scripts/02_clip_to_aoi.py:91`](/home/cantlor/uzcosmos/my_project/module_prep_data/scripts/02_clip_to_aoi.py#L91)
  - [`scripts/01_check_inputs.py:49`](/home/cantlor/uzcosmos/my_project/module_prep_data/scripts/01_check_inputs.py#L49)
  - [`scripts/03_make_patches.py:23`](/home/cantlor/uzcosmos/my_project/module_prep_data/scripts/03_make_patches.py#L23)

3. Несовместимость prepared vector layer и `vector_layer` из исходного конфига
- `01_check_inputs.py` сохраняет prepared vector всегда в layer `"fields_prepared"`.
- `03_make_patches.py` при выборе prepared GPKG передаёт в ядро `vector_layer=ds.vector_layer` (исходное значение).
- Если `ds.vector_layer` не `None` и не `"fields_prepared"`, чтение prepared файла может упасть.
- Код:
  - [`scripts/01_check_inputs.py:277`](/home/cantlor/uzcosmos/my_project/module_prep_data/scripts/01_check_inputs.py#L277)
  - [`scripts/03_make_patches.py:206`](/home/cantlor/uzcosmos/my_project/module_prep_data/scripts/03_make_patches.py#L206)
  - [`prep/patches.py:406`](/home/cantlor/uzcosmos/my_project/module_prep_data/prep/patches.py#L406)

## Medium severity

4. Семантика `extent`/`extent_ig` на выходе split неочевидна и частично конфликтует с экспортной моделью
- На split этапе `extent` берётся из `extent_ig`.
- Папка `extent_ig` в `prep_data` не создаётся.
- То есть финальный `extent` уже содержит ignore-class 255.
- Код:
  - [`scripts/04_split_dataset.py:18`](/home/cantlor/uzcosmos/my_project/module_prep_data/scripts/04_split_dataset.py#L18), [`scripts/04_split_dataset.py:21`](/home/cantlor/uzcosmos/my_project/module_prep_data/scripts/04_split_dataset.py#L21), [`scripts/04_split_dataset.py:314`](/home/cantlor/uzcosmos/my_project/module_prep_data/scripts/04_split_dataset.py#L314)

5. Значимая часть конфигурации не материализована в pipeline behavior
- Примеры: `raster_preprocess.*`, `patching.sampling.mode`, `patching.samples_per_feature`, `patching.sampling.near_nodata.*`, `reporting.save_summary_csv/save_previews`, `performance.*`, `logging.log_file`.
- `config.py` их парсит, но pipeline их не применяет.
- Код:
  - Парсинг: [`prep/config.py:330`](/home/cantlor/uzcosmos/my_project/module_prep_data/prep/config.py#L330), [`prep/config.py:351`](/home/cantlor/uzcosmos/my_project/module_prep_data/prep/config.py#L351), [`prep/config.py:441`](/home/cantlor/uzcosmos/my_project/module_prep_data/prep/config.py#L441)
  - Фактическое отсутствие использования в scripts/prep (поиск по репозиторию).

6. `run_prep_all.sh` жёстко фиксирует пути split-этапа
- Независимо от `paths`/`export` в YAML, split получает:
  - `PATCHES_ALL=${PROJ_ROOT}/output_data/module_prep_data_work/patches_all`
  - `OUT_PREP=${PROJ_ROOT}/prep_data`
- Это ломает ожидаемую перенастраиваемость окружений.
- Код:
  - [`scripts/run_prep_all.sh:154`](/home/cantlor/uzcosmos/my_project/module_prep_data/scripts/run_prep_all.sh#L154), [`scripts/run_prep_all.sh:155`](/home/cantlor/uzcosmos/my_project/module_prep_data/scripts/run_prep_all.sh#L155)

7. Domain intent `near_nodata` заявлен, но выборка near-NoData не реализована
- Конфиг содержит quota/threshold near-NoData.
- Ядро patching никак не использует эти поля для стратифицированного sampling.
- Риск: dataset bias не соответствует spec/ожиданию.
- Код:
  - [`prep/config.py:363`](/home/cantlor/uzcosmos/my_project/module_prep_data/prep/config.py#L363)
  - В `prep/patches.py` нет логики квотирования по `nodata_frac`.

## Low severity / design smells

8. Частичное дублирование и рассинхрон path resolution
- AOI path resolving и work_dir resolving реализованы отдельно в разных скриптах.
- Это повышает риск расхождения behavior между стадиями.

9. Скрипты делают `sys.path.insert`, что затрудняет packaging/CLI entrypoints
- Код:
  - [`scripts/01_check_inputs.py:17`](/home/cantlor/uzcosmos/my_project/module_prep_data/scripts/01_check_inputs.py#L17),
  - [`scripts/02_clip_to_aoi.py:11`](/home/cantlor/uzcosmos/my_project/module_prep_data/scripts/02_clip_to_aoi.py#L11),
  - [`scripts/03_make_patches.py:11`](/home/cantlor/uzcosmos/my_project/module_prep_data/scripts/03_make_patches.py#L11)

10. Тесты зависимы от реального filesystem state, не unit-style
- `tests/test_prep_pipeline.py` ожидает уже существующие `output_data/...` и `prep_data/...`.
- Для CI/изолированного окружения это хрупко.
- Код:
  - [`tests/test_prep_pipeline.py:17`](/home/cantlor/uzcosmos/my_project/module_prep_data/tests/test_prep_pipeline.py#L17), [`tests/test_prep_pipeline.py:66`](/home/cantlor/uzcosmos/my_project/module_prep_data/tests/test_prep_pipeline.py#L66)


# 6. Recommended improvements

## 6.1 Structure improvements

1. Перенести stage logic из `scripts/` в library-сервисы
- Добавить модули:
  - `prep/stages/check_inputs.py`
  - `prep/stages/clip_aoi.py`
  - `prep/stages/make_patches.py`
  - `prep/stages/split_dataset.py`
- `scripts/*.py` оставить тонкими CLI-обёртками.

2. Вынести общие резолверы и контракты артефактов
- Единый `prep/paths.py`:
  - resolve raw raster/vector,
  - resolve AOI raster,
  - resolve prepared vector,
  - resolve manifests.
- Убрать дубли из 01/02/03.

3. Формализовать manifest/domain модели
- Ввести dataclasses/Pydantic для:
  - `AoiManifest`
  - `PatchManifest` (`summary`, `patches[]`)
  - `SplitManifest`
  - `PatchMeta`
- Явные схемы версионировать (`schema_version`), чтобы downstream имел стабильный контракт.

4. Согласовать экспортный слой с `export` config
- `split_dataset` должен читать `cfg.export.structure` и `cfg.export.folders`.
- Контракт `extent` vs `extent_ig` сделать явным и конфигурируемым.

## 6.2 Architecture improvements

1. Сделать конфиг реальным API, а не «декларацией без эффекта»
- Для каждого поля либо:
  - реализовать behavior, либо
  - удалить/пометить deprecated.

2. Устранить hidden coupling между стадиями
- Переходить от naming convention к явным stage outputs:
  - 02 пишет canonical manifest,
  - 01/03 читают его через общий резолвер.

3. Декомпозировать `prep/patches.py`
- Разделить на:
  - `sampling.py`
  - `label_builder.py`
  - `writers.py`
  - `manifest.py`
- Это улучшит testability и читаемость без изменения доменной логики.

4. Реализовать/валидировать near-NoData policy
- Либо реально добавить near-NoData sampling quota,
- либо убрать поле из конфига до появления реализации.

5. Добавить reproducibility safeguards
- В `split_manifest` и `patch manifest` сохранять:
  - хеш конфига,
  - версию кода/commit SHA (если доступно),
  - stage timestamps,
  - run_id.


# 7. Refactoring roadmap

## Фаза 1 (Low risk, high value)

1. Явно задокументировать текущие контракты артефактов
- Добавить `docs/pipeline_contract.md`.
- Зафиксировать фактическую структуру `patches_all` и `prep_data`.

2. Убрать расхождения AOI resolution
- Вынести общий resolver AOI path.
- В 01/03 читать canonical AOI manifest из шага 02.

3. Починить prepared vector layer contract
- При выборе prepared GPKG принудительно использовать layer `"fields_prepared"` или хранить layer в отдельном manifest.

4. Сделать split configurable из `cfg.export`/`cfg.split`
- Подключить `load_config` в 04 скрипт.
- Убрать hardcoded `SUBDIRS/COPY_RULES` где возможно.

## Фаза 2 (Medium risk)

5. Разделить бизнес-логику и CLI
- Перенести core из `scripts/` в `prep/stages`.
- Скрипты сделать thin adapters.

6. Ввести typed manifests + schema versioning
- Добавить валидацию при чтении manifest на каждом следующем шаге.

7. Пересмотреть финальную label-схему
- Выбрать одно из:
  - хранить `extent` и `extent_ig` отдельно до downstream,
  - или явно объявить что `extent` в split это `extent_ig`.

## Фаза 3 (Medium/High risk, но стратегически важно)

8. Декомпозировать `patches.py` на компоненты
- Переход через адаптер, чтобы не ломать CLI.

9. Реализовать `near_nodata` sampling и/или raster preprocess stage
- Добавить stage `raster_preprocess` до patching.
- Интегрировать `nodata_to_value` и band stats по spec.

10. Перестроить тестовую пирамиду
- Unit-тесты на sampling/label building/split assignment.
- Интеграционные e2e в temp dir с синтетическим растром и полигонами.


# 8. Final verdict

Архитектура модуля уже решает ключевую практическую задачу и содержит сильное доменное ядро, особенно в части NoData-policy, valid-mask, label generation и in-memory vector QA. Для текущего проекта и ограниченного масштаба это рабочее и полезное решение.

Пределы текущей версии проявятся при росте числа датасетов, вариантов конфигурации и требований к воспроизводимости: слишком много stage-контрактов сейчас держатся на неявных соглашениях (пути, имена, слой в GPKG), а не на формализованных интерфейсах.

Что я бы точно сохранил:
- доменные инварианты внутри `prep/patches.py`,
- in-memory политику для GT,
- append-safe split идею с группировкой по полям.

Что я бы переделал в первую очередь:
- устранение config drift,
- формализация manifest contracts между стадиями,
- унификация path/artifact resolution,
- вынос orchestration logic из `scripts/` в библиотечный слой.

