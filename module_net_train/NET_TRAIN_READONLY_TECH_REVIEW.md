# READ-ONLY Technical Review: `module_net_train`

Дата анализа: 2026-03-13  
Режим: **read-only** (без изменения кода/конфигов/артефактов)  
Источник истины: текущий worktree + существующие run-артефакты  
Repo baseline: branch `main`, commit `b57e484`, worktree **dirty** по `module_net_train/*`.

---

## 1. Repo architecture overview

### Executive summary (15 пунктов)
1. Репозиторий архитектурно разделён на понятные подсистемы: `data`, `models`, `losses`, `metrics`, `train`, `infer`, `scripts` ([`net_train/__init__.py`](./net_train/__init__.py), package wiring).  
2. Входной контракт `prep_data` реализован через индексатор и sample records с явными путями `img/valid/extent/boundary_bwbl/meta` ([`net_train/data/index.py`](./net_train/data/index.py), `SampleRecord`, L10-L20, `build_index_for_split`, L39-L79).  
3. Data-layer уже содержит важные guardrails: shape checks, channel checks, valid re-binarization, NoData→ignore enforcement в таргетах ([`net_train/data/dataset.py`](./net_train/data/dataset.py), `_read_quad`, L89-L138; `__getitem__`, L191-L213).  
4. Контракт 9-канального входа (8 спектральных + `valid`) поддержан end-to-end и принудительно синхронизируется со `model.in_channels` в train/eval/infer scripts ([`scripts/02_train.py`](./scripts/02_train.py), L127-L141; [`scripts/03_predict_aoi.py`](./scripts/03_predict_aoi.py), L74-L82; [`scripts/04_eval.py`](./scripts/04_eval.py), L87-L95).  
5. Loss-политика по ignore реализована корректно: `extent` игнор 255, `boundary` игнор 2 ([`net_train/losses/extent_loss.py`](./net_train/losses/extent_loss.py), L42-L49; [`net_train/losses/bwbl_loss.py`](./net_train/losses/bwbl_loss.py), L53-L71).  
6. Train/eval parity улучшена: eval и predict приоритетно читают `run_dir/config_resolved.yaml` ([`net_train/config.py`](./net_train/config.py), `resolve_run_train_config_path`, L119-L128; scripts `03/04`, L43-L53 и L52-L63).  
7. Inference корректно подавляет предсказания в NoData-зоне (`valid==0 => prob=0`) ([`net_train/infer/predict_aoi.py`](./net_train/infer/predict_aoi.py), L255-L257).  
8. Поддержаны strict + legacy manifest-форматы AOI, поэтому есть backward compatibility ([`net_train/infer/predict_aoi.py`](./net_train/infer/predict_aoi.py), `_resolve_from_manifest_obj`, L18-L52; `resolve_aoi_path`, L54-L82).  
9. Boundary-метрика считается по dilation-aware F1 на нескольких порогах ([`net_train/metrics/boundary_metrics.py`](./net_train/metrics/boundary_metrics.py), `boundary_f1_dilated`, L21-L67).  
10. По run-артефактам baseline работает как pipeline (есть `history.csv`, `eval_test.json`, `predict_manifest.json`), но boundary сильно чувствителен к threshold: `@0.50` почти 0, `@0.15` существенно выше (artifact evidence ниже).  
11. В анализируемом run лучшая модель выбиралась по `val/boundary_f1@0.50`, что ухудшает selection при такой threshold-чувствительности (artifact `config_resolved.yaml`, L93-L97 + `history.csv`).  
12. В текущем коде/конфиге этот риск уже адресован: monitor = `val/boundary_f1_max` ([`configs/train_config.yaml`](./configs/train_config.yaml), L137-L142; [`net_train/train/loop.py`](./net_train/train/loop.py), L262-L270).  
13. Для маленького batch (`1`) с `BatchNorm` есть архитектурный риск деградации boundary-качества; в train-скрипте это уже предупреждается ([`scripts/02_train.py`](./scripts/02_train.py), L233-L238; artifact `hardware.json`, `runtime_plan.batch_size=1`).  
14. Reproducibility не fully-locked: default `deterministic=false`, нет `worker_init_fn`/generator в DataLoader, из-за чего мультиворкерная стохастика может плавать между запусками ([`configs/train_config.yaml`](./configs/train_config.yaml), L109-L111; [`scripts/02_train.py`](./scripts/02_train.py), `_make_loader`, L33-L45).  
15. Модуль является рабочим baseline, но boundary-quality ограничивают calibration/imbalance/monitoring choices и runtime-детерминизм.

### Краткая архитектурная карта
- Config/runtime layer: [`net_train/config.py`](./net_train/config.py), [`net_train/hardware.py`](./net_train/hardware.py), [`configs/*.yaml`](./configs).
- Data layer: [`net_train/data/index.py`](./net_train/data/index.py), [`net_train/data/dataset.py`](./net_train/data/dataset.py), [`net_train/data/stats.py`](./net_train/data/stats.py), [`net_train/data/transforms.py`](./net_train/data/transforms.py).
- Model/loss/metrics: [`net_train/models/unet_multitask.py`](./net_train/models/unet_multitask.py), [`net_train/losses/*`](./net_train/losses), [`net_train/metrics/*`](./net_train/metrics).
- Train loop/checkpoint: [`net_train/train/loop.py`](./net_train/train/loop.py), [`net_train/train/checkpoint.py`](./net_train/train/checkpoint.py), [`net_train/train/optim.py`](./net_train/train/optim.py).
- Inference: [`net_train/infer/predict_aoi.py`](./net_train/infer/predict_aoi.py), [`net_train/infer/tiling.py`](./net_train/infer/tiling.py).
- CLI/orchestration: [`scripts/01_check_prep_data.py`](./scripts/01_check_prep_data.py), [`02_train.py`](./scripts/02_train.py), [`03_predict_aoi.py`](./scripts/03_predict_aoi.py), [`04_eval.py`](./scripts/04_eval.py), run wrappers.

---

## 2. End-to-end data contract check

### E2E dataflow (фактический)
1. `scripts/01_check_prep_data.py` строит index по `meta_*` и проверяет наличие/допустимые значения масок + базовые статистики.  
2. `scripts/02_train.py`:
   - грузит config + hardware,
   - строит runtime plan,
   - фиксирует `config_resolved.yaml` и `hardware.json`,
   - индексирует train/val,
   - считает `band_stats.npz`,
   - обучает и пишет checkpoints/history,
   - опционально запускает AOI inference.
3. `scripts/04_eval.py`:
   - приоритетно берёт `run_dir/config_resolved.yaml`,
   - использует `band_stats.npz` + checkpoint,
   - считает test-метрики и пишет `metrics/eval_test.json`.
4. `scripts/03_predict_aoi.py`:
   - приоритетно берёт `run_dir/config_resolved.yaml`,
   - резолвит AOI через manifest,
   - делает tiled inference и пишет `predict_manifest.json`.

### Contract-check table

| Expected contract | Actual implementation | Status | Evidence |
|---|---|---|---|
| Model input = `8 spectral + 1 valid = 9` | Dataset формирует `[img8 + valid]`; scripts принудительно выравнивают `model.in_channels` с dataset contract | OK | [`/home/cantlor/uzcosmos/my_project/module_net_train/net_train/data/dataset.py`](./net_train/data/dataset.py), `PatchDataset.__getitem__`, L146-L149, L202-L213; [`/home/cantlor/uzcosmos/my_project/module_net_train/scripts/02_train.py`](./scripts/02_train.py), `main`, L127-L141 |
| `extent ignore = 255` | `valid==0` принудительно помечается `extent_ignore`; loss маскирует `target != 255` | OK | [`/home/cantlor/uzcosmos/my_project/module_net_train/net_train/data/dataset.py`](./net_train/data/dataset.py), `__getitem__`, L193-L196; [`/home/cantlor/uzcosmos/my_project/module_net_train/net_train/losses/extent_loss.py`](./net_train/losses/extent_loss.py), `extent_loss`, L42-L49 |
| `boundary ignore = 2` | `valid==0` принудительно помечается `boundary_ignore`; loss маскирует `target != 2` | OK | [`/home/cantlor/uzcosmos/my_project/module_net_train/net_train/data/dataset.py`](./net_train/data/dataset.py), `__getitem__`, L197-L199; [`/home/cantlor/uzcosmos/my_project/module_net_train/net_train/losses/bwbl_loss.py`](./net_train/losses/bwbl_loss.py), `boundary_bwbl_loss`, L53-L71 |
| NoData не участвует в loss | Реализовано через valid→ignore policy в dataset + masked losses | OK | Те же места + [`/home/cantlor/uzcosmos/my_project/module_net_train/net_train/losses/extent_loss.py`](./net_train/losses/extent_loss.py), L45-L47 |
| Inference suppression where `valid=0` | Перед blending: `e[valid==0]=0`, `b[valid==0]=0` | OK | [`/home/cantlor/uzcosmos/my_project/module_net_train/net_train/infer/predict_aoi.py`](./net_train/infer/predict_aoi.py), `predict_aoi_raster`, L255-L257 |
| Train/eval/predict используют одинаковую конфигурацию run | `03_predict` и `04_eval` приоритетно читают `run_dir/config_resolved.yaml` | OK | [`/home/cantlor/uzcosmos/my_project/module_net_train/net_train/config.py`](./net_train/config.py), `resolve_run_train_config_path`, L119-L128; [`/home/cantlor/uzcosmos/my_project/module_net_train/scripts/03_predict_aoi.py`](./scripts/03_predict_aoi.py), L43-L53; [`/home/cantlor/uzcosmos/my_project/module_net_train/scripts/04_eval.py`](./scripts/04_eval.py), L52-L63 |
| `extent` vs `extent_ig` | Используется только `extent/extent_*.tif`; `extent_ig` в net_train не читается | OK (aux не используется) | [`/home/cantlor/uzcosmos/my_project/module_net_train/net_train/data/index.py`](./net_train/data/index.py), `_expected_paths`, L30-L36; repo search |
| `boundary_bwbl` vs `boundary_raw` | Используется только `boundary_bwbl`; `boundary_raw` не участвует | OK (aux не используется) | [`/home/cantlor/uzcosmos/my_project/module_net_train/net_train/data/index.py`](./net_train/data/index.py), L35; [`/home/cantlor/uzcosmos/my_project/module_net_train/net_train/data/dataset.py`](./net_train/data/dataset.py), L102-L104 |
| Чекпоинт-монитор должен быть устойчив к threshold drift | В текущем config: `val/boundary_f1_max`; но в анализируемом run было `val/boundary_f1@0.50` | Mismatch (historical run vs current config) | [`/home/cantlor/uzcosmos/my_project/module_net_train/configs/train_config.yaml`](./configs/train_config.yaml), L137-L142; artifact [`/home/cantlor/uzcosmos/my_project/output_data/module_net_train/runs/20260312_131232/config_resolved.yaml`](../output_data/module_net_train/runs/20260312_131232/config_resolved.yaml), L93-L97 |
| AOI manifest strict path | Код поддерживает strict + legacy fallbacks | OK (compat mode) | [`/home/cantlor/uzcosmos/my_project/module_net_train/net_train/config.py`](./net_train/config.py), `resolve_inference_manifest_path`, L99-L117; [`/home/cantlor/uzcosmos/my_project/module_net_train/net_train/infer/predict_aoi.py`](./net_train/infer/predict_aoi.py), L64-L82 |

### Artifact evidence (run `20260312_131232`)
- Artifact path: `/home/cantlor/uzcosmos/my_project/output_data/module_net_train/runs/20260312_131232/config_resolved.yaml`  
  Key: `train.checkpoint.monitor`  
  Observed: `val/boundary_f1@0.50`  
  Interpretation: best checkpoint в этом run выбирался по фиксированному порогу 0.50.

- Artifact path: `/home/cantlor/uzcosmos/my_project/output_data/module_net_train/runs/20260312_131232/metrics/eval_test.json`  
  Key: `metrics.val/boundary_f1@0.50` vs `metrics.val/boundary_f1@0.15`  
  Observed: `0.000383` vs `0.268070`  
  Interpretation: качество boundary критически зависит от threshold.

- Artifact path: `/home/cantlor/uzcosmos/my_project/output_data/module_net_train/runs/20260312_131232/metrics/history.csv`  
  Row evidence: max `val/boundary_f1@0.50` at epoch 38 equals `0.000490`; max `val/boundary_f1@0.15` at epoch 38 equals `0.247733`  
  Interpretation: monitoring only `@0.50` занижает реальный сигнал boundary head.

---

## 3. File-by-file review

Ниже краткий профиль **всех обязательных файлов**: роль, что сделано хорошо, и риск.

| File | Role | Что хорошо | Риск / что проверить | Evidence |
|---|---|---|---|---|
| [`README.md`](./README.md) | Документация пайплайна | Описывает 9-канальный контракт и run artifacts | Части README могут не совпадать с историческими run-артефактами | section “Нормализация и каналы” L47-L52, “Оценка” L83-L88 |
| [`configs/train_config.yaml`](./configs/train_config.yaml) | Основной train/eval/infer config | Явные ignore values, thresholds, crop policy, monitor | Есть поля-заготовки без runtime эффекта | L57-L65, L137-L142, L170-L175 |
| [`configs/hardware_config.yaml`](./configs/hardware_config.yaml) | Runtime/hardware policy | Явный autotune plan и precision settings | Autotune heuristic, нет runtime probe OOM | L20-L30 + [`net_train/hardware.py`](./net_train/hardware.py) L134-L155 |
| [`net_train/config.py`](./net_train/config.py) | Config loading + path resolving | Чёткий parity helper для eval/infer; manifest resolver | Сохраняется legacy fallback на AOI manifest | `resolve_run_train_config_path` L119-L128; `resolve_inference_manifest_path` L99-L117 |
| [`net_train/hardware.py`](./net_train/hardware.py) | Runtime plan + AMP/device flags | Единый `RuntimePlan`, fallback при SemLock issues | Нет реального autotune через пробные step’ы; deterministic tradeoff | `build_runtime_plan` L197-L240 |
| [`net_train/paths.py`](./net_train/paths.py) | Path defaults | Простая централизация roots | Почти не используется в scripts (низкая ценность слоя) | L7-L18 |
| [`net_train/data/__init__.py`](./net_train/data/__init__.py) | Re-export API | Упрощает импорты | Нет | L3-L22 |
| [`net_train/data/index.py`](./net_train/data/index.py) | Индексация samples из `prep_data` | Явная проверка expected paths | Contract жёстко прибит к naming (`img_*`, `bwbl_*`) | `_expected_paths` L30-L36 |
| [`net_train/data/dataset.py`](./net_train/data/dataset.py) | Dataset reading/crop/augment/contract enforcement | Strong guardrails (shape/channels/NoData ignore) | `valid` санитизируется как `>0`, что может скрыть повреждённые маски | `_read_quad` L112-L117; `__getitem__` L193-L213 |
| [`net_train/data/stats.py`](./net_train/data/stats.py) | Нормализация stats + normalize | Учитывает valid/nodata policy | При плохих valid-масках можно получить смещённые stats без явного алерта о доле valid | `compute_normalization_stats` L91-L116 |
| [`net_train/data/transforms.py`](./net_train/data/transforms.py) | Crop + augment | Есть crop attempts + min boundary pixels | При неудачных попытках берётся “best available”, может быть слабый boundary-signal | `random_crop` L40-L74 |
| [`net_train/losses/__init__.py`](./net_train/losses/__init__.py) | Loss exports | Ок | Нет | L3-L9 |
| [`net_train/losses/extent_loss.py`](./net_train/losses/extent_loss.py) | BCE+Dice extent | Корректный ignore masking | Нет | `extent_loss` L29-L57 |
| [`net_train/losses/bwbl_loss.py`](./net_train/losses/bwbl_loss.py) | Boundary BWBL loss | `pos_weight=auto`, optional focal, ignore masking | Нет Dice/Tversky компоненты; `auto` pos_weight batch-зависим | `_resolve_pos_weight` L9-L37; `boundary_bwbl_loss` L39-L77 |
| [`net_train/metrics/__init__.py`](./net_train/metrics/__init__.py) | Metric exports | Есть multi-threshold export | `boundary_metrics_multi_threshold` не используется в train loop | L3-L10 |
| [`net_train/metrics/extent_metrics.py`](./net_train/metrics/extent_metrics.py) | Extent IoU/F1 | Ignore-aware метрики | Нет | `extent_binary_metrics` L9-L52 |
| [`net_train/metrics/boundary_metrics.py`](./net_train/metrics/boundary_metrics.py) | Boundary F1 с dilation | Соответствует задаче границ | CPU numpy/scipy loop может быть дорогим | `boundary_f1_dilated` L21-L67 |
| [`net_train/models/__init__.py`](./net_train/models/__init__.py) | Model registry | Чистый builder API | Поддерживается только один model name | `build_model` L7-L11 |
| [`net_train/models/unet_multitask.py`](./net_train/models/unet_multitask.py) | U-Net 2-head model | Простая стабильная baseline-архитектура | BatchNorm при small batch может ухудшать boundary | `_norm_layer` L17-L23; `UNetMultiTask` L78-L124 |
| [`net_train/train/__init__.py`](./net_train/train/__init__.py) | Train exports | Ок | Нет | L3-L16 |
| [`net_train/train/checkpoint.py`](./net_train/train/checkpoint.py) | save/load + monitor-based best | Fail-fast если monitor key отсутствует | Eval/predict могут тихо fallback на `last.pt` (не здесь, а в scripts) | `CheckpointManager.step` L43-L69 |
| [`net_train/train/optim.py`](./net_train/train/optim.py) | Optimizer/scheduler factories | Просто и прозрачно | Узкий набор optimizer/scheduler | `create_optimizer` L11-L23; `create_scheduler` L39-L63 |
| [`net_train/train/loop.py`](./net_train/train/loop.py) | Core train/val loop | Epoch-level aggregated metrics, multi-threshold boundary F1 | `val_every_epochs` config не используется; нет worker-seed handling | `validate_one_epoch` L188-L272; `run_training` L276-L349 |
| [`net_train/infer/__init__.py`](./net_train/infer/__init__.py) | Inference exports | Ок | Нет | L3-L12 |
| [`net_train/infer/tiling.py`](./net_train/infer/tiling.py) | Window generation + blending | Корректное покрытие окон, `mean/gaussian` blend | Нет adaptive/uncertainty blending | `generate_windows` L29-L40 |
| [`net_train/infer/predict_aoi.py`](./net_train/infer/predict_aoi.py) | AOI resolve + tiled prediction | NoData suppression и manifest compatibility | Legacy fallback может скрывать миграционные проблемы | `resolve_aoi_path` L54-L82; `predict_aoi_raster` L255-L257 |
| [`net_train/utils/__init__.py`](./net_train/utils/__init__.py) | Utils namespace | Ок | Нет | L1 |
| [`net_train/utils/io.py`](./net_train/utils/io.py) | JSON/CSV helpers | Простые стабильные IO helpers | `append_csv_row` зависит от консистентности ключей между эпохами | `append_csv_row` L26-L33 |
| [`net_train/utils/logging.py`](./net_train/utils/logging.py) | Logger setup | Rich + file logging | Нет структурированных JSON logs | `setup_logger` L10-L27 |
| [`net_train/utils/seed.py`](./net_train/utils/seed.py) | Global seeding | Есть deterministic toggle | Не настраивает DataLoader worker seeds отдельно | `seed_everything` L14-L28 |
| [`scripts/01_check_prep_data.py`](./scripts/01_check_prep_data.py) | Contract checker for prep_data | Проверяет маски, band count, missing files | Проверяет только subset sample masks (`max_mask_checks`) | L65-L67, L141-L158 |
| [`scripts/02_train.py`](./scripts/02_train.py) | Main training orchestrator | Сильный wiring config→data→model→loss→metrics→ckpt | Есть `sys.path.insert`; CLI layer содержит много orchestration logic | L11-L14, `main` L48-L323 |
| [`scripts/03_predict_aoi.py`](./scripts/03_predict_aoi.py) | Inference entrypoint | Uses run-resolved config, loads run stats/checkpoint | Тихий fallback `best -> last` checkpoint | L86-L93 |
| [`scripts/04_eval.py`](./scripts/04_eval.py) | Test evaluation entrypoint | Uses run-resolved config and run stats | Тихий fallback `best -> last` checkpoint | L69-L73 |
| [`scripts/run_train_all.sh`](./scripts/run_train_all.sh) | Linux orchestrator | Полный e2e workflow с flag controls | Небольшой дублирующий orchestration vs Python scripts | L122-L168 |
| [`scripts/run_train_all.ps1`](./scripts/run_train_all.ps1) | Windows orchestrator | Паритет с sh-пайплайном | Аналогично, дубли orchestration | L77-L137 |
| [`scripts/run_train_all.bat`](./scripts/run_train_all.bat) | Thin wrapper for PS | Минимальный launcher | Нет | L4-L13 |
| [`tests/test_bwbl_loss.py`](./tests/test_bwbl_loss.py) | Unit test boundary loss | Проверяет auto pos_weight и all-ignore case | Нет теста на gradient signal/imbalance dynamics | L8-L26 |
| [`tests/test_transforms_crop_policy.py`](./tests/test_transforms_crop_policy.py) | Crop-policy test | Проверяет минимальный boundary/extent coverage | Синтетика мала, нет integration test с real patch size | L8-L30 |
| [`tests/test_config_run_resolution.py`](./tests/test_config_run_resolution.py) | Config parity helper tests | Проверяет run config precedence | Нет теста на missing/corrupt run config content | L8-L30 |
| [`tests/test_infer_manifest_resolution.py`](./tests/test_infer_manifest_resolution.py) | AOI manifest resolver tests | Покрыты strict + legacy форматы | Нет теста на status!=clipped ветку | L14-L65 |

---

## 4. Findings by priority

### Critical

1. **В анализируемом run (`20260312_131232`) best checkpoint выбирался по `val/boundary_f1@0.50`, при том что boundary-качество максимум на low threshold (`0.15`).**  
Почему проблема: selection criterion не отражает целевую boundary-производительность, может фиксировать субоптимальную модель.  
Влияние: качество boundary и downstream полигонов.  
Evidence:
- File: `/home/cantlor/uzcosmos/my_project/output_data/module_net_train/runs/20260312_131232/config_resolved.yaml`  
  Symbol: `train.checkpoint.monitor`  
  Lines: L93-L97  
  Why: монитор в run зафиксирован как `val/boundary_f1@0.50`.
- Artifact: `/home/cantlor/uzcosmos/my_project/output_data/module_net_train/runs/20260312_131232/metrics/eval_test.json`  
  Key: `val/boundary_f1@0.50` vs `val/boundary_f1@0.15`  
  Observed: `0.000383` vs `0.268070`  
  Interpretation: threshold 0.50 почти обнуляет метрику.

2. **Boundary head в baseline-run явно недокалиброван по threshold.**  
Почему проблема: inference/post-processing сильно зависит от ручного выбора порога.  
Влияние: нестабильная геометрия границ.  
Evidence:
- Artifact: `.../metrics/history.csv`  
  Key/row: epoch 38  
  Observed: `val/boundary_f1@0.50=0.000490`, `val/boundary_f1@0.15=0.247733`  
  Interpretation: резкий разрыв между порогами.
- File: [`/home/cantlor/uzcosmos/my_project/module_net_train/net_train/train/loop.py`](./net_train/train/loop.py)  
  Symbol: `validate_one_epoch`  
  Lines: L262-L270  
  Why: loop поддерживает multi-threshold и `f1_max`, но historical run мониторил фиксированный threshold.

### Important

1. **Small-batch + BatchNorm риск реален на текущем железе (batch=1 в runtime artifact).**  
Влияние: шумные statistics BN, деградация тонких boundary-паттернов.  
Evidence:
- Artifact: `/home/cantlor/uzcosmos/my_project/output_data/module_net_train/runs/20260312_131232/hardware.json`  
  Key: `runtime_plan.batch_size`  
  Observed: `1`
- File: [`/home/cantlor/uzcosmos/my_project/module_net_train/net_train/models/unet_multitask.py`](./net_train/models/unet_multitask.py)  
  Symbol: `_norm_layer`  
  Lines: L17-L23  
  Why: default нормализация — BatchNorm.
- File: [`/home/cantlor/uzcosmos/my_project/module_net_train/scripts/02_train.py`](./scripts/02_train.py)  
  Symbol: `main`  
  Lines: L233-L238  
  Why: код прямо предупреждает о риске BN при batch<4.

2. **Reproducibility across workers не полностью контролируется.**  
Влияние: run-to-run variance при тех же seed/config.  
Evidence:
- File: [`/home/cantlor/uzcosmos/my_project/module_net_train/net_train/utils/seed.py`](./net_train/utils/seed.py)  
  Symbol: `seed_everything`  
  Lines: L14-L28  
  Why: global seeds задаются, но worker-specific seeding отсутствует.
- File: [`/home/cantlor/uzcosmos/my_project/module_net_train/scripts/02_train.py`](./scripts/02_train.py)  
  Symbol: `_make_loader`  
  Lines: L33-L45  
  Why: DataLoader создаётся без `worker_init_fn`/`generator`.

3. **Fallback `best.pt -> last.pt` в eval/predict выполняется без явного alert о смене режима.**  
Влияние: можно незаметно оценивать/деплоить не лучшую модель.  
Evidence:
- File: [`/home/cantlor/uzcosmos/my_project/module_net_train/scripts/03_predict_aoi.py`](./scripts/03_predict_aoi.py)  
  Symbol: `main`  
  Lines: L86-L90  
- File: [`/home/cantlor/uzcosmos/my_project/module_net_train/scripts/04_eval.py`](./scripts/04_eval.py)  
  Symbol: `main`  
  Lines: L69-L73

4. **Часть config-полей сейчас не wired в runtime.**  
Влияние: риск ложного ожидания поведения.  
Evidence:
- File: [`/home/cantlor/uzcosmos/my_project/module_net_train/configs/train_config.yaml`](./configs/train_config.yaml)  
  Sections: `dataset.mode` L15, `multi_dataset` L170-L175, `model.out_heads` L69-L71, `metrics.*.compute_*` L99-L107, `train.val_every_epochs` L135  
- Repo usage search: отсутствуют runtime references в `net_train/*` и `scripts/*` для этих ключей.

5. **`valid` маска в dataset санитизируется через `(valid > 0)` без fail-fast по unexpected values.**  
Влияние: тихое “исправление” битых масок может скрыть upstream проблемы качества.  
Evidence:
- File: [`/home/cantlor/uzcosmos/my_project/module_net_train/net_train/data/dataset.py`](./net_train/data/dataset.py)  
  Symbol: `_read_quad`  
  Lines: L113-L117.

### Nice-to-have

1. `boundary_metrics_multi_threshold` экспортируется, но не используется напрямую в loop (лишняя ветка API).  
Evidence: [`net_train/metrics/boundary_metrics.py`](./net_train/metrics/boundary_metrics.py), L71-L91; usage search.

2. `net_train/paths.py` фактически почти не участвует в scripts wiring (низкий ROI слоя).  
Evidence: [`net_train/paths.py`](./net_train/paths.py), L7-L18 + usage search.

3. Scripts используют `sys.path.insert(...)` вместо package-entrypoints.  
Evidence: [`scripts/01_check_prep_data.py`](./scripts/01_check_prep_data.py), L13-L16; [`scripts/02_train.py`](./scripts/02_train.py), L11-L14.

---

## 5. Boundary-specific analysis

### Boundary head: текущая реализация
- Target semantics: `boundary_bwbl` с `ignore_value=2` ([`configs/train_config.yaml`](./configs/train_config.yaml), L35-L37, L88-L95).
- Loss: masked BCE + optional `pos_weight` (в т.ч. `auto`) + optional focal gamma ([`net_train/losses/bwbl_loss.py`](./net_train/losses/bwbl_loss.py), L39-L77).
- Метрика: dilated boundary F1 по threshold list ([`net_train/metrics/boundary_metrics.py`](./net_train/metrics/boundary_metrics.py), L21-L67).
- Crop strategy: attempts + минимальные boundary pixels ([`net_train/data/transforms.py`](./net_train/data/transforms.py), L24-L27, L40-L55).

### Проверка 7 гипотез (verdict)

| Hypothesis | Verdict | Evidence |
|---|---|---|
| 1) Boundary слабеет из-за masked BCE + imbalance + no pos_weight/focal/dice | **Partially confirmed** | Сейчас в коде `pos_weight=auto` и optional focal есть ([`bwbl_loss.py`](./net_train/losses/bwbl_loss.py), L57-L70; config L92-L94), но Dice-компоненты нет. В historical run `config_resolved.yaml` не содержал `pos_weight`/`focal`. |
| 2) `random_crop()` наивный и часто без boundary-positive | **Partially confirmed (historical), mostly addressed now** | Теперь есть `attempts/min_boundary_pixels` ([`transforms.py`](./net_train/data/transforms.py), L24-L27, L40-L55; [`scripts/02_train.py`](./scripts/02_train.py), L174-L201). В historical run `config_resolved.yaml` имел только `sampling.crop_size` без crop_policy. |
| 3) Best checkpoint по `@0.50` неудачен | **Confirmed for analyzed run** | Artifact `config_resolved.yaml` L93-L97 + `history.csv`/`eval_test.json` разрыв между `@0.50` и `@0.15`. |
| 4) `04_eval.py` может читать исходный config вместо resolved | **Not confirmed (for current code)** | [`config.py`](./net_train/config.py), `resolve_run_train_config_path`, L119-L128; [`scripts/04_eval.py`](./scripts/04_eval.py), L52-L63; artifact `eval_test.json` key `config_used` указывает `run_dir/config_resolved.yaml`. |
| 5) BatchNorm плох при маленьком batch | **Confirmed as risk** | Model default BN ([`unet_multitask.py`](./net_train/models/unet_multitask.py), L17-L23), runtime artifact `batch_size=1`, и предупреждение в [`scripts/02_train.py`](./scripts/02_train.py), L233-L238. |
| 6) Недостаточно checks на shape/contract/parity | **Partially confirmed** | Shape/contract checks есть ([`dataset.py`](./net_train/data/dataset.py), L125-L137, L207-L213; `01_check_prep_data.py`), parity по config_resolved есть (`03/04`). Но reproducibility across workers не fully covered. |
| 7) Pipeline рабочий baseline, но с архитектурными лимитами для boundary | **Confirmed** | Run artifacts демонстрируют рабочий train/eval/infer pipeline и низкий boundary F1 на operational threshold 0.50. |

### Почему boundary F1 проседает в baseline run
1. Selection criterion в том run фиксировал `@0.50`, при этом максимум качества был на `0.15`.  
2. Для BWBL-позитивов типично высокий class imbalance; BCE даже с masking может давать слабый сигнал без устойчивой калибровки порога.  
3. Runtime `batch_size=1` + BatchNorm усиливает noise по границе.  
4. Historical run не включал явный `crop_policy` в resolved config, значит boundary-aware crop мог не применяться.

---

## 6. Train / Eval / Infer parity

### Parity check matrix

| Aspect | Train | Eval | Infer | Status |
|---|---|---|---|---|
| Config source | CLI config → write `config_resolved.yaml` | Prefer `run_dir/config_resolved.yaml` | Prefer `run_dir/config_resolved.yaml` | OK |
| Normalization stats | Compute + save `band_stats.npz` | Load same `band_stats.npz` | Load same `band_stats.npz` | OK |
| Input channels | Contract enforced (`in_channels` override) | Contract enforced (`in_channels` override) | Contract enforced (`in_channels` override) | OK |
| NoData/valid policy | valid from file (fallback compute), enforce ignore in targets | same dataset policy on test | valid computed from AOI chip; suppression before blending | OK |
| Checkpoint usage | save best/last by monitor | load best fallback last | load best fallback last | OK with risk (silent fallback) |

Evidence:
- [`/home/cantlor/uzcosmos/my_project/module_net_train/scripts/02_train.py`](./scripts/02_train.py), L89, L127-L141, L157-L170  
- [`/home/cantlor/uzcosmos/my_project/module_net_train/scripts/04_eval.py`](./scripts/04_eval.py), L52-L67, L69-L73  
- [`/home/cantlor/uzcosmos/my_project/module_net_train/scripts/03_predict_aoi.py`](./scripts/03_predict_aoi.py), L43-L56, L86-L99  
- [`/home/cantlor/uzcosmos/my_project/module_net_train/net_train/infer/predict_aoi.py`](./net_train/infer/predict_aoi.py), L147-L151, L255-L257

### Дополнительная проверка contract targets
- Реально используются: `extent`, `boundary_bwbl`, `valid`, `img`, `meta`.  
- Не используются в обучении: `extent_ig`, `boundary_raw` (aux/legacy).

---

## 7. Reproducibility and runtime risks

### 7.1 Reproducibility
- Seed задаётся глобально, но `deterministic=false` по умолчанию ([`configs/train_config.yaml`](./configs/train_config.yaml), L109-L111; [`net_train/utils/seed.py`](./net_train/utils/seed.py), L14-L28).  
- DataLoader создаётся без `worker_init_fn`/`generator`; при `num_workers>0` это источник вариативности.  
- AMP/bf16 зависит от hardware plan ([`net_train/hardware.py`](./net_train/hardware.py), L70-L93; artifact `hardware.json`: bf16 enabled).

### 7.2 Silent failure risks (без traceback, но с деградацией)
1. `best.pt` отсутствует → eval/predict тихо переходят на `last.pt`.  
2. `valid` с некорректными значениями (>1) в dataset приводится к binary `(>0)` без жёсткой ошибки.  
3. Если boundary-aware crop constraints не выполняются, `random_crop` возвращает “best available”, но не гарантирует достаточный boundary signal.  
4. Legacy AOI manifest fallback может скрывать несогласованность с новым strict manifest contract.  
5. При `BatchNorm + batch=1` модель формально обучается, но boundary head может системно терять качество.

### 7.3 Legacy / auxiliary / currently-unused

| Category | Item | Status | Evidence |
|---|---|---|---|
| Auxiliary target | `extent_ig` | currently-unused in net_train | data index expects only `extent` ([`data/index.py`](./net_train/data/index.py), L30-L36) |
| Auxiliary target | `boundary_raw` | currently-unused in net_train | data index/dataset use only `boundary_bwbl` |
| Legacy path | `aoi_rasters_manifest.json` | supported as fallback | [`config.py`](./net_train/config.py), L115-L117; [`infer/predict_aoi.py`](./net_train/infer/predict_aoi.py), L64-L69 |
| Config field | `dataset.mode` (`single|multi`) | declared, not wired | [`configs/train_config.yaml`](./configs/train_config.yaml), L15; no runtime usage |
| Config section | `multi_dataset` | declared, not wired | [`configs/train_config.yaml`](./configs/train_config.yaml), L170-L175; no runtime usage |
| Config field | `model.out_heads` | declared, not used by model builder | only in config, no code references |
| Config field | `metrics.extent.compute_iou/compute_f1` | declared, not gating behavior | only in config, no code checks |
| Config field | `metrics.boundary.compute_f1` | declared, not gating behavior | only in config, no code checks |
| Config field | `train.val_every_epochs` | declared, not used in loop control | config L135, no usage in loop |
| Config field | `paths.prep_work_root` | declared, not used in runtime wiring | only config mention |
| Logic branch | `boundary_metrics_multi_threshold()` | implemented/exported, not used by loop | [`metrics/boundary_metrics.py`](./net_train/metrics/boundary_metrics.py), L71-L91 |

---

## 8. Optimization opportunities (no code change proposals, only hypotheses)

### Low-effort
1. Проверить, что каждый новый run действительно пишет `train.checkpoint.monitor=val/boundary_f1_max` в `config_resolved.yaml`.  
2. Сравнивать `eval_test.json` на нескольких threshold и фиксировать operational threshold отдельно от monitor threshold.  
3. Ввести обязательную проверку: если fallback на `last.pt`, логировать это как warning в eval/predict pipeline report.

### Medium-effort
1. Закрепить reproducibility policy для DataLoader workers (инициализация seed per worker + deterministic profile для экспериментов сравнения).  
2. Собирать calibration diagnostics для boundary head (F1/precision/recall по threshold grid + reliability trend).  
3. Проверять влияние `instancenorm` vs `batchnorm` при batch=1 на boundary branch.

### High-effort
1. Перейти от одиночной boundary BCE к loss-композиции, устойчивой к extreme imbalance (если подтвердится на серии runs).  
2. Добавить end-to-end quality proxy ближе к polygon quality (не только pixel F1), чтобы избежать оптимизации по суррогату.

---

## 9. What information ChatGPT should see next

Чтобы внешний ревьюер не перечитывал весь репозиторий, дайте ему этот минимальный пакет:

### 9.1 Обязательные файлы
1. `/home/cantlor/uzcosmos/my_project/module_net_train/configs/train_config.yaml`
2. `/home/cantlor/uzcosmos/my_project/module_net_train/net_train/data/dataset.py`
3. `/home/cantlor/uzcosmos/my_project/module_net_train/net_train/losses/bwbl_loss.py`
4. `/home/cantlor/uzcosmos/my_project/module_net_train/net_train/train/loop.py`
5. `/home/cantlor/uzcosmos/my_project/module_net_train/net_train/infer/predict_aoi.py`
6. `/home/cantlor/uzcosmos/my_project/module_net_train/scripts/02_train.py`
7. `/home/cantlor/uzcosmos/my_project/module_net_train/scripts/04_eval.py`

### 9.2 Обязательные run-артефакты
1. `/home/cantlor/uzcosmos/my_project/output_data/module_net_train/runs/20260312_131232/config_resolved.yaml`
2. `/home/cantlor/uzcosmos/my_project/output_data/module_net_train/runs/20260312_131232/hardware.json`
3. `/home/cantlor/uzcosmos/my_project/output_data/module_net_train/runs/20260312_131232/metrics/history.csv`
4. `/home/cantlor/uzcosmos/my_project/output_data/module_net_train/runs/20260312_131232/metrics/eval_test.json`
5. `/home/cantlor/uzcosmos/my_project/output_data/module_net_train/runs/20260312_131232/logs/train.log`
6. `/home/cantlor/uzcosmos/my_project/output_data/module_net_train/runs/20260312_131232/pred/first_raster/predict_manifest.json`

### 9.3 Read-only команды для внешнего ревьюера
```bash
cd /home/cantlor/uzcosmos/my_project/module_net_train

# 1) Проверить monitor и boundary thresholds в resolved run config
nl -ba /home/cantlor/uzcosmos/my_project/output_data/module_net_train/runs/20260312_131232/config_resolved.yaml | sed -n '85,110p'

# 2) Проверить динамику boundary на разных threshold
head -n 3 /home/cantlor/uzcosmos/my_project/output_data/module_net_train/runs/20260312_131232/metrics/history.csv
tail -n 5 /home/cantlor/uzcosmos/my_project/output_data/module_net_train/runs/20260312_131232/metrics/history.csv

# 3) Проверить финальные test метрики
nl -ba /home/cantlor/uzcosmos/my_project/output_data/module_net_train/runs/20260312_131232/metrics/eval_test.json

# 4) Проверить parity config_used/checkpoint в eval/predict
nl -ba /home/cantlor/uzcosmos/my_project/output_data/module_net_train/runs/20260312_131232/pred/first_raster/predict_manifest.json

# 5) Проверить ключевые контракты в коде
nl -ba net_train/data/dataset.py | sed -n '180,230p'
nl -ba net_train/losses/bwbl_loss.py
nl -ba net_train/train/loop.py | sed -n '240,280p'
nl -ba net_train/infer/predict_aoi.py | sed -n '248,262p'
```

### 9.4 Чего не хватает для следующего этапа
1. Несколько run-артефактов (а не один) для статистически честного сравнения гипотез.
2. Отдельный отчёт по постпроцессингу в полигон (если он в другом модуле) для связи pixel-boundary и polygon quality.
3. Срезы примеров предсказаний boundary (визуальные QC) для диагностики calibration vs structure errors.

---

## 10. Final verdict

`module_net_train` в текущем состоянии — **рабочий baseline модуль**, архитектурно уже достаточно зрелый для e2e train/eval/infer и совместим с контрактом `module_prep_data` (9-channel input, ignore semantics, NoData suppression на инференсе).

Главный ограничитель качества на наблюдаемом run — не поломка пайплайна, а сочетание boundary-specific факторов: сильная threshold-чувствительность, historical checkpoint monitor по `@0.50`, small-batch BatchNorm и неполная reproducibility-политика для мультиворкера.

Текущее состояние кода уже частично адресует исторические риски (crop policy, `boundary_f1_max`, run-resolved config parity). Следующий этап анализа стоит фокусировать не на “переписывании”, а на подтверждении улучшений по серии runs и стабилизации boundary calibration/monitoring.

