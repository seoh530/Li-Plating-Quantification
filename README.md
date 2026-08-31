# QCML Photo Jig Vision

Battery Photo Jig 이미지의 QC, ML 마스킹, 라벨링, 모델 학습, 결과 분석을 위한 Tkinter 기반 도구입니다.

## 기본 실행

메인 앱 실행:

```powershell
python main.py
```

설정 파일은 기본적으로 `config/thresholds.yaml`을 사용합니다.

## 일반 작업 순서

1. `main.py` 실행 후 Workflow 탭에서 이미지 QC/ML 처리
2. 라벨 데이터가 필요하면 `tools/label_tools.py`로 픽셀 라벨링
3. 새 모델을 만들 때는 `train/train_model.py` 또는 알고리즘별 학습 스크립트 사용
4. 모델 생성 후 `data/test` 이미지가 있으면 자동으로 마스크와 퍼센트 저장
5. 저장된 모델을 나중에 다시 테스트할 때는 `tools/test_model.py` 사용
6. 테스트 모델 결과 비교는 `test_dashboard.py` 사용
7. 운영 결과 추세 확인은 Analysis 탭 또는 `monitoring_app.py` 사용

## 자주 쓰는 파일

### `main.py`

Tkinter 메인 앱 진입점입니다.

언제 사용:
- QC/ML Workflow 탭을 실행할 때
- Analysis 탭에서 결과를 확인할 때

실행:

```powershell
python main.py
```

### `tools/label_tools.py`

학습용 픽셀 라벨 CSV를 만드는 도구입니다. `data/labeled/<AnalysisID>/Cell #N/FRONT` 또는 `BACK` 폴더의 이미지를 선택해서 `b/l/n` 라벨을 찍습니다.

새로 만든 CSV에는 라벨링에 필요한 기본 컬럼만 저장됩니다.

저장 컬럼:
- `filename`
- `x`
- `y`
- `label`

Enhanced feature는 라벨링이 끝난 뒤 `train/train_model.py`로 별도 생성합니다.

언제 사용:
- 새 학습 데이터를 직접 라벨링할 때
- 기존에 라벨링한 CSV가 있는 폴더를 피하고 싶을 때

실행:

```powershell
python tools\label_tools.py
```

이미 CSV가 있는 폴더는 선택 목록에서 `completed`가 번호 앞에 표시됩니다.

### `train/train_model.py`

Unified training entrypoint. This is the only training command operators should use.
It trains LightGBM, CatBoost, XGBoost, Random Forest, or Logistic Regression from enhanced CSVs.

Run:

```powershell
python train\train_model.py
```

Common options:

```powershell
python train\train_model.py --model lightgbm --model-name my_model
python train\train_model.py --model catboost --base-model logical_lightgbm_1
python train\train_model.py --model xgboost --base-model logical_lightgbm_1 --add-csv-indices 1,3
python train\train_model.py --skip-test-masks
```

Behavior:
- Uses enhanced features only.
- Generates missing `enhanced_*.csv` files automatically from label CSVs.
- Can reuse an existing model's `source_csvs.txt` and show only additional CSV candidates.
- Saves models under `models/<model-name>/`.
- Writes `metadata.json`, `report.txt`, `source_csvs.txt`, and `feature_importance.csv`.
- Runs `data/test` masks automatically unless `--skip-test-masks` is passed.

Internal modules under `train/`:
- `train_lightGBM.py`: LightGBM trainer implementation.
- `train_classical_model.py`: CatBoost/XGBoost/Random Forest/Logistic Regression implementation.
- `enhance_training_features.py`: enhanced CSV generation.
- `training_utils.py`: split/metrics utilities.

### `tools/test_model.py`

저장된 모델을 선택해서 `data/test` 안의 이미지를 다시 마스킹하고 리튬염 퍼센트를 계산하는 도구입니다.

언제 사용:
- 이미 만든 모델을 같은 test 이미지로 다시 비교할 때
- 새 test 이미지를 넣고 기존 모델 결과를 확인할 때
- 모델 학습 없이 마스크와 `test_results.csv`만 다시 만들 때

실행:

```powershell
python tools\test_model.py
```

모델 번호 입력에서 여러 모델을 쉼표로 선택할 수 있습니다.

```text
Select model(s) [1-5], example 1,2,3: 1,2,3
```

실행하면 `models/` 안의 모델 폴더가 번호로 표시되고, 선택한 모델로 `data/test` 이미지를 처리합니다. 이미 `data/test/<model-name>` 안에 해당 이미지의 마스크가 있으면 재처리하지 않고, 기존 결과가 없는 이미지만 새로 추론해서 추가 저장합니다.

직접 모델명 지정:

```powershell
python tools\test_model.py --model-name my_model
python tools\test_model.py --model-name model_a,model_b,model_c
```

저장 위치:
- 마스크 이미지: `data/test/<model-name>/*_mask.png`
- 결과 CSV: `data/test/<model-name>/test_results.csv`

`test_results.csv`에는 파일별 pixel count와 `background_percent`, `nonactive_percent`, `li_salt_percent`, `edge_percent`가 포함됩니다.

### `test_dashboard.py`

`tools/test_model.py`와 학습 스크립트들이 만든 기존 결과를 읽어서 모델별 테스트 결과를 비교하는 Streamlit 대시보드입니다. 기존 출력 구조는 변경하지 않고 `data/test/<model-name>/test_results.csv`와 마스크 이미지만 읽습니다.

언제 사용:
- 같은 test 이미지에 대해 여러 모델의 마스크와 퍼센트를 비교할 때
- 모델별 Lithium Salt/Background/Nonactive 차이를 볼 때
- 이미지별/모델별 variability와 disagreement를 확인할 때

실행:

```powershell
streamlit run test_dashboard.py
```

주요 기능:
- 모델 입력 원본 이미지와 모델별 prediction mask 비교
- Lithium Salt, Background, Nonactive grouped bar chart
- 이미지별/모델별 variability chart와 heatmap
- pairwise model difference heatmap
- class mask 복원이 가능한 경우 pixel-level disagreement 분석
- long-format result table과 CSV 다운로드

### `tools/compute_reference.py`

색상 정규화와 drift 기준값을 만들 때 쓰는 보조 도구입니다.

언제 사용:
- reference 이미지 기준으로 background/inactive 색상 통계를 다시 잡을 때
- `baseline_stats.json` 또는 reference mean/std를 갱신해야 할 때

### `monitoring_app.py`

SQLite 로그를 읽는 Streamlit 모니터링 대시보드입니다. QC/ML을 실행하지 않고 DB만 읽습니다.

실행:

```powershell
streamlit run monitoring_app.py
```

## GUI 파일

### `gui/workflow_tab.py`

메인 Workflow 탭입니다. QC, ML 추론, 마스크 저장, 결과 CSV 생성, SQLite logging이 연결되는 핵심 화면입니다.

언제 수정:
- 실제 운영 플로우 버튼/테이블/상태 표시를 바꿀 때
- QC 후 ML 처리 흐름을 바꿀 때

### `gui/analysis_tab.py`

결과 CSV와 마스크 이미지를 불러와 분석 그래프와 이미지 미리보기를 보여주는 탭입니다.

언제 수정:
- 결과 비교 방식, 그래프, heatmap, 이미지 리뷰 UI를 바꿀 때

### `gui/qc_tab.py`, `gui/ml_tab.py`

이전/개별 탭 기반 QC, ML UI입니다. 현재 메인 앱은 `WorkflowTab`과 `AnalysisTab` 중심입니다.

## 핵심 모듈

### `modules/qc/crop_engine.py`

원본 이미지를 QC-ready crop 이미지로 만드는 엔진입니다.

### `modules/qc/folder_qc.py`

AnalysisID 폴더 구조, 이미지 개수, FRONT/BACK 구성 등 폴더 단위 QC를 담당합니다.

### `modules/qc/color_normalizer.py`

QC 단계에서 reference 기준 색상 정규화를 수행합니다.

### `modules/ml/feature_extractor.py`

픽셀 단위 feature를 만드는 핵심 모듈입니다. `basic`, `enhanced` 학습과 추론 feature가 여기서 나옵니다.

### `modules/ml/inference.py`

저장된 LightGBM 모델을 로드하고 이미지 feature를 label mask로 예측합니다. edge/nonactive 후처리도 여기서 적용됩니다.

### `modules/ml/quantifier.py`

label mask를 색상 마스크 PNG로 저장하고, li-salt/background/nonactive 비율을 계산합니다.

### `modules/ml/background_normalizer.py`, `modules/ml/drift_detector.py`

추론 중 배경 색상 drift를 감지하고 필요하면 LAB 기반 보정을 적용하는 모듈입니다.

### `modules/storage/sqlite_logger.py`

Workflow 실행 기록과 예측 결과를 SQLite DB에 저장합니다.

## 데이터 폴더

- `data/labeled/`: 라벨링 이미지와 학습 CSV
- `data/test/`: 새 모델 생성 시 자동 마스킹할 테스트 이미지
- `models/`: 학습된 모델 저장 위치
- `config/thresholds.yaml`: QC, ML, feature extraction, 저장 경로 설정

## 추천 사용 예

새 라벨링부터 enhanced 모델 학습과 테스트까지:

```powershell
python tools\label_tools.py
python train\train_model.py
python train\train_model.py --model-name enhanced_test
python tools\test_model.py --model-name enhanced_test
```

구형 CSV를 enhanced로 변환 후 학습:

```powershell
python train\train_model.py
python train\train_model.py --model-name enhanced_from_old_csv
python tools\test_model.py --model-name enhanced_from_old_csv
```

basic/enhanced 비교 실험:

```powershell
python train\train_model.py --model-name basic_color
python train\train_model.py --model-name enhanced_color
python train\train_model.py --model logistic_regression --model-name logistic_regression_color
python train\train_model.py --model random_forest --model-name random_forest_color
python train\train_model.py --model xgboost --model-name xgboost_color
python train\train_model.py --model catboost --model-name catboost_color
python tools\test_model.py --model-name basic_color
python tools\test_model.py --model-name enhanced_color
python tools\test_model.py --model-name logistic_regression_color,random_forest_color,xgboost_color,catboost_color
```
