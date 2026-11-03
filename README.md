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
2. 배포 PC에서 PhotoJig 저장 위치가 다르면 Settings 탭에서 root path 저장
3. 라벨 데이터가 필요하면 `tools/label_tools.py`로 픽셀 라벨링
4. 새 모델을 만들 때는 `tools/train_lightGBM.py` 또는 알고리즘별 학습 스크립트 사용
5. 모델 생성 후 `data/test` 이미지가 있으면 자동으로 마스크와 퍼센트 저장
6. 저장된 모델을 나중에 다시 테스트할 때는 `tools/test_model.py` 사용
7. 테스트 모델 결과 비교는 `test_dashboard.py` 사용
8. 운영 결과 추세 확인은 Analysis 탭 또는 `monitoring_app.py` 사용

## 자주 쓰는 파일

### `main.py`

Tkinter 메인 앱 진입점입니다.

언제 사용:
- QC/ML Workflow 탭을 실행할 때
- Analysis 탭에서 결과를 확인할 때
- Settings 탭에서 PhotoJig root path를 저장할 때

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

Enhanced feature는 라벨링이 끝난 뒤 `tools/enhance_training_features.py`로 별도 생성합니다.

언제 사용:
- 새 학습 데이터를 직접 라벨링할 때
- 기존에 라벨링한 CSV가 있는 폴더를 피하고 싶을 때

실행:

```powershell
python tools\label_tools.py
```

이미 CSV가 있는 폴더는 선택 목록에서 `completed`가 번호 앞에 표시됩니다.

### `tools/enhance_training_features.py`

기본 라벨 CSV에 enhanced feature를 추가해서 `enhanced_<원본>.csv`를 만드는 후처리 도구입니다.

언제 사용:
- `label_tools.py`로 만든 기본 CSV를 enhanced 학습에 쓰고 싶을 때
- 기존 CSV를 `--feature-set enhanced` 학습에 쓰고 싶을 때

실행:

```powershell
python tools\enhance_training_features.py
```

`--feature-set enhanced` 학습은 `enhanced_*.csv` 파일만 사용합니다. 새 `label_tools.py`로 만든 CSV를 enhanced 학습에 쓰려면 먼저 `tools/enhance_training_features.py`로 `enhanced_<원본>.csv`를 만드세요.

### 모델 학습 스크립트

학습 CSV로 여러 알고리즘 모델을 만들 수 있습니다. LightGBM 학습은 `tools/train_lightGBM.py`가 담당합니다.

`tools/train_model_LAB.py`는 `tools/train_lightGBM.py`와 같은 동작을 하던 호환용 이름이라 제거했습니다. 앞으로는 LightGBM 학습 시 `tools/train_lightGBM.py`만 사용하세요.

`tools/train_lightGBM.py`가 다른 알고리즘 실행 파일보다 긴 이유는 CSV 선택, feature set 처리, 모델 폴더 생성, test mask/result CSV 생성 같은 공통 학습 유틸이 이 파일에 많이 들어 있기 때문입니다. `train_logistic_regression.py`, `train_random_forest.py`, `train_xgboost.py`, `train_catboost.py`는 `tools/train_classical_model.py`의 공통 로직을 호출하는 짧은 wrapper입니다. 출력 구조는 모델 파일 확장자만 다르고 대시보드 입력은 동일합니다.

지원 feature set:
- `basic`: `r,g,b,h,s,v,l_lab,a_lab,b_lab`
- `enhanced`: `basic + x_norm,y_norm + residual/robust/edge feature`

`--feature-set enhanced`는 `enhanced_*.csv` 파일만 학습 입력으로 사용합니다.

실행:

```powershell
python tools\train_lightGBM.py
```

실행하면 다음 순서로 선택합니다.

1. `basic` 또는 `enhanced`
2. `all` 또는 `custom`
3. `custom`이면 CSV 번호 입력, 예: `1,2,4`
4. 모델명 입력

직접 지정:

```powershell
python tools\train_lightGBM.py --feature-set enhanced --model-name my_model
```

custom CSV 지정:

```powershell
python tools\train_lightGBM.py --feature-set enhanced --csv-selection custom --csv-indices 1,2,4 --model-name my_model
```

다른 알고리즘 학습:

```powershell
python tools\train_logistic_regression.py --feature-set enhanced --model-name logistic_regression_test
python tools\train_random_forest.py --feature-set enhanced --model-name random_forest_test
python tools\train_xgboost.py --feature-set enhanced --model-name xgboost_test
python tools\train_catboost.py --feature-set enhanced --model-name catboost_test
```

저장 위치:
- 모델: `models/<model-name>/`
- 테스트 마스크: `data/test/<model-name>/`

모델 폴더에 저장되는 파일:
- `model.txt`: LightGBM 모델
- `model.joblib`: Logistic Regression, Random Forest, XGBoost, CatBoost 모델
- `metadata.json`: feature set, feature 컬럼, 사용 CSV 목록
- `report.txt`: validation 성능 리포트
- `source_csvs.txt`: 학습에 사용한 CSV 목록
- `feature_importance.csv`: feature importance

`data/test` 안에 테스트 이미지가 있으면 모델 생성 직후 자동으로 마스킹 이미지와 `test_results.csv`가 저장됩니다. `test_results.csv`에는 파일별 pixel count와 `background_percent`, `nonactive_percent`, `li_salt_percent`, `edge_percent`가 포함됩니다.

학습만 하고 테스트를 건너뛰고 싶을 때:

```powershell
python tools\train_lightGBM.py --feature-set enhanced --model-name my_model --skip-test-masks
```

### `tools/train_model.py`

구버전 학습 스크립트입니다. RGB/HSV 6개 feature만 사용합니다.

현재는 주요 사용 대상이 아닙니다. 새 모델은 `tools/train_lightGBM.py` 또는 알고리즘별 학습 스크립트를 사용하세요.

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

결과 CSV와 마스크 이미지를 불러와 분석 그래프와 이미지 미리보기를 보여주는 탭입니다. 선택 결과 또는 현재 날짜 전체 결과를 CSV/Excel로 내보내고, 현재 Analysis 화면을 PNG로 캡처할 수 있습니다.

언제 수정:
- 결과 비교 방식, 그래프, heatmap, 이미지 리뷰 UI를 바꿀 때
- export 또는 화면 캡처 방식을 바꿀 때

### `gui/settings_tab.py`

배포 후 운영자가 root path를 저장하는 탭입니다. 저장된 값은 `config/thresholds.yaml`에 남고, Workflow/Analysis/Guide 탭이 즉시 새 root path를 다시 읽습니다.

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

Workflow 실행 기록과 예측 결과는 Python 내장 `sqlite3`로 사용자별 로컬 DB에 저장합니다. 별도 SQLite 프로그램은 필요하지 않으며 DB 경로는 `%LOCALAPPDATA%/Li-Plating_Quant_System/photo_jig_vision.db`입니다.

## 데이터 폴더

- `data/labeled/`: 라벨링 이미지와 학습 CSV
- `data/test/`: 새 모델 생성 시 자동 마스킹할 테스트 이미지
- `models/`: 학습된 모델 저장 위치
- `config/thresholds.yaml`: QC, ML, feature extraction, 저장 경로 설정

## 추천 사용 예

새 라벨링부터 enhanced 모델 학습과 테스트까지:

```powershell
python tools\label_tools.py
python tools\enhance_training_features.py
python tools\train_lightGBM.py --feature-set enhanced --model-name enhanced_test
python tools\test_model.py --model-name enhanced_test
```

구형 CSV를 enhanced로 변환 후 학습:

```powershell
python tools\enhance_training_features.py
python tools\train_lightGBM.py --feature-set enhanced --model-name enhanced_from_old_csv
python tools\test_model.py --model-name enhanced_from_old_csv
```

basic/enhanced 비교 실험:

```powershell
python tools\train_lightGBM.py --feature-set basic --model-name basic_color
python tools\train_lightGBM.py --feature-set enhanced --model-name enhanced_color
python tools\train_logistic_regression.py --feature-set enhanced --model-name logistic_regression_color
python tools\train_random_forest.py --feature-set enhanced --model-name random_forest_color
python tools\train_xgboost.py --feature-set enhanced --model-name xgboost_color
python tools\train_catboost.py --feature-set enhanced --model-name catboost_color
python tools\test_model.py --model-name basic_color
python tools\test_model.py --model-name enhanced_color
python tools\test_model.py --model-name logistic_regression_color,random_forest_color,xgboost_color,catboost_color
```
