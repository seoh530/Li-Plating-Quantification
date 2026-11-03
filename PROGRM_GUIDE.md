# QCML main.py Program Guide

이 문서는 `python main.py`로 실행되는 데스크톱 프로그램을 처음 쓰는 사람도 전체 기능을 이해할 수 있도록 정리한 사용 가이드입니다.

`main.py`는 모델 학습 도구가 아니라, 이미 준비된 모델을 이용해 PhotoJig 이미지의 QC, crop, ML 마스킹, 리튬염 비율 계산, 결과 확인을 실행하는 메인 화면입니다.

## 1. 실행 전 확인

### 실행 위치

항상 프로젝트 루트에서 실행합니다.

```powershell
python main.py
```

프로젝트 루트 예시:

```text
QCML_3_develop/
  main.py
  config/thresholds.yaml
  models/
  gui/
  modules/
```

### 설정 파일

`main.py`는 기본으로 아래 설정 파일을 사용합니다.

```text
config/thresholds.yaml
```

가장 먼저 확인할 값:

```yaml
app:
  root_path: "C:/Users/SKBA_FA/Documents/VIISAN OfficeCam"

ml:
  model_path: "models/lightgbm"
```

- `app.root_path`: PhotoJig 원본 이미지가 저장되는 최상위 폴더입니다. main.py 실행 후 Settings 탭에서 바꿀 수 있고, 저장한 값은 다시 변경하기 전까지 유지됩니다.
- `ml.model_path`: 기본 ML 모델 위치입니다.
- Workflow 탭에서는 `models/` 안의 모델 목록을 선택할 수 있습니다.

## 2. main.py를 켜면 보이는 화면

프로그램을 실행하면 네 개의 탭이 있습니다.

```text
Workflow
Analysis
Settings
Guide
```

### Workflow 탭

실제 작업을 실행하는 메인 탭입니다.

여기서 할 수 있는 일:

- 날짜별 또는 Analysis ID별 작업 목록 보기
- Folder QC 실행
- crop 이미지 생성
- ML 마스킹 실행
- Lithium Salt %, Background, Nonactive 비율 계산
- 마스크 이미지 저장
- 결과 CSV 저장
- SQLite monitoring DB 저장
- QC 실패 이미지 확인
- retake 완료 처리
- QC 실패 이미지를 강제로 crop 처리

### Analysis 탭

이미 만들어진 ML 결과를 시각적으로 확인하는 탭입니다.

여기서 할 수 있는 일:

- 날짜별 ML 결과 CSV 탐색
- Cell별 Li-salt / Nonactive 요약 확인
- Layer별 Li-salt % 그래프 확인
- FRONT, BACK, TEST, FRONT + BACK, Average 보기 전환
- 선택한 layer의 crop 이미지와 mask 이미지 비교
- Li-salt / Nonactive heatmap 확인
- 현재 Analysis 화면 PNG 캡처
- 선택 결과 또는 현재 날짜 전체 결과를 CSV/Excel로 내보내기

### Settings 탭

배포 후 운영자가 바꿀 수 있는 설정 탭입니다.

여기서 할 수 있는 일:

- PhotoJig root path 입력 또는 Browse로 선택
- 설정 저장 후 Workflow, Analysis, Guide 탭 즉시 reload
- 현재 Windows 사용자 기준 SQLite DB 저장 위치 확인

## 3. 입력 폴더 구조

`main.py`는 `config/thresholds.yaml`의 `app.root_path` 아래에서 날짜 폴더와 AnalysisID 폴더를 찾습니다.

일반 제품 이미지 구조:

```text
<root_path>/
  2026-07-24/
    AHG12345/
      Cell #1/
        FRONT/
          image files...
        BACK/
          image files...
      Cell #2/
        FRONT/
        BACK/
```

허용되는 날짜 폴더 형식:

```text
YYYY-MM-DD
M-D-YYYY
M-D-YY
```

AnalysisID 형식:

```text
대문자 3개 + 숫자 5개
예: AHG06113
```

지원되는 고객 코드는 AnalysisID의 두 번째 글자로 판단합니다.

```text
H: HKMC
F: FORD
V: VW
E: ESS
S: Slate
T: Test
```

일반 제품의 FRONT/BACK 원본 이미지 개수 기준:

```text
HKMC: 30장
FORD: 38장
VW: 23장
ESS: 40장
Slate: 개수 제한 없음
Test: 개수 제한 없음
```

QC 후 ML에 들어가는 이미지는 보통 첫 이미지와 마지막 이미지를 제외한 crop 결과입니다.

## 4. TEST AnalysisID 구조

테스트용 AnalysisID는 아래 이름을 사용할 수 있습니다.

```text
TEST
TEST01
TEST02
...
```

TEST 구조는 FRONT/BACK 정규 구조가 없어도 됩니다.

예시:

```text
<root_path>/
  2026-07-24/
    TEST/
      sample_01.png
      sample_02.png
```

또는 여러 묶음으로 나눌 수 있습니다.

```text
<root_path>/
  2026-07-24/
    TEST/
      group_a/
        image files...
      group_b/
        image files...
```

TEST를 Workflow에서 실행하면 출력 폴더가 자동으로 만들어집니다.

```text
TEST/
  _pass/
  _mask/
```

여러 source folder가 있으면 아래처럼 만들어집니다.

```text
TEST/
  group_a/
  group_a_pass/
  group_a_mask/
  group_b/
  group_b_pass/
  group_b_mask/
```

## 5. Workflow 탭 사용법

### View

`View`는 작업 목록을 어떻게 볼지 정합니다.

```text
By Date
```

선택한 날짜 폴더 안의 AnalysisID만 보여줍니다.

```text
By Analysis ID
```

모든 날짜 폴더를 훑어서 AnalysisID 기준으로 보여줍니다.

### Date

`By Date` 모드에서 처리할 날짜를 선택합니다.

프로그램은 오늘 날짜 폴더가 있으면 우선 선택하고, 없으면 가장 최근 날짜 폴더를 선택합니다.

### Refresh

폴더 상태를 다시 스캔합니다.

사용할 때:

- 새 이미지가 들어왔을 때
- QC/ML 결과 파일을 수동으로 지웠거나 옮겼을 때
- 다른 프로그램에서 결과가 새로 생성되었을 때

### Model

ML에 사용할 모델을 선택합니다.

모델은 `models/` 아래에서 인식됩니다. 모델 폴더에는 보통 아래 파일이 있어야 합니다.

```text
model.txt
model.joblib
metadata.json
```

모델 학습은 `main.py`에서 하지 않습니다. 새 모델은 `tools/train_lightGBM.py` 또는 알고리즘별 학습 스크립트로 만듭니다.

## 6. Workflow 실행 버튼

### Run Selected QC

선택한 AnalysisID에 대해 QC와 crop만 실행합니다.

결과:

- FolderQC 통과 여부 확인
- FRONT/BACK 원본 이미지 crop
- `FRONT_pass`, `BACK_pass` 폴더 생성
- ML 마스킹은 실행하지 않음

사용할 때:

- 이미지 개수나 crop 상태만 먼저 확인하고 싶을 때
- ML 전에 QC 실패 이미지를 검토하고 싶을 때

### Run Selected ML

선택한 AnalysisID에 대해 ML만 실행합니다.

전제:

- QC가 이미 완료되어 있어야 합니다.
- `FRONT_pass`, `BACK_pass` 또는 TEST의 pass 폴더가 있어야 합니다.

결과:

- 선택 모델로 마스크 생성
- 리튬염/비활성/배경 비율 계산
- 결과 CSV 저장
- SQLite monitoring DB 저장

사용할 때:

- QC는 이미 끝났고 모델만 바꿔 다시 돌리고 싶을 때
- DB 저장이 누락된 결과를 다시 채우고 싶을 때

### Run Selected QC + ML

선택한 AnalysisID에 대해 QC부터 ML까지 순서대로 실행합니다.

가장 일반적인 실행 버튼입니다.

처리 순서:

1. FolderQC 실행
2. crop 이미지 생성
3. QC가 통과하면 ML 실행
4. mask 이미지 저장
5. 결과 CSV 저장
6. SQLite monitoring DB 저장

### Run Missing QC + ML

현재 목록에서 QC 또는 ML 결과가 없는 항목만 실행합니다.

사용할 때:

- 오늘 촬영된 것 중 아직 처리 안 된 것만 돌리고 싶을 때
- 중간에 프로그램을 멈췄다가 이어서 처리하고 싶을 때

### Force Selected QC + ML

FolderQC가 실패해도 기존 FRONT/BACK 이미지를 사용해서 crop과 ML을 강제로 실행합니다.

주의:

- 이미지 개수나 폴더 구조가 기준과 맞지 않아도 진행할 수 있습니다.
- 정상 운영 결과로 쓰기 전에 결과 이미지를 꼭 확인해야 합니다.
- QC 기준을 무시하는 기능이므로 테스트나 예외 처리 상황에서만 사용하세요.

### Stop

실행 중인 작업을 중단합니다.

즉시 끊는 방식이 아니라 현재 처리 중인 AnalysisID 또는 이미지가 끝난 뒤 멈춥니다.

## 7. Workflow 목록 컬럼 의미

Workflow 중앙 테이블에는 아래 컬럼이 표시됩니다.

### Date

촬영 날짜 폴더명입니다.

### Analysis ID

처리 대상 AnalysisID입니다.

### QC

QC 상태입니다.

```text
DONE
```

QC가 끝났고 pass 이미지가 준비되어 있습니다.

```text
NEEDED
```

QC가 아직 필요하거나 pass 이미지가 없습니다.

```text
HOLD
```

QC 중 실패 이미지가 있어 사용자가 확인해야 합니다.

### ML / DB

ML 결과와 SQLite DB 저장 상태입니다.

```text
DONE
```

ML CSV와 SQLite DB row가 모두 있습니다.

```text
ML missing
```

QC는 준비됐지만 ML 결과 CSV가 없습니다.

```text
DB missing
```

CSV는 있지만 SQLite monitoring DB row가 없습니다.

```text
WAIT
```

QC가 아직 끝나지 않아 ML을 기다리는 상태입니다.

### Li-salt %

해당 AnalysisID의 평균 Lithium Salt 비율입니다.

결과 CSV 파일명에 있는 값을 우선 사용하고, 없으면 CSV 안의 `li_salt_ratio` 평균을 사용합니다.

### Correction

색 보정 적용 여부 요약입니다.

```text
-
```

관련 DB row가 없거나 확인할 내용이 없습니다.

```text
NO
```

DB row는 있으나 색 보정이 적용되지 않았습니다.

```text
Mixed
```

일부 이미지에 색 보정이 적용되었습니다.

### Summary

현재 상태의 상세 설명입니다.

예:

```text
QC ready, ML result missing
QC + ML + DB complete
QC hold: 2 NG image(s)
Cell #1 FRONT_pass count 27/28
```

## 8. Workflow 색상 의미

목록 row 색상은 빠르게 상태를 구분하기 위한 표시입니다.

```text
연한 빨강: QC missing/fail
연한 노랑: ML/DB missing
연한 파랑: Complete
```

## 9. Preview와 Before / After Review

오른쪽에는 처리 중이거나 선택한 AnalysisID의 이미지를 확인하는 영역이 있습니다.

### Preview 체크박스

켜져 있으면 실행 중 Before/After 이미지를 보여줍니다.

끄면 화면 업데이트가 줄어들어 대량 처리 시 조금 더 가볍게 돌릴 수 있습니다.

### Previous / Next

QC 실패 이미지 또는 리뷰 이미지를 앞뒤로 이동합니다.

### Retake done

선택한 QC 실패 이미지를 다시 확인합니다.

사용 흐름:

1. 실제 이미지 파일을 다시 촬영하거나 교체합니다.
2. `Retake done`을 누릅니다.
3. 해당 이미지가 다시 QC/crop 처리됩니다.
4. 통과하면 hold 목록에서 빠집니다.

### Crop anyway

QC 실패 사유를 무시하고 해당 이미지를 crop 결과로 저장합니다.

주의:

- 이미지가 잘렸거나 경계가 이상한 경우에도 강제로 저장될 수 있습니다.
- 실제 양산 결과로 쓰기 전에는 반드시 Before/After를 확인하세요.

## 10. Workflow 출력 파일

### QC crop 출력

일반 AnalysisID:

```text
Cell #1/
  FRONT_pass/
    *.png
  BACK_pass/
    *.png
```

TEST AnalysisID:

```text
TEST/
  _pass/
    *.png
```

또는:

```text
TEST/
  group_a_pass/
    *.png
```

### ML mask 출력

일반 AnalysisID:

```text
Cell #1/
  FRONT_mask/
    *_mask.png
  BACK_mask/
    *_mask.png
```

TEST AnalysisID:

```text
TEST/
  _mask/
    *_mask.png
```

또는:

```text
TEST/
  group_a_mask/
    *_mask.png
```

마스크 색상:

```text
초록: Background
빨강: Lithium Salt
파랑: Nonactive
회색: Edge
```

### ML 결과 CSV

AnalysisID 폴더 아래에 저장됩니다.

```text
<AnalysisID>_LI-SALT_<평균값>%.csv
```

예:

```text
AHG12345_LI-SALT_12.34%.csv
```

CSV 주요 컬럼:

```text
AnalysisID
Cell
Face
ImageNum
total_pixel
background_ratio
nonactive_ratio
li_salt_ratio
correction_applied
```

주의:

- `background_ratio`, `nonactive_ratio`, `li_salt_ratio`는 ratio 값입니다.
- 화면이나 파일명에서는 `%`로 변환되어 표시될 수 있습니다.

### SQLite monitoring DB

Workflow에서 ML을 실행하면 monitoring DB에도 결과가 저장됩니다.

기본 설정:

```yaml
storage:
  sqlite:
    path: "photo_jig_vision.db"
```

DB 파일명은 `photo_jig_vision.db`로 고정되며 데이터 root path와 분리됩니다.

```text
%LOCALAPPDATA%/Li-Plating_Quant_System/photo_jig_vision.db
```

DB는 Windows 사용자별 로컬 경로만 사용합니다. Python 내장 `sqlite3`가 최초 실행 시 파일과 스키마를 자동 생성하므로 별도 SQLite 설치가 필요하지 않습니다.

DB에는 이미지별 평균 색상, confidence, drift, correction 적용 여부, mask path, pass image path 등이 함께 저장됩니다.

## 11. SQLite 중복 경고

이미 같은 `AnalysisID + Cell` 결과가 DB에 있으면 실행 전에 확인 창이 뜹니다.

```text
Replace existing SQLite rows?
```

선택 의미:

```text
Yes
```

기존 DB row를 지우고 새 결과로 저장합니다.

```text
No
```

CSV와 mask는 생성되지만, 이미 있는 DB row는 덮어쓰지 않습니다.

## 12. Analysis 탭 사용법

### Date

결과를 볼 날짜를 선택합니다.

Analysis 탭은 해당 날짜 폴더 아래의 AnalysisID 결과 CSV를 찾습니다.

찾는 파일:

```text
<AnalysisID>_LI-SALT_*.csv
<AnalysisID>_result.csv
```

### View

그래프에 표시할 면을 선택합니다.

```text
FRONT
BACK
TEST
FRONT + BACK
Average
```

### ML Results 목록

왼쪽 목록에서 Cell 결과를 클릭하면 그래프에 추가됩니다.

다시 클릭하면 선택이 해제됩니다.

표시되는 값:

```text
Analysis ID
Cell
Li-salt / Nonactive
```

### Graph 탭

선택한 Cell의 layer별 Li-salt %를 선 그래프로 보여줍니다.

그래프 점을 클릭하면 아래의 `Selected layer images` 영역에 해당 layer의 crop 이미지와 mask 이미지가 표시됩니다.

같은 위치에 여러 점이 겹치면 선택 드롭다운이 나타납니다.

### Heatmap 탭

선택한 Cell의 mask 이미지들을 합쳐 위치별 경향을 보여줍니다.

Mask 옵션:

```text
Li-salt
Nonactive
Li-salt + Nonactive
```

Heatmap은 보통 첫 layer와 마지막 layer를 제외한 중간 layer mask를 기준으로 만듭니다.

### Clear graph

현재 그래프 선택을 모두 지웁니다.

### Export data

선택한 Cell 결과가 있으면 선택 결과만, 선택이 없으면 현재 Date의 전체 결과를 병합해 저장합니다.

지원 형식:

```text
CSV
Excel .xlsx
```

Excel export는 배포 패키지에 `openpyxl`이 포함되어 있어야 합니다.

### Capture screen

현재 Analysis 탭 화면 영역을 PNG로 저장합니다.

화면 캡처는 배포 패키지에 `Pillow`가 포함되어 있어야 합니다.

## 13. main.py에서 하지 않는 일

아래 작업은 `main.py`가 아니라 별도 도구를 사용합니다.

### 픽셀 라벨링

```powershell
python tools\label_tools.py
```

학습용 CSV를 만들 때 사용합니다.

### 기존 라벨 CSV에 enhanced feature 추가

```powershell
python tools\enhance_training_features.py
```

예전 CSV를 enhanced 학습에 쓰기 위한 전처리 도구입니다.

### 새 모델 학습

```powershell
python tools\train_lightGBM.py
```

basic/enhanced 선택, CSV 전체/커스텀 선택, 모델명 입력을 통해 새 LightGBM 모델을 만듭니다.
다른 알고리즘은 `tools\train_logistic_regression.py`, `tools\train_random_forest.py`, `tools\train_xgboost.py`, `tools\train_catboost.py`를 사용합니다.
`tools\train_model_LAB.py`는 `tools\train_lightGBM.py`와 같은 동작을 하던 호환용 이름이라 제거했습니다.

LightGBM 스크립트가 다른 알고리즘 실행 파일보다 긴 이유는 CSV 선택, feature set 처리, 모델 저장, `data/test` 마스크 생성 같은 공통 기능이 많이 들어 있기 때문입니다. 다른 알고리즘 실행 파일은 `tools\train_classical_model.py`의 공통 로직을 호출하는 짧은 wrapper입니다.

### data/test 이미지로 모델 단독 테스트

```powershell
python tools\test_model.py
```

이미 학습된 모델을 선택해 `data/test` 이미지에 대해 마스크와 퍼센트를 생성합니다.
모델 번호는 `1,2,3`처럼 쉼표로 여러 개를 선택할 수 있습니다.

### 여러 모델 테스트 결과 비교 대시보드

```powershell
streamlit run test_dashboard.py
```

`data/test/<model-name>/test_results.csv` 결과를 읽어 여러 모델을 비교합니다.

### 운영 monitoring 대시보드

```powershell
streamlit run monitoring_app.py
```

SQLite DB에 저장된 운영 결과 추세를 확인합니다.

## 14. 자주 쓰는 작업 흐름

### 새 촬영 데이터 처리

1. Settings 탭에서 PhotoJig `root_path`가 맞는지 확인합니다.
2. `python main.py`를 실행합니다.
3. Workflow 탭에서 날짜를 선택합니다.
4. 사용할 모델을 선택합니다.
5. 처리할 AnalysisID를 선택합니다.
6. `Run Selected QC + ML`을 누릅니다.
7. QC hold가 뜨면 오른쪽 Review에서 실패 이미지를 확인합니다.
8. 필요하면 retake 후 `Retake done`을 누릅니다.
9. 결과 CSV, mask, DB 저장 상태가 `DONE`인지 확인합니다.
10. Analysis 탭에서 그래프와 mask 이미지를 확인합니다.
11. 필요하면 `Export data` 또는 `Capture screen`으로 결과를 저장합니다.

### 처리 안 된 것만 이어서 실행

1. Workflow 탭에서 날짜를 선택합니다.
2. `Refresh`를 누릅니다.
3. `Run Missing QC + ML`을 누릅니다.

### 모델만 바꿔서 다시 ML 실행

1. Workflow 탭에서 모델을 새로 선택합니다.
2. 이미 QC가 완료된 AnalysisID를 선택합니다.
3. `Run Selected ML`을 누릅니다.
4. SQLite 중복 경고가 뜨면 DB를 덮어쓸지 선택합니다.

### QC 실패했지만 테스트 목적으로 강제 실행

1. Workflow 탭에서 대상 AnalysisID를 선택합니다.
2. 모델을 선택합니다.
3. `Force Selected QC + ML`을 누릅니다.
4. 결과 mask와 CSV를 직접 확인합니다.

## 15. 문제 해결

### 프로그램 시작 시 config 파일 오류

확인할 것:

```text
config/thresholds.yaml
```

`main.py`는 프로젝트 루트 기준으로 이 파일을 찾습니다. `tools` 폴더 안에서 실행하지 말고 프로젝트 루트에서 실행하세요.

### 날짜나 AnalysisID가 보이지 않음

확인할 것:

- Settings 탭의 root path가 실제 PhotoJig 폴더인지
- 날짜 폴더명이 허용 형식인지
- AnalysisID가 대문자 3개 + 숫자 5개 형식인지
- TEST 데이터라면 폴더명이 `TEST`, `TEST01` 같은 형식인지

### QC가 계속 NEEDED로 남음

가능한 원인:

- FRONT/BACK 폴더가 없음
- 이미지 개수가 고객별 기준과 맞지 않음
- pass 폴더가 생성되지 않음
- crop 실패 이미지가 있음

Workflow의 `Summary` 컬럼과 오른쪽 Review 영역을 확인하세요.

### ML missing으로 남음

가능한 원인:

- QC가 아직 DONE이 아님
- 선택한 모델 폴더가 잘못됨
- `metadata.json` 또는 `model.txt`/`model.joblib`이 없음
- pass 이미지가 없음

### DB missing으로 남음

가능한 원인:

- CSV는 있지만 SQLite 저장이 누락됨
- 중복 경고에서 DB overwrite를 거부함
- DB 파일 접근 권한 문제가 있음
- 다른 Windows 로그인 계정으로 실행해 별도의 사용자 DB를 보고 있음

해결:

1. 대상 AnalysisID를 선택합니다.
2. `Run Selected ML`을 다시 실행합니다.
3. 중복 경고가 뜨면 필요한 경우 `Yes`를 선택합니다.

### 마스크 이미지가 저장되지 않음

확인할 설정:

```yaml
ml:
  save_mask_images: true
```

`false`이면 CSV/DB 중심으로만 실행되고 mask PNG 저장이 생략될 수 있습니다.

마스크 PNG 저장 시간이 병목인지 확인하려면 로그의 `mask_write` 시간을 봅니다. 한 이미지가 20~30초 걸리고 로그상 `predict_total` 또는 `Inference timing ... predict`가 대부분이면 마스크 저장보다 모델 추론/픽셀 수가 병목입니다.

### 결과가 이상해 보임

확인 순서:

1. Workflow 오른쪽 Before/After 이미지 확인
2. Analysis 탭에서 layer별 Li-salt % 그래프 확인
3. mask 이미지에서 빨강/파랑/초록 영역 확인
4. 모델이 올바른지 확인
5. 필요하면 `test_dashboard.py`로 여러 모델 결과 비교

## 16. 핵심 요약

- `main.py`는 운영용 QC + ML 실행 프로그램입니다.
- 일반 작업은 Workflow 탭에서 `Run Selected QC + ML`을 사용합니다.
- 결과 확인은 Analysis 탭에서 합니다.
- 새 모델 학습은 `main.py`가 아니라 `tools/train_lightGBM.py` 또는 알고리즘별 학습 스크립트에서 합니다.
- 여러 모델의 `data/test` 결과 비교는 `test_dashboard.py`에서 합니다.
- 문제가 생기면 Workflow의 `QC`, `ML / DB`, `Summary` 컬럼을 먼저 확인하세요.
