---
name: FolderQC FAIL does not invoke RetakeManager
description: FolderQC FAIL은 RetakeManager와 무관 — FAIL 사유는 qc_tab 대시보드에 표시
type: feedback
---

FolderQC FAIL 시 RetakeManager를 호출하지 않는다.
FAIL 사유는 FolderQCResult.fail_reasons 리스트에 담겨 qc_tab 대시보드에 표시된다.

**Why:** RetakeManager는 CropEngine이 만든 NG 이미지를 삭제/재스캔하는 도구다.
FolderQC는 이미지 내용이 아닌 폴더 구조와 파일 수를 검사하므로,
FAIL 시 오퍼레이터가 구조를 수정하거나 재촬영하는 것은 GUI 대시보드 안내로 충분하다.

**How to apply:** folder_qc.py 구현 시 RetakeManager import 또는 호출 코드를 넣지 않는다.
qc_tab.py 구현 시 FolderQCResult.fail_reasons를 읽어 UI에 직접 표시한다.
