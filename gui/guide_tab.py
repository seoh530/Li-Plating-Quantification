"""Operator guide for the current desktop workflow."""

from pathlib import Path
import tkinter as tk
from tkinter import ttk


SECTIONS = [
    (
        "1. 처음 실행과 Settings",
        [
            (
                "Root folder",
                "처음 실행하면 Choose your workspace 화면에서 분석 폴더가 들어 있는 루트 폴더를 선택합니다. 설정을 저장한 다음 Main 화면을 엽니다. 취소하면 다른 폴더를 선택할 수 있습니다.",
            ),
            (
                "설정 변경",
                "Settings에서 Root folder와 Model path를 변경하고 Save settings를 누릅니다. 저장값은 사용자별로 유지됩니다. Main과 Single Mask는 저장한 모델 설정을 사용합니다.",
            ),
            (
                "폴더 검색",
                "Main은 루트 아래 Analysis ID 폴더를 하위 경로까지 검색합니다. 날짜 폴더를 반드시 만들 필요는 없습니다. Analysis ID의 두 번째 글자로 고객사를 자동 구분합니다: H=HKMC, F=Ford, V=VW, E=ESS.",
            ),
        ],
    ),
    (
        "2. Main · Li-Salt Quantification",
        [
            (
                "작업 선택",
                "Refresh로 목록을 갱신하거나 분석 폴더를 남색 드롭 영역에 놓습니다. FRONT/BACK 폴더도 입력할 수 있습니다. 원본 및 결과 폴더 구조를 확인한 뒤 처리할 행을 선택합니다. 행을 선택하지 않으면 현재 대상 목록을 처리합니다.",
            ),
            (
                "실행",
                "QC / ML을 선택하고 Run analysis를 누릅니다. 둘 다 선택하면 QC 후 ML로 이어집니다. Stop은 현재 처리 작업이 정리된 뒤 중단합니다.",
            ),
            (
                "완료 작업",
                "완료된 작업은 기본 목록에서 숨깁니다. Show completed를 켜면 함께 볼 수 있습니다. 기존 완료 작업을 다시 처리할 때는 Reprocess completed를 사용합니다.",
            ),
            (
                "Combined",
                "Workflow가 QC/ML 산출물을 기반으로 PASS combined와 mask_combined를 자동 생성합니다. ML에서 제외한 첫 번째·마지막 이미지의 마스크 자리는 비워 두며, 중간 이미지 순서를 당기지 않습니다.",
            ),
        ],
    ),
    (
        "3. QC Hold Images · 재촬영 및 Crop",
        [
            (
                "이미지 선택",
                "QC Hold Images 탭에서 실패 이미지를 선택합니다. Before / After Review는 셀 전체가 들어오도록 원래 비율을 유지합니다. 창 크기나 패널 너비를 바꾸면 자동으로 다시 맞춥니다.",
            ),
            (
                "Retake done",
                "같은 경로의 원본 파일을 재촬영 이미지로 교체한 다음 Retake done을 누르면 해당 이미지를 다시 검사합니다.",
            ),
            (
                "Crop anyway",
                "검토 중인 이미지를 QC 실패 조건과 관계없이 Crop하도록 요청합니다. 실패 사유와 촬영 상태를 먼저 확인합니다.",
            ),
            (
                "Crop selected / Crop all",
                "선택한 Hold 이미지 또는 Hold 목록 전체에 강제 Crop을 적용합니다. 이 네 개의 조치 버튼은 QC Hold Images 탭에서만 나타나며, 처리 중에는 비활성화됩니다.",
            ),
            (
                "프리뷰 탐색",
                "프리뷰를 클릭한 다음 키보드 왼쪽·오른쪽 방향키로 이전·다음 이미지를 봅니다. QC Hold에서는 목록의 다른 이미지를 직접 선택할 수도 있습니다.",
            ),
        ],
    ),
    (
        "4. Single Mask Mode",
        [
            (
                "입력과 Crop",
                "한 장의 이미지를 드롭하거나 Choose image로 선택합니다. 원본은 자동 Crop한 뒤 분석하며, 이미 투명 배경으로 Crop된 이미지는 기존 좌표를 유지합니다.",
            ),
            (
                "두 화면",
                "왼쪽은 Crop 원본 위 마스크 오버레이, 오른쪽은 순수 마스크입니다. 빨강=Li-salt, 파랑=Nonactive, 초록=Background, 회색=Edge입니다.",
            ),
            (
                "Overlay opacity",
                "상단 슬라이더로 오버레이의 진하기를 조절합니다. 0%는 원본, 100%는 마스크 색상입니다. 표시만 바뀌며 예측 결과와 비율은 변하지 않습니다.",
            ),
            (
                "확대와 이동",
                "휠로 확대하고 마우스 오른쪽 드래그로 두 화면을 이동합니다. Fit to view로 전체 셀 보기로 돌아갑니다.",
            ),
            (
                "Export images + CSV",
                "Crop 원본, 마스크 PNG와 Li-salt / Nonactive / Background 비율 CSV를 선택한 폴더에 저장합니다.",
            ),
        ],
    ),
    (
        "5. Developer mode · 라벨링 및 학습",
        [
            (
                "라벨링",
                "Developer mode를 켜면 사진 위쪽에 도구가 나타납니다. Li-salt / Background / Nonactive를 선택하고 왼쪽 오버레이에 점을 찍거나 드래그합니다. 투명한 셀 바깥 영역은 라벨링하지 않습니다. Undo point로 마지막 점을 되돌립니다.",
            ),
            (
                "Add train set",
                "Crop 이미지와 라벨 좌표에 대응하는 enhanced 학습 CSV를 data/labeled에 저장합니다. 배포 프로그램에서는 사용자별 쓰기 가능한 데이터 위치를 사용합니다. 원본 촬영본은 captures에 별도 보관합니다.",
            ),
            (
                "Train",
                "기존 학습 데이터와 새 라벨을 사용해 새 모델을 학습하고 Single Mask 결과를 갱신합니다. 기존 모델은 보존되며, 이 세션의 Single Mask에서 새 모델을 사용합니다. 다른 화면이나 다음 실행에서도 쓰려면 Settings에서 사용할 모델을 선택합니다.",
            ),
            (
                "라벨 기준",
                "실제 클래스에 맞게 라벨링합니다. 리튬염으로 잘못 잡히는 경계·얼룩도 올바른 클래스로 추가합니다. 라벨링하지 않은 픽셀은 새 정답 데이터로 사용하지 않습니다. label_tools에서도 enhanced 특징을 포함한 CSV를 바로 저장합니다.",
            ),
        ],
    ),
    (
        "6. Analysis / Report",
        [
            (
                "Analysis",
                "All folders 또는 날짜 범위에서 Analysis ID·Cell·Face별 결과를 조회합니다. Layer별 비율, 원본/마스크와 heatmap을 확인하고 데이터 또는 화면을 내보냅니다.",
            ),
            (
                "Workflow > Combine",
                "QC Hold Images 옆 Combine 탭에서 Analysis ID·Cell과 Rows/Columns, PASS/MASK/BOTH를 선택하고 Create combined를 누릅니다. 기존 PASS는 재사용하고, 없는 PASS는 QC Crop을 시도합니다. 마스크가 없으면 ML을 먼저 실행합니다. 첫·마지막 제외 마스크 자리는 비워 둡니다.",
            ),
            (
                "Report",
                "QC Hold와 PASS·마스크·CSV·combined 완성 여부를 확인한 Analysis ID만 표시합니다. Cell이 여러 개여도 하나로 묶습니다. 선택하면 PDF 미리보기를 생성하고 Export PDF로 저장합니다. 라인 그래프, 평균 막대그래프, 레이어별 실제 값 및 mean/median/min/max, FRONT/BACK 마스크 누적 지도와 combined가 포함됩니다.",
            ),
        ],
    ),
    (
        "7. 문제 해결",
        [
            (
                "목록이 비어 있음",
                "Settings의 Root folder와 원본 Analysis ID 폴더 구조를 확인하고 Refresh합니다. 완료 작업을 찾는 경우 Show completed를 켭니다. _pass / _mask 등 결과 폴더는 원본 작업으로 중복 표시하지 않습니다.",
            ),
            (
                "Drag and drop unavailable",
                "드롭 기능을 지원하는 실행 환경과 tkinterdnd2 설치 상태를 확인합니다. Main은 Settings의 Root folder로도 작업할 수 있고 Single Mask는 Choose image로 파일을 선택할 수 있습니다.",
            ),
            (
                "모델 로드 오류",
                "Settings에서 Model path를 확인합니다. 모델 폴더에는 metadata.json과 model.txt 또는 model.joblib이 있어야 합니다. 학습에 사용한 특징과 모델 메타데이터도 일치해야 합니다.",
            ),
            (
                "Crop 또는 QC 오류",
                "전극 전체가 촬영되었는지, 흰 배경·초점·조명·각도와 경계가 정상인지 확인합니다. QC Hold에서 실패 사유를 확인하고 재촬영 후 다시 검사합니다.",
            ),
            (
                "저장 오류",
                "출력 폴더의 접근 권한, 디스크 공간, 네트워크와 파일 잠금 상태를 확인합니다. SQLite는 사용자별 로컬 데이터 위치에 저장됩니다.",
            ),
        ],
    ),
]


class GuideTab(ttk.Frame):
    def __init__(self, parent, config_path: str | Path):
        super().__init__(parent)
        self._config_path = Path(config_path)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        ttk.Label(self, text="Guide", font=("Segoe UI", 19, "bold")).grid(
            row=0, column=0, sticky=tk.W, padx=24, pady=(18, 14)
        )
        canvas = tk.Canvas(self, bg="#f4f6f8", highlightthickness=0)
        canvas.grid(row=1, column=0, sticky=tk.NSEW)
        scroll = ttk.Scrollbar(self, orient=tk.VERTICAL, command=canvas.yview)
        scroll.grid(row=1, column=1, sticky=tk.NS)
        canvas.configure(yscrollcommand=scroll.set)
        content = ttk.Frame(canvas, padding=(24, 0, 24, 18))
        content.columnconfigure(0, weight=1)
        window = canvas.create_window(0, 0, window=content, anchor=tk.NW)
        text_labels = []
        for row, (title, items) in enumerate(SECTIONS):
            card = ttk.Frame(content, style="Card.TFrame", padding=18)
            card.grid(row=row, column=0, sticky=tk.EW, pady=(0, 12))
            card.columnconfigure(0, weight=1)
            ttk.Label(
                card, text=title, font=("Segoe UI", 12, "bold"), style="Panel.TLabel"
            ).grid(row=0, column=0, sticky=tk.W, pady=(0, 10))
            for index, (label, text) in enumerate(items, 1):
                ttk.Label(
                    card,
                    text=label,
                    style="Panel.TLabel",
                    font=("Segoe UI", 10, "bold"),
                ).grid(row=index * 2 - 1, column=0, sticky=tk.W, pady=(8, 4))
                body = ttk.Label(
                    card,
                    text=text,
                    style="Panel.TLabel",
                    justify=tk.LEFT,
                    wraplength=900,
                )
                body.grid(row=index * 2, column=0, sticky=tk.EW)
                text_labels.append(body)
        content.bind(
            "<Configure>",
            lambda _event: canvas.configure(scrollregion=canvas.bbox("all")),
        )

        def resize(event):
            canvas.itemconfigure(window, width=event.width)
            for label in text_labels:
                label.configure(wraplength=max(200, event.width - 90))

        canvas.bind("<Configure>", resize)

        # Bind locally: scrolling the guide must not replace Single Mask's zoom binding.
        def wheel(event):
            if self.winfo_ismapped():
                canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")
                return "break"

        for widget in [canvas, content, *self._descendants(content)]:
            widget.bind("<MouseWheel>", wheel)
            widget.bind("<Button-4>", lambda _event: canvas.yview_scroll(-1, "units"))
            widget.bind("<Button-5>", lambda _event: canvas.yview_scroll(1, "units"))

    @staticmethod
    def _descendants(widget):
        children = widget.winfo_children()
        return children + [
            nested for child in children for nested in GuideTab._descendants(child)
        ]

    def reload_settings(self):
        pass  # The guide describes behavior; no user paths are copied into it.
