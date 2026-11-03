from pathlib import Path
import tkinter as tk
from tkinter import ttk

import yaml


class GuideTab(ttk.Frame):
    """Operator guide and troubleshooting page."""

    def __init__(self, parent, config_path: str | Path) -> None:
        super().__init__(parent)
        self._config_path = Path(config_path)
        self._load_config()
        self._build_ui()

    def _load_config(self) -> None:
        with open(self._config_path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        folder_qc = cfg.get("folder_qc", {})
        self._root_path = str(cfg.get("app", {}).get("root_path", ""))
        self._customer_codes = dict(folder_qc.get("customer_codes", {}))
        self._image_count_rules = dict(folder_qc.get("image_count_rules", {}))

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0)
        scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky=tk.NSEW)
        scrollbar.grid(row=0, column=1, sticky=tk.NS)

        content = ttk.Frame(canvas, padding=(18, 16, 28, 20))
        window_id = canvas.create_window((0, 0), window=content, anchor=tk.NW)
        content.columnconfigure(0, weight=1)

        def update_scroll_region(_event=None) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def update_content_width(event) -> None:
            canvas.itemconfigure(window_id, width=event.width)

        content.bind("<Configure>", update_scroll_region)
        canvas.bind("<Configure>", update_content_width)
        canvas.bind("<Enter>", lambda _e: self._bind_mousewheel(canvas))
        canvas.bind("<Leave>", lambda _e: self._unbind_mousewheel(canvas))

        row = 0
        row = self._add_title(content, row)
        row = self._add_program_overview(content, row)
        row = self._add_folder_structure(content, row)
        row = self._add_workflow_guide(content, row)
        row = self._add_analysis_guide(content, row)
        row = self._add_report_guide(content, row)
        row = self._add_settings_guide(content, row)
        row = self._add_troubleshooting(content, row)
        content.rowconfigure(row, weight=1)

    def reload_settings(self) -> None:
        for child in self.winfo_children():
            child.destroy()
        self._load_config()
        self._build_ui()

    def _bind_mousewheel(self, canvas: tk.Canvas) -> None:
        canvas.bind_all(
            "<MouseWheel>",
            lambda event: canvas.yview_scroll(int(-1 * (event.delta / 120)), "units"),
        )

    def _unbind_mousewheel(self, canvas: tk.Canvas) -> None:
        canvas.unbind_all("<MouseWheel>")

    def _add_title(self, parent: ttk.Frame, row: int) -> int:
        ttk.Label(
            parent,
            text="프로그램 사용 가이드",
            font=("TkDefaultFont", 18, "bold"),
        ).grid(row=row, column=0, sticky=tk.W)
        row += 1
        ttk.Label(
            parent,
            text=(
                "처음 실행할 때부터 결과 확인과 오류 조치까지, 실제 작업 순서에 맞춰 "
                "정리한 운영 가이드입니다."
            ),
            foreground="#555555",
            wraplength=1050,
        ).grid(row=row, column=0, sticky=tk.EW, pady=(4, 14))
        return row + 1

    def _add_program_overview(self, parent: ttk.Frame, row: int) -> int:
        items = [
            (
                "프로그램 역할",
                "촬영 이미지의 폴더 QC와 crop을 수행하고, PASS 이미지에 ML을 적용해 CSV, mask, SQLite 결과를 생성합니다. Report 탭에서는 별도의 combined 이미지를 만들 수 있습니다.",
            ),
            (
                "1. 입력 준비",
                f"Settings의 Root path 아래에 유효한 DATE 폴더와 작업 폴더를 준비합니다. 현재 Root path: {self._root_path or '-'}",
            ),
            (
                "2. 작업 선택",
                "Workflow에서 View, Date, Customer를 선택하고 Refresh를 누른 뒤 처리할 행을 선택합니다. 행을 선택하지 않으면 현재 목록 전체가 대상입니다.",
            ),
            (
                "3. 실행",
                "QC와 ML 중 필요한 항목을 체크하고 Run을 누릅니다. 둘 다 체크하면 QC 완료 후 생성된 PASS 이미지에 ML을 실행합니다.",
            ),
            (
                "4. NG 처리",
                "QC hold가 발생하면 오른쪽 Review에서 이미지를 확인합니다. 재촬영 후 Retake done을 우선 사용하고, 운영자가 허용한 경우에만 Crop anyway를 사용합니다.",
            ),
            (
                "5. 결과 확인",
                "Workflow 상태를 확인한 뒤 Analysis에서 그래프·mask·heatmap을 검토하고, 필요하면 Report에서 combined 이미지를 생성합니다.",
            ),
        ]
        return self._add_section(parent, row, "1. 프로그램 개요 및 기본 사용 순서", items)

    def _add_folder_structure(self, parent: ttk.Frame, row: int) -> int:
        customer_lines = []
        for code, customer in self._customer_codes.items():
            if str(customer).upper() not in {"HKMC", "FORD", "VW", "ESS"}:
                continue
            count = self._image_count_rules.get(code)
            if count is not None:
                customer_lines.append((str(customer), f"QC 실행 시 source set별 {count}장"))

        items = [
            (
                "DATE 폴더",
                "Root path 바로 아래 날짜 폴더만 검색합니다. 허용 예: 2026-08-08, 8-8-2026, 8-8-26.",
            ),
            (
                "작업 폴더",
                "DATE 아래 폴더명은 자유롭게 사용할 수 있습니다. Analysis ID 형식이면 기존 ID로, 아니면 폴더명을 Work ID 및 표시 ID로 사용합니다.",
            ),
            (
                "CELL 구조",
                "CELL #1, CELL #2 폴더가 있으면 Cell별로 처리합니다. CELL 폴더가 없으면 작업 폴더 전체를 single cell로 처리합니다.",
            ),
            (
                "FRONT / BACK",
                "둘 다 있으면 모두 처리하고, 한쪽만 있으면 존재하는 방향만 처리합니다. 반대쪽 폴더가 없다는 이유만으로 QC Fail 처리하지 않습니다.",
            ),
            (
                "직접 이미지",
                "FRONT와 BACK이 모두 없고 작업 폴더 또는 CELL 폴더에 이미지가 직접 있으면 하나의 SOURCE set으로 처리합니다. 이 구조는 Workflow에서 사용할 수 있지만 Analysis 탭에서는 사용할 수 없습니다.",
            ),
            (
                "결과 폴더 제외",
                "이름이 _pass 또는 _mask로 끝나는 DATE 하위 폴더는 대소문자와 관계없이 Workflow 작업 목록에서 제외합니다.",
            ),
            (
                "지원 이미지",
                "jpg, jpeg, png, bmp, tif, tiff 확장자를 지원합니다. 각 source set의 첫 이미지와 마지막 이미지는 기존 규칙대로 일반 ML 입력에서 제외됩니다.",
            ),
            *customer_lines,
            (
                "ETC",
                "이미지 수 검사를 수행하지 않습니다. Workflow의 기존 상태는 Customer나 예상 장수 대신 PASS, mask, CSV 산출물 존재 여부로 판단합니다.",
            ),
        ]
        return self._add_section(parent, row, "2. 폴더 구조 및 이미지 기준", items)

    def _add_workflow_guide(self, parent: ttk.Frame, row: int) -> int:
        items = [
            (
                "목적",
                "작업 폴더를 찾아 QC/crop과 ML을 한 화면에서 실행하고 진행 상태와 QC NG 이미지를 관리합니다.",
            ),
            (
                "입력",
                "View, Date, Customer, 작업 행, QC/ML 체크 상태를 선택합니다. Workflow ML 모델은 models/V2F로 고정되어 별도의 모델 선택은 없습니다.",
            ),
            (
                "주요 기능",
                "QC와 ML 개별 또는 연속 실행, Stop, Force Run, 다중 행 선택, 진행률, Before/After Preview, QC hold 재처리를 제공합니다. 상태는 PASS, mask, CSV, DB 존재 여부로 구분하며 Customer 변경으로 기존 상태가 바뀌지 않습니다.",
            ),
            (
                "실행 순서",
                "① View와 Date 선택 → ② Customer 선택 → ③ Refresh → ④ 작업 행 선택 → ⑤ QC/ML 체크 → ⑥ Run. QC+ML이면 이번 QC가 정상 완료된 작업의 PASS 이미지만 ML로 이어집니다.",
            ),
            (
                "출력 결과",
                "QC는 FRONT_pass/BACK_pass 또는 SOURCE의 _pass를 생성합니다. ML은 mask 폴더, 작업별 Li-salt CSV, SQLite 기록을 생성합니다. QC hold가 있으면 Review에 NG 이미지가 표시됩니다.",
            ),
            (
                "상태 해석",
                "PASS 없음: QC NEEDED · PASS 있음/CSV 없음: ML missing · CSV 있음/mask 없음: Mask missing · DB 없음: DB missing · 모두 있음: DONE.",
            ),
        ]
        return self._add_section(parent, row, "3. Workflow 탭 사용법", items)

    def _add_analysis_guide(self, parent: ttk.Frame, row: int) -> int:
        items = [
            (
                "목적",
                "완료된 ML 결과를 Cell과 Face별로 조회하고 Li-salt 추세, mask, 평균 heatmap을 검토합니다.",
            ),
            (
                "입력",
                "Date와 View(FRONT/BACK 등)를 선택하고 왼쪽 ML Results에서 Analysis ID와 Cell 결과를 선택합니다. FRONT/BACK이 없는 직접 이미지 SOURCE 구조는 지원하지 않습니다.",
            ),
            (
                "주요 기능",
                "Layer별 그래프, 선택 layer 원본/mask 비교, 평균 heatmap, Clear graph, Export data, Capture screen을 제공합니다.",
            ),
            (
                "실행 순서",
                "① Date 선택 → ② Refresh → ③ ML Result 선택 → ④ Face View 선택 → ⑤ 그래프 지점 또는 Heatmap 선택 → ⑥ 필요한 결과를 Export/Capture.",
            ),
            (
                "출력 결과",
                "화면에서 Li-salt 및 Nonactive 비율과 mask를 확인할 수 있습니다. Export data는 CSV 또는 Excel을, Capture screen은 현재 Analysis 화면 PNG를 저장합니다.",
            ),
        ]
        return self._add_section(parent, row, "4. Analysis 탭 사용법", items)

    def _add_report_guide(self, parent: ttk.Frame, row: int) -> int:
        items = [
            (
                "목적",
                "원본 FRONT/BACK 이미지를 다시 crop하고 여러 layer를 한 장에 배치한 보고서용 combined 이미지를 생성합니다.",
            ),
            (
                "입력",
                "View, Date, Analysis ID, Cell, Columns/Rows를 선택합니다. 지원 환경에서는 FRONT/BACK 폴더, Analysis ID 폴더 또는 이미지 파일을 drag and drop할 수 있습니다.",
            ),
            (
                "주요 기능",
                "기존 combined 결과 확인, Grid Preview, Cell 선택, 외부 경로 처리, 결과 파일 및 저장 위치 표시, 화면 로그 초기화를 제공합니다. FRONT와 BACK은 섞지 않습니다.",
            ),
            (
                "실행 순서",
                "① View/Date/Analysis ID 선택 또는 파일 drop → ② Cell 선택 → ③ Columns/Rows와 Preview 확인 → ④ Create Report Images → ⑤ 덮어쓰기 안내가 나오면 정책 선택.",
            ),
            (
                "출력 결과",
                "FRONT_combined와 BACK_combined 폴더에 페이지별 combined PNG를 저장합니다. Clear report log는 화면 목록만 지우며 실제 파일은 삭제하지 않습니다.",
            ),
        ]
        return self._add_section(parent, row, "5. Report 탭 사용법", items)

    def _add_settings_guide(self, parent: ttk.Frame, row: int) -> int:
        items = [
            (
                "목적",
                "배포 PC에서 사용할 촬영 Root path, 모델 경로, SQLite 저장 위치를 확인하고 저장합니다.",
            ),
            (
                "입력",
                "Root path, Model path, 필요 시 SQLite DB 경로를 입력하거나 Browse로 선택합니다. Workflow 자체는 models/V2F를 고정 사용합니다.",
            ),
            (
                "주요 기능",
                "경로 선택, 모델 상태 확인, Save settings, 저장값 Reload를 제공합니다. SQLite DB는 사용자별 LOCALAPPDATA 아래 Li-Plating_Quant_System/photo_jig_vision.db에 저장됩니다.",
            ),
            (
                "실행 순서",
                "① 경로 입력/Browse → ② 경로와 Model status 확인 → ③ Save settings → ④ 각 탭에서 Refresh하여 날짜와 작업 목록 확인. 실행 중에는 설정을 변경하지 않는 것을 권장합니다.",
            ),
            (
                "출력 결과",
                "설정은 config/thresholds.yaml에 저장되고 관련 탭이 새 값을 다시 읽습니다. Reload는 저장된 값을 화면에 다시 불러옵니다.",
            ),
        ]
        return self._add_section(parent, row, "6. Settings 탭 사용법", items)

    def _add_troubleshooting(self, parent: ttk.Frame, row: int) -> int:
        items = [
            (
                "날짜 또는 작업이 보이지 않음",
                "Root path, DATE 형식 또는 source 폴더를 찾지 못했습니다.",
                "Settings의 Root path와 날짜 폴더명을 확인합니다. 작업 폴더에 FRONT/BACK, CELL 하위 source 또는 직접 이미지가 있는지 확인하고 Refresh합니다.",
            ),
            (
                "_pass / _mask 폴더가 목록에 안 보임",
                "결과 폴더를 원본 작업으로 중복 표시하지 않도록 정상적으로 제외한 상태입니다.",
                "처리할 원본 작업 폴더를 선택합니다. 결과 폴더명을 작업 폴더명으로 사용하지 않습니다.",
            ),
            (
                "Select QC and/or ML",
                "QC와 ML이 모두 체크 해제되어 실행할 작업이 없습니다.",
                "QC 또는 ML 중 하나 이상을 체크한 뒤 Run을 누릅니다.",
            ),
            (
                "QC NEEDED / PASS missing",
                "처리 가능한 PASS PNG가 아직 없습니다.",
                "QC를 체크해 실행합니다. QC hold나 이미지 오류가 발생했다면 해당 문제를 먼저 해결합니다.",
            ),
            (
                "Image count mismatch",
                "이번 QC 실행에서 선택 Customer의 source set별 이미지 수 규칙과 실제 수가 다릅니다.",
                "Customer 선택과 촬영 이미지 누락·중복을 확인합니다. 기존 목록 상태는 이 메시지가 아니라 산출물 존재 여부로 판정됩니다.",
            ),
            (
                "QC hold",
                "angle, contour, boundary, aspect ratio 또는 pixel count 검사에서 NG가 발생해 ML이 보류됐습니다.",
                "Review에서 원본을 확인하고 재촬영 후 Retake done을 사용합니다. 승인 가능한 경우에만 Crop anyway를 사용합니다.",
            ),
            (
                "ML missing",
                "PASS는 있지만 결과 CSV가 없습니다.",
                "ML을 체크하고 Run을 실행합니다. models/V2F에 metadata.json과 model.txt 또는 model.joblib이 있는지 확인합니다.",
            ),
            (
                "Mask missing",
                "CSV는 있지만 PASS 이미지에 대응하는 mask PNG가 없습니다.",
                "ML을 다시 실행하고 mask 폴더 쓰기 권한, 디스크 용량, 파일 잠금 상태를 확인합니다.",
            ),
            (
                "DB missing",
                "PASS, mask, CSV는 있지만 SQLite 기록이 없습니다.",
                "ML을 다시 실행합니다. 중복 안내가 나오면 기존 행 교체 여부를 작업 정책에 맞게 선택하고 DB 경로·잠금 상태를 확인합니다.",
            ),
            (
                "Model files are missing / load failed",
                "models/V2F 모델 파일이 없거나 metadata와 모델 파일이 맞지 않습니다.",
                "배포본의 models/V2F 폴더에 정상 학습된 metadata.json과 model.txt 또는 model.joblib을 배치합니다.",
            ),
            (
                "Image load / contour / crop 오류",
                "이미지가 손상됐거나 전극 윤곽, 각도, 경계 또는 crop 영역이 정상적이지 않습니다.",
                "파일 열기 여부와 촬영 상태를 확인합니다. 흰 배경, 조명, 초점, 전극 전체 포함 여부와 수평을 점검한 뒤 재촬영합니다.",
            ),
            (
                "Failed to write output / SQLite 오류",
                "결과 폴더 또는 DB 파일에 쓸 수 없습니다.",
                "폴더 권한, 디스크 용량, 네트워크 연결, 다른 프로그램의 파일 잠금을 확인한 뒤 다시 실행합니다.",
            ),
        ]
        return self._add_troubleshooting_table(parent, row, "7. 오류 및 조치", items)

    def _add_section(
        self,
        parent: ttk.Frame,
        row: int,
        title: str,
        items: list[tuple[str, str]],
    ) -> int:
        frame = ttk.LabelFrame(parent, text=title, padding=12)
        frame.grid(row=row, column=0, sticky=tk.EW, pady=(0, 12))
        frame.columnconfigure(1, weight=1)

        for index, (label, text) in enumerate(items):
            ttk.Label(
                frame,
                text=label,
                font=("TkDefaultFont", 10, "bold"),
                width=24,
                anchor=tk.NW,
            ).grid(row=index, column=0, sticky=tk.NW, padx=(0, 12), pady=4)
            ttk.Label(
                frame,
                text=text,
                wraplength=880,
                justify=tk.LEFT,
            ).grid(row=index, column=1, sticky=tk.EW, pady=4)
        return row + 1

    def _add_troubleshooting_table(
        self,
        parent: ttk.Frame,
        row: int,
        title: str,
        items: list[tuple[str, str, str]],
    ) -> int:
        frame = ttk.LabelFrame(parent, text=title, padding=12)
        frame.grid(row=row, column=0, sticky=tk.EW, pady=(0, 12))
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(2, weight=2)

        headings = ("화면 메시지 / 상태", "의미", "조치")
        for column, heading in enumerate(headings):
            ttk.Label(
                frame,
                text=heading,
                font=("TkDefaultFont", 10, "bold"),
                foreground="#333333",
            ).grid(row=0, column=column, sticky=tk.EW, padx=(0, 10), pady=(0, 6))

        for index, (message, meaning, action) in enumerate(items, start=1):
            ttk.Separator(frame, orient=tk.HORIZONTAL).grid(
                row=index * 2 - 1,
                column=0,
                columnspan=3,
                sticky=tk.EW,
                pady=(2, 6),
            )
            ttk.Label(
                frame,
                text=message,
                wraplength=260,
                justify=tk.LEFT,
                font=("TkDefaultFont", 9, "bold"),
            ).grid(row=index * 2, column=0, sticky=tk.NW, padx=(0, 10), pady=(0, 6))
            ttk.Label(
                frame,
                text=meaning,
                wraplength=300,
                justify=tk.LEFT,
            ).grid(row=index * 2, column=1, sticky=tk.NW, padx=(0, 10), pady=(0, 6))
            ttk.Label(
                frame,
                text=action,
                wraplength=430,
                justify=tk.LEFT,
            ).grid(row=index * 2, column=2, sticky=tk.NW, pady=(0, 6))
        return row + 1
