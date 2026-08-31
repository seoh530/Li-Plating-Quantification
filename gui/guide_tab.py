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

    def reload_config(self) -> None:
        self._load_config()
        for child in self.winfo_children():
            child.destroy()
        self._build_ui()

    def _load_config(self) -> None:
        with open(self._config_path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        folder_qc = cfg.get("folder_qc", {}) or {}
        self._root_path = str((cfg.get("app", {}) or {}).get("root_path", ""))
        self._customer_codes = dict(folder_qc.get("customer_codes", {}) or {})
        self._image_count_rules = dict(folder_qc.get("image_count_rules", {}) or {})

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
        row = self._add_quick_start(content, row)
        row = self._add_quick_path_mode(content, row)
        row = self._add_folder_rules(content, row)
        row = self._add_outputs(content, row)
        row = self._add_status_guide(content, row)
        row = self._add_settings(content, row)
        row = self._add_troubleshooting(content, row)
        row = self._add_deployment_notes(content, row)
        content.rowconfigure(row, weight=1)

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
            text="Cell Li-Plating Quantification System 사용 가이드",
            font=("TkDefaultFont", 18, "bold"),
        ).grid(row=row, column=0, sticky=tk.W)
        row += 1
        ttk.Label(
            parent,
            text=(
                "Photo Jig 이미지에서 crop, ML mask, Li-salt 정량 결과를 생성하는 운영용 프로그램입니다. "
                "일반 QC workflow와 팀룸 Path 기반 Quick Crop + ML 사용법을 함께 설명합니다."
            ),
            foreground="#555555",
            wraplength=1080,
        ).grid(row=row, column=0, sticky=tk.EW, pady=(4, 14))
        return row + 1

    def _add_quick_start(self, parent: ttk.Frame, row: int) -> int:
        steps = [
            ("1. Settings 열기", "처음 실행하면 먼저 Settings 탭으로 이동합니다."),
            ("2. Root Path 고정", "사진이 저장되는 최상단 폴더를 Root Path에 입력합니다. 일반 workflow는 이 폴더 아래의 날짜/AnalysisID 폴더를 기준으로 찾습니다."),
            ("3. Model 고정", "같이 사용할 모델을 Model에서 선택합니다. 모델 폴더에는 metadata.json과 model.txt 또는 model.joblib이 있어야 합니다."),
            ("4. Save Settings", "Root Path와 Model 조합을 저장합니다. 아래 preset 목록에서 현재 사용 중인 조합이 파란색으로 표시되는지 확인합니다."),
            ("5. Workflow 이동", "Workflow 탭으로 이동해서 일반 작업은 View를 By Date 또는 By Analysis ID로 두고 처리할 행을 선택합니다."),
            ("6. 일반 실행", "QC와 ML 체크 상태를 확인하고 Run을 누릅니다. QC 통과 후 crop, ML mask, CSV, SQLite 저장이 진행됩니다."),
            ("7. 팀룸 Quick 실행", "팀룸 또는 임의 폴더는 Quick Crop + ML 영역의 Paths 입력칸에 T:\\ 경로를 붙여넣거나 드롭하고 Load Path를 누릅니다."),
            ("8. Quick 실행", "IMAGE 컬럼의 상대경로를 확인한 뒤 필요한 이미지만 선택하거나 선택 없이 Run Quick Crop + ML을 누릅니다."),
        ]
        return self._add_section(parent, row, "처음 실행 순서", steps)

    def _add_quick_path_mode(self, parent: ttk.Frame, row: int) -> int:
        body = [
            ("입력 가능한 Path", "로컬 폴더, 이미지 파일, 팀룸 T:\\ 경로, 드래그앤드롭 경로를 사용할 수 있습니다. 공백과 한글이 포함된 경로도 지원합니다."),
            ("Load Path 동작", "폴더 전체를 복사하지 않고 파일 목록만 조회합니다. 실제 이미지 복사는 Run Quick Crop + ML을 누른 뒤 파일별 TEMP 폴더로 수행됩니다."),
            ("팀룸 이미지 처리", "팀룸 파일은 OpenCV로 직접 읽지 않습니다. safe_copy_file()로 로컬 TEMP에 복사한 사본을 crop/ML 처리한 뒤 결과만 원본 폴더 쪽으로 복사합니다."),
            ("QC skip 기준", "Quick Crop + ML은 폴더 QC를 수행하지 않고 crop + ML inference + 결과 저장만 수행합니다. 일반 workflow는 기존 QC 로직을 유지합니다."),
            ("입력칸 유지", "Load Path 후에도 입력한 Path는 사라지지 않습니다. 같은 위치를 다시 확인하거나 경로를 수정해서 재사용할 수 있습니다."),
        ]
        return self._add_section(parent, row, "Path 기반 Quick Crop + ML", body)

    def _add_folder_rules(self, parent: ttk.Frame, row: int) -> int:
        customer_lines = []
        for code, customer in self._customer_codes.items():
            count = self._image_count_rules.get(code)
            count_text = "이미지 수 제한 없음" if count is None else f"FRONT/BACK 각각 {count}장"
            customer_lines.append((f"{code}: {customer}", count_text))

        body = [
            ("현재 root_path", self._root_path or "Settings 탭에서 root_path를 설정하세요."),
            ("일반 AnalysisID", "폴더명은 영문 3글자 + 숫자 5자리 형식을 사용합니다. 예: AHG06113, TEG30301"),
            ("일반 폴더 구조", "root_path\\날짜\\AnalysisID 아래에 FRONT와 BACK 폴더를 둡니다. Cell #1, Cell #2 같은 하위 cell 폴더 구조도 지원합니다."),
            ("TEST 폴더", "TEST, TEST01 같은 폴더는 테스트 작업으로 처리합니다. FRONT/BACK이 없어도 이미지가 직접 있으면 처리할 수 있습니다."),
            *customer_lines,
        ]
        return self._add_section(parent, row, "입력 폴더 규칙", body)

    def _add_outputs(self, parent: ttk.Frame, row: int) -> int:
        body = [
            ("일반 QC crop", "FRONT 원본은 FRONT_pass, BACK 원본은 BACK_pass 폴더에 crop PNG로 저장됩니다."),
            ("일반 ML 결과", "ML mask와 AnalysisID 단위 CSV가 기존 workflow 규칙에 맞게 저장되고 SQLite monitoring DB에도 기록됩니다."),
            ("Quick crop 결과", "원본 폴더가 FRONT이면 crop 이미지는 원본의 형제 폴더 FRONT_PASS에 image01.png 형식으로 저장됩니다."),
            ("Quick mask 결과", "ML mask는 FRONT_MASK 또는 BACK_MASK 폴더에 image01_mask.png 형식으로 저장됩니다."),
            ("Quick CSV 결과", "CSV는 mask 폴더에 image01_LI_SALT_12.35%.csv 형식으로 저장됩니다. 퍼센트는 ML 결과 li_salt_ratio를 100배 한 값입니다."),
            ("저장 실패", "팀룸에 결과를 복사할 수 없으면 오류 메시지에 실패 대상 폴더와 로컬 fallback TEMP 위치가 표시됩니다."),
        ]
        return self._add_section(parent, row, "결과 저장 위치", body)

    def _add_status_guide(self, parent: ttk.Frame, row: int) -> int:
        body = [
            ("빨간색", "_PASS 결과가 없거나 QC/처리 결과가 아직 없는 상태입니다."),
            ("노란색", "_PASS는 있지만 _MASK 또는 ML 결과가 없는 상태입니다."),
            ("파란색", "_MASK와 CSV까지 생성된 완료 상태입니다. STATUS에 Li-salt 퍼센트가 함께 표시됩니다."),
            ("QC hold", "일반 workflow에서 NG 이미지가 있어 ML 실행이 보류된 상태입니다. 오른쪽 Review에서 이미지를 확인합니다."),
            ("Quick 목록 Status", "Ready, Running, PASS, FAIL로 표시됩니다. FAIL이면 Output 컬럼의 오류 내용을 확인합니다."),
        ]
        return self._add_section(parent, row, "상태 색상과 의미", body)

    def _add_settings(self, parent: ttk.Frame, row: int) -> int:
        body = [
            ("Root Path", "일반 workflow에서 날짜/AnalysisID를 찾는 기준 폴더입니다. 팀룸 Quick 실행은 Paths 입력칸에 직접 넣을 수 있습니다."),
            ("Model", "현재 선택된 모델이 Workflow의 Model 목록에 표시됩니다. Settings에서 저장하면 Workflow, Analysis, Guide가 설정을 다시 읽습니다."),
            ("Preset 목록", "Settings에서 저장했던 Root Path / Model 조합이 아래 목록에 남습니다. 현재 사용 중인 조합은 파란색으로 표시됩니다."),
        ]
        return self._add_section(parent, row, "Settings 탭", body)

    def _add_troubleshooting(self, parent: ttk.Frame, row: int) -> int:
        rows = [
            ("No supported image files found", "입력한 Path 아래에서 지원 이미지 파일을 찾지 못했습니다.", "Path가 정확한지 확인하고, FRONT/BACK처럼 이미지가 들어 있는 하위 폴더가 있는지 확인합니다. _PASS, _MASK, _MASKS 폴더는 원본 이미지 탐색에서 제외됩니다."),
            ("Access denied", "팀룸 권한 또는 Windows 폴더 목록 조회 권한이 막혀 있습니다.", "같은 경로를 파일 탐색기에서 열 수 있는지 확인합니다. 계속 실패하면 이미지가 들어 있는 더 낮은 폴더를 직접 Load Path에 넣습니다."),
            ("Failed to copy source image into local temp", "팀룸 원본 이미지를 로컬 TEMP로 복사하지 못했습니다.", "파일 잠금 또는 권한 문제일 수 있습니다. 잠시 후 다시 실행하고, 계속 실패하면 오류에 표시된 원본 파일 경로를 확인합니다."),
            ("Crop failed", "전극 contour를 찾지 못했거나 crop 기준을 통과하지 못했습니다.", "원본 이미지에서 전극 전체, 흰 배경, 초점, 조명이 정상인지 확인합니다. Quick mode는 폴더 QC는 건너뛰지만 crop 실패는 표시합니다."),
            ("Model files are missing", "선택한 모델 폴더에 필요한 metadata.json, model.txt 또는 model.joblib이 없습니다.", "Settings 또는 Workflow의 Model 목록에서 정상 모델을 선택합니다."),
            ("Failed to save result back to source folders", "PASS/MASK/CSV 결과를 원본 팀룸 폴더로 복사하지 못했습니다.", "권한, 파일 잠금, 디스크 용량을 확인합니다. 메시지에 표시된 fallback TEMP 폴더에는 로컬 결과가 남아 있습니다."),
            ("DB missing / SQLite insert failed", "일반 workflow 결과의 SQLite 저장이 누락되었거나 실패했습니다.", "공유 DB 경로 연결과 파일 잠금을 확인한 뒤 같은 모델로 ML을 다시 실행합니다."),
        ]
        return self._add_troubleshooting_table(parent, row, "오류 메시지와 조치", rows)

    def _add_deployment_notes(self, parent: ttk.Frame, row: int) -> int:
        body = [
            ("배포 전 확인", "config\\thresholds.yaml, models 폴더, 필요한 Python 패키지가 빌드 환경에 있어야 합니다."),
            ("팀룸 사용", "배포 PC에서 T:\\ 드라이브가 같은 계정으로 연결되어 있어야 합니다. 관리자 권한/일반 권한 차이로 드라이브 매핑이 안 보일 수 있습니다."),
            ("실행 파일 테스트", "exe를 만든 뒤 로컬 폴더 1개와 팀룸 폴더 1개를 각각 Load Path로 확인하고, 결과가 *_PASS와 *_MASK에 저장되는지 확인합니다."),
        ]
        return self._add_section(parent, row, "배포 전 체크", body)

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
                wraplength=900,
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
                wraplength=440,
                justify=tk.LEFT,
            ).grid(row=index * 2, column=2, sticky=tk.NW, pady=(0, 6))
        return row + 1
