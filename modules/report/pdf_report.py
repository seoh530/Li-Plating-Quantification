"""Analysis-level PDF: trends, exact CSV values, mask accumulation, combined sheets."""

from datetime import datetime
from io import BytesIO
from pathlib import Path
import colorsys
from reportlab.pdfbase.pdfmetrics import stringWidth
import os
import tempfile

import cv2
import numpy as np
from PIL import Image
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.ticker import MaxNLocator
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib.utils import ImageReader
from reportlab.lib.colors import HexColor

from modules.report.report_data import ReportDataset, summary_frame, load_dataset
from modules.utils.image_io import quiet_imread

COLORS = (
    "#2563eb",
    "#f97316",
    "#16a34a",
    "#9333ea",
    "#0891b2",
    "#db2777",
    "#854d0e",
    "#475569",
)


def accumulate_masks(side):
    """Use per-pixel valid layer counts, with transparent pixels excluded."""
    li = non = denom = None
    for path in side.masks:
        image = quiet_imread(path)
        if image is None or image.ndim != 3 or image.shape[2] not in (3, 4):
            raise ValueError(f"Cannot read mask: {path.name}")
        valid = (
            image[:, :, 3] > 0
            if image.shape[2] == 4
            else np.ones(image.shape[:2], bool)
        )
        yy, xx = np.where(valid)
        if not len(xx):
            raise ValueError(f"Mask has no valid pixels: {path.name}")
        image = image[yy.min() : yy.max() + 1, xx.min() : xx.max() + 1]
        valid = valid[yy.min() : yy.max() + 1, xx.min() : xx.max() + 1]
        salt = np.all(image[:, :, :3] == (0, 0, 255), axis=2) & valid
        inactive = np.all(image[:, :, :3] == (255, 0, 0), axis=2) & valid
        if denom is None:
            h, w = image.shape[:2]
            scale = min(1, 1600 / max(h, w))
            shape = (max(1, round(w * scale)), max(1, round(h * scale)))
            li = np.zeros(shape[::-1], np.float32)
            non = li.copy()
            denom = li.copy()
            reference_ratio = w / h
        elif abs((image.shape[1] / image.shape[0]) / reference_ratio - 1) > 0.2:
            raise ValueError(
                f"Cell #{side.cell} {side.face}: masks have inconsistent orientation / aspect ratio."
            )
        li += cv2.resize(
            salt.astype(np.float32), shape, interpolation=cv2.INTER_NEAREST
        )
        non += cv2.resize(
            inactive.astype(np.float32), shape, interpolation=cv2.INTER_NEAREST
        )
        denom += cv2.resize(
            valid.astype(np.float32), shape, interpolation=cv2.INTER_NEAREST
        )
    if denom is None:
        raise ValueError("No masks to accumulate.")
    a = np.divide(li, denom, out=np.zeros_like(li), where=denom > 0)
    b = np.divide(non, denom, out=np.zeros_like(non), where=denom > 0)
    result = np.full((*a.shape, 3), 255, np.float32)
    result = (
        result * (1 - a - b)[:, :, None]
        + np.array([220, 38, 38]) * a[:, :, None]
        + np.array([37, 99, 235]) * b[:, :, None]
    )
    result[denom == 0] = (241, 245, 249)
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def _chart_image(figure):
    stream = BytesIO()
    FigureCanvasAgg(figure).print_png(stream)
    figure.clear()
    stream.seek(0)
    return stream


def line_chart(dataset):
    faces = sorted(
        {s.face for s in dataset.sides},
        key=lambda f: ({"FRONT": 0, "BACK": 1}.get(f, 2), f),
    )
    fig = Figure(figsize=(11, 5.2), dpi=160, layout="constrained")
    for index, face in enumerate(faces):
        ax = fig.add_subplot(len(faces), 1, index + 1)
        for color_index, cell in enumerate(dataset.cells):
            rows = dataset.frame[
                (dataset.frame.Cell == cell) & (dataset.frame.Face == face)
            ].sort_values("ImageNum")
            if len(rows):
                ax.plot(
                    rows.ImageNum,
                    rows.li_salt_ratio * 100,
                    label=f"Cell #{cell}",
                    color=COLORS[color_index]
                    if len(dataset.cells) <= len(COLORS)
                    else colorsys.hsv_to_rgb(
                        color_index / len(dataset.cells), 0.72, 0.78
                    ),
                    linewidth=1.8,
                    marker="o",
                    markersize=3,
                )
        ax.set_title(face, loc="left", fontsize=11, fontweight="bold")
        ax.set_ylabel("Li-salt (%)")
        ax.set_xlabel("Layer number")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.18)
        ax.set_ylim(bottom=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="upper right", fontsize=8, ncol=min(4, len(dataset.cells)))
    return _chart_image(fig)


def bar_chart(dataset):
    means = (
        dataset.frame.groupby(["Cell", "Face"], sort=False)[
            ["nonactive_ratio", "li_salt_ratio"]
        ].mean()
        * 100
    )
    fig = Figure(figsize=(11, 2.8), dpi=160, layout="constrained")
    ax = fig.add_subplot(111)
    labels = [f"Cell #{c}\n{face}" for c, face in means.index]
    x = np.arange(len(means))
    width = 0.36
    ax.bar(
        x - width / 2, means.nonactive_ratio, width, label="Nonactive", color="#2563eb"
    )
    ax.bar(x + width / 2, means.li_salt_ratio, width, label="Li-salt", color="#dc2626")
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylabel("Mean layer area (%)")
    ax.grid(axis="y", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, ncol=2)
    return _chart_image(fig)


class PDFReport:
    def __init__(self, dataset: ReportDataset, destination: Path, created=None):
        self.data = dataset
        self.destination = Path(destination)
        self.created = created or datetime.now()
        self.width, self.height = landscape(A4)
        self.margin = 36
        self.y = 0
        self.page = 0

    def _text(self, x, y, text, size=10, color="#334155", bold=False):
        self.pdf.setFillColor(HexColor(color))
        self.pdf.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        self.pdf.drawString(x, y, str(text))

    def _new_page(self, title):
        if self.page:
            self.pdf.showPage()
        self.page += 1
        w, h = self.width, self.height
        m = self.margin
        self.pdf.setFillColor(HexColor("#153b63"))
        self.pdf.rect(0, h - 9, w, 9, fill=1, stroke=0)
        # Wrap long multi-cell titles instead of clipping the cell list.
        y = h - 38

        def lines(text, size):
            result = []
            line = ""
            for word in text.split():
                candidate = (line + " " + word).strip()
                if line and stringWidth(candidate, "Helvetica-Bold", size) > w - 2 * m:
                    result.append(line)
                    line = word
                else:
                    line = candidate
            return result + [line]

        for line in lines(title, 19):
            self._text(m, y, line, 19, "#0f233d", True)
            y -= 23
        identity = (
            self.data.analysis_id
            + " | "
            + ", ".join(f"Cell #{c}" for c in self.data.cells)
        )
        if title != identity:
            for line in lines(identity, 9):
                self._text(m, y + 4, line, 9)
                y -= 12
        self._text(
            m,
            y,
            f"Report date: {self.created:%Y-%m-%d %H:%M}   |   created by Seyeong Oh",
            9,
            "#64748b",
        )
        self.pdf.setStrokeColor(HexColor("#d7dde5"))
        self.pdf.line(m, 31, w - m, 31)
        self._text(m, 18, "Li-Salt Quantification", 8, "#64748b")
        self._text(w - m - 50, 18, f"Page {self.page}", 8, "#64748b")
        self.y = y - 24

    def _image(self, source, x, top, w, h, border=False):
        if isinstance(source, Path):
            array = quiet_imread(source)
            if array is None:
                raise ValueError(f"Cannot read report image: {source}")
            if array.ndim == 2:
                im = Image.fromarray(array)
            elif array.shape[2] == 4:
                im = Image.fromarray(cv2.cvtColor(array, cv2.COLOR_BGRA2RGBA))
            else:
                im = Image.fromarray(cv2.cvtColor(array, cv2.COLOR_BGR2RGB))
            im.thumbnail((2800, 2800))
            reader = ImageReader(im)
        elif isinstance(source, Image.Image):
            reader = ImageReader(source)
        else:
            reader = ImageReader(source)
        iw, ih = reader.getSize()
        scale = min(w / iw, h / ih)
        dw, dh = iw * scale, ih * scale
        left, bottom = x + (w - dw) / 2, top - h + (h - dh) / 2
        self.pdf.drawImage(reader, left, bottom, width=dw, height=dh, mask="auto")
        if border:
            self.pdf.setStrokeColor(HexColor("#cbd5e1"))
            self.pdf.setLineWidth(0.5)
            self.pdf.rect(left, bottom, dw, dh, stroke=1, fill=0)

    def _table(self, headers, rows, widths, continuation):
        row_height = 18

        def header():
            self.pdf.setFillColor(HexColor("#e8eff8"))
            self.pdf.rect(
                self.margin,
                self.y - row_height,
                sum(widths),
                row_height,
                fill=1,
                stroke=0,
            )
            x = self.margin
            for name, width in zip(headers, widths):
                self._text(x + 6, self.y - 12, name, 8, "#153b63", True)
                x += width
            self.y -= row_height

        if self.y < 82:
            self._new_page(continuation)
        header()
        for index, row in enumerate(rows):
            if self.y < 62:
                self._new_page(continuation)
                header()
            if index % 2 == 0:
                self.pdf.setFillColor(HexColor("#f6f8fb"))
                self.pdf.rect(
                    self.margin,
                    self.y - row_height,
                    sum(widths),
                    row_height,
                    fill=1,
                    stroke=0,
                )
            x = self.margin
            for value, width in zip(row, widths):
                self._text(x + 6, self.y - 12, value, 8)
                x += width
            self.y -= row_height
        self.y -= 16

    def render(self):
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            suffix=".pdf", prefix=".report_", dir=self.destination.parent
        )
        os.close(fd)
        try:
            self.pdf = canvas.Canvas(
                tmp, pagesize=(self.width, self.height), pageCompression=1
            )
            self.pdf.setTitle(
                self.data.analysis_id
                + " - "
                + ", ".join(f"Cell #{c}" for c in self.data.cells)
            )
            self.pdf.setAuthor("Seyeong Oh")
            self._new_page(
                self.data.analysis_id
                + " | "
                + ", ".join(f"Cell #{c}" for c in self.data.cells)
            )
            self._text(self.margin, self.y, "Li-salt by layer", 12, "#153b63", True)
            self._image(
                line_chart(self.data),
                self.margin,
                self.y - 10,
                self.width - 72,
                self.y - 57,
            )
            self._new_page("Nonactive & Li-salt")
            self._image(bar_chart(self.data), self.margin, self.y, self.width - 72, 195)
            self.y -= 210
            self._text(self.margin, self.y, "Layer statistics (%)", 11, "#153b63", True)
            self.y -= 10
            stats = summary_frame(self.data)
            statrows = [
                [
                    f"#{r.Cell}",
                    r.Face,
                    r.Metric,
                    str(r.N),
                    *(f"{getattr(r, k):.3f}" for k in ("Mean", "Median", "Min", "Max")),
                ]
                for r in stats.itertuples()
            ]
            self._table(
                [
                    "Cell",
                    "Face",
                    "Class",
                    "Layers",
                    "Mean %",
                    "Median %",
                    "Min %",
                    "Max %",
                ],
                statrows,
                [55, 75, 100, 60, 100, 100, 100, self.width - 72 - 590],
                "Layer statistics (%)",
            )
            if self.y < 100:
                self._new_page("Actual layer values (%)")
            self._text(
                self.margin, self.y, "Actual layer values (%)", 11, "#153b63", True
            )
            self.y -= 12
            raw = [
                [
                    f"#{r.Cell}",
                    r.Face,
                    str(r.ImageNum),
                    f"{r.li_salt_ratio * 100:.3f}",
                    f"{r.nonactive_ratio * 100:.3f}",
                ]
                for r in self.data.frame.itertuples()
            ]
            self._table(
                ["Cell", "Face", "Layer", "Li-salt %", "Nonactive %"],
                raw,
                [70, 100, 100, 240, self.width - 72 - 510],
                "Actual layer values (%)",
            )
            # Each face gets its own pages; never mix FRONT and BACK maps.
            for face in sorted(
                {s.face for s in self.data.sides},
                key=lambda f: ({"FRONT": 0, "BACK": 1}.get(f, 2), f),
            ):
                sides = sorted(
                    (s for s in self.data.sides if s.face == face), key=lambda s: s.cell
                )
                for start in range(0, len(sides), 6):
                    self._new_page(f"Mask accumulation | {face}")
                    self._text(
                        self.margin,
                        self.y,
                        "Red: Li-salt   Blue: Nonactive   White: no detection. Color intensity = fraction of valid layers.",
                        9,
                    )
                    self.y -= 18
                    self._text(
                        self.margin,
                        self.y,
                        "Masks only. Normalized electrode coordinates; no geometric registration.",
                        8,
                        "#64748b",
                    )
                    self.y -= 20
                    tile_w = (self.width - 72 - 24) / 3
                    for index, side in enumerate(sides[start : start + 6]):
                        row, col = divmod(index, 3)
                        top = self.y - row * 177
                        x = self.margin + col * (tile_w + 12)
                        self._text(
                            x,
                            top,
                            f"Cell #{side.cell} | {len(side.masks)} layers",
                            10,
                            "#153b63",
                            True,
                        )
                        self._image(
                            accumulate_masks(side),
                            x,
                            top - 14,
                            tile_w,
                            145,
                            border=True,
                        )
            # One cell at a time: PASS/MASK pair across; FRONT above BACK.
            for cell in self.data.cells:
                byface = {s.face: s for s in self.data.sides if s.cell == cell}
                pages = max(
                    max(len(s.combined), len(s.mask_combined)) for s in byface.values()
                )
                for index in range(pages):
                    self._new_page(
                        f"Combined images | Cell #{cell} | Sheet {index + 1}"
                    )
                    faces = sorted(
                        byface, key=lambda f: ({"FRONT": 0, "BACK": 1}.get(f, 2), f)
                    )
                    row_h = (self.y - 45) / max(len(faces), 1)
                    tile_w = (self.width - 84) / 2
                    for row, face in enumerate(faces):
                        side = byface[face]
                        top = self.y - row * row_h
                        for col, (title, paths) in enumerate(
                            [("PASS", side.combined), ("MASK", side.mask_combined)]
                        ):
                            x = self.margin + col * (tile_w + 12)
                            self._text(x, top, f"{face} | {title}", 10, "#153b63", True)
                            if index < len(paths):
                                self._image(
                                    paths[index], x, top - 12, tile_w, row_h - 25
                                )
                            else:
                                self._text(
                                    x, top - 35, "No sheet at this index", 9, "#64748b"
                                )
            self.pdf.save()
            os.replace(tmp, self.destination)
        finally:
            if Path(tmp).exists():
                Path(tmp).unlink()
        return self.destination


def create_pdf(analysis_path, config_path, destination):
    return PDFReport(
        load_dataset(Path(analysis_path), Path(config_path)), Path(destination)
    ).render()
