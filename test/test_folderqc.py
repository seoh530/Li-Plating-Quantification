# test_folderqc.py  (delete after testing)
import logging
from modules.qc.folder_qc import FolderQC

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

qc = FolderQC("config/thresholds.yaml")
result = qc.validate(
    root_path="C:/PhotoJig",
    date_str="2026-02-09",
    analysis_id="AHG06113",
)
print(f"\nStatus  : {result.status}")
print(f"Customer: {result.customer}")
for reason in result.fail_reasons:
    print(f"  FAIL  → {reason}")
for cell in result.cells:
    print(f"  Cell '{cell.cell_name}': FRONT={cell.front_count}  BACK={cell.back_count}  {cell.status}")