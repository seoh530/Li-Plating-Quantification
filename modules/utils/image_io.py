import contextlib
import os
import time
import uuid
from pathlib import Path

import cv2
import numpy as np


@contextlib.contextmanager
def _suppress_stderr():
    stderr_fd = None
    devnull_fd = None
    saved_fd = None
    try:
        stderr_fd = os.dup(2)
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            devnull_fd = os.dup(devnull.fileno())
            saved_fd = os.dup(2)
            os.dup2(devnull_fd, 2)
            yield
    finally:
        if saved_fd is not None:
            os.dup2(saved_fd, 2)
            os.close(saved_fd)
        elif stderr_fd is not None:
            os.dup2(stderr_fd, 2)
        if devnull_fd is not None:
            os.close(devnull_fd)
        if stderr_fd is not None:
            os.close(stderr_fd)


def quiet_imread(path: str | Path, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray | None:
    with _suppress_stderr():
        return cv2.imread(str(path), flags)


def atomic_imwrite_png(
    path: str | Path,
    img: np.ndarray,
    compression: int = 1,
) -> bool:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ok, buf = cv2.imencode(
        ".png",
        img,
        [cv2.IMWRITE_PNG_COMPRESSION, int(compression)],
    )
    if not ok:
        return False

    payload = buf.tobytes()
    last_error: Exception | None = None
    for attempt in range(10):
        tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            with open(tmp, "wb") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
            return True
        except PermissionError as exc:
            last_error = exc
            tmp.unlink(missing_ok=True)
            time.sleep(0.05 * (attempt + 1))
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

    if last_error is not None:
        raise last_error
    return False
