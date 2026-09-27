"""Color-coded console logging plus an optional plain-text log file.

Usage:
    from logger import get_logger, log_step
    log = get_logger(__name__)
    with log_step(log, "Blocking"):
        ...
"""
import logging
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

ROOT_NAME = "ber"

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
COLORS = {
    logging.DEBUG: "\033[36m",           # cyan
    logging.INFO: "\033[32m",            # green
    logging.WARNING: "\033[33m",         # yellow
    logging.ERROR: "\033[31m",           # red
    logging.CRITICAL: "\033[1;41;97m",   # bold white on red
}
STEP_COLOR = "\033[1;35m"                # bold magenta for step banners
METRIC_COLOR = "\033[1;34m"              # bold blue for key metrics

# Custom levels: STEP (banners) and METRIC (headline numbers) sit at INFO level
# priority so they are shown by default but stand out visually.
STEP = logging.INFO + 1
METRIC = logging.INFO + 2
logging.addLevelName(STEP, "STEP")
logging.addLevelName(METRIC, "METRIC")
COLORS[STEP] = STEP_COLOR
COLORS[METRIC] = METRIC_COLOR


def _enable_windows_ansi() -> bool:
    """Turn on ANSI escape processing in the Windows console (Win10+)."""
    if os.name != "nt":
        return True
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
    except Exception:
        return False


class ColorFormatter(logging.Formatter):
    def __init__(self, use_color: bool):
        super().__init__(datefmt="%H:%M:%S")
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        ts = self.formatTime(record, self.datefmt)
        level = f"{record.levelname:<7}"
        module = record.name.removeprefix(ROOT_NAME + ".")
        msg = record.getMessage()
        if record.exc_info:
            msg += "\n" + self.formatException(record.exc_info)
        if not self.use_color:
            return f"{ts} | {level} | {module:<10} | {msg}"
        c = COLORS.get(record.levelno, "")
        body = f"{c}{msg}{RESET}" if record.levelno >= logging.WARNING or record.levelno in (STEP, METRIC) else msg
        return f"{DIM}{ts}{RESET} | {c}{BOLD}{level}{RESET} | {DIM}{module:<10}{RESET} | {body}"


class PlainFormatter(logging.Formatter):
    def __init__(self):
        super().__init__("%(asctime)s | %(levelname)-7s | %(name)-14s | %(message)s",
                         datefmt="%Y-%m-%d %H:%M:%S")


def setup_logging(level: str = "INFO", log_file: Path | None = None, color: bool = True) -> logging.Logger:
    """Configure the package root logger once, from main()."""
    root = logging.getLogger(ROOT_NAME)
    root.setLevel(logging.DEBUG)
    root.handlers.clear()
    root.propagate = False

    use_color = color and sys.stdout.isatty() and _enable_windows_ansi() and "NO_COLOR" not in os.environ
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(getattr(logging, level.upper(), logging.INFO))
    console.setFormatter(ColorFormatter(use_color))
    root.addHandler(console)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setLevel(logging.DEBUG)  # the file always gets full detail
        fh.setFormatter(PlainFormatter())
        root.addHandler(fh)

    # Quiet noisy third-party loggers
    for name in ("lightgbm", "sentence_transformers", "urllib3", "filelock", "huggingface_hub"):
        logging.getLogger(name).setLevel(logging.WARNING)
    return root


def get_logger(name: str) -> logging.Logger:
    """Module logger under the package root, e.g. 'ber.blocking'."""
    short = name.rsplit(".", 1)[-1]
    return logging.getLogger(f"{ROOT_NAME}.{short}")


def metric(log: logging.Logger, msg: str, *args) -> None:
    log.log(METRIC, msg, *args)


@contextmanager
def log_step(log: logging.Logger, title: str):
    """Banner at start/end of a pipeline step with elapsed time; logs failures."""
    log.log(STEP, "=" * 12 + f" {title} " + "=" * 12)
    t0 = time.perf_counter()
    try:
        yield
    except Exception as e:
        if not getattr(e, "_ber_logged", False):
            log.exception("%s failed after %.1fs", title, time.perf_counter() - t0)
            try:
                e._ber_logged = True   # avoid repeating the traceback in outer handlers
            except AttributeError:
                pass
        raise
    log.log(STEP, "%s done in %.1fs", title, time.perf_counter() - t0)


def fmt_int(n) -> str:
    return f"{int(n):,}"


def fmt_pct(x) -> str:
    return f"{100 * float(x):.2f}%"
