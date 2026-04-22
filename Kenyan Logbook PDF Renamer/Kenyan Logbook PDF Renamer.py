import argparse
import csv
import logging
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Optional

import cv2
import easyocr
import numpy as np
import pytesseract
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance
from tqdm import tqdm


# ─────────────────────────────────────────────
# CONFIGURATION DATACLASS
# ─────────────────────────────────────────────

@dataclass
class Config:
    input_dir:        str   = "input_docs"
    success_dir:      str   = "success_docs"
    partial_dir:      str   = "partial_docs"
    failed_dir:       str   = "failed_docs"
    log_dir:          str   = "logs"

    # Windows: r"C:\Users\...\poppler-xx\Library\bin"
    # Linux:   "" (poppler is on PATH)
    poppler_path:     str   = ""

    # Windows: r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    # Linux:   "" (tesseract is on PATH)
    tesseract_cmd:    str   = ""

    languages:        list  = field(default_factory=lambda: ['en'])

    dpi_fast:         int   = 150   # first pass — quick
    dpi_full:         int   = 300   # retry pass — detailed

    min_confidence:   float = 0.40  # EasyOCR tokens below this are dropped
    fallback_conf:    float = 0.60  # trigger Tesseract if mean conf below this

    workers:          int   = 3     # parallel PDFs (raise carefully — memory hungry)

    skip_existing:    bool  = True  # skip PDFs already renamed in output folders
    gpu:              bool  = False


# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

def setup_logging(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"run_{ts}.log")

    logger = logging.getLogger("logbook_renamer")
    logger.setLevel(logging.DEBUG)

    # File handler — verbose
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

    # Console handler — INFO only (tqdm will handle the progress bar)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ─────────────────────────────────────────────
# IMAGE PREPROCESSING PIPELINE
# ─────────────────────────────────────────────

def deskew(image: np.ndarray) -> np.ndarray:
    """
    Detect and correct scan skew using minAreaRect on thresholded pixels.
    Skewed text is one of the largest accuracy killers for OCR.
    Skips correction when the detected angle is < 0.5° (noise).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    inverted = cv2.bitwise_not(gray)
    _, thresh = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) == 0:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if abs(angle) < 0.5:
        return image
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def boost_contrast(gray: np.ndarray, factor: float = 1.8) -> np.ndarray:
    """PIL contrast enhancement on a grayscale numpy array."""
    pil = Image.fromarray(gray)
    pil = ImageEnhance.Contrast(pil).enhance(factor)
    return np.array(pil)


def denoise(gray: np.ndarray) -> np.ndarray:
    """
    Non-local means denoising.
    Preserves edges better than Gaussian blur — important for thin strokes.
    h=10, templateWindowSize=7, searchWindowSize=21 is a well-tested balance.
    """
    return cv2.fastNlMeansDenoising(gray, h=10,
                                    templateWindowSize=7,
                                    searchWindowSize=21)


def adaptive_binarize(gray: np.ndarray) -> np.ndarray:
    """
    Adaptive Gaussian thresholding handles uneven scanner lighting.
    Global Otsu fails on pages with dark margins or shadows.
    blockSize=15, C=8 works for most logbook scans.
    """
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=15,
        C=8,
    )


def sharpen(image: np.ndarray) -> np.ndarray:
    """Unsharp mask kernel — sharpens character edges without blowing out noise."""
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)


def preprocess(pil_image: Image.Image) -> np.ndarray:
    """
    Full preprocessing pipeline.
    Returns a single-channel (grayscale) uint8 numpy array ready for OCR.
    Pipeline order is intentional:
      1. Deskew on colour (needs colour for rotation quality)
      2. Convert to gray
      3. Boost contrast before binarizing
      4. Denoise before binarizing (not after — denoising binary is less effective)
      5. Adaptive binarize
      6. Sharpen for crisper strokes
    """
    img = np.array(pil_image.convert("RGB"))
    img = deskew(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = boost_contrast(gray)
    gray = denoise(gray)
    binary = adaptive_binarize(gray)
    binary = sharpen(binary)
    return binary


# ─────────────────────────────────────────────
# OCR CORRECTION TABLES
# ─────────────────────────────────────────────

LETTER_TO_NUMBER = {
    'A': '4', 'B': '8', 'S': '5', 'Z': '2',
    'O': '0', 'Q': '0', 'I': '1', 'L': '1',
    'G': '6', 'T': '7', 'D': '0', 'C': '0',
}
NUMBER_TO_LETTER = {
    '0': 'O', '1': 'I', '2': 'Z', '5': 'S',
    '8': 'B', '4': 'A', '6': 'G', '7': 'T',
}
POS2_CORRECTION = {
    'O': 'G', 'S': 'G', 'U': 'G', 'Q': 'G',
    '0': 'G', '6': 'G', '8': 'G', 'C': 'G',
}


# ─────────────────────────────────────────────
# EXTRACTION LOGIC
# ─────────────────────────────────────────────

def correct_registration(raw: str) -> Optional[str]:
    """
    Apply position-aware OCR corrections to a raw 8-char candidate.
    Returns corrected string if it passes validation, else None.
    """
    if not raw or len(raw) != 8 or not raw.startswith('KM'):
        return None

    chars = list(raw)

    # Position 2: must be a letter (typically G) — fix common misreads
    if chars[2] in POS2_CORRECTION:
        chars[2] = POS2_CORRECTION[chars[2]]

    # Positions 4-6: must be digits
    for i in range(4, 7):
        if chars[i].isalpha() and chars[i] in LETTER_TO_NUMBER:
            chars[i] = LETTER_TO_NUMBER[chars[i]]

    # Position 7: must be a letter
    if chars[7].isdigit() and chars[7] in NUMBER_TO_LETTER:
        chars[7] = NUMBER_TO_LETTER[chars[7]]

    corrected = ''.join(chars)
    if corrected[4:7].isdigit() and corrected[7].isalpha():
        return corrected
    return None


def score_registration(candidate: str) -> int:
    """
    Heuristic score for how likely a candidate is the true registration.
    Used to pick the best match when multiple candidates are found.
    """
    if len(candidate) != 8:
        return 0
    score = 0
    if candidate[:2] == 'KM': score += 40
    if candidate[2] == 'G':   score += 20
    if candidate[3].isalpha(): score += 10
    if candidate[4:7].isdigit(): score += 20
    if candidate[7].isalpha(): score += 10
    return score


def extract_registration(text: str) -> Optional[str]:
    if not text:
        return None
    upper = text.upper()

    patterns = [
        r'\b(KM[A-Z0-9]{6})\b',
        r'\b(KMGU[A-Z0-9]{4})\b',
        r'REGISTRATION\s*[:.\s]*([A-Z0-9]{8})',
        r'REG\.?\s*NO\.?\s*[:.\s]*([A-Z0-9]{8})',
        r'PLATE\s*[:.\s]*([A-Z0-9]{8})',
    ]

    best, best_score = None, 0
    for pattern in patterns:
        for match in re.findall(pattern, upper):
            corrected = correct_registration(match)
            if corrected:
                s = score_registration(corrected)
                if s > best_score:
                    best, best_score = corrected, s

    return best


def extract_serial(text: str) -> Optional[str]:
    if not text:
        return None
    upper = text.upper()

    # Strategy 1: clean exact match
    m = re.search(r'\b(N\d{7}[A-Z])\b', upper)
    if m:
        return m.group(1)

    # Strategy 2: space between digits and letter  (N8091011 C)
    m = re.search(r'N(\d{7})\s+([A-Z])', upper)
    if m:
        return f"N{m.group(1)}{m.group(2)}"

    # Strategy 3: separator character  (N8091014,C  N8091014-C)
    m = re.search(r'N(\d{7})[,;:\-_]([A-Z0-9])', upper)
    if m:
        last = m.group(2)
        if last.isdigit():
            last = NUMBER_TO_LETTER.get(last, 'X')
        return f"N{m.group(1)}{last}"

    # Strategy 4: digit where letter should be  (N8091051 0)
    m = re.search(r'N(\d{7})\s*([0-9])', upper)
    if m:
        return f"N{m.group(1)}{NUMBER_TO_LETTER.get(m.group(2), 'X')}"

    # Strategy 5: char substitution on 9-char N-pattern
    for candidate in re.findall(r'N[A-Z0-9]{8}', upper):
        body = candidate[1:8]
        body = re.sub(r'[OQILZSBTG]', lambda x: LETTER_TO_NUMBER.get(x.group(), x.group()), body)
        last = candidate[8]
        if last.isdigit():
            last = NUMBER_TO_LETTER.get(last, 'X')
        fixed = f"N{body}{last}"
        if re.match(r'^N\d{7}[A-Z]$', fixed):
            return fixed

    # Strategy 6: scan first 10 lines for N + digits + nearby letter
    for line in upper.split('\n')[:10]:
        nm = re.search(r'N([0-9]{7,8})', line.strip())
        if nm:
            digits = nm.group(1)[:7]
            after  = line[nm.end():nm.end() + 5]
            lm     = re.search(r'([A-Z])', after)
            if lm:
                return f"N{digits}{lm.group(1)}"

    return None


def extract_info(text: str) -> dict:
    return {
        'registration': extract_registration(text),
        'serial':       extract_serial(text),
    }


# ─────────────────────────────────────────────
# OCR ENGINES
# ─────────────────────────────────────────────

def run_easyocr(image: np.ndarray, reader: easyocr.Reader,
                min_conf: float) -> tuple[str, float]:
    """
    Run EasyOCR with per-token confidence.
    Tokens below min_conf are dropped before joining — removes garbage text
    that confuses the regex extractors.
    Returns (joined_text, mean_confidence_of_kept_tokens).
    """
    results = reader.readtext(image, detail=1, paragraph=False)
    if not results:
        return "", 0.0

    kept = [(text, conf) for _, text, conf in results if conf >= min_conf]
    if not kept:
        return "", 0.0

    text      = "\n".join(t for t, _ in kept)
    mean_conf = sum(c for _, c in kept) / len(kept)
    return text, mean_conf


def run_tesseract(image: np.ndarray) -> str:
    """
    Tesseract with OEM 3 (LSTM), PSM 6 (single text block).
    Character whitelist avoids garbage symbols — logbooks only contain
    uppercase letters and digits.
    """
    config = (
        "--oem 3 --psm 6 "
        "-c tessedit_char_whitelist="
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
    )
    return pytesseract.image_to_string(image, config=config)


def ocr_with_fallback(pil_image: Image.Image, reader: easyocr.Reader,
                      cfg: Config, logger: logging.Logger) -> str:
    """
    1. Preprocess image.
    2. Run EasyOCR with confidence filtering.
    3. If mean confidence < fallback_conf AND extraction failed → run Tesseract.
    4. Return whichever produced more extracted fields.
    """
    processed = preprocess(pil_image)

    easy_text, easy_conf = run_easyocr(processed, reader, cfg.min_confidence)
    logger.debug(f"    EasyOCR mean confidence: {easy_conf:.2f}")

    easy_info = extract_info(easy_text)
    easy_hits = sum(1 for v in easy_info.values() if v)

    if easy_hits == 2:
        # Perfect — no need for Tesseract
        return easy_text

    if easy_conf < cfg.fallback_conf:
        logger.debug("    Low confidence — running Tesseract fallback")
        tess_text = run_tesseract(processed)
        tess_info = extract_info(tess_text)
        tess_hits = sum(1 for v in tess_info.values() if v)

        if tess_hits > easy_hits:
            logger.debug("    Tesseract outperformed EasyOCR — using Tesseract result")
            return tess_text

    return easy_text


# ─────────────────────────────────────────────
# PDF → PIL IMAGE
# ─────────────────────────────────────────────

def pdf_to_pil(pdf_path: str, dpi: int,
               poppler_path: Optional[str]) -> Optional[Image.Image]:
    """Convert only the first page of a PDF to a PIL Image at the given DPI."""
    kwargs = dict(first_page=1, last_page=1, dpi=dpi, fmt="jpeg")
    if poppler_path:
        kwargs["poppler_path"] = poppler_path
    try:
        pages = convert_from_path(pdf_path, **kwargs)
        return pages[0] if pages else None
    except Exception as e:
        return None


# ─────────────────────────────────────────────
# FILE HELPERS
# ─────────────────────────────────────────────

def sanitize(name: str) -> str:
    for ch in r'<>:"/\|?*':
        name = name.replace(ch, '')
    return name


def unique_path(target: str) -> str:
    """Append _1, _2 … if the target path already exists."""
    if not os.path.exists(target):
        return target
    stem, ext = os.path.splitext(target)
    for i in range(1, 10_000):
        candidate = f"{stem}_{i}{ext}"
        if not os.path.exists(candidate):
            return candidate
    return target


def already_processed(filename: str, *output_dirs: str) -> bool:
    """Return True if this filename already exists in any output folder."""
    for d in output_dirs:
        if os.path.exists(os.path.join(d, filename)):
            return True
    return False


# ─────────────────────────────────────────────
# SINGLE PDF PROCESSOR
# ─────────────────────────────────────────────

# Thread-safe counter for progress tracking
_lock = Lock()
_counters: dict = {}


def process_pdf(pdf_path: Path, cfg: Config, reader: easyocr.Reader,
                success_dir: str, partial_dir: str, failed_dir: str,
                logger: logging.Logger) -> dict:
    """
    Process one PDF end-to-end.
    Returns a result dict for the CSV report.
    Thread-safe — reader.readtext is safe for concurrent CPU calls.
    """
    start = time.time()
    result = {
        "file":         pdf_path.name,
        "registration": None,
        "serial":       None,
        "status":       "failed",
        "dest":         "",
        "elapsed_s":    0.0,
        "passes":       0,
    }

    logger.debug(f"  ▶ {pdf_path.name}")

    # Pass 1: fast DPI
    pil = pdf_to_pil(str(pdf_path), cfg.dpi_fast,
                     cfg.poppler_path or None)
    if pil is None:
        logger.warning(f"  ❌ PDF conversion failed: {pdf_path.name}")
        shutil.move(str(pdf_path), unique_path(os.path.join(failed_dir, pdf_path.name)))
        result["elapsed_s"] = round(time.time() - start, 2)
        return result

    text = ocr_with_fallback(pil, reader, cfg, logger)
    info = extract_info(text)
    result["passes"] = 1

    # Pass 2: full DPI only if extraction is incomplete
    if not (info.get("registration") and info.get("serial")):
        logger.debug(f"    Pass 1 incomplete — retrying at {cfg.dpi_full} DPI")
        pil_full = pdf_to_pil(str(pdf_path), cfg.dpi_full,
                               cfg.poppler_path or None)
        if pil_full:
            text_full = ocr_with_fallback(pil_full, reader, cfg, logger)
            info_full  = extract_info(text_full)
            result["passes"] = 2

            # Merge: accept any field that improved
            if info_full.get("registration") and not info.get("registration"):
                info["registration"] = info_full["registration"]
            if info_full.get("serial") and not info.get("serial"):
                info["serial"] = info_full["serial"]

    reg    = info.get("registration")
    serial = info.get("serial")
    result.update({"registration": reg, "serial": serial})

    # Determine destination
    if reg and serial:
        new_name    = sanitize(f"{reg}_@{serial}.pdf")
        target_dir  = success_dir
        status      = "complete"
    elif reg:
        new_name    = sanitize(f"{reg}_@NO_SERIAL.pdf")
        target_dir  = partial_dir
        status      = "partial_reg"
    elif serial:
        new_name    = sanitize(f"NO_REG_@{serial}.pdf")
        target_dir  = partial_dir
        status      = "partial_serial"
    else:
        new_name    = pdf_path.name
        target_dir  = failed_dir
        status      = "failed"

    dest = unique_path(os.path.join(target_dir, new_name))
    shutil.move(str(pdf_path), dest)

    result.update({
        "status":    status,
        "dest":      os.path.basename(dest),
        "elapsed_s": round(time.time() - start, 2),
    })

    icon = {"complete": "✅", "partial_reg": "⚠️",
            "partial_serial": "⚠️", "failed": "❌"}.get(status, "?")
    logger.debug(f"  {icon} {status.upper():15s}  reg={reg}  serial={serial}  "
                 f"({result['elapsed_s']}s, {result['passes']} pass(es))")
    return result


# ─────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────

def write_csv_report(results: list[dict], log_dir: str):
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(log_dir, f"report_{ts}.csv")
    fields = ["file", "registration", "serial", "status", "dest",
              "elapsed_s", "passes"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    return path


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def parse_args() -> Config:
    p = argparse.ArgumentParser(
        description="Kenyan Logbook PDF Renamer — Enhanced Version"
    )
    p.add_argument("--input-dir",      default="input_docs")
    p.add_argument("--success-dir",    default="success_docs")
    p.add_argument("--partial-dir",    default="partial_docs")
    p.add_argument("--failed-dir",     default="failed_docs")
    p.add_argument("--log-dir",        default="logs")
    p.add_argument("--poppler-path",   default="",
                   help="Path to poppler bin folder (Windows only)")
    p.add_argument("--tesseract-cmd",  default="",
                   help="Path to tesseract.exe (Windows only)")
    p.add_argument("--dpi-fast",       type=int,   default=150)
    p.add_argument("--dpi-full",       type=int,   default=300)
    p.add_argument("--min-confidence", type=float, default=0.40)
    p.add_argument("--fallback-conf",  type=float, default=0.60)
    p.add_argument("--workers",        type=int,   default=3)
    p.add_argument("--gpu",            action="store_true")
    p.add_argument("--no-skip",        action="store_true",
                   help="Re-process files even if output already exists")
    args = p.parse_args()

    return Config(
        input_dir       = args.input_dir,
        success_dir     = args.success_dir,
        partial_dir     = args.partial_dir,
        failed_dir      = args.failed_dir,
        log_dir         = args.log_dir,
        poppler_path    = args.poppler_path,
        tesseract_cmd   = args.tesseract_cmd,
        dpi_fast        = args.dpi_fast,
        dpi_full        = args.dpi_full,
        min_confidence  = args.min_confidence,
        fallback_conf   = args.fallback_conf,
        workers         = args.workers,
        gpu             = args.gpu,
        skip_existing   = not args.no_skip,
    )


def main():
    cfg    = parse_args()
    logger = setup_logging(cfg.log_dir)

    # Apply paths for Windows tools
    if cfg.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = cfg.tesseract_cmd

    base = os.path.dirname(os.path.abspath(__file__))

    input_dir   = os.path.join(base, cfg.input_dir)
    success_dir = os.path.join(base, cfg.success_dir)
    partial_dir = os.path.join(base, cfg.partial_dir)
    failed_dir  = os.path.join(base, cfg.failed_dir)

    if not os.path.isdir(input_dir):
        logger.error(f"Input folder not found: {input_dir}")
        sys.exit(1)

    for d in [success_dir, partial_dir, failed_dir]:
        os.makedirs(d, exist_ok=True)

    # Collect PDFs (case-insensitive, deduplicated)
    seen: dict[str, Path] = {}
    for p in Path(input_dir).glob("*"):
        if p.suffix.lower() == ".pdf":
            key = p.name.lower()
            if key not in seen:
                seen[key] = p
    pdf_files = list(seen.values())

    if not pdf_files:
        logger.info("No PDF files found in input folder.")
        sys.exit(0)

    # Skip already-processed files
    if cfg.skip_existing:
        before = len(pdf_files)
        pdf_files = [
            p for p in pdf_files
            if not already_processed(p.name, success_dir, partial_dir, failed_dir)
        ]
        skipped = before - len(pdf_files)
        if skipped:
            logger.info(f"Skipping {skipped} file(s) already in output folders.")

    total = len(pdf_files)
    if total == 0:
        logger.info("Nothing to process.")
        sys.exit(0)

    # ── Banner ────────────────────────────────
    logger.info("\n" + "═" * 68)
    logger.info("  KENYAN LOGBOOK RENAMER — ENHANCED LOCAL VERSION")
    logger.info("═" * 68)
    logger.info(f"  Input folder : {input_dir}")
    logger.info(f"  Files to run : {total}")
    logger.info(f"  Fast DPI     : {cfg.dpi_fast}  |  Full DPI: {cfg.dpi_full}")
    logger.info(f"  Workers      : {cfg.workers}")
    logger.info(f"  GPU          : {cfg.gpu}")
    logger.info("═" * 68 + "\n")

    # ── Initialise OCR (once, shared across threads) ──────────────────
    logger.info("Initialising EasyOCR …")
    reader = easyocr.Reader(cfg.languages, gpu=cfg.gpu, verbose=False)
    logger.info("EasyOCR ready.\n")

    # ── Parallel processing ───────────────────────────────────────────
    results   = []
    run_start = time.time()

    with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
        futures = {
            pool.submit(
                process_pdf,
                pdf_path, cfg, reader,
                success_dir, partial_dir, failed_dir,
                logger,
            ): pdf_path
            for pdf_path in pdf_files
        }

        with tqdm(total=total, unit="pdf", dynamic_ncols=True,
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
                  ) as pbar:
            for future in as_completed(futures):
                pdf_path = futures[future]
                try:
                    res = future.result()
                except Exception as exc:
                    logger.error(f"  ❌ Unhandled error for {pdf_path.name}: {exc}")
                    res = {"file": pdf_path.name, "status": "error",
                           "registration": None, "serial": None,
                           "dest": "", "elapsed_s": 0.0, "passes": 0}
                results.append(res)
                pbar.set_postfix_str(
                    f"{res['status']}: {res['file'][:30]}", refresh=True
                )
                pbar.update(1)

    # ── Summary ───────────────────────────────────────────────────────
    total_time = round(time.time() - run_start, 1)
    by_status  = {}
    for r in results:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1

    complete       = by_status.get("complete",       0)
    partial_reg    = by_status.get("partial_reg",    0)
    partial_serial = by_status.get("partial_serial", 0)
    failed_count   = by_status.get("failed",         0) + by_status.get("error", 0)
    avg_time       = round(total_time / total, 1) if total else 0

    logger.info("\n" + "═" * 68)
    logger.info("  RESULTS")
    logger.info("═" * 68)
    logger.info(f"  ✅  Complete  (reg + serial) : {complete}")
    logger.info(f"  ⚠️   Partial  (reg only)      : {partial_reg}")
    logger.info(f"  ⚠️   Partial  (serial only)   : {partial_serial}")
    logger.info(f"  ❌  Failed                   : {failed_count}")
    logger.info(f"  ─────────────────────────────────")
    logger.info(f"  📈  True success rate        : {complete/total*100:.1f}%")
    logger.info(f"  ⏱   Total time               : {total_time}s  ({avg_time}s/file)")
    logger.info("═" * 68)

    # ── CSV report ────────────────────────────────────────────────────
    report_path = write_csv_report(results, os.path.join(base, cfg.log_dir))
    logger.info(f"\n  📄 CSV report: {report_path}")
    logger.info(f"  ✅  Complete  → {success_dir}")
    logger.info(f"  ⚠️   Partial   → {partial_dir}")
    logger.info(f"  ❌  Failed    → {failed_dir}")
    logger.info("═" * 68 + "\n")


if __name__ == "__main__":
    main()