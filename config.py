from pathlib import Path

BASE_DIR = Path(__file__).parent

RAW_DATA_DIR = BASE_DIR / "data" / "raw"

PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

LOG_DIR = BASE_DIR / "logs"

SUPPORTED_EXTENSIONS = [
    ".pdf",
    ".docx",
    ".txt",
    ".png",
    ".jpg",
    ".jpeg"
]