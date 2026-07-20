"""
Utilities Module

This module provides common utility functions for the RAG system including:
- Logging setup and configuration
- Input validation and sanitization  
- Output formatting and sanitization
- File operations and path handling
- Performance monitoring utilities
"""

import os
import re
import logging
import json
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from pathlib import Path

# Rich for beautiful console output
try:
    from rich.console import Console
    from rich.logging import RichHandler
    RICH_AVAILABLE = True
    console = Console(stderr=True)
except ImportError:
    RICH_AVAILABLE = False
    console = None

# Colorama for cross-platform colored output
try:
    from colorama import init, Fore, Style
    init()  # Initialize colorama
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False


def setup_logging(log_level: str = "INFO", log_dir: str = "logs") -> logging.Logger:
    """
    Set up logging configuration for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory to store log files
        
    Returns:
        Configured logger instance
    """
    try:
        # Create logs directory if it doesn't exist
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        # Configure logging format
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        date_format = "%Y-%m-%d %H:%M:%S"
        
        # Clear any existing handlers and close file descriptors.
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            handler.close()
        
        # Set up file handler
        log_file = os.path.join(log_dir, f"rag_system_{datetime.now().strftime('%Y%m%d')}.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_formatter = logging.Formatter(log_format, date_format)
        file_handler.setFormatter(file_formatter)
        
        # Set up console handler
        if RICH_AVAILABLE:
            console_handler = RichHandler(console=console, rich_tracebacks=True)
        else:
            console_handler = logging.StreamHandler()
        
        console_handler.setLevel(getattr(logging, log_level.upper()))
        
        if not RICH_AVAILABLE:
            console_formatter = logging.Formatter(log_format, date_format)
            console_handler.setFormatter(console_formatter)
        
        # Configure root logger
        root_logger.setLevel(getattr(logging, log_level.upper()))
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        # Log startup message
        logger = logging.getLogger(__name__)
        logger.info(f"Logging initialized - Level: {log_level}, File: {log_file}")
        
        return logger
        
    except Exception as e:
        print(f"Error setting up logging: {e}")
        # Fallback to basic logging
        logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO))
        return logging.getLogger(__name__)


def validate_input(text: str, config: Dict) -> bool:
    """
    Validate user input for safety and compliance.
    
    Args:
        text: Input text to validate
        config: System configuration
        
    Returns:
        True if input is valid, False otherwise
    """
    try:
        if not isinstance(text, str):
            return False
        
        # Check if input validation is enabled
        if not config.get("security", {}).get("enable_input_validation", True):
            return True
        
        # Length validation
        max_length = config.get("security", {}).get("max_query_length", 1000)
        if len(text) > max_length:
            logging.warning(f"Input exceeds maximum length: {len(text)} > {max_length}")
            return False
        
        # Empty input check
        if not text.strip():
            return False
        
        # Potentially harmful patterns
        dangerous_patterns = [
            r'<script[^>]*>.*?</script>',  # Script tags
            r'javascript:',               # JavaScript URLs
            r'vbscript:',                # VBScript URLs
            r'onload\s*=',               # Event handlers
            r'onerror\s*=',
            r'onclick\s*=',
            r'data:text/html',           # Data URLs
            r'eval\s*\(',                # Code evaluation
            r'exec\s*\(',
            r'system\s*\(',
            r'import\s+os',              # System imports
            r'import\s+subprocess',
            r'__import__',
            r'DROP\s+TABLE',             # SQL injection
            r'DELETE\s+FROM',
            r'UPDATE\s+.*\s+SET',
            r'INSERT\s+INTO',
            r';\s*DROP\s+',
            r'UNION\s+SELECT',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                logging.warning(f"Potentially dangerous pattern detected: {pattern}")
                return False
        
        # Character validation - allow only printable characters and common unicode
        if not all(c.isprintable() or c.isspace() for c in text):
            logging.warning("Non-printable characters detected in input")
            return False
        
        return True
        
    except Exception as e:
        logging.error(f"Error validating input: {e}")
        return False


def sanitize_output(text: str) -> str:
    """
    Sanitize output text for safe display.
    
    Args:
        text: Output text to sanitize
        
    Returns:
        Sanitized text
    """
    try:
        if not isinstance(text, str):
            text = str(text)
        
        # Remove potential HTML/script tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove excessive horizontal whitespace while preserving markdown line breaks.
        lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in text.splitlines()]
        text = "\n".join(lines)
        text = re.sub(r'\n{3,}', '\n\n', text).strip()
        
        # Limit length to prevent extremely long outputs
        max_output_length = 10000
        if len(text) > max_output_length:
            text = text[:max_output_length] + "... (truncated)"
        
        return text
        
    except Exception as e:
        logging.error(f"Error sanitizing output: {e}")
        return "Error processing response"


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.2f} {size_names[i]}"


def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """
    Safely load JSON string with fallback.
    
    Args:
        json_str: JSON string to parse
        default: Default value if parsing fails
        
    Returns:
        Parsed JSON or default value
    """
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default


def ensure_directory(path: str) -> bool:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path
        
    Returns:
        True if directory exists or was created successfully
    """
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logging.error(f"Error creating directory {path}: {e}")
        return False


def get_file_hash(file_path: str) -> Optional[str]:
    """
    Get SHA-256 hash of a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File hash or None if error
    """
    try:
        import hashlib
        
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        
        return hash_sha256.hexdigest()
        
    except Exception as e:
        logging.error(f"Error calculating file hash: {e}")
        return None


def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace and normalizing.
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    try:
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Normalize quotes
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r"[''']", "'", text)
        
        # Remove control characters except newlines and tabs
        text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)
        
        return text
        
    except Exception as e:
        logging.error(f"Error cleaning text: {e}")
        return text


def format_timestamp(timestamp: Optional[str] = None) -> str:
    """
    Format timestamp for display.
    
    Args:
        timestamp: ISO timestamp string or None for current time
        
    Returns:
        Formatted timestamp string
    """
    try:
        if timestamp:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        else:
            dt = datetime.now()
        
        return dt.strftime("%Y-%m-%d %H:%M:%S")
        
    except Exception:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def colorize_text(text: str, color: str = "white") -> str:
    """
    Add color to text for console output.
    
    Args:
        text: Text to colorize
        color: Color name (red, green, blue, yellow, etc.)
        
    Returns:
        Colorized text if colorama is available, otherwise plain text
    """
    if not COLORAMA_AVAILABLE:
        return text
    
    color_map = {
        'red': Fore.RED,
        'green': Fore.GREEN,
        'blue': Fore.BLUE,
        'yellow': Fore.YELLOW,
        'magenta': Fore.MAGENTA,
        'cyan': Fore.CYAN,
        'white': Fore.WHITE,
        'bright_red': Fore.LIGHTRED_EX,
        'bright_green': Fore.LIGHTGREEN_EX,
        'bright_blue': Fore.LIGHTBLUE_EX,
    }
    
    color_code = color_map.get(color.lower(), Fore.WHITE)
    return f"{color_code}{text}{Style.RESET_ALL}"


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to specified length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def extract_urls(text: str) -> List[str]:
    """
    Extract URLs from text.
    
    Args:
        text: Text to extract URLs from
        
    Returns:
        List of URLs found
    """
    url_pattern = r'https?://[^\s<>"{\}|\\^`\[\]]+'
    return re.findall(url_pattern, text)


def is_valid_email(email: str) -> bool:
    """
    Check if email address is valid.
    
    Args:
        email: Email address to validate
        
    Returns:
        True if valid email format
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def measure_performance(func):
    """
    Decorator to measure function performance.
    
    Args:
        func: Function to measure
        
    Returns:
        Wrapper function that logs performance
    """
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        
        try:
            result = func(*args, **kwargs)
            success = True
        except Exception as e:
            result = None
            success = False
            raise e
        finally:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            status = "SUCCESS" if success else "FAILED"
            logging.info(f"Performance - {func.__name__}: {duration:.4f}s [{status}]")
        
        return result
    
    return wrapper


def create_backup(source_file: str, backup_dir: str = "backups") -> Optional[str]:
    """
    Create a backup of a file.
    
    Args:
        source_file: File to backup
        backup_dir: Directory to store backup
        
    Returns:
        Path to backup file or None if failed
    """
    try:
        import shutil
        
        ensure_directory(backup_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = Path(source_file).name
        backup_path = os.path.join(backup_dir, f"{timestamp}_{filename}")
        
        shutil.copy2(source_file, backup_path)
        logging.info(f"Created backup: {backup_path}")
        
        return backup_path
        
    except Exception as e:
        logging.error(f"Error creating backup: {e}")
        return None


def get_system_info() -> Dict[str, Any]:
    """
    Get system information for debugging.
    
    Returns:
        Dictionary with system information
    """
    try:
        import platform
        import psutil
        
        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "memory_available_gb": psutil.virtual_memory().available / (1024**3),
            "disk_free_gb": psutil.disk_usage('/').free / (1024**3) if os.name != 'nt' else psutil.disk_usage('C:\\').free / (1024**3)
        }
        
    except ImportError:
        return {
            "platform": "Unknown",
            "python_version": "Unknown",
            "note": "psutil not available for detailed system info"
        }
    except Exception as e:
        return {"error": str(e)}


_OLLAMA_ENV_MAP = {
    "keep_alive": "OLLAMA_KEEP_ALIVE",
    "kv_cache_type": "OLLAMA_KV_CACHE_TYPE",
    "num_gpu": "OLLAMA_NUM_GPU",
}


def _probe_gpu_vram_gb() -> float:
    """Best-effort GPU VRAM probe.

    Tries ``nvidia-smi`` first (no Python deps, fast). Returns 0.0 if the
    binary isn't on PATH or no NVIDIA GPU is present. We deliberately do
    not import ``torch`` here - pulling torch in for a hardware probe
    would defeat the purpose on torch-less installs.
    """
    import subprocess
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            timeout=2,
            stderr=subprocess.DEVNULL,
        )
        lines = out.decode().strip().splitlines()
        if lines:
            # Value is in MiB; convert to GB.
            return float(lines[0].strip()) / 1024.0
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError, ValueError):
        pass
    return 0.0


def detect_hardware() -> Dict[str, Any]:
    """Snapshot the host's hardware for tuning decisions."""
    info: Dict[str, Any] = {
        "gpu_vram_gb": _probe_gpu_vram_gb(),
        "ram_gb": 0.0,
        "cpu_count": os.cpu_count() or 0,
    }
    try:
        import psutil  # local import - keeps utils.py import-cheap
        info["ram_gb"] = psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        pass
    return info


def _vram_tier(vram_gb: float) -> str:
    """Bucket VRAM into qualitative tiers used by the auto-tuner."""
    if vram_gb <= 0.0:
        return "cpu"
    if vram_gb < 5.0:
        return "tight"  # 4 GB-class cards (e.g., GTX 1650)
    if vram_gb < 9.0:
        return "mid"    # 6-8 GB cards
    return "ample"      # 12 GB+ cards


def auto_tune_defaults(hardware: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Pick reasonable Ollama settings given the detected hardware.

    Returns a dict shaped like the ``system.ollama_env`` config block plus
    a top-level ``num_ctx``. Callers merge user-supplied overrides over
    the top so explicit config always wins.
    """
    hw = hardware or detect_hardware()
    tier = _vram_tier(hw["gpu_vram_gb"])

    if tier == "cpu":
        return {
            "num_ctx": 2048,
            "ollama_env": {
                "keep_alive": "10m",
                "kv_cache_type": "f16",
                "num_gpu": 0,
            },
        }
    if tier == "tight":
        # Aggressive VRAM diet - q8 KV cache, capped context, keep model loaded.
        return {
            "num_ctx": 2048,
            "ollama_env": {
                "keep_alive": "24h",
                "kv_cache_type": "q8_0",
                "num_gpu": 999,
            },
        }
    if tier == "mid":
        return {
            "num_ctx": 4096,
            "ollama_env": {
                "keep_alive": "24h",
                "kv_cache_type": "f16",
                "num_gpu": 999,
            },
        }
    # Ample VRAM
    return {
        "num_ctx": 8192,
        "ollama_env": {
            "keep_alive": "24h",
            "kv_cache_type": "f16",
            "num_gpu": 999,
        },
    }


def _is_auto(value: Any) -> bool:
    """Treat None and the literal string ``"auto"`` as "please auto-tune"."""
    return value is None or (isinstance(value, str) and value.strip().lower() == "auto")


def resolve_ollama_tuning(config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge config-provided values with hardware-derived defaults.

    Rules:
    - Missing keys (or ``"auto"``) get filled from ``auto_tune_defaults()``.
    - Explicit user values win - auto-tune never overrides them.
    - When ``system.auto_tune`` is ``false``, no detection runs; only
      config-provided values are used.

    Returns ``{"num_ctx": int, "ollama_env": {...}, "hardware": {...},
    "auto_tune": bool, "tier": str}`` - the last three are for logging.
    """
    system_cfg = config.get("system") or {}
    llm_cfg = config.get("llm") or {}
    auto_tune = bool(system_cfg.get("auto_tune", True))

    user_env = dict(system_cfg.get("ollama_env") or {})
    user_num_ctx = llm_cfg.get("num_ctx")

    hardware = detect_hardware() if auto_tune else {
        "gpu_vram_gb": 0.0, "ram_gb": 0.0, "cpu_count": os.cpu_count() or 0
    }

    if auto_tune:
        defaults = auto_tune_defaults(hardware)
    else:
        defaults = {"num_ctx": user_num_ctx or 2048, "ollama_env": {}}

    # Start from auto-tuned defaults, layer user values on top (user wins).
    resolved_env = {**defaults["ollama_env"]}
    for key in _OLLAMA_ENV_MAP:
        if key in user_env and not _is_auto(user_env[key]):
            resolved_env[key] = user_env[key]

    resolved_num_ctx = (
        defaults["num_ctx"] if _is_auto(user_num_ctx) else user_num_ctx
    )

    return {
        "num_ctx": int(resolved_num_ctx),
        "ollama_env": resolved_env,
        "hardware": hardware,
        "auto_tune": auto_tune,
        "tier": _vram_tier(hardware["gpu_vram_gb"]),
    }


def apply_ollama_env(ollama_env: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """Export Ollama tuning env vars unless the user already set them.

    ``keep_alive`` → ``OLLAMA_KEEP_ALIVE``, ``kv_cache_type`` →
    ``OLLAMA_KV_CACHE_TYPE``, ``num_gpu`` → ``OLLAMA_NUM_GPU``.
    Returns the dict of variables that were actually set.
    """
    if not ollama_env:
        return {}

    exported: Dict[str, str] = {}
    for key, env_name in _OLLAMA_ENV_MAP.items():
        if key not in ollama_env or _is_auto(ollama_env[key]):
            continue
        if env_name in os.environ:
            continue
        os.environ[env_name] = str(ollama_env[key])
        exported[env_name] = os.environ[env_name]
    return exported


def reciprocal_rank_fusion(
    rankings: List[List[Any]], k: int = 60
) -> List[Tuple[Any, float]]:
    """Fuse several ranked id lists with Reciprocal Rank Fusion.

    Each input is an ordered list of ids (best first). A document's fused score
    is ``sum(1 / (k + rank))`` over the lists it appears in, where ``rank`` is
    its 0-based position in that list. The constant ``k`` damps the influence of
    top ranks so a single list cannot dominate the fusion.

    Args:
        rankings: List of ranked id lists from each retriever leg.
        k: RRF damping constant (typically 60).

    Returns:
        ``(doc_id, score)`` pairs sorted by fused score descending.
    """
    scores: Dict[Any, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


class PerformanceMonitor:
    """Simple performance monitoring utility."""
    
    def __init__(self):
        self.metrics = {}
    
    def start_timer(self, name: str):
        """Start timing an operation."""
        self.metrics[name] = {"start": datetime.now()}
    
    def end_timer(self, name: str):
        """End timing an operation."""
        if name in self.metrics and "start" in self.metrics[name]:
            end_time = datetime.now()
            duration = (end_time - self.metrics[name]["start"]).total_seconds()
            self.metrics[name]["duration"] = duration
            self.metrics[name]["end"] = end_time
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics."""
        return self.metrics.copy()
    
    def reset(self):
        """Reset all metrics."""
        self.metrics.clear()


# Global performance monitor instance
performance_monitor = PerformanceMonitor()
