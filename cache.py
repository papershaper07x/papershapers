# cache.py
"""
In-memory cache replacement for Redis-backed cache.
- Thread-safe.
- TTL-aware.
- LRU eviction when item count exceeds MAX_ITEMS.
- Background cleaner thread that purges expired keys.
- Public API:
    create_cache_key(board, class_label, subject) -> str
    get_from_cache(key) -> Optional[dict]
    set_to_cache(key, value) -> bool
    cache_status() -> dict
    clear_cache() -> None
"""

from __future__ import annotations
import os
import time
import json
import logging
import threading
from collections import OrderedDict
from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger("cache")
if not logger.handlers:
    # basic config if not already configured by app
    logging.basicConfig(level=logging.INFO)

# Configuration (override via environment variables)
DEFAULT_TTL = int(os.getenv("CACHE_EXPIRATION_SECONDS", "3600"))  # 1 hour default
MAX_ITEMS = int(os.getenv("INMEMORY_CACHE_MAX_ITEMS", "1000"))    # max number of cached keys
CLEANER_INTERVAL = int(os.getenv("INMEMORY_CACHE_CLEANER_INTERVAL", "60"))  # seconds

# Internal structures
# OrderedDict key -> (payload_json_str, expire_timestamp)
_cache: "OrderedDict[str, Tuple[str, float]]" = OrderedDict()
_lock = threading.RLock()
_hits = 0
_misses = 0
_sets = 0
_cleaner_thread: Optional[threading.Thread] = None
_stop_cleaner = False

def _start_cleaner():
    global _cleaner_thread, _stop_cleaner
    if _cleaner_thread and _cleaner_thread.is_alive():
        return
    _stop_cleaner = False
    def _cleaner():
        logger.info("In-memory cache cleaner started (interval=%ds)", CLEANER_INTERVAL)
        while not _stop_cleaner:
            try:
                time.sleep(CLEANER_INTERVAL)
                now = time.time()
                removed = []
                with _lock:
                    keys = list(_cache.keys())
                    for k in keys:
                        _, exp = _cache.get(k, (None, 0.0))
                        if exp and exp <= now:
                            _cache.pop(k, None)
                            removed.append(k)
                if removed:
                    logger.debug("Cleaner removed expired keys: %s", removed)
            except Exception as e:
                logger.exception("Cache cleaner error: %s", e)
        logger.info("In-memory cache cleaner stopped.")
    _cleaner_thread = threading.Thread(target=_cleaner, name="InMemoryCacheCleaner", daemon=True)
    _cleaner_thread.start()

def _stop_cleaner_thread():
    global _stop_cleaner, _cleaner_thread
    _stop_cleaner = True
    if _cleaner_thread:
        _cleaner_thread.join(timeout=1.0)
        _cleaner_thread = None

# Start cleaner at import
_start_cleaner()

def create_cache_key(board: str, class_label: str, subject: str) -> str:
    """
    Normalize and create a cache key.
    """
    board_safe = board.strip().lower().replace(" ", "_")
    class_safe = class_label.strip().lower().replace(" ", "_")
    subject_safe = subject.strip().lower().replace(" ", "_")
    return f"paper:{board_safe}:{class_safe}:{subject_safe}"

def _evict_if_needed():
    """
    Evict least-recently-used items until len(_cache) <= MAX_ITEMS.
    """
    while len(_cache) > MAX_ITEMS:
        k, _ = _cache.popitem(last=False)
        logger.debug("Evicted LRU cache key: %s", k)

def get_from_cache(key: str) -> Optional[dict]:
    """
    Return the stored object (JSON-decoded) or None if not present/expired.
    Does not raise on errors; logs exceptions and returns None.
    """
    global _hits, _misses
    now = time.time()
    try:
        with _lock:
            item = _cache.get(key)
            if not item:
                _misses += 1
                logger.debug("CACHE MISS (not found) for key: %s", key)
                return None
            payload_json, exp = item
            if exp and exp <= now:
                # expired
                _cache.pop(key, None)
                _misses += 1
                logger.debug("CACHE MISS (expired) for key: %s", key)
                return None
            # Move to end to mark recent use (LRU)
            _cache.move_to_end(key, last=True)
            _hits += 1
            try:
                return json.loads(payload_json)
            except Exception:
                # in case the stored payload is not JSON, attempt eval fallback? no: return raw string wrapped
                logger.exception("Failed to decode JSON value from cache for key %s", key)
                return None
    except Exception as e:
        logger.exception("get_from_cache error for key %s: %s", key, e)
        return None

def set_to_cache(key: str, value: dict, ttl: Optional[int] = None) -> bool:
    """
    Store value (dict) in cache with TTL seconds. Returns True on success.
    """
    global _sets
    ttl = int(ttl) if ttl is not None else DEFAULT_TTL
    exp = time.time() + ttl if ttl > 0 else None
    try:
        # JSON serialize to keep behavior consistent with external cache
        payload_json = json.dumps(value)
    except Exception as e:
        logger.exception("Value not JSON serializable for key %s: %s", key, e)
        try:
            payload_json = json.dumps({"_cache_error_nonserializable": str(value)})
        except Exception:
            logger.exception("Failed to create fallback JSON for key %s", key)
            return False

    try:
        with _lock:
            _cache[key] = (payload_json, exp)
            # mark as recently used
            _cache.move_to_end(key, last=True)
            _evict_if_needed()
            _sets += 1
        logger.debug("CACHE SET for key: %s (ttl=%s)", key, ttl)
        return True
    except Exception as e:
        logger.exception("set_to_cache error for key %s: %s", key, e)
        return False

def cache_status() -> Dict[str, Any]:
    """
    Return information about the in-memory cache for debugging/metrics.
    """
    with _lock:
        now = time.time()
        total = len(_cache)
        # count expired
        expired = sum(1 for (_, exp) in _cache.values() if exp and exp <= now)
        keys_preview = list(_cache.keys())[-10:]  # last 10 most recent keys
        return {
            "backend": "in-memory",
            "items": total,
            "expired": expired,
            "max_items": MAX_ITEMS,
            "default_ttl": DEFAULT_TTL,
            "hits": _hits,
            "misses": _misses,
            "sets": _sets,
            "recent_keys": keys_preview,
            "cleaner_interval": CLEANER_INTERVAL,
        }

def clear_cache():
    """
    Clear all cached items.
    """
    with _lock:
        _cache.clear()
    logger.info("In-memory cache cleared.")

# For graceful shutdown hooks (if your app calls these)
def shutdown():
    _stop_cleaner_thread()
    clear_cache()

# Export only intended names
__all__ = [
    "create_cache_key",
    "get_from_cache",
    "set_to_cache",
    "cache_status",
    "clear_cache",
    "shutdown",
]
