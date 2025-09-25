# cache.py (replace your current file with this)
from __future__ import annotations
import os
import json
import time
import logging
from typing import Optional
import redis
import urllib.parse

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- Configurable ---
CACHE_EXPIRATION_SECONDS = int(os.getenv("CACHE_EXPIRATION_SECONDS", "3600"))
# Railway and common env var names we should try
ENV_TRY_ORDER = [
    "REDIS_URL",            # common canonical form
    "REDIS_TLS_URL",        # some providers
    "REDIS_INTERNAL_URL",   # Railway-style private URL if present
    "RAILWAY_REDIS_URL",    # sometimes plugin names
    # fallback to host/port/password env combos below
]
REDISHOST = os.getenv("REDISHOST")
REDISPORT = os.getenv("REDISPORT")
REDISPASSWORD = os.getenv("REDISPASSWORD")
REDISUSER = os.getenv("REDISUSER", None)
# End config

# Internal state (lazy)
_redis_pool: Optional[redis.ConnectionPool] = None
_redis_client: Optional[redis.Redis] = None
_last_connect_attempt = 0.0
_connect_backoff_seconds = 2.0
_max_retries_on_op = 3

def _mask_url(url: str) -> str:
    try:
        p = urllib.parse.urlparse(url)
        userinfo = p.netloc
        if "@" in userinfo:
            masked = userinfo.split("@")[-1]  # host:port
            return f"{p.scheme}://<hidden>@{masked}{p.path or ''}"
        return url
    except Exception:
        return url

def _determine_connection_url() -> Optional[str]:
    # 1) explicit canonical names in order
    for name in ENV_TRY_ORDER:
        val = os.getenv(name)
        if val:
            logger.info("Using %s from environment.", name)
            return val

    # 2) host/port/password combination (Railway sometimes provides host+port)
    if REDISHOST and REDISPORT:
        if REDISPASSWORD:
            user = (REDISUSER + ":") if REDISUSER else ""
            logger.info("Building Redis URL from REDISHOST/REDISPORT/REDISPASSWORD (hidden).")
            return f"redis://{user}{REDISPASSWORD}@{REDISHOST}:{REDISPORT}"
        else:
            logger.info("Building Redis URL from REDISHOST/REDISPORT without password.")
            return f"redis://{REDISHOST}:{REDISPORT}"

    # 3) default dev fallback
    logger.info("No Redis environment found. Falling back to local redis://localhost:6379/0 (dev).")
    return "redis://localhost:6379/0"

def _create_client_from_url(url: str) -> redis.Redis:
    pool = redis.ConnectionPool.from_url(
        url,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5,
        health_check_interval=30,
        socket_keepalive=True
    )
    return redis.Redis(connection_pool=pool)

def _try_connect(force: bool = False) -> bool:
    """
    Lazily (re)create client. Returns True if client is reachable.
    Uses backoff between attempts to avoid rapid retries.
    """
    global _redis_client, _redis_pool, _last_connect_attempt

    now = time.time()
    if not force and (now - _last_connect_attempt) < _connect_backoff_seconds:
        # recently tried
        return _redis_client is not None

    _last_connect_attempt = now
    url = _determine_connection_url()
    if not url:
        logger.warning("No Redis URL could be determined.")
        return False

    # create client
    try:
        logger.info("Attempting to connect to Redis at %s", _mask_url(url))
        client = _create_client_from_url(url)
        # quick ping with tiny retry
        for i in range(2):
            try:
                if client.ping():
                    # success: replace global client/pool
                    _redis_client = client
                    logger.info("Redis ping successful.")
                    return True
            except Exception as e:
                logger.debug("Redis ping attempt %d failed: %s", i + 1, e)
                time.sleep(0.2 * (i + 1))
    except Exception as e:
        logger.exception("Failed to create redis client: %s", e)

    logger.warning("Could not connect to Redis at startup/resolution.")
    _redis_client = None
    return False

def get_redis_connection() -> redis.Redis:
    """
    Return a Redis client. Attempts lazy connect if client is None/unreachable.
    Raises ConnectionError if unable to connect after retries.
    """
    # quick check
    if _redis_client:
        try:
            _redis_client.ping()
            return _redis_client
        except Exception:
            # attempt reconnect
            logger.info("Existing redis client appears dead; trying to reconnect.")
            _try_connect(force=True)

    # try to connect
    if not _redis_client:
        ok = _try_connect(force=True)
        if not ok:
            raise ConnectionError("Redis cache is not enabled or reachable (lazy connect failed).")
    # final check
    try:
        _redis_client.ping()
        return _redis_client
    except Exception as e:
        logger.exception("Redis unreachable after connect: %s", e)
        raise ConnectionError("Redis cache is not enabled or reachable.") from e

def create_cache_key(board: str, class_label: str, subject: str) -> str:
    board_safe = board.strip().lower().replace(" ", "_")
    class_safe = class_label.strip().lower().replace(" ", "_")
    subject_safe = subject.strip().lower().replace(" ", "_")
    return f"paper:{board_safe}:{class_safe}:{subject_safe}"

def get_from_cache(key: str) -> Optional[dict]:
    """
    Returns JSON-decoded object or None.
    On any Redis failure returns None (do not raise) so caller will regenerate.
    """
    # try to ensure client exists
    try:
        r = get_redis_connection()
    except Exception as e:
        logger.debug("get_from_cache: redis not available: %s", e)
        return None

    # try to read with small retry loop
    for attempt in range(1, _max_retries_on_op + 1):
        try:
            raw = r.get(key)
            if raw:
                logger.info("CACHE HIT for key: %s", key)
                try:
                    return json.loads(raw)
                except Exception as e:
                    logger.exception("Failed to decode JSON from cache for key %s: %s", key, e)
                    # treat as miss
                    return None
            logger.info("CACHE MISS for key: %s", key)
            return None
        except Exception as e:
            logger.exception("Redis read error attempt %d for key %s: %s", attempt, key, e)
            time.sleep(0.2 * attempt)
            # attempt reconnect
            _try_connect(force=True)
    return None

def set_to_cache(key: str, value: dict) -> bool:
    """
    Sets key with expiration. Returns True on success, False otherwise.
    Failures are logged but do not raise to avoid breaking app flows.
    """
    try:
        r = get_redis_connection()
    except Exception as e:
        logger.debug("set_to_cache: redis not available: %s", e)
        return False

    payload = None
    try:
        payload = json.dumps(value)
    except Exception as e:
        # fallback: try sanitized string
        try:
            payload = json.dumps({"_error_nonserializable": str(value)})
            logger.warning("Value not JSON-serializable for key %s. Saving fallback.", key)
        except Exception:
            logger.exception("Failed to serialize cache value for key %s: %s", key, e)
            return False

    for attempt in range(1, _max_retries_on_op + 1):
        try:
            # If payload is large, setex still works, but watch for max memory on Redis side.
            r.setex(key, CACHE_EXPIRATION_SECONDS, payload)
            logger.info("CACHE SET for key: %s", key)
            return True
        except Exception as e:
            logger.exception("Redis write error attempt %d for key %s: %s", attempt, key, e)
            time.sleep(0.2 * attempt)
            _try_connect(force=True)
    return False

def cache_status() -> dict:
    """
    Useful for debugging: returns a small dict showing if redis is reachable and masked URL.
    """
    url = _determine_connection_url()
    connected = False
    try:
        connected = bool(_redis_client and _redis_client.ping())
    except Exception:
        connected = False
    return {"resolved_url": _mask_url(url) if url else None, "connected": connected}
