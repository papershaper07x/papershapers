# cache.py (replace your current file with this)
from __future__ import annotations
import os
import json
import time
import logging
import ssl
import threading
import urllib.parse
from typing import Optional
import redis

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- Configurable ---
CACHE_EXPIRATION_SECONDS = int(os.getenv("CACHE_EXPIRATION_SECONDS", "86400"))
ENV_TRY_ORDER = [
    "REDIS_URL",
    "REDIS_TLS_URL",
    "REDIS_INTERNAL_URL",
    "RAILWAY_REDIS_URL",
]
REDISHOST = os.getenv("REDISHOST")
REDISPORT = os.getenv("REDISPORT")
REDISPASSWORD = os.getenv("REDISPASSWORD")
REDISUSER = os.getenv("REDISUSER", None)
REDIS_MAX_CONNECTIONS = int(os.getenv("REDIS_MAX_CONNECTIONS", "10"))
REDIS_INSECURE_SKIP_VERIFY = os.getenv("REDIS_INSECURE_SKIP_VERIFY", "false").lower() in (
    "1",
    "true",
    "yes",
)
# End config

# Internal state (lazy)
_redis_pool: Optional[redis.ConnectionPool] = None
_redis_client: Optional[redis.Redis] = None
_last_connect_attempt = 0.0
_connect_backoff_seconds = 2.0
_max_retries_on_op = 3
_connect_lock = threading.Lock()


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
    for name in ENV_TRY_ORDER:
        val = os.getenv(name)
        if val:
            logger.info("Using %s from environment.", name)
            return val

    if REDISHOST and REDISPORT:
        if REDISPASSWORD:
            user = (REDISUSER + ":") if REDISUSER else ""
            logger.info(
                "Building Redis URL from REDISHOST/REDISPORT/REDISPASSWORD (hidden)."
            )
            return f"redis://{user}{REDISPASSWORD}@{REDISHOST}:{REDISPORT}"
        else:
            logger.info("Building Redis URL from REDISHOST/REDISPORT without password.")
            return f"redis://{REDISHOST}:{REDISPORT}"

    logger.info(
        "No Redis environment found. Falling back to local redis://localhost:6379/0 (dev)."
    )
    return "redis://localhost:6379/0"


def _is_tls_scheme(url: str) -> bool:
    try:
        scheme = urllib.parse.urlparse(url).scheme or ""
        scheme = scheme.lower()
        return scheme.startswith("rediss") or "ssl" in scheme
    except Exception:
        return False


def _create_client_from_url(url: str) -> redis.Redis:
    """
    Build ConnectionPool/Redis client from the URL.
    Tries to pass an SSLContext for TLS and falls back if the installed redis lib
    doesn't accept ssl-related kwargs.
    """
    global _redis_pool

    is_tls = _is_tls_scheme(url)

    base_kwargs = {
        "decode_responses": True,
        "socket_connect_timeout": 5,
        "socket_timeout": 5,
        "health_check_interval": 30,
        "max_connections": REDIS_MAX_CONNECTIONS,
    }

    # Build SSLContext if needed
    ssl_ctx = None
    if is_tls:
        if REDIS_INSECURE_SKIP_VERIFY:
            # INSECURE: testing only
            ssl_ctx = ssl._create_unverified_context()
        else:
            ssl_ctx = ssl.create_default_context()

    # Try constructing pool with SSLContext (robust against redis-py versions)
    pool_kwargs = base_kwargs.copy()
    if ssl_ctx is not None:
        pool_kwargs["ssl"] = ssl_ctx

    try:
        pool = redis.ConnectionPool.from_url(url, **pool_kwargs)
    except TypeError as te:
        # Some redis versions might not accept 'ssl' or related kwargs via this path;
        # retry without passing SSL kwargs (best-effort fallback).
        logger.debug(
            "ConnectionPool.from_url TypeError (retrying without ssl kwargs): %s", te
        )
        # remove ssl-related keys and retry
        for k in ("ssl", "ssl_cert_reqs", "ssl_certfile", "ssl_keyfile", "ssl_ca_certs"):
            pool_kwargs.pop(k, None)
        pool = redis.ConnectionPool.from_url(url, **pool_kwargs)

    # store pool in module state
    _redis_pool = pool
    return redis.Redis(connection_pool=pool)


def _try_connect(force: bool = False) -> bool:
    """
    Lazily (re)create client. Returns True if reachable.
    Uses backoff and a lock to avoid concurrent creation.
    """
    global _redis_client, _redis_pool, _last_connect_attempt

    with _connect_lock:
        now = time.time()
        if not force and (now - _last_connect_attempt) < _connect_backoff_seconds:
            return _redis_client is not None

        _last_connect_attempt = now
        url = _determine_connection_url()
        if not url:
            logger.warning("No Redis URL could be determined.")
            return False

        try:
            logger.info("Attempting to connect to Redis at %s", _mask_url(url))
            client = _create_client_from_url(url)
            # quick ping with tiny retry
            for i in range(2):
                try:
                    if client.ping():
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
    global _redis_client

    # quick check
    if _redis_client:
        try:
            _redis_client.ping()
            return _redis_client
        except Exception:
            logger.info("Existing redis client appears dead; trying to reconnect.")
            _try_connect(force=True)

    if not _redis_client:
        ok = _try_connect(force=True)
        if not ok:
            raise ConnectionError(
                "Redis cache is not enabled or reachable (lazy connect failed)."
            )

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
    try:
        r = get_redis_connection()
    except Exception as e:
        logger.debug("get_from_cache: redis not available: %s", e)
        return None

    for attempt in range(1, _max_retries_on_op + 1):
        try:
            raw = r.get(key)
            if raw:
                logger.info("CACHE HIT for key: %s", key)
                try:
                    return json.loads(raw)
                except Exception as e:
                    logger.exception(
                        "Failed to decode JSON from cache for key %s: %s", key, e
                    )
                    return None
            logger.info("CACHE MISS for key: %s", key)
            return None
        except Exception as e:
            logger.exception(
                "Redis read error attempt %d for key %s: %s", attempt, key, e
            )
            time.sleep(0.2 * attempt)
            _try_connect(force=True)
    return None


def set_to_cache(key: str, value: dict) -> bool:
    try:
        r = get_redis_connection()
    except Exception as e:
        logger.debug("set_to_cache: redis not available: %s", e)
        return False

    payload = None
    try:
        payload = json.dumps(value)
    except Exception as e:
        try:
            payload = json.dumps({"_error_nonserializable": str(value)})
            logger.warning("Value not JSON-serializable for key %s. Saving fallback.", key)
        except Exception:
            logger.exception(
                "Failed to serialize cache value for key %s: %s", key, e
            )
            return False

    for attempt in range(1, _max_retries_on_op + 1):
        try:
            r.setex(key, CACHE_EXPIRATION_SECONDS, payload)
            logger.info("CACHE SET for key: %s", key)
            return True
        except Exception as e:
            logger.exception(
                "Redis write error attempt %d for key %s: %s", attempt, key, e
            )
            time.sleep(0.2 * attempt)
            _try_connect(force=True)
    return False


def cache_status() -> dict:
    url = _determine_connection_url()
    connected = False
    try:
        connected = bool(_redis_client and _redis_client.ping())
    except Exception:
        connected = False
    return {"resolved_url": _mask_url(url) if url else None, "connected": connected}
