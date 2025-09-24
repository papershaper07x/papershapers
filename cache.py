# cache.py

import redis
import json
import os

# --- Redis Connection (Best Practice for Railway Private Networking) ---

# Railway injects these variables into your application container automatically.

"""
Environment-aware Redis cache helper.
Behavior:
 - Uses REDIS_URL if present (Railway or standard env).
 - Otherwise builds a URL from REDISHOST/REDISPORT/REDISPASSWORD.
 - Falls back to localhost Redis for dev.
 - Exposes get_redis_connection(), create_cache_key(), get_from_cache(), set_to_cache().
 - Safe: will disable caching (return None) if Redis is unreachable.
"""

from __future__ import annotations
import os
import json
import time
import logging
from typing import Optional
import redis

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- Configurable ---
CACHE_EXPIRATION_SECONDS = int(os.getenv("CACHE_EXPIRATION_SECONDS", "3600"))
REDIS_URL = os.getenv("REDIS_URL")  # preferred canonical url (e.g. set by Railway)
REDISHOST = os.getenv("REDISHOST")
REDISPORT = os.getenv("REDISPORT")
REDISPASSWORD = os.getenv("REDISPASSWORD")
REDISUSER = os.getenv("REDISUSER", "default")
# --- End config ---

def _determine_connection_url() -> Optional[str]:
    """Decide connection url using env vars, with sensible fallbacks."""
    # 1) explicit canonical URL
    if REDIS_URL:
        logger.info("Using REDIS_URL from environment.")
        return REDIS_URL

    # 2) try host/port/password combo (useful on Railway if REDIS_URL not set)
    if REDISHOST and REDISPORT:
        if REDISPASSWORD:
            # Don't log the password itself
            logger.info("Building Redis connection URL from REDISHOST/REDISPORT/REDISPASSWORD (hidden).")
            return f"redis://{REDISUSER}:{REDISPASSWORD}@{REDISHOST}:{REDISPORT}"
        else:
            logger.info("Building Redis connection URL from REDISHOST/REDISPORT without password.")
            return f"redis://{REDISHOST}:{REDISPORT}"

    # 3) local fallback for development
    logger.info("No Redis env found. Falling back to local redis://localhost:6379/0 (dev).")
    return "redis://localhost:6379/0"


_CONNECTION_URL = _determine_connection_url()
_redis_pool: Optional[redis.ConnectionPool] = None
_redis_client: Optional[redis.Redis] = None
_cache_enabled = False

if _CONNECTION_URL:
    try:
        # create pool with sensible timeouts so failures are fast & recoverable
        _redis_pool = redis.ConnectionPool.from_url(
            _CONNECTION_URL,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            health_check_interval=30
        )
        _redis_client = redis.Redis(connection_pool=_redis_pool)
    except Exception as e:
        logger.exception("Failed to create Redis connection pool: %s", e)
        _redis_pool = None
        _redis_client = None

def _ping_redis(client: redis.Redis, retries: int = 3, backoff: float = 0.5) -> bool:
    """Ping Redis with small retry/backoff. Returns True if reachable."""
    for attempt in range(1, retries + 1):
        try:
            if client.ping():
                logger.info("Redis ping successful.")
                return True
        except Exception as e:
            logger.debug("Redis ping attempt %d failed: %s", attempt, e)
        time.sleep(backoff * attempt)
    logger.warning("Redis is not reachable after %d attempts.", retries)
    return False

# Confirm connectivity at import time but do not raise if unavailable.
if _redis_client:
    try:
        _cache_enabled = _ping_redis(_redis_client, retries=2, backoff=0.2)
    except Exception:
        _cache_enabled = False

if not _cache_enabled:
    logger.warning("Caching disabled. Redis is not available or failed health check.")

# Public API

def get_redis_connection() -> redis.Redis:
    """Return a Redis client. Raises ConnectionError if caching disabled."""
    if not _cache_enabled or not _redis_client:
        raise ConnectionError("Redis cache is not enabled or reachable.")
    return _redis_client

def create_cache_key(board: str, class_label: str, subject: str) -> str:
    board_safe = board.strip().lower().replace(" ", "_")
    class_safe = class_label.strip().lower().replace(" ", "_")
    subject_safe = subject.strip().lower().replace(" ", "_")
    return f"paper:{board_safe}:{class_safe}:{subject_safe}"

def get_from_cache(key: str) -> Optional[dict]:
    if not _cache_enabled:
        logger.debug("get_from_cache: cache disabled -> returning None for key %s", key)
        return None
    try:
        r = get_redis_connection()
        raw = r.get(key)
        if raw:
            logger.info("CACHE HIT for key: %s", key)
            return json.loads(raw)
        logger.info("CACHE MISS for key: %s", key)
        return None
    except Exception as e:
        logger.exception("CACHE ERROR (read) for key %s: %s", key, e)
        return None

def set_to_cache(key: str, value: dict) -> bool:
    if not _cache_enabled:
        logger.debug("set_to_cache: cache disabled -> not setting key %s", key)
        return False
    try:
        r = get_redis_connection()
        r.setex(key, CACHE_EXPIRATION_SECONDS, json.dumps(value))
        logger.info("CACHE SET for key: %s", key)
        return True
    except Exception as e:
        logger.exception("CACHE ERROR (write) for key %s: %s", key, e)
        return False
