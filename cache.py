# cache.py

import redis
import json
import os

# --- Redis Connection ---
# These variable names (REDISHOST, REDISPORT, REDISPASSWORD) are provided
# AUTOMATICALLY by Railway to your FastAPI service.
REDIS_HOST = os.getenv("REDISHOST", "localhost")
REDIS_PORT = int(os.getenv("REDISPORT", 6379))
REDIS_PASSWORD = os.getenv("REDISPASSWORD", None) # <-- ADD THIS LINE

CACHE_EXPIRATION_SECONDS = 3600

# This creates a connection pool, which is more efficient for web apps.
# We will now pass the password to the connection pool if it exists.
redis_pool = redis.ConnectionPool(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD, # <-- ADD THIS LINE
    db=0,
    decode_responses=True
)

def get_redis_connection():
    """Gets a Redis connection from the connection pool."""
    return redis.Redis(connection_pool=redis_pool)

# No other changes are needed in this file. The get/set functions will work correctly.

def create_cache_key(board: str, class_label: str, subject: str) -> str:
    """Creates a consistent, unique key for a generation request."""
    board_safe = board.strip().lower().replace(" ", "_")
    class_safe = class_label.strip().lower().replace(" ", "_")
    subject_safe = subject.strip().lower().replace(" ", "_")
    return f"paper:{board_safe}:{class_safe}:{subject_safe}"

def get_from_cache(key: str) -> dict | None:
    """
    Retrieves and deserializes a JSON object from the Redis cache.
    Returns None if the key doesn't exist.
    """
    try:
        r = get_redis_connection()
        cached_result = r.get(key)
        if cached_result:
            print(f"CACHE HIT for key: {key}")
            return json.loads(cached_result)
        print(f"CACHE MISS for key: {key}")
        return None
    except Exception as e:
        print(f"CACHE ERROR: Could not read from cache. Key: {key}, Error: {e}")
        return None

def set_to_cache(key: str, value: dict):
    """
    Serializes a Python dict to JSON and stores it in the cache with an expiration.
    """
    try:
        r = get_redis_connection()
        json_value = json.dumps(value)
        r.setex(key, CACHE_EXPIRATION_SECONDS, json_value)
        print(f"CACHE SET for key: {key}")
    except Exception as e:
        print(f"CACHE ERROR: Could not write to cache. Key: {key}, Error: {e}")