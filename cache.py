# cache.py

import redis
import json
import os

# --- Redis Connection (Best Practice for Railway) ---

# Railway injects the full connection URL as a single environment variable.
# This is more robust than using separate host, port, and password variables.
REDIS_CONNECTION_URL = os.getenv("REDIS_URL")

# This is a fallback for local development with your Docker container.
if not REDIS_CONNECTION_URL:
    print("REDIS_URL not found, falling back to local Docker setup.")
    REDIS_CONNECTION_URL = "redis://localhost:6379"

CACHE_EXPIRATION_SECONDS = 3600

print(f"Connecting to Redis at: {REDIS_CONNECTION_URL.split('@')[-1]}") # Log without showing password

try:
    # Create the connection pool directly from the URL.
    # This handles the host, port, user, and password all in one go.
    redis_pool = redis.ConnectionPool.from_url(REDIS_CONNECTION_URL, decode_responses=True)
except Exception as e:
    print(f"FATAL: Could not create Redis connection pool. Error: {e}")
    redis_pool = None # Ensure pool is None if connection fails

def get_redis_connection():
    """Gets a Redis connection from the connection pool."""
    if not redis_pool:
        raise ConnectionError("Redis connection pool is not available.")
    return redis.Redis(connection_pool=redis_pool)

# ... The rest of the file (create_cache_key, get_from_cache, set_to_cache)
# remains exactly the same.

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