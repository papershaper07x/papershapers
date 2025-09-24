# cache.py

import redis
import json
import os

# --- Redis Connection (Best Practice for Railway Private Networking) ---

# Railway injects these variables into your application container automatically.
# The `REDISHOST` variable should resolve to the correct service name (e.g., 'redis').
REDISHOST = os.getenv("REDISHOST")
REDISPORT = os.getenv("REDISPORT")
REDISPASSWORD = os.getenv("REDISPASSWORD")
REDISUSER = os.getenv("REDISUSER", "default") # The user is typically 'default'

# The main `REDIS_URL` is also available, we'll use it for logging/fallback.
RAILWAY_REDIS_URL = os.getenv("REDIS_URL")

# Initialize connection_url to None
connection_url = None

if RAILWAY_REDIS_URL:
    # We log this to see what Railway is providing us.
    print(f"Railway provided REDIS_URL: {RAILWAY_REDIS_URL}")

# **This is the critical part.**
# We will manually build the connection URL from individual parts. This is the most
# reliable method as it ensures we are using the service name from REDISHOST.
if REDISHOST and REDISPORT and REDISPASSWORD:
    print(f"Building Redis connection string from individual variables. Host: {REDISHOST}")
    connection_url = f"redis://{REDISUSER}:{REDISPASSWORD}@{REDISHOST}:{REDISPORT}"
else:
    # This block is for LOCAL development (when Railway variables aren't present)
    print("Railway Redis variables not found. Falling back to local Docker Redis.")
    connection_url = "redis://localhost:6379"

# Log the final connection address (hiding the password)
safe_log_url = connection_url.split('@')[-1]
print(f"Attempting to connect to Redis at: {safe_log_url}")

# --- End of connection logic ---

CACHE_EXPIRATION_SECONDS = 3600

try:
    # Create the connection pool directly from the URL we constructed.
    redis_pool = redis.ConnectionPool.from_url(connection_url, decode_responses=True)
except Exception as e:
    print(f"FATAL: Could not create Redis connection pool. Error: {e}")
    redis_pool = None # Ensure pool is None if connection fails

def get_redis_connection():
    """Gets a Redis connection from the connection pool."""
    if not redis_pool:
        raise ConnectionError("Redis connection pool is not available or failed to initialize.")
    return redis.Redis(connection_pool=redis_pool)


def create_cache_key(board: str, class_label: str, subject: str) -> str:
    """Creates a consistent, unique key for a generation request."""
    board_safe = board.strip().lower().replace(" ", "_")
    class_safe = class_label.strip().lower().replace(" ", "_")
    subject_safe = subject.strip().lower().replace(" ", "_")
    return f"paper:{board_safe}:{class_safe}:{subject_safe}"

def get_from_cache(key: str) -> dict | None:
    """Retrieves and deserializes a JSON object from the Redis cache."""
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
    """Serializes a Python dict to JSON and stores it in the cache."""
    try:
        r = get_redis_connection()
        json_value = json.dumps(value)
        r.setex(key, CACHE_EXPIRATION_SECONDS, json_value)
        print(f"CACHE SET for key: {key}")
    except Exception as e:
        print(f"CACHE ERROR: Could not write to cache. Key: {key}, Error: {e}")