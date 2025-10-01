# config.py
import os
import logging

# It's good practice to have a logger available for config-level warnings.
LOG = logging.getLogger("uvicorn.error")

# =============================================================================
# 1. Environment & API Keys
# =============================================================================
# Load sensitive keys and environment-specific settings from environment variables.
# For local development, you can use a .env file with a library like python-dotenv.

# Google API Key for accessing Gemini models. This is the most critical key.
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    LOG.warning(
        "CRITICAL: GOOGLE_API_KEY environment variable not set. "
        "All generative AI calls will fail."
    )

# Tavily API Key for the research agent's web search capabilities.
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    LOG.warning(
        "WARNING: TAVILY_API_KEY environment variable not set. "
        "The /research endpoint will fail."
    )
# Set the key in the environment for the tavily-python library to pick it up automatically.
if TAVILY_API_KEY:
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY


# =============================================================================
# 2. File & Data Paths
# =============================================================================
# Centralize all file paths used in the application.
# Using os.getenv allows you to override these paths in different environments.

# Path to the CSV containing the content/textbook data.
CONTENT_CSV_PATH = os.environ.get("CONTENT_CSV_PATH", "text_files_data2.csv")

# Path to the CSV containing prompt templates.
PROMPT_CSV_PATH = os.environ.get("PROMPT_CSV_PATH", "prompt_data.csv")

# Path to the CSV containing the paper generation schemas (previously an implicit import).
INPUT_CSV_PATH = os.environ.get("INPUT_CSV_PATH", "instructions.csv")

# Path for logging user requests and responses.
REQUEST_LOG_PATH = os.environ.get("REQUEST_LOG_PATH", "/tmp/request_log.txt")


# =============================================================================
# 3. Application Performance & Behavior Tuning
# =============================================================================
# Constants that affect the performance, resource usage, and output of the application.

# Number of worker threads for the background task executor.
EXECUTOR_WORKERS = 4

# The maximum number of characters to include from long text fields in the final JSON response.
# This prevents the API response from becoming excessively large. Set to 0 to disable truncation.
MAX_TEXT_CHARS = 1000


# =============================================================================
# 4. LLM Researcher Agent Settings
# =============================================================================
# Constants that control the behavior of the /research agent.

# The number of web search results to retrieve per sub-query.
NO_OF_SOURCEURLS = 5

# The number of sub-queries the agent should generate to explore the main topic.
NO_OF_SUBQUERIES = 3


# =============================================================================
# 5. Cache (Redis) Configuration
# =============================================================================
# Settings for connecting to the Redis cache used for the stale-while-revalidate strategy.

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))



# In config.py, add a new section for model names

# =============================================================================
# 5. Model Configuration
# =============================================================================

# Model for the main paper generation pipeline
GENERATOR_MODEL_NAME = "models/gemini-2.5-flash-lite"

# Model for the LLM researcher agent
RESEARCHER_MODEL_NAME = "gemini-2.5-flash-lite"

# Model for text embeddings
EMBEDDING_MODEL_NAME = "models/text-embedding-004"

# --- NEW ---
# Multimodal model for the document processing endpoint
DOCUMENT_PROCESSING_MODEL_NAME = "gemini-2.5-flash-lite"