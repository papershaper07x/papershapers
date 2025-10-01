# logger.py
import logging

# This hooks into the logger configured by Uvicorn, which is the best practice for FastAPI.
# All modules in the application can now safely import this single 'log' instance.
log = logging.getLogger("uvicorn.error")