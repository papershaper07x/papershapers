# main.py
import os

import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# --- New modular imports ---
# Import the router that contains all our API endpoints
from endpoints import router as api_router
# Import the services module to call initialization functions
import services
# Import the config module for application settings
import config
from logger import log  # <--- IMPORT FROM THE NEW FILE

# -------- Logging --------
# Set up the logger, which will be used across the application
LOG = logging.getLogger("uvicorn.error")
logging.basicConfig(level=logging.INFO)


# -------- App Initialization --------
app = FastAPI(
    title="Optimized RAG and Research API",
    description="A modular FastAPI application with separate layers for routing, services, and utilities.",
    version="1.0.0"
)

# -------- Middleware --------
# Allow CORS for local development; adjust origins as required in a production environment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, this should be a list of allowed domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- Global Executor --------
# The global thread pool executor for running synchronous, blocking tasks.
# It is initialized during the startup event.
executor: ThreadPoolExecutor = None


# -------- Application Lifecycle Events --------
@app.on_event("startup")
async def startup_event():
    """
    Handles application startup logic.
    - Initializes the global ThreadPoolExecutor.
    - Triggers the loading of heavy resources (like ML models and CSVs) in the background.
    """
    global executor
    executor = ThreadPoolExecutor(max_workers=config.EXECUTOR_WORKERS)
    LOG.info(f"ThreadPoolExecutor initialized with {config.EXECUTOR_WORKERS} workers.")

    # Delegate the loading of heavy resources to the services module.
    # This keeps the main entrypoint clean and focused.
    loop = asyncio.get_event_loop()
    
    # Trigger the embedding model to be loaded in the executor to avoid blocking.
    services.set_executor(executor) # Pass the executor instance to the services module
    await loop.run_in_executor(executor, services.load_heavy_models_and_data)
    
    LOG.info("Startup complete: Heavy models and data loading has been initiated.")


@app.on_event("shutdown")
def shutdown_event():
    """
    Handles application shutdown logic.
    - Gracefully shuts down the ThreadPoolExecutor.
    """
    global executor
    if executor:
        executor.shutdown(wait=False)
        LOG.info("ThreadPoolExecutor shutdown initiated.")


# -------- API Router Inclusion --------
# Include all the API endpoints defined in the endpoints.py file.
# This is the core of the modular design.
app.include_router(api_router)


# -------- Health Check Endpoint --------
@app.get("/health", tags=["Monitoring"])
def health() -> Dict[str, Any]:
    """
    A simple health check endpoint to confirm the API is running
    and to get a basic status of loaded data.
    """
    # We can call a service function to get the status, keeping the endpoint clean.
    content_rows, prompt_rows = services.get_data_status()
    return {
        "status": "ok",
        "content_rows": content_rows,
        "prompt_rows": prompt_rows,
    }