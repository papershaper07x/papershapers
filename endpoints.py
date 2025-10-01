# endpoints.py
import json
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Request, UploadFile, status
from pydantic import ValidationError

# --- New modular imports ---
# Import the Pydantic models that define our request/response data shapes
import models
# Import the services module which contains all the business logic
import services
# Import the logger from the main application file
from logger import log # <--- ADD THIS IMPORT

# -------- API Router Setup --------
# All endpoints defined in this file will be prefixed with what's set in main.py (if any).
# We use tags to group related endpoints in the auto-generated API docs (e.g., Swagger UI).
router = APIRouter()


# -------- Paper Generation Endpoints --------

@router.post("/generate_full", response_model=models.PaperResponse, tags=["Paper Generation"])
async def generate_full_paper(
    req: models.GenerateRequest,
    background_tasks: BackgroundTasks,
    version_id: Optional[str] = None
):
    """
    Orchestrates the full RAG pipeline with a Stale-While-Revalidate cache.
    - Immediately returns a cached version if available.
    - Triggers a background task to refresh the cache with a new version.
    - If no cache exists, performs a synchronous generation, caches the result, and returns it.
    """
    try:
        # Delegate the entire complex logic to the service layer
        paper_response = await services.handle_generate_full(
            req=req,
            background_tasks=background_tasks,
            version_id=version_id
        )
        return paper_response
    except services.SchemaNotFoundError as e:
        log.error(f"Schema not found for request: {req.dict()}. Error: {e}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        log.exception(f"Unhandled error in /generate_full endpoint for request: {req.dict()}")
        # The original code provided a raw LLM output on failure, which is good for debugging.
        # We'll have the service layer provide this context in its exceptions.
        detail = f"An internal error occurred during paper generation: {e}"
        if hasattr(e, 'raw_output'):
            detail += f". Raw LLM Output: {e.raw_output}"
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=detail)


# -------- Legacy/Simple Generation Endpoints --------

@router.post("/process", response_model=models.ProcessResponse, tags=["Legacy Generation"])
async def process_request(input_data: models.InputData):
    """
    Handles simple, prompt-based generation or question answering based on CSV data.
    logs the request and response.
    """
    try:
        # Delegate logic to the service layer
        response_text, hit_count = await services.handle_process_request(input_data)

        return {
            "id": input_data.id,
            "result": response_text,
            "hit_count": hit_count,
        }
    except ValidationError as ve:
        # Pydantic validation errors are caught automatically by FastAPI,
        # but this is here for completeness if manual validation were needed.
        log.error("Validation error in /process: %s", ve)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=json.loads(ve.json()))
    except HTTPException:
        # Re-raise known HTTP exceptions (e.g., 401 Unauthorized from the service layer)
        raise
    except Exception as e:
        log.exception("Unhandled error in /process endpoint")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An internal error occurred")


@router.post("/generate", response_model=models.ProcessResponse, tags=["Legacy Generation"])
async def generate_legacy(request: Request):
    """
    A legacy endpoint for simple prompt-based generation.
    This endpoint demonstrates handling raw request bodies.
    """
    try:
        payload = await request.json()
        input_data = models.InputData(**payload)

        # Delegate logic to the service layer
        response_text, hit_count = await services.handle_legacy_generate(input_data)
        
        return {
            "id": input_data.id,
            "result": response_text,
            "hit_count": hit_count,
        }
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=json.loads(e.json()))
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Failed during /generate endpoint: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An internal error occurred")


# -------- AI Research Endpoint --------

@router.post("/research", response_model=models.ResearchResponse, tags=["AI Research"])
async def research_endpoint(request: Request):
    """
    Conducts web research for a given query using an agentic workflow.
    - Determines the best agent for the task.
    - Generates sub-queries.
    - Scrapes web pages and performs contextual compression.
    - Generates a final report.
    """
    try:
        payload = await request.json()
        # Handle cases where payload might be a JSON string
        if isinstance(payload, str):
            payload = json.loads(payload)
        
        input_data = models.ResearchInput(**payload)
        
        # Delegate the entire research workflow to the service layer
        report = await services.handle_research_request(input_data.query)
        
        return {"response": report}
    except ValidationError as ve:
        log.error("Validation error in /research: %s", ve)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=ve.json())
    except Exception as e:
        log.exception("Error in /research endpoint")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred in the research workflow"
        )
    

from fastapi import File, UploadFile
# --- END NEW IMPORTS ---


# ... (existing router setup and endpoints for Paper Generation, Legacy, and AI Research)


# -------- Document Intelligence Endpoints --------

@router.post("/upload-and-process/", status_code=202, tags=["Document Processing"])
async def upload_and_process_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Upload a document (PDF, DOCX, XLSX, PNG, JPG, ZIP) and start
    processing it in the background to generate a summary.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    task_id = await services.handle_upload_and_process(background_tasks, file)
    
    return {"message": "File upload successful, processing in the background.", "task_id": task_id}


@router.get("/task-status/{task_id}", response_model=models.TaskStatusResponse, tags=["Document Processing"])
async def get_task_status(task_id: str):
    """
    Get the status and result of a background processing task.
    The status can be 'processing', a dictionary with results, or an error object.
    """
    status_result = services.handle_get_task_status(task_id)
    return {"task_id": task_id, "status": status_result}




# In endpoints.py


@router.post(
    "/analyze-resume/", 
    response_model=models.ResumeAnalysisResponse, 
    tags=["Resume Analysis"]
)
async def analyze_resume(
    file: UploadFile = File(..., description="The resume file (.pdf, .docx, .txt)."),
    analysis_type: models.AnalysisType = Form(
        ..., 
        alias="analysisType",  # <--- THIS IS THE FIX
        description="The type of analysis to perform."
    )
):
    """
    Upload a resume and select an analysis type to receive instant,
    AI-powered feedback in a structured JSON format.
    """
    analysis_result_dict = await services.handle_resume_analysis(file, analysis_type)
    
    return {"success": True, "data": analysis_result_dict}
