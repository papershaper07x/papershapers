# from fastapi import FastAPI

# app = FastAPI()

# @app.get("/")
# async def root():
#     return {"greeting": "Hello, World!", "message": "Welcome to FastAPI!"}


# app.py
import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from filelock import FileLock
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
import google.generativeai as genai

# -------- Configuration & Logging --------
LOG = logging.getLogger("uvicorn.error")
logging.basicConfig(level=logging.INFO)

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    LOG.warning(
        "GOOGLE_API_KEY not found in environment. Generative calls will fail until you set it."
    )

# Path to CSVs (change to appropriate paths if needed)
CONTENT_CSV_PATH = os.environ.get("CONTENT_CSV", "text_files_data2.csv")
PROMPT_CSV_PATH = os.environ.get("PROMPT_CSV", "prompt_data.csv")
REQUEST_LOG_PATH = os.environ.get("REQUEST_LOG", "/tmp/request_log.txt")

# -------- FastAPI App Setup --------
app = FastAPI(title="Converted Lambda -> FastAPI")

# Allow CORS for local development; adjust origins as required
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- DataFrames to be loaded on startup --------
df_content = pd.DataFrame()
df_prompt = pd.DataFrame()

@app.on_event("startup")
def load_csv_data():
    global df_content, df_prompt
    try:
        df_content = pd.read_csv(CONTENT_CSV_PATH)
        LOG.info(f"Loaded content CSV from {CONTENT_CSV_PATH} ({len(df_content)} rows)")
    except Exception as e:
        LOG.error(f"Could not load content CSV at {CONTENT_CSV_PATH}: {e}")
        df_content = pd.DataFrame()

    try:
        df_prompt = pd.read_csv(PROMPT_CSV_PATH)
        LOG.info(f"Loaded prompt CSV from {PROMPT_CSV_PATH} ({len(df_prompt)} rows)")
    except Exception as e:
        LOG.error(f"Could not load prompt CSV at {PROMPT_CSV_PATH}: {e}")
        df_prompt = pd.DataFrame()

    # Configure Google Generative API client if key available
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)

# -------- Pydantic model (same as your Lambda) --------
class InputData(BaseModel):
    id: str
    Board: str
    Class: str
    Subject: str
    Chapter: str
    Prompt_Type: str
    hit_count: int
    is_logedIn: bool
    answer: bool = False
    question_paper: Optional[str] = None

# -------- Helper functions (converted) --------
def prepare_message(prompt: str, document_content: str, query_type: str) -> str:
    # Keep your template logic
    user_prompt = prompt.format(
        **{"Document_content": document_content, "Mock Paper": query_type}
    )
    return user_prompt

def generate_response(model: str, msg: str) -> str:
    """
    Call the google.generativeai wrapper.
    Note: genai must be configured with an API key via env var GOOGLE_API_KEY.
    """
    if not GOOGLE_API_KEY:
        raise RuntimeError("Google API key is not configured (set GOOGLE_API_KEY).")

    # Re-configure each call to be safe in some deployments (no-op if already configured)
    genai.configure(api_key=GOOGLE_API_KEY)

    # The code matches your original approach; if your genai version expects a different call,
    # adjust here (e.g., genai.generate_text(...) or model.generate_text(...))
    model_obj = genai.GenerativeModel(model)
    resp = model_obj.generate_content(msg)

    # Try to pull a textual representation - be conservative & robust
    if hasattr(resp, "text"):
        return resp.text
    try:
        return str(resp)
    except Exception:
        return json.dumps({"response": "unserializable response object"})

def get_answer(Board, Class, Subject, Chapter, Prompt_Type, input_data: InputData) -> str:
    # Filter the DataFrames based on selection
    prompt_df = df_prompt[
        (df_prompt["Board"] == Board)
        & (df_prompt["Class"] == Class)
        & (df_prompt["Prompt_Type"] == Prompt_Type)
    ]

    document_content_df = df_content[
        (df_content["Board"] == Board)
        & (df_content["Class"] == Class)
        & (df_content["Subject"] == Subject)
        & (df_content["Chapter"] == Chapter)
    ]

    query_type_df = document_content_df.copy()  # in your original code query_type used same filtering

    if document_content_df.empty:
        raise HTTPException(status_code=404, detail="No matching records found document_content_df")
    if query_type_df.empty:
        raise HTTPException(status_code=404, detail="No matching records found query_type_df")
    if prompt_df.empty:
        raise HTTPException(status_code=404, detail="No matching records found prompt_df")

    prompt = prompt_df["Prompt_Data"].values[0]
    document_content = document_content_df["File_Data"].tolist()[0]
    query_type = query_type_df["File_Data"].tolist()[0]

    answer_paper = f"""You are a Expert teacher and you teach the students in a school. Your aim is to help students prepare well for their final examinations with atmost precision and clearity.
You are provided with a sample paper from previous year/s your task is to solve the paper provided with at most precision and accuracy. 
Note:- Be clear and simple.Guideline for Question paper:- {input_data.question_paper}

Textbook content:-{document_content}
"""
    response_text = generate_response(model="models/gemini-1.5-flash-8b", msg=answer_paper)
    return response_text

def get_response(Board, Class, Subject, Chapter, Prompt_Type) -> str:
    prompt_df = df_prompt[
        (df_prompt["Board"] == Board)
        & (df_prompt["Class"] == Class)
        & (df_prompt["Prompt_Type"] == Prompt_Type)
    ]

    document_content_df = df_content[
        (df_content["Board"] == Board)
        & (df_content["Class"] == Class)
        & (df_content["Subject"] == Subject)
        & (df_content["Chapter"] == Chapter)
    ]

    query_type_df = document_content_df.copy()

    if document_content_df.empty:
        raise HTTPException(status_code=404, detail="No matching records found document_content_df")
    if query_type_df.empty:
        raise HTTPException(status_code=404, detail="No matching records found query_type_df")
    if prompt_df.empty:
        raise HTTPException(status_code=404, detail="No matching records found prompt_df")

    prompt = prompt_df["Prompt_Data"].values[0]
    document_content = document_content_df["File_Data"].tolist()[0]
    query_type = query_type_df["File_Data"].tolist()[0]

    msg = prepare_message(prompt, document_content, query_type)
    # fallback model name from your original code
    response = generate_response(model="gemini-2.0-flash-exp", msg=msg)
    return response

# -------- Logging helper --------
def log_request(input_data: InputData, response_text: str):
    log_file = Path(REQUEST_LOG_PATH)
    lock_file = str(log_file) + ".lock"
    # Ensure the log directory exists
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(lock_file):
        with open(log_file, "a") as f:
            log_entry = {
                "id": input_data.id,
                "Board": input_data.Board,
                "Class": input_data.Class,
                "Subject": input_data.Subject,
                "Chapter": input_data.Chapter,
                "Prompt_Type": input_data.Prompt_Type,
                "hit_count": input_data.hit_count,
                "is_logedIn": input_data.is_logedIn,
                "answer": input_data.answer,
                "question_paper": input_data.question_paper,
                "response": response_text,
            }
            f.write(json.dumps(log_entry) + "\n")

# -------- Routes --------
@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "content_rows": len(df_content), "prompt_rows": len(df_prompt)}

@app.post("/process")
async def process(input_data: InputData):
    """
    Replace your lambda_handler with this single POST endpoint.
    Body: JSON matching InputData model.
    """
    try:
        # Authentication check (same as Lambda)
        if not input_data.is_logedIn:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not authenticated")

        if input_data.answer:
            if not input_data.question_paper:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Missing 'question_paper' when answer is True",
                )
            response_text = get_answer(
                input_data.Board,
                input_data.Class,
                input_data.Subject,
                input_data.Chapter,
                input_data.Prompt_Type,
                input_data,
            )
        else:
            response_text = get_response(
                input_data.Board,
                input_data.Class,
                input_data.Subject,
                input_data.Chapter,
                input_data.Prompt_Type,
            )

        # Update hit count and log
        input_data.hit_count += 1
        try:
            log_request(input_data, response_text)
        except Exception as e:
            LOG.error(f"Failed to write request log: {e}")

        return {"id": input_data.id, "result": response_text, "hit_count": input_data.hit_count}

    except ValidationError as ve:
        # Pydantic validation - should normally be handled by FastAPI, but kept for parity
        LOG.error("Validation error: %s", ve)
        raise HTTPException(status_code=422, detail=json.loads(ve.json()))
    except HTTPException:
        # Re-raise FastAPI HTTP exceptions
        raise
    except Exception as e:
        LOG.exception("Unhandled error in /process")
        raise HTTPException(status_code=500, detail="An internal error occurred")

# Optional: separate endpoints if you want to split behavior
@app.post("/generate")
async def generate_endpoint(request: Request):
    """
    Lightweight wrapper if you want to call get_response only.
    Accepts same InputData JSON but ignores `answer` flag and uses get_response.
    """
    payload = await request.json()
    try:
        input_data = InputData(**payload)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=json.loads(e.json()))

    if not input_data.is_logedIn:
        raise HTTPException(status_code=401, detail="User not authenticated")

    response_text = get_response(
        input_data.Board,
        input_data.Class,
        input_data.Subject,
        input_data.Chapter,
        input_data.Prompt_Type,
    )
    input_data.hit_count += 1
    log_request(input_data, response_text)
    return {"id": input_data.id, "result": response_text, "hit_count": input_data.hit_count}
