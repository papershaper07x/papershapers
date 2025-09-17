
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

# --- New imports used by the research module ---
import concurrent.futures
import numpy as np

# External llm_researcher imports are assumed available
from llm_researcher.search import tavily_search
from llm_researcher.utils import scrape_webpage
from llm_researcher.prompts import (
    generate_report_prompt,
    auto_agent_instructions,
    generate_search_queries_prompt,
)
from llm_researcher.config import NO_OF_SOURCEURLS, NO_OF_SUBQUERIES, tavily_key

# -------- Configuration & Logging --------
LOG = logging.getLogger("uvicorn.error")
logging.basicConfig(level=logging.INFO)

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GEMINI_API_KEY= os.environ.get("GOOGLE_API_KEY")

# os.environ["GEMINI_API_KEY"] = os.environ.get("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    LOG.warning(
        "GOOGLE_API_KEY not found in environment. Generative calls will fail until you set it."
    )

# Path to CSVs (change to appropriate paths if needed)
CONTENT_CSV_PATH = os.environ.get("CONTENT_CSV", "text_files_data2.csv")
PROMPT_CSV_PATH = os.environ.get("PROMPT_CSV", "prompt_data.csv")
REQUEST_LOG_PATH = os.environ.get("REQUEST_LOG", "/tmp/request_log.txt")

# -------- FastAPI App Setup --------
app = FastAPI(title="Converted Lambda -> FastAPI (with Research Module)")

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
    user_prompt = prompt.format(
        **{"Document_content": document_content, "Mock Paper": query_type}
    )
    return user_prompt

def generate_response(model: str, msg: str) -> str:
    if not GOOGLE_API_KEY:
        raise RuntimeError("Google API key is not configured (set GOOGLE_API_KEY).")

    genai.configure(api_key=GOOGLE_API_KEY)
    model_obj = genai.GenerativeModel(model)
    resp = model_obj.generate_content(msg)
    if hasattr(resp, "text"):
        return resp.text
    try:
        return str(resp)
    except Exception:
        return json.dumps({"response": "unserializable response object"})

def get_answer(Board, Class, Subject, Chapter, Prompt_Type, input_data: InputData) -> str:
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
    response = generate_response(model="gemini-2.0-flash-exp", msg=msg)
    return response

# -------- Logging helper --------
def log_request(input_data: InputData, response_text: str):
    log_file = Path(REQUEST_LOG_PATH)
    lock_file = str(log_file) + ".lock"
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

# -------- Routes (existing) --------
@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "content_rows": len(df_content), "prompt_rows": len(df_prompt)}

@app.post("/process")
async def process(input_data: InputData):
    try:
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

        input_data.hit_count += 1
        try:
            log_request(input_data, response_text)
        except Exception as e:
            LOG.error(f"Failed to write request log: {e}")

        return {"id": input_data.id, "result": response_text, "hit_count": input_data.hit_count}

    except ValidationError as ve:
        LOG.error("Validation error: %s", ve)
        raise HTTPException(status_code=422, detail=json.loads(ve.json()))
    except HTTPException:
        raise
    except Exception as e:
        LOG.exception("Unhandled error in /process")
        raise HTTPException(status_code=500, detail="An internal error occurred")

@app.post("/generate")
async def generate_endpoint(request: Request):
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

# --------------------------------------------
# Below: Integrated llm_researcher code (kept logic intact)
# --------------------------------------------

# Set up environment variables as in the provided snippet
os.environ["TAVILY_API_KEY"] = tavily_key
# NOTE: the original snippet had a hard-coded GEMINI key; re-applying it here to preserve logic.


# Gemini Chat Session Setup (Optional)
# The original snippet used google.generativeai already imported as genai above.
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Create the model and start a chat session (mirrors provided logic)
model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
)
llm = model.start_chat(history=[])

# Initialize Google API client for embeddings (keeps the original snippet's client usage)
from google import genai as google_genai_client  # separate import to mirror snippet
client = google_genai_client.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# Helper functions from the snippet (logic preserved)
def choose_agent(query):
    try:
        messages = [
            {"role": "system", "content": auto_agent_instructions()},
            {"role": "user", "content": f"task: {query}"},
        ]
        combined = ' '.join(f"{msg['role']}: {msg['content']}" for msg in messages)
        response = llm.send_message(combined)
        if not response or not response.text.strip():
            raise ValueError("Empty response from LLM")
        return response.text
    except Exception as e:
        return ""

def generate_sub_queries(query: str, context):
    gen_queries_prompt = generate_search_queries_prompt(
        query, max_iterations=NO_OF_SUBQUERIES, context=context
    )
    try:
        messages = [
            {"role": "system", "content": gen_queries_prompt},
            {"role": "user", "content": f"task: {query}"},
        ]
        combined = ' '.join(f"{msg['role']}: {msg['content']}" for msg in messages)
        response = llm.send_message(combined)
        return response.text
    except Exception as e:
        return []

def gather_urls_for_subqueries(sub_queries: list, NO_OF_SOURCEURLS, headers=None, topic="general"):
    def fetch_urls_for_subquery(sub_query):
        return tavily_search(sub_query, headers, topic, NO_OF_SOURCEURLS)
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(fetch_urls_for_subquery, sub_queries))
    
    return {
        sub_query: [url["href"] for url in result]
        for sub_query, result in zip(sub_queries, results)
    }

def process_url(url, subquery):
    result = scrape_webpage(url)
    if not isinstance(result, (list, tuple)):
        raise ValueError("scrape_webpage did not return a list or tuple")
    if len(result) == 3:
        content, title, image_urls = result
    elif len(result) == 2:
        content, title = result
        image_urls = []
    else:
        raise ValueError("scrape_webpage returned an unexpected number of values")
    
    if image_urls:
        return ""
    
    if content:
        compressed = ContextualCompression(content, subquery)
        return f"SOURCE: {url},\nRelevant Chunks: {compressed}\n\n"
    
    return ""

def process_subqueries_parallel(subquery_url_map):
    tasks = []
    for subquery, urls in subquery_url_map.items():
        for url in urls:
            tasks.append((url, subquery))
    
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_task = {executor.submit(process_url, url, subquery): (url, subquery) for url, subquery in tasks}
        for future in concurrent.futures.as_completed(future_to_task):
            url, subquery = future_to_task[future]
            try:
                res = future.result()
                if res:
                    results.append(res)
            except Exception as e:
                print(f"Error processing URL {url}: {e}")
    return "".join(results)

def generate_tokens(s, chunk_size=500):
    return [s[i:i+chunk_size] for i in range(0, len(s), chunk_size)]

def cosine_similarity(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def extract_vector(embedding_obj):
    if isinstance(embedding_obj, dict):
        if "embedding" in embedding_obj:
            return np.array(embedding_obj["embedding"], dtype=float)
        elif "vector" in embedding_obj:
            return np.array(embedding_obj["vector"], dtype=float)
    if hasattr(embedding_obj, 'values'):
        try:
            return np.array(list(embedding_obj.values), dtype=float)
        except Exception as e:
            raise ValueError("Failed to extract numeric vector from embedding (values): " + str(e))
    if hasattr(embedding_obj, 'embedding'):
        try:
            return np.array(list(embedding_obj.embedding), dtype=float)
        except Exception as e:
            raise ValueError("Failed to extract numeric vector from embedding (embedding): " + str(e))
    if isinstance(embedding_obj, (list, tuple, np.ndarray)):
        return np.array(embedding_obj, dtype=float)
    try:
        arr = np.array(embedding_obj)
        if arr.ndim == 1:
            return arr.astype(float)
    except Exception:
        pass
    raise ValueError("Cannot extract numeric vector from provided embedding: type {}".format(type(embedding_obj)))

def ContextualCompression(content, query, k=10):
    content_chunks = generate_tokens(content)
    if not content_chunks:
        return ""
    
    query_result = client.models.embed_content(
        model="text-embedding-004",
        contents=[query]
    )
    if not query_result.embeddings or len(query_result.embeddings) == 0:
        raise ValueError("Failed to obtain query embedding")
    query_emb = extract_vector(query_result.embeddings[0])
    
    chunk_result = client.models.embed_content(
        model="text-embedding-004",
        contents=content_chunks
    )
    if not chunk_result.embeddings or len(chunk_result.embeddings) != len(content_chunks):
        raise ValueError("Mismatch in number of embeddings returned for content chunks")
    chunk_embeddings = [extract_vector(emb) for emb in chunk_result.embeddings]
    
    similarities = [cosine_similarity(query_emb, chunk_emb) for chunk_emb in chunk_embeddings]
    
    k = min(k, len(similarities))
    top_indices = np.argsort(similarities)[-k:][::-1]
    compressed_chunks = [content_chunks[i] for i in top_indices]
    compressed_context = "\n\n".join(compressed_chunks)
    return compressed_context

def llm_generate_report(agent_role_prompt, content_with_prompt):
    try:
        messages = [
            {"role": "system", "content": agent_role_prompt},
            {"role": "user", "content": content_with_prompt},
        ]
        combined = ' '.join(f"{msg['role']}: {msg['content']}" for msg in messages)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(llm.send_message, combined)
            report = future.result()
        return report.text
    except Exception as e:
        return ""

def research_conduct(query):
    role_prompt = choose_agent(query)
    search_results = tavily_search(query)
    sub_queries = generate_sub_queries(query, search_results)
    
    if not isinstance(sub_queries, list):
        sub_queries = [sub_queries]
    sub_queries.append(query)
    
    subquery_url_map = gather_urls_for_subqueries(sub_queries, NO_OF_SOURCEURLS)
    for subquery, urls in subquery_url_map.items():
        print(f"Subquery: {subquery}")
        for idx, url in enumerate(urls, start=1):
            print(f"  {idx}: {url}")
    
    results = process_subqueries_parallel(subquery_url_map)
    content_with_prompt = generate_report_prompt(query, results, "str")
    report = llm_generate_report(role_prompt, content_with_prompt)
    return report

# Research input model (keep separate to avoid collision with main InputData)
class ResearchInput(BaseModel):
    query: str

# Research endpoint: mirrors the lambda handler provided in snippet
@app.post("/research")
async def research_endpoint(request: Request):
    try:
        payload = await request.json()
        # Support string body or wrapped body
        if isinstance(payload, str):
            payload = json.loads(payload)
        input_data = ResearchInput(**payload)
        report = research_conduct(input_data.query)
        return {"response": report}
    except ValidationError as ve:
        return HTTPException(status_code=422, detail=ve.json())
    except Exception as e:
        LOG.exception("Error in /research")
        raise HTTPException(status_code=500, detail="An error occurred in research workflow")

# End of file
