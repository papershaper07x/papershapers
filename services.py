# services.py
import os
import json
import logging
import asyncio
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from typing import Any, Dict, List, Optional
from datetime import datetime

import numpy as np
import pandas as pd
from filelock import FileLock
from fastapi import BackgroundTasks, HTTPException, UploadFile, status
import fitz
# Third-party AI clients
import google.generativeai as genai
from google import genai as google_genai_client

# --- New modular imports ---
import config
import models
import utils
from cache import (
    create_cache_key,
    add_cache_version,
    get_latest_cache_version,
    get_cache_version_by_id,
)
import io
import zipfile
from PIL import Image
import pandas as pd
from docx import Document


# Import specific functions from the original codebase structure
# These would ideally be in their own modules too, but for this refactoring, we keep them here.
from full_paper.run_full_pipeline import (
    load_bge,
    derive_plan_from_filedata,
    build_retrieval_objective,
    retrieve_from_pinecone,
    mmr_and_stratified_sample,
    build_generator_prompt_questions_only,
    call_gemini,
    parse_generator_response,
    embed_texts_bge,
    load_schema_row,
)
from llm_researcher.search import tavily_search
from llm_researcher.utils import scrape_webpage
from llm_researcher.prompts import (
    generate_report_prompt,
    auto_agent_instructions,
    generate_search_queries_prompt,
)
from full_paper.retrieval_and_summarization import (
    generate_dense_evidence_summary_with_llm,
)
from logger import log  # <--- ADD THIS IMPORT

from pydantic import ValidationError # <--- Make sure this is imported at the top

# -------- Module-level State --------
LOG = logging.getLogger("uvicorn.error")
_executor: ThreadPoolExecutor = None
df_content = pd.DataFrame()
df_prompt = pd.DataFrame()

# For LLM Researcher
llm_chat_model = None
embedding_client = None

background_task_status: Dict[str, Any] = {}
# -------- Custom Exceptions --------
class SchemaNotFoundError(Exception):
    """Custom exception for when a schema is not found."""

    pass


class GenerationError(Exception):
    """Custom exception for failures during the generation process."""

    def __init__(self, message, raw_output=None):
        super().__init__(message)
        self.raw_output = raw_output


# -------- Initialization and State Management --------
def set_executor(executor: ThreadPoolExecutor):
    """Receives the executor instance from main.py during startup."""
    global _executor
    _executor = executor


def load_heavy_models_and_data():
    """
    Loads all heavy resources: CSVs, embedding models, and configures API clients.
    This is a blocking function intended to be run in the executor during startup.
    """
    global df_content, df_prompt, llm_chat_model, embedding_client
    LOG.info("Initiating loading of heavy models and data...")

    # 1. Load CSV data
    try:
        df_content = pd.read_csv(config.CONTENT_CSV_PATH)
        LOG.info(
            f"Loaded content CSV from {config.CONTENT_CSV_PATH} ({len(df_content)} rows)"
        )
    except Exception as e:
        LOG.error(f"Could not load content CSV at {config.CONTENT_CSV_PATH}: {e}")

    try:
        df_prompt = pd.read_csv(config.PROMPT_CSV_PATH)
        LOG.info(
            f"Loaded prompt CSV from {config.PROMPT_CSV_PATH} ({len(df_prompt)} rows)"
        )
    except Exception as e:
        LOG.error(f"Could not load prompt CSV at {config.PROMPT_CSV_PATH}: {e}")

    # 2. Configure Google Generative AI clients
    if config.GOOGLE_API_KEY:
        genai.configure(api_key=config.GOOGLE_API_KEY)
        LOG.info("Configured google.generativeai client.")
        try:
            # For LLM Researcher part
            generation_config = {
                "temperature": 1,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
                "response_mime_type": "text/plain",
            }
            model = genai.GenerativeModel(
                model_name="gemini-2.5-flash-lite",  # Using a stable model name
                generation_config=generation_config,
            )
            llm_chat_model = model.start_chat(history=[])
            embedding_client = google_genai_client.Client(api_key=config.GOOGLE_API_KEY)
            LOG.info("LLM research model and client initialized.")
        except Exception as e:
            LOG.exception("Failed to initialize research LLM or client: %s", e)
    else:
        LOG.warning("GOOGLE_API_KEY not found. Generative calls will fail.")

    # 3. Load the embedding model (heavy, blocking call)
    LOG.info("Loading BGE embedding model...")
    load_bge()
    LOG.info("BGE embedding model loaded successfully.")


def get_data_status():
    """Returns the row counts of the loaded dataframes for the health check."""
    return len(df_content), len(df_prompt)


# -------- Core Service for /generate_full --------

# In services.py
# In services.py
from pydantic import ValidationError # Make sure this is imported
from copy import deepcopy # <--- ADD THIS IMPORT at the top of the file

# ...

async def handle_generate_full(req: models.GenerateRequest, background_tasks: BackgroundTasks, version_id: Optional[str] = None):
    """
    Service logic with a self-healing AND self-patching cache migration strategy.
    """
    loop = asyncio.get_event_loop()
    cache_key = create_cache_key(req.board, req.class_label, req.subject)

    # --- CACHE HIT LOGIC ---
    latest_version = None
    if not version_id:
        latest_version = get_latest_cache_version(cache_key)
    else:
        latest_version = get_cache_version_by_id(cache_key, version_id)

    if latest_version:
        try:
            # Attempt to validate the cached data directly
            validated_response = models.PaperResponse(**latest_version)
            log.info("Serving latest VALID cached version %s for %s", latest_version.get("version_id"), cache_key)
            if not version_id:
                background_tasks.add_task(_generate_and_cache_background, req)
            return validated_response
        except ValidationError as e:
            # --- SELF-PATCHING MIGRATION LOGIC ---
            log.warning(f"Cache data for key {cache_key} is invalid. Attempting in-place migration.")
            
            migrated_version = deepcopy(latest_version) # Work on a copy
            is_migrated = False

            # Check if the specific error we expect is present
            if 'value' in migrated_version and isinstance(migrated_version['value'], dict):
                paper_data = migrated_version['value']
                # THE CORE MIGRATION: Check for 'class' and rename it to 'class_label'
                if 'class' in paper_data and 'class_label' not in paper_data:
                    paper_data['class_label'] = paper_data.pop('class')
                    is_migrated = True
                    log.info(f"Successfully migrated 'class' to 'class_label' for key {cache_key}.")

            if is_migrated:
                try:
                    # Retry validation with the migrated data
                    validated_response = models.PaperResponse(**migrated_version)
                    log.info(f"Migration successful. Serving patched version for key {cache_key}.")
                    
                    # IMPORTANT: Save the fixed version back to the cache.
                    # This uses the existing version_id to overwrite the bad entry.
                    # NOTE: This requires a new function in cache.py, or careful handling.
                    # For simplicity, let's add a new version, the old one will eventually expire/be trimmed.
                    add_cache_version(cache_key, migrated_version['value'])
                    log.info(f"Saved newly patched version back to cache for key {cache_key}.")

                    if not version_id:
                        background_tasks.add_task(_generate_and_cache_background, req) # Still refresh in background
                    
                    return validated_response
                except ValidationError as final_e:
                    log.error(f"Migration attempt failed validation for key {cache_key}. Falling back. Error: {final_e}")
            else:
                 log.warning(f"Could not automatically migrate cache for key {cache_key}. Falling back.")
            
            # If we reach here, migration failed, so we fall through to the cache miss logic.

    # --- CACHE MISS LOGIC (Unchanged) ---
    log.info("Cache miss. Running synchronous generation for %s", cache_key)
    try:
        generated_paper_object = await _run_full_generation_pipeline(req)
        json_safe_paper = utils.sanitize_for_json(generated_paper_object)
        new_version_id = add_cache_version(cache_key, json_safe_paper)
        log.info("Synchronous generation saved as version %s for %s", new_version_id, cache_key)

        response_object = {
            "version_id": new_version_id,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "value": generated_paper_object
        }
        
        return response_object

    except Exception as e:
        log.error(f"Failed during synchronous generation: {e}", exc_info=True)
        raw_output = getattr(e, 'raw_output', '[Could not get raw text]')
        raise GenerationError(f"Failed to generate paper: {e}", raw_output=raw_output) from e
# In services.py

def _generate_and_cache_background(req: models.GenerateRequest):
    """
    Self-contained background task to run the generation pipeline and update the cache.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    log.info(f"BACKGROUND TASK: Starting generation for {req.board} {req.class_label} {req.subject}...")
    try:
        # 1. Generate the raw, Pydantic-compatible paper object
        final_paper_object = loop.run_until_complete(_run_full_generation_pipeline(req))
        
        # 2. Sanitize it for storage in Redis
        json_safe_paper = utils.sanitize_for_json(final_paper_object)
        cache_key = create_cache_key(req.board, req.class_label, req.subject)
        
        version_id = add_cache_version(cache_key, json_safe_paper)
        log.info("BACKGROUND TASK: Successfully updated cache key %s with new version %s", cache_key, version_id)
    except Exception as e:
        log.error(f"BACKGROUND TASK FAILED for {req.subject}. Error: {e}", exc_info=True)
    finally:
        loop.close()


async def _run_full_generation_pipeline(req: models.GenerateRequest) -> Dict[str, Any]:
    """
    The core, reusable logic for generating a paper from scratch.
    """
    loop = asyncio.get_event_loop()

    # 1. Load Schema
    row = await loop.run_in_executor(
        _executor,
        lambda: load_schema_row(
            config.INPUT_CSV_PATH, req.board, req.class_label, req.subject
        ),
    )
    if not row:
        raise SchemaNotFoundError(
            f"Schema not found for {req.board}, {req.class_label}, {req.subject}"
        )

    # 2. Derive Plan
    file_data = row.get("File_Data", "")
    plan = derive_plan_from_filedata(file_data, req.subject)

    # 3. Process Sections Concurrently
    section_tasks = [
        loop.run_in_executor(
            _executor,
            _process_section_sync,
            sec,
            file_data,
            req.class_label,
            req.subject,
        )
        for sec in plan["sections"]
    ]
    sections_results = await asyncio.gather(*section_tasks)

    # 4. Build Prompt
    slot_summaries = [
        {
            "slot_id": r["section_id"],
            "slot_meta": r.get("slot_meta", ""),
            "summaries": r.get("summaries", []),
        }
        for r in sections_results
    ]
    planner_text = plan.get("planner_text", "Generate a standard exam paper.")
    prompt = build_generator_prompt_questions_only(planner_text, slot_summaries, plan)

    # 5. Call LLM
    gen_resp = None
    try:
        gen_resp = await loop.run_in_executor(
            _executor,
            lambda: call_gemini(prompt, model_name="models/gemini-2.5-flash-lite"),
        )
        raw_text = gen_resp.get("text", "")
        parsed_llm_json = parse_generator_response(raw_text, call_gemini)
    except Exception as e:
        raw_output = gen_resp.get("text", "") if gen_resp else "[LLM call failed]"
        raise GenerationError(
            f"LLM call or parsing failed: {e}", raw_output=raw_output
        ) from e

    # 6. Clean and Finalize
    cleaned_questions = utils._post_process_and_clean_questions(
        parsed_llm_json.get("questions", [])
    )
    final_paper = {
        "paper_id": parsed_llm_json.get("paper_id", "unknown-id"),
        "board": req.board,
        "class_label": req.class_label,
        "subject": req.subject,
        "total_marks": plan.get("total_marks"),
        "time_allowed_minutes": plan.get("time_minutes"),
        "general_instructions": plan.get("general_instructions"),
        "questions": cleaned_questions,
        "retrieval_metadata": sections_results,
    }
    return final_paper


def _process_section_sync(
    sec: Dict[str, Any], file_data: str, class_label: str, subject: str
) -> Dict[str, Any]:
    """
    Synchronous function to process a single section of the paper plan. Intended to be run in the executor.
    (This function is copied verbatim from the original main.py, but now calls utils.standardize_subject_name)
    """
    try:
        objective = build_retrieval_objective(
            sec, subject_guidelines=file_data, user_mode="balanced"
        )
        specific_objective = (
            f"For {class_label} {subject}, find content for: {objective}"
        )
        LOG.info(f"Running retrieval for Section {sec.get('section_id')}...")

        class_val = "".join(re.findall(r"\d+", class_label))
        subject_val = utils.standardize_subject_name(subject)
        filters = {"class": {"$eq": class_val}, "subject": {"$eq": subject_val}}

        candidates = retrieve_from_pinecone(specific_objective, filters, top_k=20)
        if not candidates:
            LOG.warning(f"No candidates found for Section {sec.get('section_id')}")
            return {
                "section_id": sec.get("section_id"),
                "summaries": [{"id": "N/A", "summary": "No relevant evidence found."}],
                "slot_meta": sec.get("title", ""),
            }

        valid_candidates = [c for c in candidates if c.get("embedding") is not None]
        if not valid_candidates:
            LOG.warning(
                f"Candidates for Section {sec.get('section_id')} had no embeddings."
            )
            return {
                "section_id": sec.get("section_id"),
                "summaries": [{"id": "N/A", "summary": "No valid evidence processed."}],
                "slot_meta": sec.get("title", ""),
            }

        emb_matrix = np.vstack([c["embedding"] for c in valid_candidates])
        query_emb = embed_texts_bge([specific_objective], batch_size=1)[0]
        desired_snippets = min(8, max(5, int(sec.get("num_questions", 5)) + 2))

        picks_indices = mmr_and_stratified_sample(
            query_emb,
            emb_matrix,
            [{"id": c.get("snippet_id")} for c in valid_candidates],
            metadata=[c.get("metadata", {}) for c in valid_candidates],
            n_samples=desired_snippets,
        )
        selected_snips = [valid_candidates[i] for i in picks_indices]

        dense_summary = generate_dense_evidence_summary_with_llm(
            selected_snips, specific_objective, call_gemini
        )
        final_summaries = [
            {
                "id": ",".join([str(s.get("snippet_id", "")) for s in selected_snips]),
                "summary": dense_summary,
            }
        ]

        return {
            "section_id": sec.get("section_id"),
            "summaries": final_summaries,
            "slot_meta": sec.get("title", ""),
        }
    except Exception as e:
        LOG.error(
            f"FATAL ERROR in _process_section_sync for section {sec.get('section_id')}: {e}",
            exc_info=True,
        )
        return {
            "section_id": sec.get("section_id"),
            "summaries": [{"id": "ERROR", "summary": "An error occurred."}],
            "slot_meta": sec.get("title", ""),
        }


# -------- Core Services for /process and /generate (Legacy) --------


async def handle_process_request(input_data: models.InputData):
    """Service logic for the /process endpoint."""
    if not input_data.is_logedIn:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User not authenticated"
        )

    if input_data.answer:
        if not input_data.question_paper:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Missing 'question_paper' when answer is True",
            )
        response_text = await _get_answer(input_data)
    else:
        response_text = await _get_response(input_data)

    input_data.hit_count += 1
    _log_request(input_data, response_text)
    return response_text, input_data.hit_count


async def handle_legacy_generate(input_data: models.InputData):
    """Service logic for the legacy /generate endpoint."""
    if not input_data.is_logedIn:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User not authenticated"
        )

    response_text = await _get_response(input_data)
    input_data.hit_count += 1
    _log_request(input_data, response_text)
    return response_text, input_data.hit_count


async def _get_response(data: models.InputData) -> str:
    """Internal function to generate a response based on CSV data."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _get_response_sync, data)


def _get_response_sync(data: models.InputData) -> str:
    prompt_df_f = df_prompt[
        (df_prompt["Board"] == data.Board)
        & (df_prompt["Class"] == data.Class)
        & (df_prompt["Prompt_Type"] == data.Prompt_Type)
    ]
    content_df_f = df_content[
        (df_content["Board"] == data.Board)
        & (df_content["Class"] == data.Class)
        & (df_content["Subject"] == data.Subject)
        & (df_content["Chapter"] == data.Chapter)
    ]

    if content_df_f.empty or prompt_df_f.empty:
        raise HTTPException(
            status_code=404, detail="No matching records found for response generation"
        )

    prompt = prompt_df_f["Prompt_Data"].values[0]
    document_content = content_df_f["File_Data"].tolist()[0]

    msg = prompt.format(
        **{"Document_content": document_content, "Mock Paper": document_content}
    )
    return _generate_llm_response_text("models/gemini-2.5-flash-lite", msg)


async def _get_answer(data: models.InputData) -> str:
    """Internal function to generate an answer for a question paper."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _get_answer_sync, data)


def _get_answer_sync(data: models.InputData) -> str:
    content_df_f = df_content[
        (df_content["Board"] == data.Board)
        & (df_content["Class"] == data.Class)
        & (df_content["Subject"] == data.Subject)
        & (df_content["Chapter"] == data.Chapter)
    ]
    if content_df_f.empty:
        raise HTTPException(
            status_code=404,
            detail="No matching content records found for answer generation",
        )

    document_content = content_df_f["File_Data"].tolist()[0]
    answer_paper_prompt = f"""You are a Expert teacher... Guideline for Question paper:- {data.question_paper}\n\nTextbook content:-{document_content}"""
    return _generate_llm_response_text(
        "models/gemini-2.5-flash-lite", answer_paper_prompt
    )


def _generate_llm_response_text(model: str, msg: str) -> str:
    """Helper to call the Gemini API and return text."""
    if not config.GOOGLE_API_KEY:
        raise RuntimeError("Google API key is not configured.")
    model_obj = genai.GenerativeModel(model)
    resp = model_obj.generate_content(msg)
    return resp.text


def _log_request(input_data: models.InputData, response_text: str):
    """Logs request and response details to a file with a lock."""
    log_file = Path(config.REQUEST_LOG_PATH)
    lock_file = str(log_file) + ".lock"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(lock_file):
            with open(log_file, "a") as f:
                log_entry = input_data.dict()
                log_entry["response"] = response_text
                f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        LOG.error(f"Failed to write request log: {e}")


# -------- Core Service for /research --------


async def handle_research_request(query: str) -> str:
    """Service logic for the /research endpoint."""
    loop = asyncio.get_event_loop()
    # The entire research_conduct function is blocking, so run it in the executor
    report = await loop.run_in_executor(_executor, _research_conduct, query)
    return report


def _research_conduct(query: str) -> str:
    """The full, synchronous research pipeline. (Logic moved from main.py)"""
    LOG.info(f"Starting research for query: '{query}'")
    role_prompt = _choose_agent(query)
    search_results = tavily_search(query)
    sub_queries = _generate_sub_queries(query, search_results)

    if not isinstance(sub_queries, list):
        sub_queries = [sub_queries]
    sub_queries.append(query)

    subquery_url_map = _gather_urls_for_subqueries(sub_queries, config.NO_OF_SOURCEURLS)
    results = _process_subqueries_parallel(subquery_url_map)
    content_with_prompt = generate_report_prompt(query, results, "str")
    report = _llm_generate_report(role_prompt, content_with_prompt)
    LOG.info(f"Research finished for query: '{query}'")
    return report


# --- LLM Researcher Helper Functions (Internal to this service) ---


def _choose_agent(query: str) -> str:
    # ... (logic from choose_agent in main.py)
    messages = [
        {"role": "system", "content": auto_agent_instructions()},
        {"role": "user", "content": f"task: {query}"},
    ]
    combined = " ".join(f"{msg['role']}: {msg['content']}" for msg in messages)
    response = llm_chat_model.send_message(combined)
    return response.text


def _generate_sub_queries(query: str, context: Any) -> list:
    # ... (logic from generate_sub_queries in main.py)
    prompt = generate_search_queries_prompt(
        query, max_iterations=config.NO_OF_SUBQUERIES, context=context
    )
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"task: {query}"},
    ]
    combined = " ".join(f"{msg['role']}: {msg['content']}" for msg in messages)
    response = llm_chat_model.send_message(combined)
    # The original code didn't parse this, assuming it's a list-like string. A more robust implementation would parse it.
    return response.text.split("\n") if response.text else []


# in services.py
def _gather_urls_for_subqueries(
    sub_queries: list, num_urls: int
) -> Dict[str, List[str]]:
    # ...
    with concurrent.futures.ThreadPoolExecutor() as tpe:
        results = list(
            tpe.map(lambda sq: tavily_search(sq, max_results=num_urls), sub_queries)
        )

    # This new logic checks that 'res' is a list and that each item in it is a
    # dictionary containing the 'url' key before trying to access it.
    url_map = {}
    for sq, res in zip(sub_queries, results):
        if isinstance(res, list):
            # Safely extract URLs from valid dictionary objects
            valid_urls = [
                item["url"] for item in res if isinstance(item, dict) and "url" in item
            ]
            if valid_urls:
                url_map[sq] = valid_urls
    return url_map


def _process_subqueries_parallel(subquery_url_map: Dict[str, List[str]]) -> str:
    # ... (logic from process_subqueries_parallel in main.py)
    tasks = [
        (url, subquery) for subquery, urls in subquery_url_map.items() for url in urls
    ]
    results = []
    with concurrent.futures.ThreadPoolExecutor() as tpe:
        future_to_task = {
            tpe.submit(_process_url, url, subquery): (url, subquery)
            for url, subquery in tasks
        }
        for future in concurrent.futures.as_completed(future_to_task):
            try:
                res = future.result()
                if res:
                    results.append(res)
            except Exception as e:
                LOG.error(f"Error processing URL {future_to_task[future][0]}: {e}")
    return "".join(results)


def _process_url(url: str, subquery: str) -> str:
    # ... (logic from process_url in main.py)
    try:
        content, _, _ = scrape_webpage(url)  # Assuming scrape_webpage is robust
        if content:
            compressed = _contextual_compression(content, subquery)
            return f"SOURCE: {url},\nRelevant Chunks: {compressed}\n\n"
    except Exception as e:
        LOG.warning(f"Could not process URL {url}: {e}")
    return ""


def _contextual_compression(content: str, query: str, k: int = 10) -> str:
    # ... (logic from ContextualCompression in main.py, now using the initialized client)
    content_chunks = utils.generate_tokens(content)
    if not content_chunks:
        return ""

    query_result = embedding_client.embed_content(
        model="models/text-embedding-004", content=[query]
    )
    query_emb = query_result["embedding"][0]

    chunk_result = embedding_client.embed_content(
        model="models/text-embedding-004", content=content_chunks
    )
    chunk_embeddings = [emb for emb in chunk_result["embedding"]]

    similarities = [
        utils.cosine_similarity(query_emb, chunk_emb) for chunk_emb in chunk_embeddings
    ]
    top_indices = np.argsort(similarities)[-k:][::-1]
    return "\n\n".join([content_chunks[i] for i in top_indices])


def _llm_generate_report(agent_role_prompt: str, content_with_prompt: str) -> str:
    # ... (logic from llm_generate_report in main.py)
    messages = [
        {"role": "system", "content": agent_role_prompt},
        {"role": "user", "content": content_with_prompt},
    ]
    combined = " ".join(f"{msg['role']}: {msg['content']}" for msg in messages)
    response = llm_chat_model.send_message(combined)
    return response.text






# =============================================================================
# CORE SERVICES FOR DOCUMENT PROCESSING (/upload-and-process)
# =============================================================================

def _process_with_gemini(content: Any, task_type: str) -> str:
    """Internal function to process content with Google Gemini."""
    try:
        model = genai.GenerativeModel(config.DOCUMENT_PROCESSING_MODEL_NAME)

        if task_type == "summarize_text":
            prompt = f"Please provide a concise summary of the following document:\n\n{content}"
            response = model.generate_content(prompt)
            return response.text
        elif task_type in ["summarize_images", "alt_text"]:
            prompt_parts = []
            if task_type == "summarize_images":
                prompt_parts.append("Provide a detailed summary of the document shown in the following page images.")
            else:
                prompt_parts.append("Provide a concise alt text and a brief one-sentence summary for the following image.")
            
            if isinstance(content, list):
                prompt_parts.extend(content)
            else:
                prompt_parts.append(content)
            
            response = model.generate_content(prompt_parts)
            return response.text
        else:
            raise ValueError("Invalid task type specified.")
    except Exception as e:
        log.error(f"Error with Google Gemini API: {e}")
        raise Exception(f"Error with Google Gemini API: {e}")


def _process_single_document(file_content: bytes, filename: str) -> str:
    """Internal function: Processes one document and returns its summary."""
    file_extension = os.path.splitext(filename)[1].lower()

    if file_extension == ".pdf":
        pdf_doc = fitz.open(stream=file_content, filetype="pdf")
        if len(pdf_doc) > 10:
            raise ValueError(f"PDF '{filename}' exceeds the 10-page limit.")
        
        parsed_content = utils.parse_pdf(file_content, filename)
        if "text" in parsed_content:
            return _process_with_gemini(parsed_content["text"], "summarize_text")
        elif "images" in parsed_content:
            return _process_with_gemini(parsed_content["images"], "summarize_images")
    elif file_extension in [".jpg", ".jpeg", ".png"]:
        image = Image.open(io.BytesIO(file_content))
        return _process_with_gemini(image, "alt_text")
    elif file_extension == ".docx":
        doc = Document(io.BytesIO(file_content))
        text = "\n".join([para.text for para in doc.paragraphs])
        return _process_with_gemini(text, "summarize_text")
    elif file_extension in [".xls", ".xlsx"]:
        df = pd.read_excel(io.BytesIO(file_content))
        return _process_with_gemini(df.to_string(), "summarize_text")
    else:
        return f"Unsupported file type: {filename}"


def _process_file_background(file_content: bytes, filename: str, task_id: str):
    """The main background worker function for processing files."""
    try:
        file_extension = os.path.splitext(filename)[1].lower()

        if file_extension == ".zip":
            all_summaries = []
            with zipfile.ZipFile(io.BytesIO(file_content)) as zf:
                valid_files = [f for f in zf.infolist() if not f.is_dir() and not f.filename.startswith('__MACOSX/')]
                
                if len(valid_files) > 5:
                    raise ValueError(f"ZIP archive exceeds the 5-file limit.")
                
                for file_info in valid_files:
                    try:
                        summary = _process_single_document(zf.read(file_info.filename), file_info.filename)
                        all_summaries.append({"filename": file_info.filename, "summary": summary})
                    except Exception as doc_error:
                        log.error(f"Error processing {file_info.filename} in zip: {doc_error}")
                        all_summaries.append({"filename": file_info.filename, "summary": f"Could not process. Error: {doc_error}"})
            
            background_task_status[task_id] = {"summaries": all_summaries}
        else:
            summary = _process_single_document(file_content, filename)
            background_task_status[task_id] = {"summaries": [{"filename": filename, "summary": summary}]}

    except ValueError as ve:
        # Custom handling for user-facing limit errors
        if "exceeds" in str(ve) and "limit" in str(ve):
             background_task_status[task_id] = {"requires_payment": True, "reason": str(ve)}
        else:
             background_task_status[task_id] = {"error": str(ve)}
    except Exception as e:
        log.error(f"Error processing file in background for task {task_id}: {e}")
        background_task_status[task_id] = {"error": str(e)}


# --- Handler functions to be called by endpoints ---

async def handle_upload_and_process(background_tasks: BackgroundTasks, file: Any):
    """Service handler for creating and dispatching a document processing task."""
    file_content = await file.read()
    task_id = f"task_{file.filename}_{os.urandom(4).hex()}"
    background_task_status[task_id] = "processing"

    background_tasks.add_task(_process_file_background, file_content, file.filename, task_id)
    return task_id


def handle_get_task_status(task_id: str):
    """Service handler for retrieving the status of a task."""
    status = background_task_status.get(task_id)
    if not status:
        raise HTTPException(status_code=404, detail="Task not found")
    return status






# In services.py, add this new import at the top
import json

# ... (other imports)


def _get_resume_analysis_prompt(analysis_type: models.AnalysisType) -> str:
    """
    Selects the expert-level system prompt that instructs the LLM to return a structured JSON object
    that perfectly matches the frontend's data contract.
    """
    base_persona = (
        "You are an expert career coach and professional resume reviewer acting as a data extraction API. "
        "Your task is to analyze the provided resume text and return a single, valid JSON object. "
        "Do NOT add any introductory text, explanations, or Markdown formatting like ```json. "
        "Your entire output must be only the JSON object."
    )

    # This JSON structure now perfectly matches the frontend's TypeScript interfaces.
    json_structure = """
    {
      "score": {
        "overall": <integer, 0-100, a holistic score>,
        "skills": <integer, 0-100, score for skills presentation and relevance>,
        "experience": <integer, 0-100, score for impact and quality of experience section>,
        "education": <integer, 0-100, score for clarity and relevance of education>
      },
      "personalInfo": {
        "name": "<string, extracted name or null>",
        "email": "<string, extracted email or null>",
        "phone": "<string, extracted phone number or null>",
        "location": "<string, extracted city/state or null>"
      },
      "summary": "<string, a 2-4 sentence professional summary based on the resume>",
      "skills": ["<string, skill 1>", "<string, skill 2>", "..."],
      "experience": [
        {
          "position": "<string, job title>",
          "company": "<string, company name>",
          "duration": "<string, e.g., 'Jan 2022 - Present'>",
          "description": "<string, a 1-2 sentence summary of the role's key responsibilities and achievements>"
        }
      ],
      "education": [
        {
          "degree": "<string, e.g., 'Bachelor of Science in Computer Science'>",
          "institution": "<string, university name>",
          "year": "<string, e.g., '2018 - 2022'>"
        }
      ],
      "recommendations": {
        "strengths": ["<string, a key strength of the resume>"],
        "improvements": ["<string, a critical area for improvement>"],
        "suggestions": ["<string, an actionable suggestion>"]
      }
    }
    """
    
    focus_instruction = {
        models.AnalysisType.GENERAL: "Provide a balanced, general analysis focusing on overall presentation and impact.",
        models.AnalysisType.DETAILED: "Provide a detailed analysis, paying close attention to every section. Be critical in your scoring.",
        models.AnalysisType.SKILLS: "Focus your analysis heavily on the skills section. Score the 'skills' field highest. Recommendations should be skills-focused.",
        models.AnalysisType.EXPERIENCE: "Focus your analysis on the work experience section. Score the 'experience' field highest. Recommendations should be experience-focused."
    }

    task_prompt = focus_instruction.get(analysis_type)

    return (
        f"{base_persona}\n\n"
        f"Analysis Focus: {task_prompt}\n\n"
        f"Based on the resume text provided, populate the following JSON structure. Ensure all fields are filled accurately. "
        f"If a piece of information is not present, use null for optional fields and empty arrays [] for lists.\n\n"
        f"JSON Structure to populate:\n{json_structure}"
    )


async def handle_resume_analysis(file: UploadFile, analysis_type: models.AnalysisType):
    """
    Service handler for the resume analysis workflow, now returns a structured dictionary.
    """
    if file.size > config.MAX_RESUME_FILE_SIZE:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="File size exceeds 10MB.")

    try:
        content = await file.read()
        resume_text = utils.parse_document_to_text(content, file.filename)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    prompt = _get_resume_analysis_prompt(analysis_type)
    full_prompt = f"{prompt}\n\n--- RESUME TEXT TO ANALYZE ---\n\n{resume_text}"

    try:
        log.info(f"Sending resume for '{analysis_type.value}' JSON analysis to Gemini.")
        model = genai.GenerativeModel(config.RESUME_ANALYSIS_MODEL_NAME)
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            _executor, 
            lambda: model.generate_content(full_prompt)
        )
        
        # --- CRITICAL CHANGE: Parse the LLM's text response as JSON ---
        try:
            # Clean the response text to remove potential markdown backticks
            cleaned_text = response.text.strip().replace("```json", "").replace("```", "")
            analysis_json = json.loads(cleaned_text)
            log.info("Successfully received and parsed JSON analysis from Gemini.")
            return analysis_json
        except json.JSONDecodeError:
            log.error(f"Failed to parse JSON from Gemini response. Raw response: {response.text}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="AI model returned a malformed response. Please try again."
            )

    except Exception as e:
        log.error(f"Error during Gemini API call for resume analysis: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="AI model failed to generate a response.")
    """
    Service handler for the entire resume analysis workflow.
    """
    # 1. Validate File Size
    if file.size > config.MAX_RESUME_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds the limit of {config.MAX_RESUME_FILE_SIZE / (1024*1024)}MB."
        )

    # 2. Read and Parse File Content
    try:
        content = await file.read()
        resume_text = utils.parse_document_to_text(content, file.filename)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    # 3. Get the Expert Prompt
    prompt = _get_resume_analysis_prompt(analysis_type)
    full_prompt = f"{prompt}\n\n--- RESUME TEXT TO ANALYZE ---\n\n{resume_text}"

    # 4. Call the LLM for Analysis
    try:
        log.info(f"Sending resume for '{analysis_type.value}' analysis to Gemini.")
        model = genai.GenerativeModel(config.RESUME_ANALYSIS_MODEL_NAME)
        
        # Run the blocking network call in the thread pool executor
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            _executor, 
            lambda: model.generate_content(full_prompt)
        )
        
        log.info("Successfully received analysis from Gemini.")
        return response.text
    except Exception as e:
        log.error(f"Error during Gemini API call for resume analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while analyzing the resume with the AI model."
        )