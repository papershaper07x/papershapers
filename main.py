# merged_app.py
import os
import traceback
# ... [keep all existing imports] ...
from fastapi import BackgroundTasks # <--- IMPORT BackgroundTasks

# --- Import our new cache utility functions ---
from cache import create_cache_key, get_from_cache, set_to_cache
from full_paper.run_full_pipeline import call_gemini

import json
import logging
import asyncio
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from typing import Any, Dict, List, Optional
from uuid import UUID
from datetime import datetime

import numpy as np
import pandas as pd
from filelock import FileLock

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError

# Third-party AI clients used in snippets
import google.generativeai as genai
from google import genai as google_genai_client

# run_full_pipeline and llm_researcher imports (assumed available)
from full_paper.run_full_pipeline import (
    load_bge,
    derive_plan_from_filedata,
    build_retrieval_objective,
    retrieve_from_pinecone,
    mmr_and_stratified_sample,
    summarize_and_budget_snippets,
    build_generator_prompt_questions_only,
    call_gemini,
    parse_generator_response,
    embed_texts_bge,
    load_schema_row,
    INPUT_CSV_PATH,
)
from llm_researcher.search import tavily_search
from llm_researcher.utils import scrape_webpage
from llm_researcher.prompts import (
    generate_report_prompt,
    auto_agent_instructions,
    generate_search_queries_prompt,
)
from llm_researcher.config import NO_OF_SOURCEURLS, NO_OF_SUBQUERIES, tavily_key
from full_paper.retrieval_and_summarization import (
    generate_dense_evidence_summary_with_llm,
)

# -------- Logging --------
LOG = logging.getLogger("uvicorn.error")
logging.basicConfig(level=logging.INFO)

# -------- App --------
app = FastAPI(title="Merged PaperRAG + Research FastAPI")


# These are the default values if environment variables are not set
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
# Allow CORS for local development; adjust origins as required
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- Global constants & helpers (from fastapi_app.py) --------
MAX_TEXT_CHARS = (
    1000  # tune: how many chars of original snippet text to return (or 0 to drop)
)


# In main.py, after your imports


def standardize_subject_name(subject: str) -> str:
    """
    Takes a potentially messy subject string from a user request and
    maps it to the exact, clean value stored in Pinecone's metadata.
    This is case-insensitive and handles common aliases.
    """
    s_lower = subject.strip().lower()

    # --- This mapping is the key. You can expand it as needed. ---
    subject_map = {
        "maths": "Maths",
        "mathematics": "Maths",
        "math": "Maths",
        "physics": "Physics",
        "science": "Science",
        "chemistry": "Chemistry",
        "biology": "Biology",
        # Add any other aliases you anticipate
    }

    # Return the standardized name, or the title-cased original if not in the map
    return subject_map.get(s_lower, subject.strip().title())


def sanitize_for_json(obj):
    """
    Recursively convert common non-json types into json-serializable types.
    - numpy arrays -> lists
    - numpy scalars -> python scalars
    - UUID -> str
    - datetime -> isoformat
    Leaves other builtins intact.
    """
    # primitives
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj

    # numpy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # numpy scalar types
    if isinstance(obj, np.generic):
        try:
            return obj.item()
        except Exception:
            return str(obj)

    # UUID
    if isinstance(obj, UUID):
        return str(obj)

    # datetime
    if isinstance(obj, datetime):
        return obj.isoformat()

    # dict -> sanitize recursively, skip non-serializable binary fields like 'embedding'
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            # drop huge binary fields
            if k == "embedding":
                continue
            # optionally truncate the original long text
            if k == "orig_text" or (
                k == "metadata" and isinstance(v, dict) and "text" in v
            ):
                if k == "orig_text":
                    text = v or ""
                    out[k] = text[:MAX_TEXT_CHARS]
                    continue
                else:
                    # metadata: copy but truncate 'text' inside
                    md = {}
                    for mdk, mdv in v.items():
                        if mdk == "text":
                            md[mdk] = (mdv or "")[:MAX_TEXT_CHARS]
                        else:
                            md[mdk] = sanitize_for_json(mdv)
                    out[k] = md
                    continue
            out[k] = sanitize_for_json(v)
        return out

    # list/tuple -> sanitize elements
    if isinstance(obj, (list, tuple, set)):
        return [sanitize_for_json(x) for x in obj]

    # fallback: try to convert via __dict__ or str
    if hasattr(obj, "__dict__"):
        return sanitize_for_json(vars(obj))
    return str(obj)


# -------- Executor config from fastapi_app.py --------
EXECUTOR_WORKERS = 4
executor: ThreadPoolExecutor = None


# -------- Models --------
class GenerateRequest(BaseModel):
    board: str
    class_label: str
    subject: str
    chapters: List[str] = None


def sanitize_for_json(obj):
    """
    Recursively convert common non-json types into json-serializable types.
    """
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, UUID):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items() if k != "embedding"}
    if isinstance(obj, (list, tuple, set)):
        return [sanitize_for_json(x) for x in obj]
    return str(obj)


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


class ResearchInput(BaseModel):
    query: str

# in main.py
def generate_and_cache_background(req: GenerateRequest, executor: ThreadPoolExecutor):
    """
    A self-contained function to run the entire paper generation pipeline
    and save the result to the cache. To be used by BackgroundTasks.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    print(f"BACKGROUND TASK: Starting fresh generation for {req.board} {req.class_label} {req.subject}...")
    try:
        # --- This is a complete copy of the generation pipeline ---
        row = load_schema_row(INPUT_CSV_PATH, req.board, req.class_label, req.subject)
        if not row:
            print(f"BACKGROUND TASK FAILED: Schema not found for {req.subject}.")
            return

        plan = derive_plan_from_filedata(row.get("File_Data", ""), req.subject)
        
        section_tasks = [
            loop.run_in_executor(executor, process_section_sync, sec, row.get("File_Data", ""), req.class_label, req.subject)
            for sec in plan["sections"]
        ]
        sections_results = loop.run_until_complete(asyncio.gather(*section_tasks))
        
        slot_summaries = [{"slot_id": r["section_id"], "slot_meta": r.get("slot_meta", ""), "summaries": r.get("summaries", [])} for r in sections_results]
        planner_text = plan.get("planner_text", "Generate a standard exam paper.")
        prompt = build_generator_prompt_questions_only(planner_text, slot_summaries, plan)
        
        gen_resp = call_gemini(prompt, model_name="models/gemini-2.5-flash-lite")
        parsed_llm_json = parse_generator_response(gen_resp.get("text", ""), call_gemini)
        cleaned_questions = _post_process_and_clean_questions(parsed_llm_json.get("questions", []))
        
        final_paper = {
            "paper_id": parsed_llm_json.get("paper_id", "unknown-id"),
            "board": req.board, "class": req.class_label, "subject": req.subject,
            "total_marks": plan.get("total_marks"), "time_allowed_minutes": plan.get("time_minutes"),
            "general_instructions": plan.get("general_instructions"),
            "questions": cleaned_questions, "retrieval_metadata": sections_results
        }
        
        json_safe_paper = sanitize_for_json(final_paper)
        cache_key = create_cache_key(req.board, req.class_label, req.subject)
        
        # ** THE GOAL: Update the cache with the new paper **
        set_to_cache(cache_key, json_safe_paper)
        print(f"BACKGROUND TASK: Successfully updated cache for key: {cache_key}")
        
    except Exception as e:
        print(f"BACKGROUND TASK FAILED for {req.subject}. Error: {e}", exc_info=True)
    finally:
        loop.close()





# -------- CSV Dataframes (from second file) --------
df_content = pd.DataFrame()
df_prompt = pd.DataFrame()

# -------- Environment keys (from second file) --------
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    LOG.warning(
        "GOOGLE_API_KEY not found in environment. Generative calls will fail until you set it."
    )

CONTENT_CSV_PATH = os.environ.get("CONTENT_CSV", "text_files_data2.csv")
PROMPT_CSV_PATH = os.environ.get("PROMPT_CSV", "prompt_data.csv")
REQUEST_LOG_PATH = os.environ.get("REQUEST_LOG", "/tmp/request_log.txt")


# -------- Startup Events --------
@app.on_event("startup")
async def startup_event():
    """
    Starts executor and loads heavy models (embedding model) in executor to avoid blocking event loop.
    """
    global executor
    executor = ThreadPoolExecutor(max_workers=EXECUTOR_WORKERS)
    loop = asyncio.get_event_loop()
    # Load embedding model once (blocking) - run in executor
    await loop.run_in_executor(executor, load_bge)
    LOG.info("Embedding model load triggered in executor.")


@app.on_event("startup")
def load_csv_data():
    """
    Loads CSVs and configures genai client (keeps original behavior).
    """
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
        LOG.info("Configured google.generativeai client on startup.")


@app.on_event("shutdown")
def shutdown_event():
    global executor
    if executor:
        executor.shutdown(wait=False)
        LOG.info("Executor shutdown initiated.")


# -------- NEW: Post-Processing and Cleaning Logic --------
def _post_process_and_clean_questions(
    questions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Cleans a list of parsed questions to fix common LLM generation errors.

    Args:
      questions: A list of question dictionaries parsed from the LLM's JSON output.

    Returns:
      A cleaned list of question dictionaries.
    """
    if not questions:
        return []

    cleaned_questions = []
    for question in questions:
        # --- FIX 1: Collapse duplicate internal choices ---
        if (
            question.get("is_choice")
            and isinstance(question.get("q_text"), list)
            and len(question["q_text"]) > 1
        ):
            # Extract the actual text from each choice object
            choice_texts = [
                choice.get("q_text", "").strip() for choice in question["q_text"]
            ]
            # If all choice texts are identical, it's a duplicate
            if len(set(choice_texts)) == 1:
                LOG.warning(
                    f"Found and fixed duplicate internal choice for q_id: {question.get('q_id')}"
                )
                question["q_text"] = choice_texts[
                    0
                ]  # Collapse to a single question text
                del question["is_choice"]  # It's no longer a choice
                question["_cleaning_note"] = (
                    "Collapsed duplicate internal choices into a single question."
                )

        # --- FIX 2: Restructure badly formatted Case Studies ---
        # This handles the case where the LLM repeats the passage in every sub-question.
        if "case" in question.get("type", "").lower() and isinstance(
            question.get("q_text"), list
        ):
            # The incorrect format is a list of dicts, e.g., [{"q_text": "Passage... (i) sub-question"}]
            # We need to convert it to the correct format: {"passage": "...", "questions": ["(i)..."]}

            all_sub_texts = [
                item.get("q_text", "") for item in question.get("q_text", [])
            ]
            if not all_sub_texts:
                continue

            # Assume the first text contains the full passage and the first sub-question
            first_full_text = all_sub_texts[0]
            # Find where the first sub-question marker (e.g., "(i)") appears
            match = re.search(r"\(\s*[ivx]+\s*\)", first_full_text)

            if match:
                passage = first_full_text[: match.start()].strip()
                cleaned_sub_questions = []
                for text in all_sub_texts:
                    # Remove the extracted passage from each sub-question text
                    sub_question = text.replace(passage, "").strip()
                    if sub_question:
                        cleaned_sub_questions.append(sub_question)

                if passage and cleaned_sub_questions:
                    LOG.warning(
                        f"Found and restructured case study for q_id: {question.get('q_id')}"
                    )
                    # Rebuild the q_text object in the correct format
                    question["q_text"] = {
                        "passage": passage,
                        "questions": cleaned_sub_questions,
                    }
                    question["_cleaning_note"] = (
                        "Restructured case study from repeated text to a proper passage/questions format."
                    )

        cleaned_questions.append(question)

    return cleaned_questions


# In merged_app.py

# Make sure call_gemini is accessible or imported here
# from full_paper.run_full_pipeline import call_gemini
# from full_paper.retrieval_and_summarization import generate_dense_evidence_summary_with_llm

# In merged_app.py

import re  # Make sure 're' is imported at the top of your file
from typing import Dict, Any  # Ensure these are imported as well

# ... (other imports and functions) ...


def process_section_sync(
    sec: Dict[str, Any], file_data: str, class_label: str, subject: str
) -> Dict[str, Any]:
    """
    The complete and corrected function.
    It retrieves candidates, processes them, uses an LLM to summarize, and ALWAYS returns a dictionary.
    """
    try:
        # 1. Build a specific retrieval objective
        base_objective = build_retrieval_objective(
            sec, subject_guidelines=file_data, user_mode="balanced"
        )
        specific_objective = (
            f"For {class_label} {subject}, find content for: {base_objective}"
        )
        print(f"INFO: Running retrieval for Section {sec.get('section_id')}...")

        # 2. Dynamically build the robust filter
        class_value_for_pinecone = "".join(re.findall(r"\d+", class_label))
        subject_value_for_pinecone = standardize_subject_name(
            subject
        )  # <--- USE THE NEW FUNCTION
        filters = {
            "class": {"$eq": class_value_for_pinecone},
            "subject": {"$eq": subject_value_for_pinecone},
        }
        print(f"DEBUG: Using filter: {filters}")

        # 3. Query Pinecone
        candidates = retrieve_from_pinecone(specific_objective, filters, top_k=20)

        if not candidates:
            print(f"WARNING: No candidates found for Section {sec.get('section_id')}")
            return {
                "section_id": sec.get("section_id"),
                "summaries": [
                    {
                        "id": "N/A",
                        "summary": "No relevant evidence was found for this section.",
                    }
                ],
                "slot_meta": sec.get("title", ""),
            }

        # 4. Process candidates: handle missing embeddings and run MMR
        valid_candidates = [c for c in candidates if c.get("embedding") is not None]
        if not valid_candidates:
            print(
                f"WARNING: Candidates found for Section {sec.get('section_id')} but none had embeddings."
            )
            return {
                "section_id": sec.get("section_id"),
                "summaries": [
                    {"id": "N/A", "summary": "No valid evidence could be processed."}
                ],
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

        # 5. Use the LLM to create a dense, high-quality summary
        print(
            f"INFO: Summarizing {len(selected_snips)} snippets for Section {sec.get('section_id')}..."
        )
        dense_summary = generate_dense_evidence_summary_with_llm(
            selected_snips, specific_objective, call_gemini
        )

        final_summaries = [
            {
                "id": ",".join([str(s.get("snippet_id", "")) for s in selected_snips]),
                "summary": dense_summary,
            }
        ]

        # 6. ALWAYS return a properly formatted dictionary
        return {
            "section_id": sec.get("section_id"),
            "summaries": final_summaries,
            "slot_meta": sec.get("title", ""),
        }

    except Exception as e:
        print(
            f"FATAL ERROR in process_section_sync for section {sec.get('section_id')}: {e}"
        )
        traceback.print_exc()  # This will print a detailed error in your log
        # Return a fallback dictionary to prevent the entire request from crashing
        return {
            "section_id": sec.get("section_id"),
            "summaries": [
                {
                    "id": "ERROR",
                    "summary": "An error occurred during evidence processing.",
                }
            ],
            "slot_meta": sec.get("title", ""),
        }


# -------- NEW: Post-Processing and Cleaning Logic --------
def _post_process_and_clean_questions(
    questions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Cleans a list of parsed questions to fix common LLM generation errors.

    Args:
      questions: A list of question dictionaries parsed from the LLM's JSON output.

    Returns:
      A cleaned list of question dictionaries.
    """
    if not questions:
        return []

    cleaned_questions = []
    for question in questions:
        # --- FIX 1: Collapse duplicate internal choices ---
        if (
            question.get("is_choice")
            and isinstance(question.get("q_text"), list)
            and len(question["q_text"]) > 1
        ):
            # Extract the actual text from each choice object
            choice_texts = [
                choice.get("q_text", "").strip() for choice in question["q_text"]
            ]
            # If all choice texts are identical, it's a duplicate
            if len(set(choice_texts)) == 1:
                LOG.warning(
                    f"Found and fixed duplicate internal choice for q_id: {question.get('q_id')}"
                )
                question["q_text"] = choice_texts[
                    0
                ]  # Collapse to a single question text
                del question["is_choice"]  # It's no longer a choice
                question["_cleaning_note"] = (
                    "Collapsed duplicate internal choices into a single question."
                )

        # --- FIX 2: Restructure badly formatted Case Studies ---
        # This handles the case where the LLM repeats the passage in every sub-question.
        if "case" in question.get("type", "").lower() and isinstance(
            question.get("q_text"), list
        ):
            # The incorrect format is a list of dicts, e.g., [{"q_text": "Passage... (i) sub-question"}]
            # We need to convert it to the correct format: {"passage": "...", "questions": ["(i)..."]}

            all_sub_texts = [
                item.get("q_text", "") for item in question.get("q_text", [])
            ]
            if not all_sub_texts:
                continue

            # Assume the first text contains the full passage and the first sub-question
            first_full_text = all_sub_texts[0]
            # Find where the first sub-question marker (e.g., "(i)") appears
            match = re.search(r"\(\s*[ivx]+\s*\)", first_full_text)

            if match:
                passage = first_full_text[: match.start()].strip()
                cleaned_sub_questions = []
                for text in all_sub_texts:
                    # Remove the extracted passage from each sub-question text
                    sub_question = text.replace(passage, "").strip()
                    if sub_question:
                        cleaned_sub_questions.append(sub_question)

                if passage and cleaned_sub_questions:
                    LOG.warning(
                        f"Found and restructured case study for q_id: {question.get('q_id')}"
                    )
                    # Rebuild the q_text object in the correct format
                    question["q_text"] = {
                        "passage": passage,
                        "questions": cleaned_sub_questions,
                    }
                    question["_cleaning_note"] = (
                        "Restructured case study from repeated text to a proper passage/questions format."
                    )

        cleaned_questions.append(question)

    return cleaned_questions


# @app.post("/generate_full")
# async def generate(req: GenerateRequest):
#     """
#     Async wrapper endpoint. Orchestrates the full, schema-aware RAG pipeline.
#      - Loads the correct blueprint to create a detailed plan.
#      - Processes all sections concurrently to retrieve and summarize evidence.
#      - Assembles a schema-aware prompt and calls the generator LLM.
#      - Parses and CLEANS the final response into a complete paper object.
#     """
#     loop = asyncio.get_event_loop()

#     try:
#         row = await loop.run_in_executor(
#             executor,
#             lambda: load_schema_row(
#                 INPUT_CSV_PATH, req.board, req.class_label, req.subject
#             ),
#         )
#     except Exception as e:
#         LOG.error(f"Failed to load schema row: {e}")
#         raise HTTPException(
#             status_code=500, detail=f"Failed to load schema row: {str(e)}"
#         )

#     if not row:
#         raise HTTPException(
#             status_code=404,
#             detail=f"Schema not found for {req.board}, {req.class_label}, {req.subject}",
#         )

#     file_data = row.get("File_Data", "") or ""
#     plan = derive_plan_from_filedata(file_data, req.subject)
#     LOG.info(
#         f"Derived plan with {len(plan.get('sections', []))} sections for {req.subject}."
#     )

#     # --- Retrieval, Summarization, and Prompt Generation (No changes) ---
#     section_tasks = []
#     for sec in plan["sections"]:
#         section_tasks.append(
#             loop.run_in_executor(
#                 executor,
#                 process_section_sync,
#                 sec,
#                 file_data,
#                 req.class_label,
#                 req.subject,
#             )
#         )

#     sections_results = await asyncio.gather(*section_tasks)

#     slot_summaries_list = []
#     for r in sections_results:
#         slot_summaries_list.append(
#             {
#                 "slot_id": r["section_id"],
#                 "slot_meta": r.get("slot_meta", ""),
#                 "summaries": r.get("summaries", []),
#             }
#         )

#     planner_text = plan.get(
#         "planner_text", "Generate a standard exam paper based on the evidence."
#     )
#     prompt = build_generator_prompt_questions_only(
#         planner_text, slot_summaries_list, plan, gen_settings={"mode": "production"}
#     )

#     try:
#         gen_resp = await loop.run_in_executor(
#             executor,
#             lambda: call_gemini(prompt, model_name="models/gemini-2.5-flash-lite"),
#         )
#     except Exception as e:
#         LOG.error(f"LLM generation failed: {e}")
#         raise HTTPException(status_code=500, detail=f"LLM generation failed: {str(e)}")

#     # --- Parsing and NEW Cleaning Step ---
#     try:
#         parsed_llm_json = parse_generator_response(gen_resp.get("text", ""))
#     except Exception as e:
#         LOG.error(f"Failed to parse LLM JSON response: {e}")
#         raise HTTPException(
#             status_code=500,
#             detail=f"Parse failed: {e}. Raw LLM output: {gen_resp.get('text', '')}",
#         )

#     # ** HERE IS THE NEW STEP **
#     # After successful parsing, clean the content of the questions.
#     LOG.info("LLM response parsed successfully. Starting post-processing and cleaning.")
#     cleaned_questions = _post_process_and_clean_questions(
#         parsed_llm_json.get("questions", [])
#     )

#     # --- Final Response Assembly (No changes, just uses cleaned_questions) ---
#     final_paper_response = {
#         "paper_id": parsed_llm_json.get("paper_id", "unknown-paper-id"),
#         "board": req.board,
#         "class": req.class_label,
#         "subject": req.subject,
#         "total_marks": plan.get("total_marks"),
#         "time_allowed_minutes": plan.get("time_minutes"),
#         "general_instructions": plan.get("general_instructions"),
#         "paper_structure_summary": plan.get("sections"),
#         "questions": cleaned_questions,  # Use the cleaned list here
#         "retrieval_metadata": sanitize_for_json(sections_results),
#     }

#     return sanitize_for_json(final_paper_response)



@app.post("/generate_full")
async def generate(req: GenerateRequest, background_tasks: BackgroundTasks):
    """
    Orchestrates the full RAG pipeline with a Stale-While-Revalidate cache.
    """
    loop = asyncio.get_event_loop()
    cache_key = create_cache_key(req.board, req.class_label, req.subject)
    
    # 1. Check the cache first
    cached_paper = get_from_cache(cache_key)
    
    if cached_paper:
        # --- CACHE HIT ---
        print(f"Serving from cache and triggering background refresh for {cache_key}")
        # Add the background task to re-generate and update the cache.
        # The user does NOT wait for this.
        background_tasks.add_task(generate_and_cache_background, req, executor)
        # Return the stale (but instant) response.
        return cached_paper

    # --- CACHE MISS ---
    # The user must wait this one time.
    print(f"Cache miss. Running synchronous generation for {cache_key}")
    try:
        # --- Run the entire generation pipeline ---
        row = await loop.run_in_executor(executor, lambda: load_schema_row(INPUT_CSV_PATH, req.board, req.class_label, req.subject))
        if not row:
            raise HTTPException(status_code=404, detail="Schema not found")

        plan = derive_plan_from_filedata(row.get("File_Data", ""), req.subject)
        
        section_tasks = [
            loop.run_in_executor(executor, process_section_sync, sec, row.get("File_Data", ""), req.class_label, req.subject)
            for sec in plan["sections"]
        ]
        sections_results = await asyncio.gather(*section_tasks)
        
        slot_summaries = [{"slot_id": r["section_id"], "slot_meta": r.get("slot_meta", ""), "summaries": r.get("summaries", [])} for r in sections_results]
        planner_text = plan.get("planner_text", "Generate a standard exam paper.")
        prompt = build_generator_prompt_questions_only(planner_text, slot_summaries, plan)
        
        gen_resp = await loop.run_in_executor(executor, lambda: call_gemini(prompt, model_name="models/gemini-2.5-flash-lite"))
        
        # Use the robust parser, passing the llm_caller it needs
        parsed_llm_json = parse_generator_response(gen_resp.get("text", ""), call_gemini)
        
        cleaned_questions = _post_process_and_clean_questions(parsed_llm_json.get("questions", []))
        
        final_paper = {
            "paper_id": parsed_llm_json.get("paper_id", "unknown-id"),
            "board": req.board, "class": req.class_label, "subject": req.subject,
            "total_marks": plan.get("total_marks"), "time_allowed_minutes": plan.get("time_minutes"),
            "general_instructions": plan.get("general_instructions"),
            "questions": cleaned_questions, "retrieval_metadata": sections_results
        }
        
        json_safe_paper = sanitize_for_json(final_paper)
        
        # ** THE GOAL: Save the new paper to the cache before returning **
        set_to_cache(cache_key, json_safe_paper)
        
        return json_safe_paper
        
    except Exception as e:
        LOG.error(f"Failed during synchronous generation: {e}", exc_info=True)
        raw_output = locals().get("gen_resp", {}).get("text", "[Could not get raw text]")
        raise HTTPException(status_code=500, detail=f"Failed to generate paper: {e}. Raw LLM Output: {raw_output}")
# -------- Legacy / simpler generator & other routes (from second file) --------

from cache import cache_status
@app.get("/_debug/cache")
def _debug_cache():
    return cache_status()




@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "content_rows": len(df_content),
        "prompt_rows": len(df_prompt),
    }


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


def get_answer(
    Board, Class, Subject, Chapter, Prompt_Type, input_data: InputData
) -> str:
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
        raise HTTPException(
            status_code=404, detail="No matching records found document_content_df"
        )
    if query_type_df.empty:
        raise HTTPException(
            status_code=404, detail="No matching records found query_type_df"
        )
    if prompt_df.empty:
        raise HTTPException(
            status_code=404, detail="No matching records found prompt_df"
        )

    prompt = prompt_df["Prompt_Data"].values[0]
    document_content = document_content_df["File_Data"].tolist()[0]
    query_type = query_type_df["File_Data"].tolist()[0]

    answer_paper = f"""You are a Expert teacher and you teach the students in a school. Your aim is to help students prepare well for their final examinations with atmost precision and clearity.
You are provided with a sample paper from previous year/s your task is to solve the paper provided with at most precision and accuracy. 
Note:- Be clear and simple.Guideline for Question paper:- {input_data.question_paper}

Textbook content:-{document_content}
"""
    response_text = generate_response(
        model="models/gemini-2.5-flash-lite", msg=answer_paper
    )
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
        raise HTTPException(
            status_code=404, detail="No matching records found document_content_df"
        )
    if query_type_df.empty:
        raise HTTPException(
            status_code=404, detail="No matching records found query_type_df"
        )
    if prompt_df.empty:
        raise HTTPException(
            status_code=404, detail="No matching records found prompt_df"
        )

    prompt = prompt_df["Prompt_Data"].values[0]
    document_content = document_content_df["File_Data"].tolist()[0]
    query_type = query_type_df["File_Data"].tolist()[0]

    msg = prepare_message(prompt, document_content, query_type)
    response = generate_response(model="models/gemini-2.5-flash-lite", msg=msg)
    return response


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


@app.post("/process")
async def process(input_data: InputData):
    try:
        if not input_data.is_logedIn:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not authenticated",
            )

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

        return {
            "id": input_data.id,
            "result": response_text,
            "hit_count": input_data.hit_count,
        }

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
    """
    The legacy/generic generator endpoint from the second file.
    Renamed to /generate_legacy to avoid collision with the PaperRAG /generate endpoint.
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
    try:
        log_request(input_data, response_text)
    except Exception as e:
        LOG.error(f"Failed to write request log during generate_legacy: {e}")
    return {
        "id": input_data.id,
        "result": response_text,
        "hit_count": input_data.hit_count,
    }


# --------------------------------------------
# Integrated llm_researcher code (kept logic intact)
# --------------------------------------------

# Set up environment variables as provided
os.environ["TAVILY_API_KEY"] = tavily_key

# Configure the genai client (again) - keep original snippet behavior
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Create the model and start a chat session (mirrors provided logic)
try:
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash-exp",
        generation_config=generation_config,
    )
    llm = model.start_chat(history=[])
    # Initialize Google API client for embeddings
    client = google_genai_client.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    LOG.info("LLM research model and client initialized.")
except Exception as e:
    # keep behavior permissive as original code would error on import
    LOG.exception("Failed to initialize research LLM or client on import: %s", e)
    model = None
    llm = None
    client = None


# Helper functions from the research snippet
def choose_agent(query):
    try:
        messages = [
            {"role": "system", "content": auto_agent_instructions()},
            {"role": "user", "content": f"task: {query}"},
        ]
        combined = " ".join(f"{msg['role']}: {msg['content']}" for msg in messages)
        response = llm.send_message(combined)
        if not response or not response.text.strip():
            raise ValueError("Empty response from LLM")
        return response.text
    except Exception as e:
        LOG.error("choose_agent failed: %s", e)
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
        combined = " ".join(f"{msg['role']}: {msg['content']}" for msg in messages)
        response = llm.send_message(combined)
        return response.text
    except Exception as e:
        LOG.error("generate_sub_queries failed: %s", e)
        return []


def gather_urls_for_subqueries(
    sub_queries: list, NO_OF_SOURCEURLS, headers=None, topic="general"
):
    def fetch_urls_for_subquery(sub_query):
        return tavily_search(sub_query, headers, topic, NO_OF_SOURCEURLS)

    with concurrent.futures.ThreadPoolExecutor() as tpe:
        results = list(tpe.map(fetch_urls_for_subquery, sub_queries))

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
    with concurrent.futures.ThreadPoolExecutor() as tpe:
        future_to_task = {
            tpe.submit(process_url, url, subquery): (url, subquery)
            for url, subquery in tasks
        }
        for future in concurrent.futures.as_completed(future_to_task):
            url, subquery = future_to_task[future]
            try:
                res = future.result()
                if res:
                    results.append(res)
            except Exception as e:
                LOG.error(f"Error processing URL {url}: {e}")
    return "".join(results)


def generate_tokens(s, chunk_size=500):
    return [s[i : i + chunk_size] for i in range(0, len(s), chunk_size)]


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
    if hasattr(embedding_obj, "values"):
        try:
            return np.array(list(embedding_obj.values), dtype=float)
        except Exception as e:
            raise ValueError(
                "Failed to extract numeric vector from embedding (values): " + str(e)
            )
    if hasattr(embedding_obj, "embedding"):
        try:
            return np.array(list(embedding_obj.embedding), dtype=float)
        except Exception as e:
            raise ValueError(
                "Failed to extract numeric vector from embedding (embedding): " + str(e)
            )
    if isinstance(embedding_obj, (list, tuple, np.ndarray)):
        return np.array(embedding_obj, dtype=float)
    try:
        arr = np.array(embedding_obj)
        if arr.ndim == 1:
            return arr.astype(float)
    except Exception:
        pass
    raise ValueError(
        "Cannot extract numeric vector from provided embedding: type {}".format(
            type(embedding_obj)
        )
    )


def ContextualCompression(content, query, k=10):
    content_chunks = generate_tokens(content)
    if not content_chunks:
        return ""

    query_result = client.models.embed_content(
        model="text-embedding-004", contents=[query]
    )
    if not query_result.embeddings or len(query_result.embeddings) == 0:
        raise ValueError("Failed to obtain query embedding")
    query_emb = extract_vector(query_result.embeddings[0])

    chunk_result = client.models.embed_content(
        model="text-embedding-004", contents=content_chunks
    )
    if not chunk_result.embeddings or len(chunk_result.embeddings) != len(
        content_chunks
    ):
        raise ValueError("Mismatch in number of embeddings returned for content chunks")
    chunk_embeddings = [extract_vector(emb) for emb in chunk_result.embeddings]

    similarities = [
        cosine_similarity(query_emb, chunk_emb) for chunk_emb in chunk_embeddings
    ]

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
        combined = " ".join(f"{msg['role']}: {msg['content']}" for msg in messages)
        with concurrent.futures.ThreadPoolExecutor() as tpe:
            future = tpe.submit(llm.send_message, combined)
            report = future.result()
        return report.text
    except Exception as e:
        LOG.error("llm_generate_report failed: %s", e)
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
        LOG.info(f"Subquery: {subquery}")
        for idx, url in enumerate(urls, start=1):
            LOG.info(f"  {idx}: {url}")

    results = process_subqueries_parallel(subquery_url_map)
    content_with_prompt = generate_report_prompt(query, results, "str")
    report = llm_generate_report(role_prompt, content_with_prompt)
    return report


@app.post("/research")
async def research_endpoint(request: Request):
    try:
        payload = await request.json()
        if isinstance(payload, str):
            payload = json.loads(payload)
        input_data = ResearchInput(**payload)
        report = research_conduct(input_data.query)
        return {"response": report}
    except ValidationError as ve:
        LOG.error("Validation error in /research: %s", ve)
        raise HTTPException(status_code=422, detail=ve.json())
    except Exception as e:
        LOG.exception("Error in /research")
        raise HTTPException(
            status_code=500, detail="An error occurred in research workflow"
        )


# End of merged_app.py
