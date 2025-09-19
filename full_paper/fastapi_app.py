# fastapi_app.py
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# Add near top imports
import numpy as np
from uuid import UUID
from datetime import datetime

# Add this helper somewhere above the endpoint (module level)
MAX_TEXT_CHARS = 1000  # tune: how many chars of original snippet text to return (or 0 to drop)

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
            if k == "orig_text" or (k == "metadata" and isinstance(v, dict) and "text" in v):
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



# import your existing functions (adjust import paths if module names differ)
from run_full_pipeline import (
    load_bge,
    derive_plan_from_filedata,
    build_retrieval_objective,
    retrieve_from_pinecone,
    mmr_and_stratified_sample,
    summarize_and_budget_snippets,
    build_generator_prompt_questions_only,
    call_gemini,
    parse_generator_response,embed_texts_bge,
    # grounding_check_answer  # optional; keep commented if expensive
)

# tune max workers depending on CPU/GPU; if embedding uses GPU, keep small (1-4)
EXECUTOR_WORKERS = 4

app = FastAPI(title="PaperRAG FastAPI")

executor: ThreadPoolExecutor = None

class GenerateRequest(BaseModel):
    board: str
    class_label: str
    subject: str
    chapters: List[str] = None
    # additional params can be added (e.g., top_k overrides)

@app.on_event("startup")
async def startup_event():
    global executor
    executor = ThreadPoolExecutor(max_workers=EXECUTOR_WORKERS)
    # Load embedding model once (blocking) - do this in executor to avoid blocking event loop longer than necessary
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(executor, load_bge)
    # optionally do a small embedding warmup
    # loop.run_in_executor(executor, lambda: embed_texts_bge(["warmup"], batch_size=1))

@app.on_event("shutdown")
def shutdown_event():
    global executor
    if executor:
        executor.shutdown(wait=False)

def process_section_sync(sec: Dict[str, Any], file_data: str, class_label: str, subject: str) -> Dict[str, Any]:
    """
    Blocking sync function that retrieves candidates for a section,
    runs batch-embedding if necessary, runs MMR+sampling, summarization,
    and returns selected snippets / compact summaries for the section.
    This is run inside the ThreadPoolExecutor for parallelism.
    """
    # 1) build objective
    objective = build_retrieval_objective(sec, subject_guidelines=file_data, user_mode='balanced')

    # 2) query pinecone (synchronously)
    filters = {'class': {'$eq': ''.join([c for c in class_label if c.isdigit()])}, 'subject': {'$eq': subject}}
    candidates = retrieve_from_pinecone(objective, filters, top_k=12)  # smaller top_k for speed

    if not candidates:
        return {"section_id": sec.get("section_id"), "selected": []}

    # 3) batch missing embeddings (reuse the function you added in run_full_pipeline)
    # note: retrieve_from_pinecone should already include vector values (embedding) if index supports it
    missing_texts = []
    missing_idxs = []
    for i, c in enumerate(candidates):
        if c.get("embedding") is None:
            missing_idxs.append(i)
            missing_texts.append(c.get("text", ""))
    if missing_texts:
        batch_embs = embed_texts_bge(missing_texts, batch_size=min(64, len(missing_texts)))
        for idx, emb in zip(missing_idxs, batch_embs):
            candidates[idx]["embedding"] = emb
    emb_matrix = np.vstack([c["embedding"] for c in candidates])

    # 4) compute query embedding and pick evidence
    query_emb = embed_texts_bge([objective], batch_size=1)[0]
    desired = min(6, max(3, int(sec.get("num_questions", 6))))
    picks = mmr_and_stratified_sample(query_emb, emb_matrix, [{'id': c.get('snippet_id')} for c in candidates],
                                      metadata=[c.get('metadata', {}) for c in candidates], n_samples=desired)
    selected_snips = [candidates[i] for i in picks]

    # 5) summarize + budget the selected snippets to compact summaries
    slot_summaries, _ = summarize_and_budget_snippets(selected_snips, objective, max_tokens=120)
    selected_summaries, _ = summarize_and_budget_snippets(selected_snips, objective, max_tokens=120)


    return {
            "section_id": sec.get("section_id"),
            "selected": selected_snips,
            "summaries": selected_summaries,
            "slot_meta": sec.get("title", "")
        }
    # return {"section_id": sec.get("section_id"), "selected": selected_snips, "summaries": summaries_wrapped}

@app.post("/generate")
async def generate(req: GenerateRequest):
    """
    Async wrapper endpoint. Orchestrates:
     - get row/file_data -> derive plan
     - process all sections concurrently in executor
     - assemble prompt and call Gemini (in executor)
    """
    loop = asyncio.get_event_loop()

    # 0) load schema row - you can keep your existing function for this
    # (we assume run_full_pipeline has load_schema_row and derive_plan_from_filedata)
    try:
        # Use your existing CSV-loading function here. I'm calling it load_schema_row for example.
        from run_full_pipeline import load_schema_row, INPUT_CSV_PATH, derive_plan_from_filedata
        row = await loop.run_in_executor(executor, lambda: load_schema_row(INPUT_CSV_PATH, req.board, req.class_label, req.subject))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not row:
        raise HTTPException(status_code=404, detail="schema row not found")

    file_data = row.get("File_Data", "") or ""
    plan = derive_plan_from_filedata(file_data)

    # 1) run section processing in parallel (executor)
    section_tasks = []
    for sec in plan["sections"]:
        # schedule blocking work in executor per section
        section_tasks.append(loop.run_in_executor(executor, process_section_sync, sec, file_data, req.class_label, req.subject))

    # gather results
    sections_results = await asyncio.gather(*section_tasks)

    # recompose into structure expected by prompt builder (slot_summaries etc.)
# convert results into the list-of-slot-dicts shape builder expects
    slot_summaries_list = []
    for r in sections_results:
        slot_summaries_list.append({
            "slot_id": r["section_id"],
            "slot_meta": r.get("slot_meta", ""),
            "summaries": r.get("summaries", [])
        })

# debug print to validate shape (remove later)
    # import pprint
    # pprint.pp(slot_summaries_list)

# pass the list to the prompt builder

    # 2) build prompt (fast, CPU)
    planner_text = plan.get("planner_text", "")  # or however you produce planner text
    gen_settings = {"mode": "production"}  # pass any generator settings here

    # sanity checks
    if not isinstance(slot_summaries_list, list):
        raise RuntimeError("slot_summaries_list must be a list")
    for slot in slot_summaries_list:
        if not isinstance(slot, dict) or "slot_id" not in slot or "summaries" not in slot:
            raise RuntimeError(f"Bad slot shape: {slot}")
        for summ in slot["summaries"]:
            if not isinstance(summ, dict) or "summary" not in summ:
                raise RuntimeError(f"Bad summary shape for slot {slot['slot_id']}: {summ}")


    prompt = build_generator_prompt_questions_only(planner_text, slot_summaries_list, gen_settings)



    # 3) call Gemini in executor (blocking)
    try:
        gen_resp = await loop.run_in_executor(executor, lambda: call_gemini(prompt, model_name="models/gemini-2.5-flash-lite"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {e}")

    # 4) parse LLM output (fast)
    try:
        parsed = parse_generator_response(gen_resp.get("text", ""))
    except Exception as e:
        # optionally retry parse by calling safe_generate or returning raw LLM output
        raise HTTPException(status_code=500, detail=f"Parse failed: {e}")

    # 5) optionally run grounding (fast if vectorized & cached); skipping for speed
    # return {"paper": parsed, "sections_meta": sections_results}
# 5) sanitize and drop heavy fields before returning
# Remove embeddings from selected_snips and create safe copy
    clean_sections = []
    for sec in sections_results:
        # deep copy not strictly necessary if we construct new dict
        sec_copy = {
            "section_id": sec.get("section_id"),
            "slot_meta": sec.get("slot_meta"),
            # sanitize summaries and selected snippets
            "summaries": sanitize_for_json(sec.get("summaries", [])),
        }
        # For selected snippets, drop embedding and optionally trim orig_text inside metadata
        selected = []
        for snip in sec.get("selected", []):
            sn = {}
            # copy fields but remove embedding
            for k, v in snip.items():
                if k == "embedding":
                    continue
                if k == "orig_text":
                    sn[k] = (v or "")[:MAX_TEXT_CHARS]
                elif k == "metadata" and isinstance(v, dict):
                    md = dict(v)  # shallow copy
                    if "text" in md:
                        md["text"] = (md["text"] or "")[:MAX_TEXT_CHARS]
                    sn[k] = sanitize_for_json(md)
                else:
                    sn[k] = sanitize_for_json(v)
            selected.append(sn)
        sec_copy["selected"] = selected
        clean_sections.append(sec_copy)

    # sanitize parsed LLM output too (it might contain numpy types if you attached any)
    safe_parsed = sanitize_for_json(parsed)

    return {"paper": safe_parsed, "sections_meta": clean_sections}
