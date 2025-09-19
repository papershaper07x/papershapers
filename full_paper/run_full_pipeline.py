# run_full_pipeline.py
"""
End-to-end pipeline:
 - loads CSV schema rows for (board,class,subject)
 - derives a planner template from File_Data heuristically
 - builds retrieval objectives per slot
 - queries Pinecone with embeddings (BAAI/bge-large-en-v1.5 assumed)
 - MMR + stratified sampling to pick evidence snippets
 - compact summaries to fit token budget
 - build a batched RAG prompt and call Gemini
 - parse + grounding check
 - save result to last_generated_paper.json

Environment:
 - PINECONE_API_KEY (required)
 - PINECONE_INDEX_NAME (optional; default papershapers)
 - GOOGLE_API_KEY (required)
 - INPUT_CSV_PATH (optional default './schema.csv')
 - DRY_RUN=1 to avoid external calls (for dev)

Dependencies:
 pip install torch transformers pinecone-client google-generativeai pandas numpy
(plus the helper modules you have created.)
"""
import os
import os

# Set an environment variable

import re
import time
import json
import traceback
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd

# HF & Pinecone & Gemini
import torch
from transformers import AutoTokenizer, AutoModel
from pinecone import Pinecone
import google.generativeai as genai

# helper modules you created
from full_paper.mmr_and_sample import mmr_and_stratified_sample
from full_paper.retrieval_and_summarization import build_retrieval_objective, summarize_and_budget_snippets
from full_paper.batched_prompt_builder import  parse_generator_response, safe_generate,build_generator_prompt_questions_only
# from full_paper.TRASH.grounding_check import grounding_check_answer

# -------------------------
# Config (env)
# -------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "papershapers2")
INPUT_CSV_PATH = "instructions.csv"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")  # MUST match index dim
DRY_RUN = os.getenv("DRY_RUN", "0") == "1"

# -------------------------
# Small logger
# -------------------------
def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)



# -------------------------
# CSV loader & planner extractor
# -------------------------
def load_schema_row(csv_path: str, board: str, class_label: str, subject: str) -> Optional[Dict[str, Any]]:
    """
    Load the CSV and return the best matching row for board/class/subject.
    Preference order:
      1) exact match with Chapter == 'Mock Paper' for this subject
      2) exact subject row (chapter may be the subject-specific file)
      3) first row matching board+class+subject
    Returns None if nothing found.
    """
    df = pd.read_csv(csv_path)
    # normalize keys
    df = df.fillna("")
    mask = (df['Board'].str.strip().str.lower() == board.strip().lower()) & \
           (df['Class'].str.strip().str.lower() == class_label.strip().lower()) & \
           (df['Subject'].str.strip().str.lower() == subject.strip().lower())
    candidates = df[mask]
    if candidates.empty:
        # fallback: match board+class only
        mask2 = (df['Board'].str.strip().str.lower() == board.strip().lower()) & \
                (df['Class'].str.strip().str.lower() == class_label.strip().lower())
        candidates = df[mask2]
        if candidates.empty:
            return None
    # Prefer 'Mock Paper' chapter row
    for _, row in candidates.iterrows():
        if str(row['Chapter']).strip().lower() in ('mock paper', 'mock_paper', 'mockpaper'):
            return row.to_dict()
    # otherwise return first candidate that has 'mock' or 'guideline' words in File_Data
    for _, row in candidates.iterrows():
        fd = str(row.get('File_Data','')).lower()
        if 'mock' in fd or 'guideline' in fd or 'guidelines' in fd:
            return row.to_dict()
    # else return first candidate row
    return candidates.iloc[0].to_dict()

def derive_plan_from_filedata(file_data: str, default_total_marks: int = 80, default_time: int = 180) -> Dict[str, Any]:
    """
    Heuristic parser: attempts to find sections, marks, and question types from file_data text.
    Returns a planner dict with: total_marks, time_minutes, sections(list of dicts)
    The parser is intentionally conservative. When it cannot find details we fallback to a reasonable default.
    """
    text = (file_data or "").replace("\\r", "\n").replace("\\n", "\n")
    text = re.sub(r'\s+', ' ', text).strip()
    plan = {"total_marks": default_total_marks, "time_minutes": default_time, "sections": []}

    # quick search for explicit total marks or time
    tm = re.search(r'(total\s*marks?)[:\s]*([0-9]{2,3})', text, flags=re.IGNORECASE)
    if tm:
        plan['total_marks'] = int(tm.group(2))
    tt = re.search(r'(time\s*(in)?\s*minutes|duration)[:\s]*([0-9]{2,4})', text, flags=re.IGNORECASE)
    if tt:
        # group 3 may be present
        num = re.search(r'([0-9]{2,4})', tt.group(0))
        if num:
            plan['time_minutes'] = int(num.group(1))

    # look for sections by keywords
    # If file_data explicitly mentions MCQ / Objective / Short Answer / Long Answer, create sections
    lc = text.lower()
    sections = []
    if 'mcq' in lc or 'objective' in lc:
        sections.append({
            "section_id": "A",
            "title": "Objective",
            "marks_for_section": 20,
            "num_questions": 20,
            "question_type": "MCQ",
            "per_question_mark": 1,
            "difficulty_distribution": {"easy": 60, "medium":30, "hard":10}
        })
    if 'short answer' in lc or 'short answer' in text or 'short' in lc:
        sections.append({
            "section_id": "B",
            "title": "Short Answer",
            "marks_for_section": 30,
            "num_questions": 6,
            "question_type": "SA",
            "per_question_mark": 5,
            "difficulty_distribution": {"easy": 40, "medium":50, "hard":10}
        })
    if 'long answer' in lc or 'long' in lc or 'long answer' in text:
        sections.append({
            "section_id": "C",
            "title": "Long Answer",
            "marks_for_section": 30,
            "num_questions": 2,
            "question_type": "LA",
            "per_question_mark": 15,
            "difficulty_distribution": {"easy": 0, "medium":50, "hard":50}
        })

    # if parser didn't find any sections, create a reasonable default distribution
    if not sections:
        sections = [
            {"section_id":"A","title":"Objective","marks_for_section":20,"num_questions":20,"question_type":"MCQ","per_question_mark":1,"difficulty_distribution":{"easy":60,"medium":30,"hard":10}},
            {"section_id":"B","title":"Short Answer","marks_for_section":30,"num_questions":6,"question_type":"SA","per_question_mark":5,"difficulty_distribution":{"easy":40,"medium":50,"hard":10}},
            {"section_id":"C","title":"Long Answer","marks_for_section":30,"num_questions":2,"question_type":"LA","per_question_mark":15,"difficulty_distribution":{"easy":0,"medium":50,"hard":50}}
        ]
    plan['sections'] = sections
    return plan

# -------------------------
# Embedding model (BGE) - MUST match index dimension
# -------------------------
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_log_model_loaded = False
_tokenizer = None
_model = None

def load_bge(model_name: str = EMBED_MODEL):
    global _log_model_loaded, _tokenizer, _model
    if _model is None or _tokenizer is None:
        log(f"Loading embedding model {model_name} on device {_device} (this may take 10-60s)...")
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModel.from_pretrained(model_name).to(_device)
        _model.eval()
        _log_model_loaded = True
        log("Embedding model ready.")
    return _tokenizer, _model, _device

def embed_texts_bge(texts: List[str], batch_size: int = 8) -> np.ndarray:
    """CLS pooling, normalization, returns (n, dim) numpy array."""
    if not texts:
        return np.zeros((0, _model.config.hidden_size))
    all_embs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
            out = model(**enc)
            cls = out[0][:, 0]  # CLS pooling
            cls = torch.nn.functional.normalize(cls, p=2, dim=1)
            all_embs.append(cls.cpu().numpy())
    return np.vstack(all_embs)


tokenizer, model, device = load_bge()

# -------------------------
# Pinecone retrieval (BGE vectors)
# -------------------------
def retrieve_from_pinecone(objective_text: str, filters: Dict[str, Any], top_k: int = 20) -> List[Dict[str, Any]]:
    """
    Embed objective using BGE and query Pinecone index. Returns candidate dicts with embeddings.
    """
    if DRY_RUN:
        log("DRY_RUN: returning canned candidates.")
        return [
            {"snippet_id":"s1","text":"Photosynthesis is the process by which plants convert light into chemical energy in chloroplasts.","metadata":{"chapter":"Photosynthesis","source_id":"doc1"},"score":0.99,"embedding":None},
            {"snippet_id":"s2","text":"Transpiration occurs through stomata and helps in the movement of water.","metadata":{"chapter":"Transpiration","source_id":"doc2"},"score":0.95,"embedding":None}
        ]
    if not PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY not set in environment.")

    # embed objective with BGE
    q_emb = embed_texts_bge([objective_text], batch_size=1)[0].tolist()

    pc = Pinecone(api_key=PINECONE_API_KEY)
    idx = pc.Index(PINECONE_INDEX_NAME)
    log("Querying Pinecone with BGE vector...")
    # ensure index created earlier: index = pinecone.Index(PINECONE_INDEX_NAME)
    # Request values + metadata so we don't need to re-embed snippets
    resp = idx.query(vector=q_emb, top_k=top_k, include_values=True, include_metadata=True, filter=filters)
    matches = resp.get('matches', []) or resp.get('results', [{}])[0].get('matches', [])

    results = []
    for m in matches:
        md = m.get('metadata', {}) or {}
        text = md.get('text') or md.get('text_preview') or md.get('content') or ""
        sid = m.get('id') or md.get('snippet_id') or None
        score = m.get('score', None)
        # we do NOT re-embed here; we'll request embedding if index returns values; else compute later.
        emb = None
        if isinstance(m, dict):
            if 'values' in m and m['values']:
                emb = np.array(m['values'], dtype=float)
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
        results.append({"snippet_id": sid or f"m{i}", "text": text, "metadata": md, "score": score, "embedding": emb})
    log(f"Pinecone returned {len(results)} matches.")
    return results

# -------------------------
# Gemini caller (simple)
# -------------------------
def call_gemini(prompt: str, model_name: str = "models/gemini-2.5-flash", temperature: float = 0.0, max_retries: int = 2) -> Dict[str, Any]:
    if DRY_RUN:
        log("DRY_RUN: returning canned generator output.")
        fake = '{"paper_id":"dryrun","questions":[{"section_id":"A","q_id":"A.1","q_text":"What is photosynthesis?","type":"SA","marks":2,"difficulty":"Easy","answer":"Photosynthesis is the process by which plants convert light into chemical energy in chloroplasts.","sources":["s1"],"rationale":"Supported by s1","needs_review":false}]}'
        return {"text": "```json\n" + fake + "\n```", "raw": None}
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY not set in environment.")
    genai.configure(api_key=GOOGLE_API_KEY)
    last_exc = None
    for attempt in range(1, max_retries+1):
        try:
            log(f"Calling Gemini (attempt {attempt})...")
            llm = genai.GenerativeModel(model_name)
            resp = llm.generate_content(prompt)
            # extract text robustly
            text = None
            if hasattr(resp, "text") and isinstance(resp.text, str):
                text = resp.text
            elif isinstance(resp, dict):
                text = resp.get("text") or (resp.get("candidates") and resp["candidates"][0].get("content"))
            else:
                text = str(resp)
            return {"text": text, "raw": resp}
        except Exception as e:
            log(f"Gemini exception: {type(e).__name__}: {e}")
            last_exc = e
            time.sleep(0.7 * attempt)
    raise last_exc

# -------------------------
# Orchestrator
# -------------------------
def run_pipeline(board: str, class_label: str, subject: str, chapters: Optional[List[str]] = None):
    try:
        log("Pipeline start")
        log(f"Loading schema row for {board} | {class_label} | {subject} from {INPUT_CSV_PATH}")
        row = load_schema_row(INPUT_CSV_PATH, board, class_label, subject)
        if not row:
            log("No schema row found. Aborting.")
            return
        file_data = row.get('File_Data', '') or ''
        plan = derive_plan_from_filedata(file_data)
        log(f"Derived plan: total_marks={plan['total_marks']} time={plan['time_minutes']} sections={len(plan['sections'])}")

        # For each section produce retrieval objectives, then retrieve & pick evidence
        selected_per_section = {}
        for sec in plan['sections']:
            # build compact human objective string
            objective = build_retrieval_objective(sec, subject_guidelines=file_data, user_mode='balanced')
            log(f"Section {sec.get('section_id')} objective: {objective[:180]}...")
            filters = {'class': {'$eq': re.sub(r'[^0-9]', '', class_label)}, 'subject': {'$eq': subject}}
            candidates = retrieve_from_pinecone(objective, filters, top_k=20)

            if not candidates:
                log(f"No candidates found for section {sec.get('section_id')}. continuing.")
                selected_per_section[sec['section_id']] = []
                continue

            # prepare embeddings matrix & metadata
            meta = [c.get('metadata', {}) for c in candidates]
            ids = [c.get('snippet_id') for c in candidates]
            missing_idxs = []
            missing_texts = []
            for i, c in enumerate(candidates):
                if c.get('embedding') is None:
                    missing_idxs.append(i)
                    missing_texts.append(c.get('text', ''))
            # compute batch embeddings for missing texts (if any)
            if missing_texts:
                # pick a sensible batch size (tune: 32/64 depending on GPU)
                batch_size = min(64, len(missing_texts))
                batch_embs = embed_texts_bge(missing_texts, batch_size=batch_size)
                # assign back to candidates in original order
                for idx, emb in zip(missing_idxs, batch_embs):
                    candidates[idx]['embedding'] = emb
            # now build embs list in original order
            embs = [c.get('embedding') for c in candidates]
            emb_matrix = np.vstack(embs)
            # create candidate placeholder objects as expected by mmr_and_stratified_sample
            candidates_placeholder = [{'id': ids[i]} for i in range(len(ids))]

            # compute query emb for mmr
            query_emb = embed_texts_bge([objective], batch_size=1)[0]

            # compute number of evidence snippets desired from plan (e.g., num_questions or fixed)
            desired = min(8, max(3, int(sec.get('num_questions', 6))))  # heuristic
            picks = mmr_and_stratified_sample(query_emb, emb_matrix, candidates_placeholder, meta, n_samples=desired, chapter_weights=sec.get('chapter_allocation', {}).get('chapter_weights', None), diversity=0.7, seed=int(time.time())%10000)
            selected_snips = [candidates[i] for i in picks]
            selected_per_section[sec['section_id']] = selected_snips
            log(f"Section {sec['section_id']} selected {len(selected_snips)} snippets.")

        # flatten selected snippets for prompt building (we will include per-slot evidence)
        slot_summaries = []
        for sec in plan['sections']:
            s_id = sec['section_id']
            sel = selected_per_section.get(s_id, [])
            # summarize with budget per section
            max_tokens_for_section = 400  # heuristic; you may tune per section
            summaries, tok = summarize_and_budget_snippets(sel, build_retrieval_objective(sec, file_data, 'balanced'), max_tokens_for_section)
            slot_summaries.append({'slot_id': s_id, 'slot_meta': sec.get('title',''), 'summaries': summaries})
            log(f"Section {s_id} summaries chosen: {len(summaries)} (est tokens {tok})")

        # build planner_text summary
        planner_text = f"Board: {board}. Class: {class_label}. Subject: {subject}. Total marks: {plan['total_marks']}. Time: {plan['time_minutes']} mins."

        # build prompt
        # prompt = build_generator_prompt(planner_text, slot_summaries, gen_settings={'mode':'balanced','answer_style':'short'})
        prompt = build_generator_prompt_questions_only(planner_text, slot_summaries, gen_settings={'mode':'balanced'})
        # save prompt for debugging
        with open("last_generated_prompt_questions_only.txt","w",encoding="utf-8") as f:
            f.write(prompt)

        # gemini_resp = call_gemini(prompt, model_name="models/gemini-2.5-flash", temperature=0.0)
        # parsed = parse_generator_response(gemini_resp.get('text',''))

        # save prompt
        with open("last_generated_prompt.txt", "w", encoding="utf-8") as f:
            f.write(prompt)
        log("Prompt built and saved to last_generated_prompt.txt (open to inspect).")

        # call Gemini
        gen_resp = call_gemini(prompt, model_name="models/gemini-2.5-flash")
        log("Generator returned; parsing JSON...")
        parsed = parse_generator_response(gen_resp.get('text', ''))

        # # grounding check using BGE embeddings (fast-ish)
        # for q in parsed.get('questions', []):
        #     src_ids = q.get('sources', [])
        #     cited = []
        #     for s_id in src_ids:
        #         # search in selected snippets
        #         for secs in selected_per_section.values():
        #             for c in secs:
        #                 if c.get('snippet_id') == s_id:
        #                     cited.append(c)
        #     # if cited empty, try to match by id in entire flattened selected list
        #     if not cited:
        #         flat = [item for sub in selected_per_section.values() for item in sub]
        #         for c in flat:
        #             if c.get('snippet_id') in src_ids:
        #                 cited.append(c)
        #     embed_fn = lambda texts: embed_texts_bge(texts, batch_size=4)
        #     check = grounding_check_answer(q.get('answer',''), [{'snippet_id': c.get('snippet_id'), 'text': c.get('text')} for c in cited], embed_fn=embed_fn)
        #     q['_grounding_check'] = check
        #     log(f"Q {q.get('q_id')} needs_review={check['needs_review']} reasons={check['reasons']}")

        out_file = "last_generated_paper.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(parsed, f, ensure_ascii=False, indent=2)
        log(f"Saved final output to {out_file}")
        log("Pipeline done.")
    except Exception as e:
        log("Pipeline exception:")
        traceback.print_exc()

# -------------------------
# If run as script, run an example
# -------------------------
import os, time, traceback

if __name__ == "__main__":
    # quick test parameters: change as required
    BOARD = "CBSE"
    CLASS_LABEL = "Class 11th"
    SUBJECT = "Biology"
    start = time.time()


    run_pipeline(BOARD, CLASS_LABEL, SUBJECT)
    total_elapsed = time.time() - start
    print("Total elapsed time:",total_elapsed)

