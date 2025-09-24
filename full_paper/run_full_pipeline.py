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
from full_paper.batched_prompt_builder import  parse_generator_response,build_generator_prompt_questions_only
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



# In run_full_pipeline.py

import re
from typing import Dict, Any

def derive_plan_from_filedata(file_data: str, subject: str) -> Dict[str, Any]:
    """
    Robust planner that selects a known-good, hardcoded plan (blueprint)
    based on the subject to ensure maximum structural accuracy. This avoids
    brittle regex parsing of the guideline text.
    """
    
    # --- KNOWN-GOOD BLUEPRINT FOR CLASS 12 PHYSICS (CBSE) ---
    correct_physics_12_plan = {
        "total_marks": 70, "time_minutes": 180,
        "general_instructions": [
            "There are 33 questions in all. All questions are compulsory.",
            "This question paper has five sections: Section A, Section B, Section C, Section D and Section E.",
            "Section A contains sixteen questions, twelve MCQ and four Assertion-Reasoning based of 1 mark each.",
            "Section B contains five questions of two marks each.",
            "Section C contains seven questions of three marks each.",
            "Section D contains two case study-based questions of four marks each.",
            "Section E contains three long answer questions of five marks each.",
            "Internal choices are provided in some questions.",
            "Use of calculators is not allowed."
        ],
        "sections": [
            {"section_id":"A", "title": "Objective Type Questions", "num_questions":16, "internal_choices": 0, "question_breakdown": [{"type":"MCQ", "count": 12, "marks": 1}, {"type":"Assertion-Reason", "count": 4, "marks": 1}]},
            {"section_id":"B", "title": "Very Short Answer Questions", "num_questions":5, "internal_choices": 2, "question_breakdown": [{"type":"VSA", "count": 5, "marks": 2}]},
            {"section_id":"C", "title": "Short Answer Questions", "num_questions":7, "internal_choices": 2, "question_breakdown": [{"type":"SA", "count": 7, "marks": 3}]},
            {"section_id":"D", "title": "Case-Study Based Questions", "num_questions":2, "internal_choices": 2, "question_breakdown": [{"type":"Case-Study", "count": 2, "marks": 4}]},
            {"section_id":"E", "title": "Long Answer Questions", "num_questions":3, "internal_choices": 3, "question_breakdown": [{"type":"LA", "count": 3, "marks": 5}]}
        ]
    }
    
    # --- KNOWN-GOOD BLUEPRINT FOR CLASS 12 MATHS (CBSE) ---
    correct_maths_12_plan = {
        "total_marks": 80, "time_minutes": 180,
        "general_instructions": [
            "This Question paper contains 38 questions. All questions are compulsory.",
            "This Question paper is divided into five Sections - A, B, C, D and E.",
            "Section A comprises of 20 questions of 1 mark each (18 MCQs and 2 Assertion-Reason).",
            "Section B comprises of 5 questions of 2 marks each.",
            "Section C comprises of 6 questions of 3 marks each.",
            "Section D comprises of 4 questions of 5 marks each.",
            "Section E comprises of 3 case study-based questions of 4 marks each.",
            "Internal choices are provided in specific questions.",
            "Use of calculator is not allowed."
        ],
        "sections": [
            {"section_id": "A", "title": "Objective Type Questions", "num_questions": 20, "internal_choices": 0, "question_breakdown": [{"type": "MCQ", "count": 18, "marks": 1}, {"type": "Assertion-Reason", "count": 2, "marks": 1}]},
            {"section_id": "B", "title": "Very Short Answer (VSA)", "num_questions": 5, "internal_choices": 2, "question_breakdown": [{"type": "VSA", "count": 5, "marks": 2}]},
            {"section_id": "C", "title": "Short Answer (SA)", "num_questions": 6, "internal_choices": 3, "question_breakdown": [{"type": "SA", "count": 6, "marks": 3}]},
            {"section_id": "D", "title": "Long Answer (LA)", "num_questions": 4, "internal_choices": 2, "question_breakdown": [{"type": "LA", "count": 4, "marks": 5}]},
            {"section_id": "E", "title": "Case Study-Based Questions", "num_questions": 3, "internal_choices": 3, "question_breakdown": [{"type": "Case-Study", "count": 3, "marks": 4}]}
        ]
    }

    # You can add more blueprints for other subjects and classes here...
    
    plan = None
    # --- Select the correct plan based on the subject ---
    subject_lower = subject.lower()
    if "math" in subject_lower:
        plan = correct_maths_12_plan
        print("INFO: Using hardcoded blueprint for Maths for maximum accuracy.")
    elif "physics" in subject_lower:
        plan = correct_physics_12_plan
        print("INFO: Using hardcoded blueprint for Physics for maximum accuracy.")
    
    # If no specific plan is found, fall back to a generic one or the first available one
    if not plan:
        # Fallback to a default plan if subject is not recognized
        plan = correct_physics_12_plan # Example fallback
        print(f"WARNING: No specific blueprint for subject '{subject}'. Falling back to default plan.")

    # Create the human-readable planner_text summary from the chosen plan
    planner_text = f"Generate a {plan['total_marks']} marks paper for {subject}, to be completed in {plan['time_minutes']} minutes. The paper has {len(plan['sections'])} sections. "
    for sec in plan['sections']:
        breakdown_str = ", ".join([f"{item['count']} {item['type']} questions worth {item['marks']} marks each" for item in sec['question_breakdown']])
        planner_text += f"Section {sec['section_id']} ({sec['title']}) has {sec['num_questions']} questions: {breakdown_str}. "
        if sec.get('internal_choices', 0) > 0:
            planner_text += f"Provide {sec['internal_choices']} internal choices in this section. "
            
    plan['planner_text'] = planner_text.strip()
    
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



# In run_full_pipeline.py

def retrieve_from_pinecone(objective_text: str, filters: Dict[str, Any], top_k: int = 50) -> List[Dict[str, Any]]:
    """
    Embed objective and query Pinecone.
    MODIFIED: This is the definitive, robust version for extracting metadata and embeddings (values).
    """
    if DRY_RUN:
        # Dry run logic remains the same
        return [...] 
    if not PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY not set in environment.")

    # 1. Embed the query
    q_emb = embed_texts_bge([objective_text], batch_size=1)[0].tolist()

    # 2. Connect and query Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    idx = pc.Index(PINECONE_INDEX_NAME)
    print(f"Querying Pinecone with BGE vector for top {top_k}...")
    
    resp = idx.query(
        vector=q_emb, 
        top_k=top_k, 
        include_values=True,  # Crucial to get the embedding vector
        include_metadata=True, 
        filter=filters
    )
    
    matches = resp.get('matches', [])
    results = []

    # 3. --- THIS IS THE FIX ---
    #    Loop through the matches and build the result dictionary correctly.
    for i, match in enumerate(matches):
        metadata = match.get('metadata', {})
        # The text snippet is INSIDE the metadata dictionary
        text = metadata.get('text', '') 
        
        # The embedding vector is in the 'values' key at the top level of the match
        embedding_values = match.get('values')
        
        results.append({
            "snippet_id": match.get('id', f'match_{i}'), 
            "text": text, 
            "metadata": metadata, 
            "score": match.get('score', 0.0), 
            # This will now be correctly populated with the vector or None
            "embedding": np.array(embedding_values, dtype=float) if embedding_values else None
        })
    # --- END OF FIX ---

    print(f"Pinecone returned {len(results)} matches. {len([r for r in results if r['embedding'] is not None])} have embeddings.")
    return results









# -------------------------
# Gemini caller (simple)
# -------------------------
def call_gemini(prompt: str, model_name: str = "models/gemini-2.5-flash-lite", temperature: float = 0.0, max_retries: int = 2) -> Dict[str, Any]:
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

        # gemini_resp = call_gemini(prompt, model_name="models/gemini-2.5-flash-lite", temperature=0.0)
        # parsed = parse_generator_response(gemini_resp.get('text',''))

        # save prompt
        with open("last_generated_prompt.txt", "w", encoding="utf-8") as f:
            f.write(prompt)
        log("Prompt built and saved to last_generated_prompt.txt (open to inspect).")

        # call Gemini
        gen_resp = call_gemini(prompt, model_name="models/gemini-2.5-flash-lite")
        log("Generator returned; parsing JSON...")
        parsed = parse_generator_response(gen_resp.get('text', ''))



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

