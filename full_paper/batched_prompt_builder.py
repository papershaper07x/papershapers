"""
batched_prompt_builder.py

Utilities to build a single batched RAG prompt for a generator LLM (e.g., Gemini)
and to robustly parse the model's JSON response.

Key functions:
 - build_generator_prompt(planner_text, slot_summaries, gen_settings) -> str
 - parse_generator_response(llm_text) -> parsed_json (raises ValueError on failure)
 - safe_generate(callable_llm, prompt, max_retries=2, retry_on_parse_error=True) -> parsed_json

Design goals:
 - Force strict JSON output from the model (instructions + few-shot example).
 - Keep the prompt compact by inserting short summaries (not full snippets).
 - Robustly extract JSON from noisy LLM outputs (code fences, leading text, trailing text).
 - Provide minimal retry/backoff behavior for parsing failures.

This module does NOT call any external APIs directly; `safe_generate` expects a
`callable_llm(prompt, **kwargs)` function which should wrap your actual LLM SDK.

"""
from typing import List, Dict, Any, Callable, Optional, Tuple
import json
import re
import textwrap
import time


# ----------------------
# Prompt builder
# ----------------------

def _shorten_text(text: str, max_chars: int) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    # attempt to cut at sentence boundary near limit
    cut = text[:max_chars]
    last_dot = max(cut.rfind('.'), cut.rfind('!'), cut.rfind('?'))
    if last_dot > int(max_chars * 0.6):
        return cut[:last_dot+1] + '...'
    return cut.rstrip() + '...'
def build_generator_prompt_questions_only(
    planner_text: str,
    slot_summaries: List[Dict[str, Any]],
    gen_settings: Optional[Dict[str, Any]] = None
) -> str:
    """
    Build a prompt that instructs the LLM to RETURN ONLY QUESTIONS (no answers).
    Each question item must include: section_id, q_id, q_text, type, marks, difficulty, sources (optional).
    The model must return STRICT JSON only.
    """
    if gen_settings is None:
        gen_settings = {}
    mode = gen_settings.get('mode', 'balanced')
    s = []
    s.append("You are an expert exam paper generator. IMPORTANT: RETURN ONLY A JSON OBJECT — DO NOT INCLUDE ANSWERS, RATIONALE, OR EXPLANATIONS.")
    s.append("Every entry in the questions list must contain only these keys: section_id, q_id, q_text, type, marks, difficulty, sources (optional).")
    if mode == 'strict':
        s.append("Mode: strict — keep questions factual and closely tied to provided evidence.")
    elif mode == 'creative':
        s.append("Mode: creative — you may paraphrase for variety, but DO NOT provide answers.")
    else:
        s.append("Mode: balanced — produce exam-style questions only.")
    s.append("\nPLANNER SUMMARY:")
    s.append(planner_text.strip()[:1200])
    s.append("\nEVIDENCE (per slot):")
    for slot in slot_summaries:
        sid = slot.get('slot_id', 'UNKNOWN')
        s.append(f"Slot {sid}: {slot.get('slot_meta','')}")
        for summ in slot.get('summaries', []):
            txt = summ.get('summary','')[:600]
            s.append(f"- [{summ.get('id','')}] {txt}")

    s.append("\nINSTRUCTIONS:")
    s.append("1) For each slot, produce the required number of questions matching the planner.  \n2) DO NOT emit answers, solution steps, or rationale.  \n3) Return ONLY the JSON object and nothing else.  \n4) Use keys exactly as requested (section_id, q_id, q_text, type, marks, difficulty, sources).")
    s.append("\nEXAMPLE QUESTION ITEM (remember: NO ANSWER):")
    s.append('''{"section_id":"A","q_id":"A.1","q_text":"Define photosynthesis.","type":"SA","marks":2,"difficulty":"Easy","sources":["s1"]}''')
    s.append("\nREQUIRED OUTPUT JSON SCHEMA:")
    s.append('''{"paper_id":"<string>","board":"CBSE","class":"Class 10th","subject":"Science","questions":[<list of question items as above>]}''')

    prompt = "\n\n".join(s)
    if len(prompt) > 15000:
        prompt = prompt[:15000]
    return prompt


# def build_generator_prompt(
#     planner_text: str,
#     slot_summaries: List[Dict[str, Any]],
#     gen_settings: Optional[Dict[str, Any]] = None
# ) -> str:
#     """Build a single batched prompt for the generator.

#     Args:
#       planner_text: a short human-readable description of the planner output
#                     (sections, target mark distribution, difficulty goals). Keep it
#                     concise (a few lines).
#       slot_summaries: list of slot dicts, each containing at least:
#          - slot_id (e.g., "A" or "B.q2")
#          - slot_meta (optional human label)
#          - summaries: list of summaries chosen for that slot; each summary is a dict
#                       with keys: id, summary, metadata (optional)
#       gen_settings: optional dict controlling generation:
#          - total_marks, time_minutes, language, mode ('strict'|'balanced'|'creative'),
#            max_questions (optional), answer_style ('short'|'detailed')

#     Returns: a string prompt ready to send to the LLM.
#     """
#     s = []
#     s.append("You are an expert exam-paper generator. Follow the instructions exactly.")
#     s.append("Do NOT use any knowledge beyond the provided evidence. Use the provided snippets/summaries as the only supporting material.")
#     s.append("Output requirement: Return a single JSON object only. No explanatory text. The JSON must contain the paper metadata and a list of questions. See the required schema example at the end.")

#     # generation settings hints
#     if not gen_settings:
#         gen_settings = {}
#     mode = gen_settings.get('mode', 'balanced')
#     if mode == 'strict':
#         s.append("Mode: strict — prefer direct factual questions closely tied to sources. Low creativity.")
#     elif mode == 'creative':
#         s.append("Mode: creative — you may paraphrase and include application-style questions; still ground answers in provided evidence.")
#     else:
#         s.append("Mode: balanced — produce exam-style questions grounded in evidence.")

#     # include planner summary
#     s.append('\nPLANNER SUMMARY:')
#     s.append(_shorten_text(planner_text.strip(), 1000))

#     # include per-slot compacted evidence
#     s.append('\nEVIDENCE (per slot):')
#     for slot in slot_summaries:
#         slot_id = slot.get('slot_id') or slot.get('section_id') or 'UNKNOWN'
#         label = slot.get('slot_meta', '')
#         header = f"Slot {slot_id}: {label}" if label else f"Slot {slot_id}:"
#         s.append(header)
#         summaries = slot.get('summaries', [])
#         if not summaries:
#             s.append("  - No relevant summaries provided — mark as needs_review if you cannot answer grounded.")
#             continue
#         for summ in summaries:
#             sid = summ.get('id') or summ.get('snippet_id') or ''
#             txt = _shorten_text(summ.get('summary', ''), 600)
#             meta = summ.get('metadata') or {}
#             meta_str = ', '.join(f"{k}={v}" for k, v in meta.items() if k in ('chapter','source_id'))
#             s.append(f"  - [{sid}] {txt}" + (f" (meta: {meta_str})" if meta_str else ""))

#     # explicit generation instructions and schema
#     s.append('\nINSTRUCTIONS:')
#     s.append("1) Following the planner, produce questions for each slot. Respect marks, counts, and question types.")
#     s.append("2) For each question include: section_id, q_id, q_text, type, marks, difficulty, answer, sources (list of evidence ids) and a short rationale line linking answer to sources.")
#     s.append("3) If evidence is insufficient for a factual answer, do NOT hallucinate. Instead set needs_review=true for that question and provide a short reason.")
#     s.append("4) Keep answers concise (one or two sentences) unless answer_style in settings is 'detailed'.")
#     s.append("5) Use the provided evidence only. You may paraphrase question wording but ensure the answer is grounded in cited evidence.")

#     # small few-shot example (very short) — show one example question JSON entry only
#     example = textwrap.dedent(
#         """
#         EXAMPLE_QUESTION_ITEM:
#         {
#           "section_id": "A",
#           "q_id": "A.1",
#           "q_text": "What is photosynthesis?",
#           "type": "SA",
#           "marks": 2,
#           "difficulty": "Easy",
#           "answer": "Photosynthesis is the process by which green plants convert light energy into chemical energy (glucose), occurring in chloroplasts.",
#           "sources": ["s1"],
#           "rationale": "Supported by source s1 which defines photosynthesis and mentions chloroplasts",
#           "needs_review": false
#         }
#         """
#     )
#     s.append(example)

#     # final output wrapper schema
#     s.append('\nREQUIRED OUTPUT JSON SCHEMA:')
#     s.append(textwrap.dedent(
#         """
#         {
#           "paper_id": "<string>",
#           "board": "CBSE",
#           "class": "Class 10th",
#           "subject": "Science",
#           "total_marks": 80,
#           "time_minutes": 180,
#           "questions": [ <list of question items like EXAMPLE_QUESTION_ITEM> ]
#         }
#         """
#     ))

#     prompt = "\n\n".join(s)
#     # final safety: trim overall prompt to a reasonable length (e.g., 15000 chars)
#     if len(prompt) > 15000:
#         prompt = prompt[:15000]
#     return prompt


# ----------------------
# Robust JSON extraction / parsing
# ----------------------

def _find_json_substring(text: str) -> str:
    """Attempt to extract the largest plausible JSON substring from `text`.

    Strategy:
      - First, try to parse the whole text as JSON.
      - Then search for fenced code blocks with ```json or ```.
      - Otherwise search for the first '[' or '{' and then find the matching closing bracket using a stack.

    Returns the JSON substring (string) or raises ValueError if none found.
    """
    text = text.strip()
    # 1) Direct parse
    try:
        json.loads(text)
        return text
    except Exception:
        pass

    # 2) fenced code blocks (```json ... ``` or ``` ... ```)
    fenced = re.search(r"```(?:json)?\s*(.*)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1).strip()
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            # continue to other heuristics
            pass

    # 3) find first JSON-like char
    first_brace = min([pos for pos in [text.find('['), text.find('{')] if pos != -1] + [len(text)])
    if first_brace >= len(text):
        raise ValueError('No JSON object or array found in LLM output')

    start = first_brace
    stack = []
    pairs = {'{': '}', '[': ']'}
    i = start
    while i < len(text):
        ch = text[i]
        if ch in pairs.keys():
            stack.append(pairs[ch])
        elif stack and ch == stack[-1]:
            stack.pop()
            if not stack:
                # include up to i (inclusive)
                candidate = text[start:i+1]
                # try parse
                try:
                    json.loads(candidate)
                    return candidate
                except Exception:
                    # continue searching for next plausible end if any
                    pass
        i += 1

    raise ValueError('Failed to locate a parsable JSON substring in LLM output')


def parse_generator_response(llm_text: str) -> Any:
    """Parse LLM text output and return the JSON object.

    Raises ValueError if parsing fails.
    """
    if not llm_text or not llm_text.strip():
        raise ValueError('Empty LLM output')

    # try incremental strategies
    # 1) direct parse
    try:
        return json.loads(llm_text)
    except Exception:
        pass

    # 2) try to find substring
    candidate = _find_json_substring(llm_text)
    try:
        return json.loads(candidate)
    except Exception as e:
        # last-ditch: try to clean up trailing commas
        cleaned = re.sub(r",\s*,+", ",", candidate)
        cleaned = re.sub(r",\s*\]", "]", cleaned)
        cleaned = re.sub(r",\s*\}", "}", cleaned)
        try:
            return json.loads(cleaned)
        except Exception:
            raise ValueError(f'Failed to parse JSON from LLM output. Original error: {e}')


# ----------------------
# Safe generate wrapper (expects an LLM callable)
# ----------------------

def safe_generate(
    callable_llm: Callable[[str], Dict[str, Any]],
    prompt: str,
    max_retries: int = 2,
    retry_on_parse_error: bool = True,
    backoff_seconds: float = 0.5
) -> Any:
    """Call the LLM via `callable_llm(prompt)` and parse the JSON response robustly.

    callable_llm must be a function that accepts the prompt and returns an object
    with either a `.text` attribute (string) or a dict with 'text' key.

    Returns parsed JSON or raises the last encountered exception.
    """
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            response = callable_llm(prompt)
            # extract text
            if isinstance(response, dict):
                text = response.get('text') or response.get('output') or ''
            else:
                # assume object with .text
                text = getattr(response, 'text', '')

            parsed = parse_generator_response(text)
            return parsed
        except Exception as e:
            last_exc = e
            # If parsing error and we allow retry, try once with relaxed settings
            if attempt < max_retries and retry_on_parse_error:
                time.sleep(backoff_seconds * attempt)
                # maybe try again with a small prompt tweak instruction appended
                prompt = prompt + "\n\nNote: If you returned non-JSON text, return ONLY the JSON object in the next response."
                continue
            else:
                break

    # if we reach here, re-raise last exception
    raise last_exc


# ----------------------
# quick self-test (not full integration)
# ----------------------
if __name__ == '__main__':
    # small demo
    planner = 'Sections: A - 20 one-mark MCQs; B - 6 short answers; emphasis on photosynthesis and transpiration'
    slot_summaries = [
        {'slot_id': 'A', 'slot_meta': 'Objective MCQs', 'summaries': [
            {'id': 's1', 'summary': 'Photosynthesis is the process by which green plants convert light energy into chemical energy.'},
            {'id': 's3', 'summary': 'Transpiration is the loss of water vapour from plant aerial parts, mainly leaves.'}
        ]},
        {'slot_id': 'B', 'slot_meta': 'Short Answers', 'summaries': [
            {'id': 's2', 'summary': 'Stomata regulate gas exchange and are the main route for transpiration.'}
        ]}
    ]
    # prompt = build_generator_prompt(planner, slot_summaries, gen_settings={'mode':'balanced'})
    print('\n--- Generated Prompt ---\n')
    # print(prompt[:2000])
    # demo parse
    fake_llm_out = 'Here is the JSON:\n```json\n{"paper_id":"p1","questions":[{"section_id":"A","q_id":"A.1","q_text":"What is photosynthesis?","type":"SA","marks":1,"difficulty":"Easy","answer":"Process by which plants make food.","sources":["s1"],"rationale":"Defined in s1","needs_review":false}]}\n```'
    parsed = parse_generator_response(fake_llm_out)
    print('\n--- Parsed JSON ---\n', parsed)
