"""
batched_prompt_builder.py

Utilities to build a single batched RAG prompt for a generator LLM (e.g., Gemini)
and to robustly parse the model's JSON response.
"""

from typing import List, Dict, Any, Callable, Optional
import json
import re
import textwrap
import time

# ----------------------
# Prompt builder (No changes from previous version)
# ----------------------


def build_generator_prompt_questions_only(
    planner_text: str,
    slot_summaries: List[Dict[str, Any]],
    plan: Dict[str, Any],
    gen_settings: Optional[Dict[str, Any]] = None,
) -> str:
    if gen_settings is None:
        gen_settings = {}

    plan_str = json.dumps(plan)
    has_ar = "Assertion-Reason" in plan_str
    has_case_study = "Case-Study" in plan_str
    has_internal_choice = any(
        sec.get("internal_choices", 0) > 0 for sec in plan.get("sections", [])
    )

    s = []
    s.append(
        "You are an expert exam paper generator. IMPORTANT: RETURN ONLY A SINGLE VALID JSON OBJECT — DO NOT INCLUDE ANSWERS, RATIONALE, OR ANY TEXT OUTSIDE THE JSON."
    )
    s.append(
        "Follow the planner summary and the specific question formats and counts for each section."
    )

    s.append("\nPLANNER SUMMARY:")
    s.append(planner_text.strip()[:1500])

    s.append("\nEVIDENCE (per slot):")
    for slot in slot_summaries:
        sid = slot.get("slot_id", "UNKNOWN")
        s.append(f"Slot {sid}: {slot.get('slot_meta','')}")
        for summ in slot.get("summaries", []):
            txt = summ.get("summary", "")[:600]
            s.append(f"- [{summ.get('id','')}] {txt}")

    s.append("\nINSTRUCTIONS:")
    s.append(
        "1) Produce the required number and types of questions exactly as specified in the planner."
    )
    s.append(
        "2) CRITICAL: The `questions` key must be a JSON list. Each question in the list MUST be a separate, complete, and valid JSON object `{...}`. Ensure each object is properly separated by a comma."
    )
    s.append(
        "3) CRITICAL: For questions with an internal choice (`is_choice: true`), the `q_text` key MUST be an array of objects. DO NOT add a separate string-based `q_text` key in the same question object, as this creates invalid JSON."
    )
    s.append(
        "4) Ensure all backslashes inside JSON strings are properly escaped (e.g., use `\\\\` for a literal backslash in LaTeX)."
    )
    s.append("5) Return ONLY the JSON object and nothing else.") # Renumbered this

    s.append(
        "\nEXAMPLE QUESTION FORMATS (use these structures when required by the plan):"
    )

    if has_internal_choice:
        s.append(
            textwrap.dedent(
                r"""
    - Question with Internal Choice (Correct Structure):
      {"section_id":"E","q_id":"E.1","type":"LA","marks":5,"difficulty":"Hard","is_choice":true,"q_text":[{"q_text":"Explain X.","sources":[...]},{"q_text":"[OR] Explain Y.","sources":[...]}]}
    """
            )
        )

    s.append("\nREQUIRED OUTPUT JSON SCHEMA:")
    s.append(
        """{"paper_id":"<string>","board":"CBSE","class":"<string>","subject":"<string>","questions":[<list of question items following the example formats>]}"""
    )

    prompt = "\n\n".join(s)
    if len(prompt) > 25000:
        prompt = prompt[:25000]
    return prompt


# ----------------------
# Robust JSON extraction / parsing (NEWEST, MOST ROBUST VERSION)
# ----------------------

def _find_json_substring(text: str) -> str:
    """Finds the most likely JSON substring using a robust "greedy grab" method."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    start_brace = text.find("{")
    if start_brace == -1:
        raise ValueError("Could not find an opening brace '{' to start JSON parsing.")
    end_brace = text.rfind("}")
    if end_brace == -1:
        raise ValueError("Could not find a closing brace '}' to end JSON parsing.")
    return text[start_brace : end_brace + 1]


def _repair_json_with_llm(broken_text: str, llm_caller: Callable) -> str:
    """
    Uses an LLM as a fallback to repair a severely malformed JSON string.
    
    Args:
      broken_text: The malformed string that needs fixing.
      llm_caller: The function to be used for making the LLM call (e.g., call_gemini).
    """
    print("--- PARSER FALLBACK: Attempting to repair JSON with an LLM call. ---")
    prompt = f"""
    The following text is a broken JSON object from an AI. Your task is to analyze its structure, identify the separate question objects, and reconstruct it into a single, valid JSON object.
    - Fix syntax errors, missing commas, duplicate keys, and incorrect nesting.
    - Ensure the `questions` key contains a valid list of well-formed question objects `{{...}}`.
    - Do NOT change any of the text content, IDs, or values. Only fix the JSON structure.
    - CRITICAL: Your final output must ONLY be the repaired JSON object. Do not add explanations or markdown.

    BROKEN JSON TEXT:
    ```json
    {broken_text}
    ```

    Repaired and valid JSON object:
    """
    try:
        # Use the provided llm_caller function
        response = llm_caller(prompt, model_name="models/gemini-2.5-flash-lite", temperature=0.0)
        repaired_text = response.get("text", "")
        return _find_json_substring(repaired_text)
    except Exception as e:
        print(f"LLM Repair call failed: {e}")
        raise ValueError(f"The LLM-based JSON repair failed. Original text: {broken_text[:500]}")


def parse_generator_response(llm_text: str, llm_caller: Callable) -> Any:
    """
    Parse LLM text output and return the JSON object. Now requires an 'llm_caller'
    to be passed for the LLM-powered repair mechanism.
    
    Args:
      llm_text: The raw text output from the primary generator LLM.
      llm_caller: The function to be used for the fallback repair call.
    """
    if not llm_text or not llm_text.strip():
        raise ValueError("LLM output is empty or contains only whitespace.")

    try:
        candidate = _find_json_substring(llm_text)
    except ValueError as e:
        raise ValueError(f"Failed to locate any potential JSON substring. Error: {e}")

    # Attempts to parse using standard methods first
    try:
        return json.loads(candidate, strict=False)
    except json.JSONDecodeError as e:
        final_regular_error = e

    # If standard parsing fails, fall back to the LLM repair function
    try:
        repaired_by_llm = _repair_json_with_llm(candidate, llm_caller)
        # Attempt one final, strict parse on the LLM's repaired output
        return json.loads(repaired_by_llm)
    except (json.JSONDecodeError, ValueError) as llm_repair_failure:
        raise ValueError(
            "Failed to parse JSON after all repair attempts, including an LLM-based fix.\n"
            f"Original structural error: {final_regular_error}\n"
            f"Final error after LLM repair: {llm_repair_failure}\n"
            f"--- Snippet of final text attempted ---\n{candidate[:1000]}..."
        )
# ----------------------
# Safe generate wrapper (No changes needed)
# ----------------------
def safe_generate(
    callable_llm: Callable[[str], Dict[str, Any]],
    prompt: str,
    max_retries: int = 2,
    retry_on_parse_error: bool = True,
    backoff_seconds: float = 0.5,
) -> Any:
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            response = callable_llm(prompt)
            text = getattr(response, "text", "") or (
                response.get("text") if isinstance(response, dict) else ""
            )
            return parse_generator_response(text)
        except Exception as e:
            last_exc = e
            if attempt < max_retries and retry_on_parse_error:
                time.sleep(backoff_seconds * attempt)
                prompt += "\n\nNote: Your last response had a JSON formatting error. Please ensure you return ONLY a single, valid JSON object with no duplicate keys and correctly escaped backslashes."
                continue
            else:
                break
    raise last_exc


# ----------------------
# quick self-test
# ----------------------
if __name__ == "__main__":
    # Test case with the invalid backslash error
    latex_error_json = r"""
    {
      "q_text": "The value of $\sin(\pi/2)$ is 1." 
    }
    """
    print("--- Testing parser on JSON with invalid backslash (LaTeX) ---")
    try:
        parsed_data = parse_generator_response(latex_error_json)
        print("Successfully parsed JSON with invalid escape using `strict=False`!")
        assert parsed_data["q_text"] == "The value of $\sin(\\pi/2)$ is 1."
        print("Assertion passed: Content is preserved correctly.")
    except ValueError as e:
        print("TEST FAILED: Could not parse the LaTeX JSON.")
        print(e)

    # Test case with both duplicate key AND invalid backslash
    combined_error_json = r"""
    ```json
    {
      "questions": [
        {
          "q_id": "C.4",
          "q_text": "This is a bad key with a bad backslash: \pi",
          "is_choice": true,
          "q_text": [
            { "q_text": "This is the first choice." }
          ]
        }
      ]
    }
    ```
    """
    print("\n--- Testing parser on combined duplicate key and backslash error ---")
    try:
        parsed_data = parse_generator_response(combined_error_json)
        print("Successfully parsed and repaired the combined error JSON!")
        assert isinstance(parsed_data["questions"][0]["q_text"], list)
        print("Assertion passed: The repaired 'q_text' is correctly a list.")
    except ValueError as e:
        print("TEST FAILED: Could not repair the combined error JSON.")
        print(e)
