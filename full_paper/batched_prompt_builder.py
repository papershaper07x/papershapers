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
        "2) CRITICAL: For questions with an internal choice (`is_choice: true`), the `q_text` key MUST be an array of objects. DO NOT add a separate string-based `q_text` key in the same question object, as this creates invalid JSON."
    )
    s.append(
        "3) Ensure all backslashes inside JSON strings are properly escaped (e.g., use `\\\\` for a literal backslash in LaTeX)."
    )
    s.append("4) Return ONLY the JSON object and nothing else.")

    s.append(
        "\nEXAMPLE QUESTION FORMATS (use these structures when required by the plan):"
    )

    if has_internal_choice:
        s.append(
            textwrap.dedent(
                """
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
    """
    Finds the most likely JSON substring using a robust "greedy grab" method.
    """
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    start_brace = text.find("{")
    start_bracket = text.find("[")

    if start_brace == -1 and start_bracket == -1:
        # If no JSON markers, maybe the whole string is the object (less likely but possible)
        return text

    start_pos = -1
    if start_brace != -1 and start_bracket != -1:
        start_pos = min(start_brace, start_bracket)
    elif start_brace != -1:
        start_pos = start_brace
    else:
        start_pos = start_bracket

    end_brace = text.rfind("}")
    end_bracket = text.rfind("]")

    if end_brace == -1 and end_bracket == -1:
        raise ValueError("Found an opening brace/bracket but no closing one.")

    end_pos = max(end_brace, end_bracket)

    if end_pos < start_pos:
        raise ValueError(
            "Mismatched brackets: last closing bracket/brace appears before the first opening one."
        )

    return text[start_pos : end_pos + 1]


def parse_generator_response(llm_text: str) -> Any:
    """
    Parse LLM text output and return the JSON object. Applies a multi-layered
    defense of non-strict parsing and targeted repairs for common errors.
    """
    if not llm_text or not llm_text.strip():
        raise ValueError("Empty LLM output")

    try:
        candidate = _find_json_substring(llm_text)
    except ValueError as e:
        raise ValueError(
            f"Failed to locate any potential JSON substring. Original error: {e}"
        )

    try:
        # Attempt 1: Strict parse (for perfectly formed JSON)
        return json.loads(candidate)
    except json.JSONDecodeError:
        try:
            # Attempt 2: Non-strict parse. This allows invalid control characters
            # (like unescaped backslashes in LaTeX) and is the primary fix for the
            # "Invalid \escape" error.
            return json.loads(candidate, strict=False)
        except json.JSONDecodeError as e:
            # Attempt 3: If even non-strict parsing fails, it's likely a structural
            # error. Apply our regex repairs and try one last time.
            original_error = e
            repaired_text = candidate

            # REPAIR 1: Fix duplicate 'q_text' key in choice questions
            pattern = re.compile(
                r'("q_text":\s*".*?",\s*)(?=[^\{]*?"is_choice":\s*true)', re.DOTALL
            )
            repaired_text = pattern.sub("", repaired_text)

            # REPAIR 2: Fix trailing commas
            repaired_text = re.sub(r",\s*([\]\}])", r"\1", repaired_text)

            # REPAIR 3: Fix double commas
            repaired_text = re.sub(r",\s*,", ",", repaired_text)

            try:
                # Use strict=False again on the repaired text as a final safeguard
                return json.loads(repaired_text, strict=False)
            except json.JSONDecodeError as final_error:
                raise ValueError(
                    f"Failed to parse JSON after all repair attempts.\n"
                    f"Original structural error: {original_error}\n"
                    f"Final parsing error after repairs: {final_error}\n"
                    f"--- Snippet of final text attempted ---\n{repaired_text[:1000]}..."
                )


# ----------------------
# Safe generate wrapper
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
    latex_error_json = """
    {
      "q_text": "The value of $\sin(\pi/2)$ is 1." 
    }
    """
    print("--- Testing parser on JSON with invalid backslash (LaTeX) ---")
    try:
        parsed_data = parse_generator_response(latex_error_json)
        print("Successfully parsed JSON with invalid escape using `strict=False`!")
        assert parsed_data["q_text"] == "The value of $\sin(\pi/2)$ is 1."
        print("Assertion passed: Content is preserved correctly.")
    except ValueError as e:
        print("TEST FAILED: Could not parse the LaTeX JSON.")
        print(e)

    # Test case with both duplicate key AND invalid backslash
    combined_error_json = """
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
