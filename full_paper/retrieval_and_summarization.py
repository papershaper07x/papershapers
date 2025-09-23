"""
retrieval_and_summarization.py

Contains two small utilities used in the retrieval -> generation pipeline:

1) build_retrieval_objective(slot: dict, subject_guidelines: str, user_mode: str) -> str
   - Turns a planner "slot" (section info) and optional guideline text into a
     concise 1-2 line natural-language retrieval objective. This is the string
     you should embed and send to Pinecone as the semantic query for that slot.

2) summarize_and_budget_snippets(snippets: List[dict], objective: str, max_tokens: int,
   max_sentences_per_snippet: int = 2) -> List[dict]
   - Produces short extractive summaries (1-2 sentences) for each snippet,
     ranks them by relevance to the objective, and returns a token-budgeted
     subset and compact summaries suitable for inclusion in the generator prompt.

Note: This module intentionally uses lightweight heuristics (no heavy NLP
libraries) so it runs in simple environments. It is intended as a stop-gap
and can be swapped for model-based summarization later.
"""
from typing import Any, Callable, List, Dict, Tuple, Optional
import re
import math

# ----------------------
# Simple sentence splitter
# ----------------------

def sentence_split(text: str) -> List[str]:
    """Split text into sentences using punctuation heuristics (no external deps).

    Keeps short paragraphs together when punctuation is absent.
    """
    if not text:
        return []
    # Normalize newlines
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    sentences = []
    for p in paragraphs:
        # split on sentence delimiters followed by space and capital letter / digit
        parts = re.split(r'(?<=[\.!?])\s+', p)
        parts = [s.strip() for s in parts if s.strip()]
        # If paragraph has no punctuation, keep as single "sentence"
        if len(parts) == 0:
            continue
        if len(parts) == 1:
            sentences.append(parts[0])
        else:
            sentences.extend(parts)
    return sentences

# ----------------------
# Minimal token estimator
# ----------------------

# In retrieval_and_summarization.py

# ... (keep all existing functions like sentence_split, build_retrieval_objective, etc.)

# --- Add this new function ---
def generate_dense_evidence_summary_with_llm(
    snippets: List[Dict],
    objective: str,
    # This function needs access to your Gemini caller
    llm_caller: Callable[[str], Dict[str, Any]]
) -> str:
    """
    Uses a powerful LLM to read multiple snippets and synthesize a single,
    dense, fact-rich paragraph suitable for generating exam questions.

    Args:
      snippets: A list of retrieved snippet dictionaries, each with a 'text' key.
      objective: The goal for the section (e.g., "Generate Hard LA questions...").
      llm_caller: A function (like your `call_gemini`) that takes a prompt and returns a response dict.

    Returns:
      A single string containing the consolidated evidence.
    """
    if not snippets:
        return "No relevant evidence was found for this section."

    # Combine the text from all snippets
    full_text = "\n\n---\n\n".join([s.get('text', '') for s in snippets])

    # Create a specific prompt for the summarizer LLM
    prompt = f"""
    You are a subject matter expert tasked with preparing evidence for an exam paper generator.
    Your goal is to consolidate the following text snippets into a single, dense, fact-rich paragraph.
    Do not lose critical information, formulas, specific numbers, or key concepts.
    The final output should be a clean paragraph of text, not a list or a summary.
    This consolidated text will be used to create exam questions for the following objective: "{objective}"

    Provided Snippets:
    ---
    {full_text}
    ---

    Consolidated Evidence Paragraph:
    """

    try:
        # Use your existing LLM calling function
        response = llm_caller(prompt)
        consolidated_text = response.get("text", "")
        # A simple cleanup
        return consolidated_text.strip().replace("Consolidated Evidence Paragraph:", "").strip()
    except Exception as e:
        print(f"ERROR: LLM-based summarization failed: {e}")
        # Fallback to a simpler concatenation if the LLM call fails
        return " ".join([s.get('text', '') for s in snippets])[:2000] # Truncate to be safe




def estimate_tokens_from_text(text: str) -> int:
    """Rough token estimate: tokens ≈ ceil(words * 1.3).

    This is a conservative overestimate (models like GPT tokenization gives
    ~0.75-1.0 tokens per word depending on language). We multiply by 1.3 to be
    safe for prompt sizing.
    """
    if not text:
        return 0
    words = re.findall(r"\w+", text)
    return int(math.ceil(len(words) * 1.3))

# ----------------------
# Keyword extraction (very small)
# ----------------------

_STOPWORDS = set([
    'the','a','an','in','on','at','for','to','from','by','with','and','or','of',
    'is','are','was','were','this','that','these','those','be','as','it','its',
    'which','will','can','should','may','has','have','do','does','did'
])


def _extract_keywords(text: str, top_k: int = 10) -> List[str]:
    """Return up to `top_k` lowercased keywords from text (naive frequency).

    Strips punctuation and stops words. Useful to match against snippet sentences.
    """
    tokens = re.findall(r"\w+", text.lower())
    tokens = [t for t in tokens if t not in _STOPWORDS and len(t) > 2]
    if not tokens:
        return []
    freq = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    # sort by frequency and then lexicographically
    items = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    return [k for k, _ in items[:top_k]]

# ----------------------
# Build retrieval objective
# ----------------------

def build_retrieval_objective(slot: Dict, subject_guidelines: Optional[str] = None, user_mode: str = 'balanced') -> str:
    """Construct a concise retrieval objective string from a planner slot.

    slot: expected keys (some optional):
      - section_id, title, question_type, difficulty_distribution (or difficulty),
        per_question_mark, special_instructions, must_cover_learning_objective,
        chapter_allocation (with preferred_chapters and chapter_weights)

    subject_guidelines: short authoritative text to include when relevant.
    user_mode: one of 'strict', 'balanced', 'creative' -> affects phrasing.

    Returns: a 1-2 line objective suitable for embedding and semantic search.
    """
    parts = []

    qtype = slot.get('question_type') or slot.get('type') or 'question'
    title = slot.get('title') or slot.get('section_id') or ''
    marks = slot.get('per_question_mark') or slot.get('marks') or None

    # Difficulty: prefer explicit field or derive from distribution
    difficulty = slot.get('difficulty')
    if not difficulty:
        dd = slot.get('difficulty_distribution') or {}
        # pick max share
        if isinstance(dd, dict) and dd:
            difficulty = max(dd.items(), key=lambda kv: kv[1])[0]
    difficulty = (difficulty or 'medium').capitalize()

    parts.append(f"{difficulty} {qtype} items")
    if marks:
        parts.append(f"worth {marks} mark{'s' if int(marks) != 1 else ''}")

    chap_alloc = slot.get('chapter_allocation') or {}
    preferred = chap_alloc.get('preferred_chapters') if isinstance(chap_alloc, dict) else None
    if preferred:
        parts.append(f"focusing on {', '.join(preferred)}")
    elif slot.get('chapters'):
        parts.append(f"from chapters: {', '.join(slot.get('chapters'))}")

    if slot.get('must_cover_learning_objective'):
        parts.append(f"objective: {slot.get('must_cover_learning_objective')}")

    if slot.get('special_instructions'):
        parts.append(f"note: {slot.get('special_instructions')} ")

    # user mode shaping
    mode_hint = {
        'strict': 'Prefer direct factual items closely tied to source text.',
        'balanced': 'Prefer exam-style questions grounded in source material.',
        'creative': 'Allow moderate rewording and application-style items.'
    }.get(user_mode, 'Prefer exam-style questions grounded in source material.')

    # Join into a compact objective (limit length)
    core = '; '.join(parts)
    objective = f"Generate {core}. {mode_hint}"

    if subject_guidelines:
        # include a short excerpt (first 120 chars) to give context without bloating
        excerpt = subject_guidelines.strip().replace('\n', ' ')
        if len(excerpt) > 120:
            excerpt = excerpt[:117].rsplit(' ', 1)[0] + '...'
        objective += f" Guideline: {excerpt}"

    # ensure objective is compact
    if len(objective.split()) > 60:
        # truncate politely at 60 words
        words = objective.split()
        objective = ' '.join(words[:60]) + '...'

    return objective

# ----------------------
# Summarization + token budgeting
# ----------------------

def _score_sentence_against_objective(sentence: str, objective_keywords: List[str]) -> float:
    """Score sentence by keyword overlap (normalized) and length penalty."""
    s_tokens = re.findall(r"\w+", sentence.lower())
    if not s_tokens:
        return 0.0
    # count matching keywords
    matches = sum(1 for t in s_tokens if t in objective_keywords)
    # score = matches / sqrt(len(sentence_words)) to penalize long sentences
    return matches / math.sqrt(len(s_tokens))


def summarize_and_budget_snippets(
    snippets: List[Dict],
    objective: str,
    max_tokens: int,
    max_sentences_per_snippet: int = 2
) -> Tuple[List[Dict], int]:
    """Create compact summaries for snippets and enforce a token budget.

    Args:
      snippets: list of dicts with at minimum a 'text' field and optional 'id'/'metadata'.
      objective: the retrieval objective string (used to rank relevance).
      max_tokens: maximum total tokens allowed across returned summaries.
      max_sentences_per_snippet: max sentences to use when summarizing each snippet.

    Returns:
      (selected_summaries, tokens_used)
      - selected_summaries: list of dicts: { 'id','summary','orig_text','est_tokens','metadata' }
      - tokens_used: total tokens (estimated)

    Behavior:
      - For each snippet, produce top `max_sentences_per_snippet` sentences chosen by
        keyword overlap with the objective.
      - Rank snippets by a combined score (relevance + snippet length preference).
      - Return as many top summaries as fit within `max_tokens` (greedy). If the first
        single summary exceeds `max_tokens`, it will be truncated to fit.
    """
    if not snippets:
        return [], 0

    # extract objective keywords
    objective_keywords = _extract_keywords(objective, top_k=20)

    summarized = []
    for s in snippets:
        text = s.get('text', '')
        sid = s.get('snippet_id') or s.get('id') or s.get('snippet_index') or None
        sentences = sentence_split(text)
        if not sentences:
            continue

        # score each sentence
        scored = [(sent, _score_sentence_against_objective(sent, objective_keywords)) for sent in sentences]
        # sort by score desc
        scored.sort(key=lambda kv: (-kv[1], len(kv[0])))
        # take top N sentences (preserve original order roughly)
        chosen = [kv[0] for kv in scored[:max_sentences_per_snippet]]

        # keep original ordering of chosen sentences as they appear in text
        chosen_sorted = [sent for sent in sentences if sent in chosen]
        summary_text = ' '.join(chosen_sorted)
        est_tokens = estimate_tokens_from_text(summary_text)

        # small normalization: if no keywords matched (score zeros), fallback to leading sentence(s)
        if sum(k for _, k in scored[:max_sentences_per_snippet]) == 0:
            # fallback: first N sentences
            chosen_sorted = sentences[:max_sentences_per_snippet]
            summary_text = ' '.join(chosen_sorted)
            est_tokens = estimate_tokens_from_text(summary_text)

        summarized.append({
            'id': sid,
            'summary': summary_text,
            'orig_text': text,
            'est_tokens': est_tokens,
            'metadata': s.get('metadata', {})
        })

    # rank summarized snippets by a simple heuristic: (keyword_coverage * 10) - est_tokens_penalty
    def _rank_item(item: Dict) -> float:
        kw_matches = 0
        toks = re.findall(r"\w+", item['summary'].lower())
        if toks:
            kw_matches = sum(1 for t in toks if t in objective_keywords)
        # prefer shorter summaries with higher keyword matches
        return (kw_matches * 10.0) - math.log(1 + item['est_tokens'])

    summarized.sort(key=_rank_item, reverse=True)

    # Greedily pick summaries until budget exhausted
    selected = []
    tokens_used = 0
    for item in summarized:
        if tokens_used + item['est_tokens'] <= max_tokens:
            selected.append(item)
            tokens_used += item['est_tokens']
        else:
            # if nothing selected yet, try truncation of this item to fit
            if not selected:
                # truncate summary_text by words to fit
                words = re.findall(r"\w+|[^\w\s]", item['summary'])
                if not words:
                    continue
                # build truncated text word by word until token estimate fits
                truncated = []
                for w in words:
                    truncated.append(w)
                    cur_text = ' '.join(truncated)
                    if estimate_tokens_from_text(cur_text) > max_tokens:
                        # remove last word and break
                        truncated.pop()
                        break
                if truncated:
                    trunc_text = ' '.join(truncated)
                    item_copy = item.copy()
                    item_copy['summary'] = trunc_text
                    item_copy['est_tokens'] = estimate_tokens_from_text(trunc_text)
                    selected.append(item_copy)
                    tokens_used += item_copy['est_tokens']
            # budget full or we cannot fit more
            break

    return selected, tokens_used

# ----------------------
# Quick self-test
# ----------------------
if __name__ == "__main__":
    # tiny sanity check
    snippets = [
        {'snippet_id': 's1', 'text': 'Photosynthesis is the process by which green plants make food. It occurs in chloroplasts. The light reactions produce ATP.'},
        {'snippet_id': 's2', 'text': 'Transpiration helps in the movement of water. It occurs through stomata. Factors include humidity and wind.'},
        {'snippet_id': 's3', 'text': 'Chemical reactions include combination, decomposition and displacement reactions. Examples include rusting and combustion.'}
    ]
    slot = {'question_type': 'MCQ', 'per_question_mark': 1, 'difficulty_distribution': {'easy': 50, 'medium': 30, 'hard': 20}, 'chapter_allocation': {'preferred_chapters': ['Photosynthesis']}}
    obj = build_retrieval_objective(slot, subject_guidelines='Follow CBSE sample paper rules: emphasis on conceptual understanding and application.')
    print('Objective:', obj)
    selected, tok = summarize_and_budget_snippets(snippets, obj, max_tokens=60)
    print('\nSelected summaries:')
    for s in selected:
        print('-', s['id'], s['summary'], '(est_tokens=', s['est_tokens'], ')')
