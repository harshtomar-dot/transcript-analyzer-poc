import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from openai import OpenAI

CHAT_MODEL = os.environ.get("TRANSCRIPT_POC_MODEL", "gpt-4.1-mini")
EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE = 25
CALL_BATCH_SIZE = 8
MAX_CONCURRENCY = int(os.environ.get("TRANSCRIPT_POC_CONCURRENCY", "20"))

UNCLASSIFIED_CATEGORY = {
    "name": "Unclassified",
    "description": "Doesn't clearly match any of the other defined categories. Use this rather than forcing a weak or partial fit.",
}

_client: Optional[OpenAI] = None


def _get_api_key() -> Optional[str]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return api_key
    try:
        import streamlit as st

        return st.secrets.get("OPENAI_API_KEY")
    except Exception:
        return None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = _get_api_key()
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to a .env file (this app auto-loads "
                "the nearest .env, including one in a parent directory) or export it in your shell."
            )
        _client = OpenAI(api_key=api_key)
    return _client


def _chunk(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _run_concurrent(items: list, fn) -> list:
    """Run fn(item) for each item, up to MAX_CONCURRENCY in parallel, preserving order."""
    if len(items) <= 1:
        return [fn(item) for item in items]
    results = [None] * len(items)
    with ThreadPoolExecutor(max_workers=min(MAX_CONCURRENCY, len(items))) as executor:
        futures = {executor.submit(fn, item): i for i, item in enumerate(items)}
        for future, i in futures.items():
            results[i] = future.result()
    return results


def _reference_block(agent_context: Optional[str]) -> str:
    if not agent_context:
        return ""
    return (
        "\n\nReference material (the agent's own prompt / knowledge base — treat as ground "
        "truth for judging factual correctness):\n" + agent_context[:60_000] + "\n"
    )


def classify_turns(turns_with_context: list, categories: list, agent_context: Optional[str] = None) -> dict:
    """turns_with_context: list of (Turn, context_str). categories: list of {name, description}.
    Returns {turn.key: category_name}.
    """
    category_names = [c["name"] for c in categories]
    category_block = "\n".join(f"- {c['name']}: {c['description']}" for c in categories)
    client = get_client()

    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "turn_classifications",
            "schema": {
                "type": "object",
                "properties": {
                    "results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "category": {"type": "string", "enum": category_names},
                                "reason": {"type": "string"},
                            },
                            "required": ["id", "category", "reason"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["results"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }

    def classify_batch(batch):
        items_block = []
        for turn, context in batch:
            ctx = f"  Preceding context:\n    {context}\n" if context else ""
            items_block.append(f'[{turn.key}]\n{ctx}  {turn.speaker}: "{turn.text}"')
        prompt = (
            "You are labeling turns from voice-agent call transcripts.\n\n"
            f"Categories:\n{category_block}\n"
            f"{_reference_block(agent_context)}\n"
            "For each turn below, pick exactly one category and give a one-sentence reason.\n\n"
            + "\n\n".join(items_block)
        )
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format=schema,
        )
        return json.loads(resp.choices[0].message.content)["results"]

    batches = list(_chunk(turns_with_context, BATCH_SIZE))
    results: dict = {}
    for batch_results in _run_concurrent(batches, classify_batch):
        for r in batch_results:
            results[r["id"]] = {"category": r["category"], "reason": r["reason"]}

    return results


def _call_transcript_block(call, max_chars: int = 6000) -> str:
    text = "\n".join(f"{t.speaker}: {t.text}" for t in call.turns)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...[truncated]"
    return text


def classify_calls(calls: list, categories: list, agent_context: Optional[str] = None) -> dict:
    """Whole-call root-cause classification. categories: list of {name, description} — an
    'Unclassified' catch-all is added automatically. Returns {call_id: {category, reason}}.
    """
    effective_categories = list(categories) + [UNCLASSIFIED_CATEGORY]
    category_names = [c["name"] for c in effective_categories]
    category_block = "\n".join(f"- {c['name']}: {c['description']}" for c in effective_categories)
    client = get_client()

    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "call_classifications",
            "schema": {
                "type": "object",
                "properties": {
                    "results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "call_id": {"type": "string"},
                                "category": {"type": "string", "enum": category_names},
                                "reason": {"type": "string"},
                            },
                            "required": ["call_id", "category", "reason"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["results"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }

    def classify_batch(batch):
        items_block = [f"[{call.call_id}]\n{_call_transcript_block(call)}" for call in batch]
        prompt = (
            "You are performing root-cause analysis on voice-agent call transcripts. Assign each "
            "call below to exactly one category.\n\n"
            f"Categories:\n{category_block}\n"
            f"{_reference_block(agent_context)}\n"
            "Strongly prefer an existing, specific category when it genuinely applies. Only use "
            "'Unclassified' when the call truly doesn't match any defined category — never force "
            "a weak or partial fit into the wrong bucket.\n\n" + "\n\n".join(items_block)
        )
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format=schema,
        )
        return json.loads(resp.choices[0].message.content)["results"]

    batches = list(_chunk(calls, CALL_BATCH_SIZE))
    results: dict = {}
    for batch_results in _run_concurrent(batches, classify_batch):
        for r in batch_results:
            results[r["call_id"]] = {"category": r["category"], "reason": r["reason"]}

    return results


def propose_categories_from_calls(calls: list, num_categories: int = 4, agent_context: Optional[str] = None) -> list:
    """calls: list of Call. Returns list of {name, description}. No catch-all — the caller adds
    'Unclassified' separately at classify time via classify_calls."""
    client = get_client()
    blocks = [f"### Call {call.call_id}\n{_call_transcript_block(call, max_chars=3000)}" for call in calls[:60]]
    prompt = (
        "You are analyzing a set of voice-agent call transcripts to find root-cause issue "
        f"categories. Propose up to {num_categories} specific, mutually exclusive categories "
        "describing what went wrong (or notably happened) in these calls — e.g. a type of "
        "factual error, a process failure, a user complaint pattern. Each category needs a short "
        "plain name and a one-sentence behavioral description a labeler could apply consistently. "
        "Do not include a catch-all/'other' category — that's handled separately."
        f"{_reference_block(agent_context)}\n"
        "Calls:\n" + "\n\n".join(blocks)
    )
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "proposed_call_categories",
            "schema": {
                "type": "object",
                "properties": {
                    "categories": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "description": {"type": "string"},
                            },
                            "required": ["name", "description"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["categories"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format=schema,
    )
    parsed = json.loads(resp.choices[0].message.content)
    return parsed["categories"]


def propose_categories(sample_turns: list, num_categories: int = 6, agent_context: Optional[str] = None) -> list:
    """sample_turns: list of Turn. Returns list of {name, description}."""
    client = get_client()
    lines = [f'{t.speaker}: "{t.text}"' for t in sample_turns]
    prompt = (
        "You are analyzing turns from voice-agent call transcripts to design a bucketing taxonomy.\n"
        f"Propose up to {num_categories} mutually exclusive categories that describe what is "
        "happening in these turns (e.g. quality of the agent's answer, type of user intent, "
        "failure mode). Each category needs a short Title_Case_With_Underscores-free plain name "
        "and a one-sentence behavioral description a labeler could apply consistently. Include a "
        "catch-all 'Other' category."
        f"{_reference_block(agent_context)}\n"
        "Turns:\n" + "\n".join(lines[:200])
    )
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "proposed_categories",
            "schema": {
                "type": "object",
                "properties": {
                    "categories": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "description": {"type": "string"},
                            },
                            "required": ["name", "description"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["categories"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format=schema,
    )
    parsed = json.loads(resp.choices[0].message.content)
    return parsed["categories"]


def embed_texts(texts: list) -> list:
    if not texts:
        return []
    client = get_client()

    def embed_batch(batch):
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        return [d.embedding for d in resp.data]

    batches = list(_chunk(texts, 100))
    vectors: list = []
    for batch_vectors in _run_concurrent(batches, embed_batch):
        vectors.extend(batch_vectors)
    return vectors


def answer_open_question(
    question: str, calls: list, agent_context: Optional[str] = None, max_chars: int = 400_000
) -> str:
    client = get_client()
    parts = []
    used = 0
    truncated = False
    for call in calls:
        block_lines = [f"### Call {call.call_id}"]
        for t in call.turns:
            block_lines.append(f"{t.speaker}: {t.text}")
        block = "\n".join(block_lines)
        if used + len(block) > max_chars:
            truncated = True
            break
        parts.append(block)
        used += len(block)

    transcript_dump = "\n\n".join(parts)
    note = "\n\n(Note: input was truncated to fit context.)" if truncated else ""
    prompt = (
        "You are analyzing a batch of voice-agent call transcripts. Answer the question using "
        "only what's in these transcripts. Cite call ids as evidence where useful. If the "
        "transcripts don't contain enough information to answer confidently, say so."
        f"{_reference_block(agent_context)}\n"
        f"Question: {question}\n\n"
        f"Transcripts:\n{transcript_dump}{note}"
    )
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content
