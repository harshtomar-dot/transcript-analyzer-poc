import hashlib
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # walks up parent directories too, picks up repo-root .env if present

from core.filters import filter_calls
from core.llm import (
    answer_open_question,
    classify_calls,
    classify_turns,
    embed_texts,
    propose_categories,
    propose_categories_from_calls,
)
from core.models import Call
from core.parsing import parse_json_call, parse_text_call, parse_uploaded_bytes

SAMPLE_DIR = Path(__file__).parent / "sample_data"
AGENT_PROMPT_PATH = SAMPLE_DIR / "woxsen_agent_prompt.txt"

DEFAULT_CATEGORIES = [
    {"name": "No_Answer", "description": "The agent fails to address the user's question at all, deflects, or says it doesn't have the information."},
    {"name": "Wrong_Answer", "description": "The agent gives a confident answer that is factually incorrect, made up, or contradicts known policy."},
    {"name": "Correct_Answer", "description": "The agent gives an accurate, relevant, on-topic answer to what the user asked."},
    {"name": "Clarification", "description": "The agent asks a follow-up question instead of answering, to gather more information first."},
    {"name": "Escalation", "description": "The agent transfers, escalates, or promises a callback instead of answering directly."},
    {"name": "Other", "description": "Doesn't fit any of the above (greetings, small talk, closing remarks, etc.)."},
]

st.set_page_config(page_title="Transcript Analyzer POC", layout="wide")


def _hash_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _get_passcode():
    try:
        val = st.secrets.get("APP_PASSCODE")
        if val:
            return str(val)
    except Exception:
        pass
    return os.environ.get("APP_PASSCODE") or None


def require_passcode() -> bool:
    passcode = _get_passcode()
    if not passcode:
        return True  # no gate configured (e.g. local dev) — skip it
    if st.session_state.get("_unlocked"):
        return True
    st.title("🔒 Transcript Analyzer POC")
    st.text_input("Passcode", type="password", key="_passcode_input")
    if st.button("Unlock"):
        if st.session_state.get("_passcode_input") == passcode:
            st.session_state["_unlocked"] = True
            st.rerun()
        else:
            st.error("Incorrect passcode.")
    return False


def init_state():
    st.session_state.setdefault("calls", [])
    st.session_state.setdefault("categories", [dict(c) for c in DEFAULT_CATEGORIES])
    st.session_state.setdefault("custom_categories", [])
    st.session_state.setdefault("classification_cache", {})  # (scope, cat_signature) -> {turn_key: result}
    st.session_state.setdefault("embedding_cache", {})  # text_hash -> vector
    st.session_state.setdefault(
        "agent_context", AGENT_PROMPT_PATH.read_text() if AGENT_PROMPT_PATH.exists() else ""
    )
    st.session_state.setdefault("rca_taxonomy", [])
    st.session_state.setdefault("rca_proposed", [])
    st.session_state.setdefault("rca_history", [])
    st.session_state.setdefault("rca_last_run", None)
    st.session_state.setdefault("rca_last_sample", [])
    if not st.session_state["calls"] and not st.session_state.get("_auto_loaded"):
        load_samples()
        st.session_state["_auto_loaded"] = True


def load_samples():
    calls = []
    for f in sorted(SAMPLE_DIR.glob("*.json")):
        raw = f.read_text()
        import json

        calls.append(parse_json_call(json.loads(raw), fallback_id=f.stem, fallback_filename=f.name))
    st.session_state["calls"] = calls


def add_uploaded_files(files):
    calls = list(st.session_state["calls"])
    existing_ids = {c.call_id for c in calls}
    for f in files:
        content = f.read()
        fallback_id = Path(f.name).stem
        i = 1
        base = fallback_id
        while fallback_id in existing_ids:
            fallback_id = f"{base}_{i}"
            i += 1
        try:
            call = parse_uploaded_bytes(content, f.name, fallback_id=fallback_id)
        except ValueError as e:
            st.sidebar.error(str(e))
            continue
        if not call.turns:
            st.sidebar.warning(f"{f.name}: no turns could be parsed, skipping.")
            continue
        calls.append(call)
        existing_ids.add(call.call_id)
    st.session_state["calls"] = calls


def add_pasted_transcript(text: str, call_id: str):
    call = parse_text_call(text, fallback_id=call_id or "pasted_call")
    if not call.turns:
        st.sidebar.warning("Could not parse any turns. Use lines like 'Agent: ...' / 'User: ...'.")
        return
    calls = list(st.session_state["calls"])
    calls.append(call)
    st.session_state["calls"] = calls


def sidebar():
    sample_count = len(list(SAMPLE_DIR.glob("*.json")))
    st.sidebar.header("1. Load transcripts")
    if st.sidebar.button(f"Reload sample data ({sample_count} real Woxsen calls)"):
        load_samples()

    uploaded = st.sidebar.file_uploader(
        "Upload transcript files (.json or .txt)", type=["json", "txt"], accept_multiple_files=True
    )
    if uploaded:
        add_uploaded_files(uploaded)

    with st.sidebar.expander("Paste a transcript"):
        call_id = st.text_input("Call ID", key="paste_call_id")
        pasted = st.text_area(
            "Format: one line per turn, e.g.\nAgent: Hello, how can I help?\nUser: I have a question about...",
            height=150,
            key="paste_text",
        )
        if st.button("Add pasted transcript"):
            add_pasted_transcript(pasted, call_id)

    if st.sidebar.button("Clear all loaded transcripts"):
        st.session_state["calls"] = []

    with st.sidebar.expander("Reference: agent prompt / knowledge base"):
        current = st.session_state["agent_context"]
        if current:
            st.caption(f"Loaded ({len(current):,} chars) — used as ground truth for judging factual correctness.")
            st.text(current[:400] + ("..." if len(current) > 400 else ""))
        else:
            st.caption("None loaded — Wrong_Answer judging will be generic, without ground-truth facts to check against.")
        replacement = st.text_area("Paste replacement reference text", height=100, key="agent_context_paste")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Set reference", key="set_agent_context") and replacement.strip():
                st.session_state["agent_context"] = replacement.strip()
        with col_b:
            if st.button("Clear reference", key="clear_agent_context"):
                st.session_state["agent_context"] = ""

    calls = st.session_state["calls"]
    st.sidebar.markdown(f"**{len(calls)} call(s) loaded**")
    if not calls:
        return []

    st.sidebar.header("2. Filters")
    dated = [c for c in calls if c.date is not None]
    if dated:
        min_d, max_d = min(c.date for c in dated), max(c.date for c in dated)
        if min_d == max_d:
            date_range = (min_d, max_d)
            st.sidebar.caption(f"All calls on {min_d}")
        else:
            date_range = st.sidebar.slider("Date range", min_value=min_d, max_value=max_d, value=(min_d, max_d))
    else:
        date_range = None
        st.sidebar.caption("No date metadata on loaded calls — date filter skipped.")

    durationed = [c for c in calls if c.duration_seconds is not None]
    if durationed:
        min_dur, max_dur = min(c.duration_seconds for c in durationed), max(c.duration_seconds for c in durationed)
        if min_dur == max_dur:
            duration_range = (min_dur, max_dur)
        else:
            duration_range = st.sidebar.slider(
                "Duration (seconds)", min_value=float(min_dur), max_value=float(max_dur), value=(float(min_dur), float(max_dur))
            )
    else:
        duration_range = None
        st.sidebar.caption("No duration metadata on loaded calls — duration filter skipped.")

    min_t, max_t = min(c.turn_count for c in calls), max(c.turn_count for c in calls)
    if min_t == max_t:
        turn_range = (min_t, max_t)
    else:
        turn_range = st.sidebar.slider("Total turns per call", min_value=min_t, max_value=max_t, value=(min_t, max_t))

    filtered = filter_calls(calls, date_range=date_range, duration_range=duration_range, turn_count_range=turn_range)
    st.sidebar.markdown(f"**{len(filtered)} call(s) match filters**")
    return filtered


def render_call_picker(calls: list, key: str) -> list:
    df = pd.DataFrame(
        [
            {
                "call_id": c.call_id,
                "date": c.date,
                "duration_s": c.duration_seconds,
                "turns": c.turn_count,
                "source": c.source_filename or "pasted",
            }
            for c in calls
        ]
    )
    st.dataframe(df, use_container_width=True, height=min(240, 40 + 32 * len(df)))
    return calls


def tab_predefined_bucketing(calls: list):
    st.subheader("Predefined bucketing")
    st.caption("Every agent turn gets classified into one of these fixed categories. Edit them below if needed.")

    cats = st.session_state["categories"]
    edited = st.data_editor(
        pd.DataFrame(cats),
        num_rows="dynamic",
        use_container_width=True,
        key="predefined_editor",
    )
    st.session_state["categories"] = edited.dropna(subset=["name"]).to_dict("records")

    if st.button("Run classification", key="run_predefined"):
        cats = st.session_state["categories"]
        if not cats:
            st.warning("Add at least one category first.")
            return
        turns_with_context = []
        for call in calls:
            for i, t in enumerate(call.turns):
                if t.speaker != "agent":
                    continue
                turns_with_context.append((t, call.context_for(i)))
        if not turns_with_context:
            st.warning("No agent turns found in the filtered calls.")
            return
        with st.spinner(f"Classifying {len(turns_with_context)} agent turns..."):
            try:
                results = classify_turns(turns_with_context, cats, agent_context=st.session_state.get("agent_context"))
            except RuntimeError as e:
                st.error(str(e))
                return
        st.session_state["predefined_results"] = results
        st.session_state["predefined_turns"] = {t.key: t for t, _ in turns_with_context}

    results = st.session_state.get("predefined_results")
    if results:
        turns_by_key = st.session_state["predefined_turns"]
        rows = []
        for key, r in results.items():
            t = turns_by_key.get(key)
            if not t:
                continue
            rows.append({"call_id": t.call_id, "turn": t.index, "text": t.text, "category": r["category"], "reason": r["reason"]})
        df = pd.DataFrame(rows)
        counts = df["category"].value_counts()
        col1, col2 = st.columns([1, 2])
        with col1:
            st.bar_chart(counts)
        with col2:
            st.dataframe(df, use_container_width=True, height=320)


def tab_auto_bucketing(calls: list):
    st.subheader("Auto bucketing")
    st.caption("The LLM proposes categories from the data itself. Review, edit, then apply.")

    scope = st.radio("Which turns to analyze?", ["Agent turns", "User turns", "All turns"], horizontal=True, key="auto_scope")
    speaker_filter = {"Agent turns": "agent", "User turns": "user", "All turns": None}[scope]

    if st.button("Propose categories", key="propose_cats"):
        sample = [t for call in calls for t in call.turns if speaker_filter is None or t.speaker == speaker_filter]
        if not sample:
            st.warning("No turns match this scope in the filtered calls.")
            return
        with st.spinner("Asking the model to propose a taxonomy..."):
            try:
                proposed = propose_categories(sample, agent_context=st.session_state.get("agent_context"))
            except RuntimeError as e:
                st.error(str(e))
                return
        st.session_state["custom_categories"] = proposed

    if st.session_state["custom_categories"]:
        st.markdown("**Proposed categories** — edit names/descriptions, add or delete rows:")
        edited = st.data_editor(
            pd.DataFrame(st.session_state["custom_categories"]),
            num_rows="dynamic",
            use_container_width=True,
            key="auto_editor",
        )
        st.session_state["custom_categories"] = edited.dropna(subset=["name"]).to_dict("records")

        if st.button("Apply categories & classify", key="apply_auto"):
            cats = st.session_state["custom_categories"]
            turns_with_context = []
            for call in calls:
                for i, t in enumerate(call.turns):
                    if speaker_filter is not None and t.speaker != speaker_filter:
                        continue
                    turns_with_context.append((t, call.context_for(i)))
            with st.spinner(f"Classifying {len(turns_with_context)} turns..."):
                try:
                    results = classify_turns(turns_with_context, cats, agent_context=st.session_state.get("agent_context"))
                except RuntimeError as e:
                    st.error(str(e))
                    return
            st.session_state["auto_results"] = results
            st.session_state["auto_turns"] = {t.key: t for t, _ in turns_with_context}

    results = st.session_state.get("auto_results")
    if results:
        turns_by_key = st.session_state["auto_turns"]
        rows = []
        for key, r in results.items():
            t = turns_by_key.get(key)
            if not t:
                continue
            rows.append({"call_id": t.call_id, "turn": t.index, "text": t.text, "category": r["category"], "reason": r["reason"]})
        df = pd.DataFrame(rows)
        counts = df["category"].value_counts()
        col1, col2 = st.columns([1, 2])
        with col1:
            st.bar_chart(counts)
        with col2:
            st.dataframe(df, use_container_width=True, height=320)


def tab_phrase_search(calls: list):
    st.subheader("Phrase search (exact match)")
    scope = st.radio("Search in", ["Both", "Agent only", "User only"], horizontal=True, key="phrase_scope")
    speaker_filter = {"Both": None, "Agent only": "agent", "User only": "user"}[scope]
    phrase = st.text_input("Phrase to find", key="phrase_query")
    case_sensitive = st.checkbox("Case sensitive", value=False, key="phrase_case")

    if phrase:
        needle = phrase if case_sensitive else phrase.lower()
        rows = []
        for call in calls:
            for t in call.turns:
                if speaker_filter is not None and t.speaker != speaker_filter:
                    continue
                haystack = t.text if case_sensitive else t.text.lower()
                if needle in haystack:
                    rows.append({"call_id": t.call_id, "turn": t.index, "speaker": t.speaker, "text": t.text})
        st.markdown(f"**{len(rows)} matching turn(s)** across {len({r['call_id'] for r in rows})} call(s)")
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, height=320)


def _cosine_sim_matrix(query_vec, matrix):
    q = np.array(query_vec)
    m = np.array(matrix)
    q_norm = q / (np.linalg.norm(q) + 1e-10)
    m_norm = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-10)
    return m_norm @ q_norm


def tab_similar_phrase_search(calls: list):
    st.subheader("Similar phrase search (semantic)")
    scope = st.radio("Search in", ["Both", "Agent only", "User only"], horizontal=True, key="sim_scope")
    speaker_filter = {"Both": None, "Agent only": "agent", "User only": "user"}[scope]
    phrase = st.text_input("Phrase or meaning to find", key="sim_query")
    threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.55, 0.01, key="sim_threshold")
    top_n = st.number_input("Show top N", min_value=5, max_value=200, value=20, key="sim_top_n")

    if phrase and st.button("Search", key="run_sim_search"):
        target_turns = [t for call in calls for t in call.turns if speaker_filter is None or t.speaker == speaker_filter]
        if not target_turns:
            st.warning("No turns match this scope in the filtered calls.")
            return

        cache = st.session_state["embedding_cache"]
        to_embed = []
        to_embed_keys = []
        for t in target_turns:
            h = _hash_key(t.text)
            if h not in cache:
                to_embed.append(t.text)
                to_embed_keys.append(h)

        with st.spinner(f"Embedding {len(to_embed)} new turn(s)..." if to_embed else "Searching..."):
            try:
                if to_embed:
                    vectors = embed_texts(to_embed)
                    for h, v in zip(to_embed_keys, vectors):
                        cache[h] = v
                query_vec = embed_texts([phrase])[0]
            except RuntimeError as e:
                st.error(str(e))
                return

        matrix = [cache[_hash_key(t.text)] for t in target_turns]
        sims = _cosine_sim_matrix(query_vec, matrix)

        rows = [
            {"call_id": t.call_id, "turn": t.index, "speaker": t.speaker, "text": t.text, "similarity": round(float(s), 4)}
            for t, s in zip(target_turns, sims)
        ]
        df = pd.DataFrame(rows).sort_values("similarity", ascending=False)
        above = df[df["similarity"] >= threshold]
        st.markdown(f"**{len(above)} turn(s) above threshold {threshold}** (out of {len(df)} searched)")
        st.dataframe(df.head(int(top_n)), use_container_width=True, height=320)


def tab_open_ended(calls: list):
    st.subheader("Open-ended questions")
    st.caption("Ask a free-form question across all filtered calls, e.g. 'What are the top 5 questions users asked?'")
    question = st.text_area("Your question", key="open_question")
    if st.button("Ask", key="run_open") and question:
        with st.spinner("Thinking across the filtered transcripts..."):
            try:
                answer = answer_open_question(question, calls, agent_context=st.session_state.get("agent_context"))
            except RuntimeError as e:
                st.error(str(e))
                return
        st.markdown(answer)


def _merge_categories(existing: list, new: list):
    merged = list(existing)
    existing_names = {c["name"].strip().lower() for c in merged if c.get("name")}
    added = 0
    for c in new:
        name = (c.get("name") or "").strip()
        if not name or name.lower() in existing_names:
            continue
        merged.append({"name": name, "description": c.get("description", "")})
        existing_names.add(name.lower())
        added += 1
    return merged, added


def _render_rca_proposed_editor():
    proposed = st.session_state.get("rca_proposed")
    if not proposed:
        return
    st.markdown("**Pending proposed categories** — edit, then add to the master taxonomy:")
    edited = st.data_editor(
        pd.DataFrame(proposed), num_rows="dynamic", use_container_width=True, key="rca_proposed_editor"
    )
    st.session_state["rca_proposed"] = edited.dropna(subset=["name"]).to_dict("records")
    if st.button("Add to master taxonomy", key="rca_merge_proposed"):
        merged, added = _merge_categories(st.session_state["rca_taxonomy"], st.session_state["rca_proposed"])
        st.session_state["rca_taxonomy"] = merged
        st.session_state["rca_proposed"] = []
        st.success(f"Added {added} new categor{'y' if added == 1 else 'ies'} to the master taxonomy.")
        st.rerun()


def tab_rca(calls: list):
    st.subheader("RCA / Coverage")
    st.caption(
        "Discover root-cause categories from a small sample, confirm them, then test coverage on "
        "a bigger set. Calls that don't fit any known category stay 'Unclassified' — mine those "
        "to discover new categories, and repeat as your sample grows (10 → 100 → 1000 ...)."
    )
    max_n = len(calls)

    st.markdown("#### Master taxonomy")
    taxonomy = st.session_state["rca_taxonomy"]
    if taxonomy:
        edited = st.data_editor(
            pd.DataFrame(taxonomy), num_rows="dynamic", use_container_width=True, key="rca_taxonomy_editor"
        )
        st.session_state["rca_taxonomy"] = edited.dropna(subset=["name"]).to_dict("records")
    else:
        st.caption("No categories yet — discover some from a sample in step 1 below.")
    st.caption("'Unclassified' is added automatically at classification time — no need to add it yourself.")

    if st.button("Reset RCA taxonomy & history", key="rca_reset"):
        st.session_state["rca_taxonomy"] = []
        st.session_state["rca_proposed"] = []
        st.session_state["rca_history"] = []
        st.session_state["rca_last_run"] = None
        st.session_state["rca_last_sample"] = []
        st.rerun()

    _render_rca_proposed_editor()

    st.divider()
    st.markdown("#### 1. Discover categories from a sample")
    col1, col2 = st.columns([1, 1])
    with col1:
        discover_n = st.number_input(
            "Sample size", min_value=1, max_value=max_n, value=min(10, max_n), key="rca_discover_n"
        )
    with col2:
        discover_mode = st.radio("Sampling", ["First N", "Random N"], horizontal=True, key="rca_discover_mode")

    if st.button("Propose categories from this sample", key="rca_propose_sample"):
        sample = calls[:discover_n] if discover_mode == "First N" else random.sample(calls, discover_n)
        with st.spinner(f"Proposing categories from {len(sample)} call(s)..."):
            try:
                proposed = propose_categories_from_calls(sample, agent_context=st.session_state.get("agent_context"))
            except RuntimeError as e:
                st.error(str(e))
                return
        st.session_state["rca_proposed"] = proposed
        st.rerun()

    st.divider()
    st.markdown("#### 2. Test coverage on a larger set")
    if not st.session_state["rca_taxonomy"]:
        st.info("Add at least one category to the master taxonomy first (via step 1 above).")
    else:
        col3, col4 = st.columns([1, 1])
        with col3:
            test_n = st.number_input("Test size (ignored if sampling 'All in scope')", min_value=1, max_value=max_n, value=max_n, key="rca_test_n")
        with col4:
            test_mode = st.radio("Sampling", ["First N", "Random N", "All in scope"], horizontal=True, key="rca_test_mode")

        if st.button("Classify against master taxonomy", key="rca_classify"):
            if test_mode == "All in scope":
                test_sample = calls
            elif test_mode == "First N":
                test_sample = calls[:test_n]
            else:
                test_sample = random.sample(calls, test_n)

            with st.spinner(
                f"Classifying {len(test_sample)} call(s) against {len(st.session_state['rca_taxonomy'])} categor"
                f"{'y' if len(st.session_state['rca_taxonomy']) == 1 else 'ies'}..."
            ):
                try:
                    results = classify_calls(
                        test_sample, st.session_state["rca_taxonomy"], agent_context=st.session_state.get("agent_context")
                    )
                except RuntimeError as e:
                    st.error(str(e))
                    return

            st.session_state["rca_last_run"] = results
            st.session_state["rca_last_sample"] = test_sample
            n_unclassified = sum(1 for r in results.values() if r["category"] == "Unclassified")
            coverage_pct = round(100 * (len(results) - n_unclassified) / len(results), 1) if results else 0.0
            st.session_state["rca_history"].append(
                {
                    "round": len(st.session_state["rca_history"]) + 1,
                    "calls_tested": len(test_sample),
                    "categories": len(st.session_state["rca_taxonomy"]),
                    "coverage_pct": coverage_pct,
                    "unclassified": n_unclassified,
                }
            )

        last_run = st.session_state.get("rca_last_run")
        if last_run:
            rows = [{"call_id": cid, "category": r["category"], "reason": r["reason"]} for cid, r in last_run.items()]
            df = pd.DataFrame(rows)
            counts = df["category"].value_counts()
            n_unclassified = int(counts.get("Unclassified", 0))
            coverage_pct = round(100 * (len(df) - n_unclassified) / len(df), 1) if len(df) else 0.0

            m1, m2, m3 = st.columns(3)
            m1.metric("Calls tested", len(df))
            m2.metric("Coverage", f"{coverage_pct}%")
            m3.metric("Unclassified", n_unclassified)

            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.bar_chart(counts)
            with col_b:
                st.dataframe(df, use_container_width=True, height=320)

        if st.session_state["rca_history"]:
            st.markdown("**Round history**")
            hist_df = pd.DataFrame(st.session_state["rca_history"])
            st.dataframe(hist_df, use_container_width=True, height=min(240, 40 + 32 * len(hist_df)))

    st.divider()
    st.markdown("#### 3. Discover new categories from the Unclassified calls")
    last_run = st.session_state.get("rca_last_run")
    last_sample = st.session_state.get("rca_last_sample", [])
    if not last_run:
        st.caption("Run step 2 first.")
    else:
        unclassified_ids = {cid for cid, r in last_run.items() if r["category"] == "Unclassified"}
        unclassified_calls = [c for c in last_sample if c.call_id in unclassified_ids]
        st.caption(f"{len(unclassified_calls)} call(s) are currently Unclassified from the last run.")
        if unclassified_calls and st.button("Propose categories from Unclassified calls", key="rca_propose_leftover"):
            with st.spinner(f"Proposing categories from {len(unclassified_calls)} unclassified call(s)..."):
                try:
                    proposed = propose_categories_from_calls(
                        unclassified_calls, agent_context=st.session_state.get("agent_context")
                    )
                except RuntimeError as e:
                    st.error(str(e))
                    return
            st.session_state["rca_proposed"] = proposed
            st.rerun()
        st.caption("After adding new categories to the master taxonomy above, go back to step 2 and re-run to see updated coverage.")


def main():
    init_state()
    if not require_passcode():
        return
    st.title("Transcript Analyzer POC")
    filtered_calls = sidebar()

    if not st.session_state["calls"]:
        st.info("Load sample data or upload/paste a transcript from the sidebar to get started.")
        return

    if not filtered_calls:
        st.warning("No calls match the current filters.")
        return

    st.subheader("Calls in scope")
    render_call_picker(filtered_calls, key="main_picker")

    tabs = st.tabs(
        [
            "Predefined bucketing",
            "Auto bucketing",
            "RCA / Coverage",
            "Phrase search",
            "Similar phrase search",
            "Open-ended questions",
        ]
    )
    with tabs[0]:
        tab_predefined_bucketing(filtered_calls)
    with tabs[1]:
        tab_auto_bucketing(filtered_calls)
    with tabs[2]:
        tab_rca(filtered_calls)
    with tabs[3]:
        tab_phrase_search(filtered_calls)
    with tabs[4]:
        tab_similar_phrase_search(filtered_calls)
    with tabs[5]:
        tab_open_ended(filtered_calls)


if __name__ == "__main__":
    main()
