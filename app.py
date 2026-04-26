import uuid

import streamlit as st
from src.rag.pipeline import answer_question
from src.feedback import log_feedback


st.set_page_config(
    page_title="Medical RAG",
    page_icon=":hospital:",
    layout="centered"
)

top_k = 5

# --- Session state init ---
if "sessions" not in st.session_state:
    st.session_state.sessions = [{"messages": [], "rag_history": []}]
if "active_session" not in st.session_state:
    st.session_state.active_session = 0
if "focused_turn" not in st.session_state:
    st.session_state.focused_turn = None
if "rated_turns" not in st.session_state:
    st.session_state.rated_turns = set()


def active() -> dict:
    return st.session_state.sessions[st.session_state.active_session]


def session_label(session: dict, idx: int) -> str:
    user_msgs = [m for m in session["messages"] if m["role"] == "user"]
    if not user_msgs:
        return f"Session {idx + 1} — (empty)"
    first = user_msgs[0]["content"]
    return first if len(first) <= 40 else first[:40] + "..."


# --- Sidebar ---
with st.sidebar:
    st.markdown("### Conversations")

    if st.button("+ New conversation", use_container_width=True):
        # Don't create a duplicate empty session
        if active()["messages"]:
            st.session_state.sessions.append({"messages": [], "rag_history": []})
            st.session_state.active_session = len(st.session_state.sessions) - 1
            st.session_state.focused_turn = None
        st.rerun()

    st.divider()

    for idx, session in enumerate(st.session_state.sessions):
        label = session_label(session, idx)
        is_active = idx == st.session_state.active_session
        # Bold the active session label
        display = f"**{label}**" if is_active else label
        if st.button(display, key=f"session_{idx}", use_container_width=True):
            st.session_state.active_session = idx
            st.session_state.focused_turn = None
            st.rerun()

    st.divider()

    # History within the active session
    user_turns = [
        (i, m) for i, m in enumerate(active()["messages"])
        if m["role"] == "user"
    ]
    if user_turns:
        st.markdown("### In this session")
        for turn_idx, (_, msg) in enumerate(user_turns):
            short = msg["content"] if len(msg["content"]) <= 40 else msg["content"][:40] + "..."
            if st.button(f"{turn_idx + 1}. {short}", key=f"turn_{turn_idx}", use_container_width=True):
                st.session_state.focused_turn = turn_idx
                st.rerun()

        if st.session_state.focused_turn is not None:
            if st.button("Show full conversation", use_container_width=True):
                st.session_state.focused_turn = None
                st.rerun()


# --- Main area ---
st.title("Medical RAG")
st.caption("Answers grounded in medical guidelines, with citations.")


def session_title() -> str:
    user_msgs = [m for m in active()["messages"] if m["role"] == "user"]
    return user_msgs[0]["content"] if user_msgs else None


def render_feedback_buttons(turn_id: str, question: str, answer: str) -> None:
    if turn_id in st.session_state.rated_turns:
        st.caption("✓ Feedback recorded")
        return
    st.caption("Was this helpful?")
    col_up, col_down, *_ = st.columns([1, 1, 8])
    if col_up.button("👍", key=f"up_{turn_id}"):
        log_feedback(turn_id, question, answer, rating=1, session_title=session_title())
        st.session_state.rated_turns.add(turn_id)
        st.rerun()
    if col_down.button("👎", key=f"down_{turn_id}"):
        log_feedback(turn_id, question, answer, rating=-1, session_title=session_title())
        st.session_state.rated_turns.add(turn_id)
        st.rerun()


def render_assistant_msg(msg: dict) -> None:
    from src.rag.pipeline import OUT_OF_SCOPE_OFFTOPIC, OUT_OF_SCOPE_MEDICAL
    content = msg["content"]
    if content == OUT_OF_SCOPE_OFFTOPIC:
        st.info(content)
    elif content == OUT_OF_SCOPE_MEDICAL:
        st.warning(content)
    else:
        st.write(content)
    for i, src in enumerate(msg.get("sources") or [], start=1):
        title = src.get("title") or "Untitled source"
        with st.expander(f"[{i}] {title}", expanded=False):
            if src.get("url"):
                st.markdown(f"**URL:** {src['url']}")
            text = src.get("text", "").strip()
            if text:
                st.markdown("**Excerpt:**")
                st.write(text)


focused = st.session_state.focused_turn

if focused is not None and focused < len(user_turns):
    # Show only the focused Q&A turn
    msg_idx, user_msg = user_turns[focused]
    with st.chat_message("user"):
        st.write(user_msg["content"])
    if msg_idx + 1 < len(active()["messages"]):
        asst_msg = active()["messages"][msg_idx + 1]
        with st.chat_message("assistant"):
            render_assistant_msg(asst_msg)
            if asst_msg.get("turn_id"):
                render_feedback_buttons(
                    asst_msg["turn_id"],
                    asst_msg.get("question", user_msg["content"]),
                    asst_msg["content"],
                )
else:
    # Full conversation replay
    for msg in active()["messages"]:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                render_assistant_msg(msg)
                if msg.get("turn_id"):
                    render_feedback_buttons(
                        msg["turn_id"],
                        msg.get("question", ""),
                        msg["content"],
                    )
            else:
                st.write(msg["content"])

    prompt = st.chat_input("Ask a medical question...")
    if prompt:
        active()["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        turn_id = str(uuid.uuid4())
        with st.chat_message("assistant"):
            with st.spinner("Retrieving and generating answer..."):
                answer, sources, updated_history = answer_question(
                    prompt, top_k=top_k, history=active()["rag_history"]
                )
            render_assistant_msg({"content": answer, "sources": sources})
            render_feedback_buttons(turn_id, prompt, answer)

        active()["messages"].append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "turn_id": turn_id,
            "question": prompt,
        })
        active()["rag_history"] = updated_history
