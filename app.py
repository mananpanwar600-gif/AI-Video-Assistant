import streamlit as st
import time
from dotenv import load_dotenv
from utils.audio_processor import process_input
from src.transcriber import transcribe_all
from src.summarizer import summarize, generate_title
from src.extractor import extract_action_items, extract_key_decisions, extract_questions
from src.rag_engine import build_rag_chain, ask_question

load_dotenv()

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VISTA AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

/* ── Root Variables ── */
:root {
    --bg: #0a0a0f;
    --surface: #111118;
    --surface-2: #1a1a25;
    --border: #2a2a3a;
    --accent: #7c3aed;
    --accent-glow: #9f67ff;
    --accent-2: #06b6d4;
    --text: #e8e8f0;
    --text-muted: #7070a0;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
}

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: 'JetBrains Mono', monospace;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

.stApp {
    background: var(--bg) !important;
}

/* Animated grid background */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background-image:
        linear-gradient(rgba(124, 58, 237, 0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(124, 58, 237, 0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] * {
    color: var(--text) !important;
}

/* ── Headings ── */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Syne', sans-serif !important;
    color: var(--text) !important;
}

/* ── Hero Title ── */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2rem, 5vw, 3.5rem);
    font-weight: 800;
    line-height: 1.1;
    margin: 0;
    background: linear-gradient(135deg, #ffffff 0%, var(--accent-glow) 50%, var(--accent-2) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: var(--text-muted);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-top: 0.5rem;
}

/* ── Cards ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}

.card:hover {
    border-color: var(--accent);
}

.card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, var(--accent), var(--accent-2));
}

.card-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.card-content {
    font-size: 0.875rem;
    line-height: 1.7;
    color: var(--text);
}

/* ── Accent Badge ── */
.badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.badge-purple { background: rgba(124,58,237,0.2); color: var(--accent-glow); border: 1px solid rgba(124,58,237,0.3); }
.badge-cyan   { background: rgba(6,182,212,0.15); color: var(--accent-2);    border: 1px solid rgba(6,182,212,0.3); }
.badge-green  { background: rgba(16,185,129,0.15); color: var(--success);    border: 1px solid rgba(16,185,129,0.3); }

/* ── Input & Buttons ── */
.stTextInput > div > div > input,
.stSelectbox > div > div {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
}

.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(124,58,237,0.2) !important;
}

.stButton > button {
    background: linear-gradient(135deg, var(--accent), #5b21b6) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.875rem !important;
    letter-spacing: 0.05em !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s !important;
    text-transform: uppercase !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(124,58,237,0.4) !important;
}

/* Secondary button */
.stButton > button[kind="secondary"] {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
}

/* ── Progress / Status ── */
.status-bar {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.75rem 1rem;
    background: var(--surface-2);
    border-radius: 8px;
    margin: 0.4rem 0;
    border: 1px solid var(--border);
    font-size: 0.8rem;
}

.status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}

.dot-active   { background: var(--accent-glow); box-shadow: 0 0 8px var(--accent-glow); animation: pulse 1.5s infinite; }
.dot-done     { background: var(--success); }
.dot-pending  { background: var(--border); }

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.4; }
}

/* ── Chat ── */
.chat-container {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem;
    max-height: 420px;
    overflow-y: auto;
    margin-bottom: 1rem;
}

.chat-msg {
    margin-bottom: 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
}

.chat-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}

.chat-bubble {
    display: inline-block;
    padding: 0.6rem 1rem;
    border-radius: 10px;
    font-size: 0.85rem;
    line-height: 1.6;
    max-width: 90%;
}

.user-label  { color: var(--accent-glow); }
.bot-label   { color: var(--accent-2); }

.user-bubble { background: rgba(124,58,237,0.15); border: 1px solid rgba(124,58,237,0.25); align-self: flex-end; }
.bot-bubble  { background: rgba(6,182,212,0.1);  border: 1px solid rgba(6,182,212,0.2);   align-self: flex-start; }

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 1.5rem 0 !important;
}

/* ── Transcript box ── */
.transcript-box {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.25rem;
    font-size: 0.82rem;
    line-height: 1.8;
    max-height: 300px;
    overflow-y: auto;
    color: var(--text-muted);
    white-space: pre-wrap;
    word-break: break-word;
}

/* ── Stale Streamlit elements ── */
.stProgress > div > div > div { background: var(--accent) !important; }
.stSpinner > div { border-top-color: var(--accent) !important; }
[data-testid="stMarkdownContainer"] p { color: var(--text) !important; }
label { color: var(--text-muted) !important; font-size: 0.8rem !important; }

/* scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }

/* ── Workspace polish ── */
.block-container {
    max-width: 1440px;
    padding-top: 2.5rem;
    padding-bottom: 4rem;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #11111b 0%, #0d0d14 100%) !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding: 1.5rem 1.15rem;
}

.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.35rem;
}

.brand-icon {
    display: grid;
    place-items: center;
    width: 2.7rem;
    height: 2.7rem;
    border: 1px solid rgba(159, 103, 255, 0.35);
    border-radius: 12px;
    background: linear-gradient(145deg, rgba(124, 58, 237, 0.3), rgba(6, 182, 212, 0.12));
    font-size: 1.25rem;
}

.sidebar-caption {
    color: var(--text-muted);
    font-size: 0.7rem;
    line-height: 1.6;
    margin: 0 0 1.25rem 3.45rem;
}

.section-label {
    color: #8888ad;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    margin: 1.1rem 0 0.55rem;
    text-transform: uppercase;
}

.app-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 1rem;
    margin: 0.5rem 0 1.5rem;
}

.app-eyebrow {
    color: var(--accent-glow);
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    margin-bottom: 0.5rem;
    text-transform: uppercase;
}

.header-status {
    border: 1px solid var(--border);
    border-radius: 999px;
    background: rgba(26, 26, 37, 0.8);
    color: #b7b7cc;
    font-size: 0.7rem;
    letter-spacing: 0.04em;
    padding: 0.45rem 0.8rem;
    white-space: nowrap;
}

.card {
    background: linear-gradient(145deg, rgba(22, 22, 32, 0.98), rgba(17, 17, 24, 0.98));
    border-radius: 14px;
    padding: 1.4rem 1.5rem;
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.12);
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0.4rem;
    border-bottom: 1px solid var(--border);
}

.stTabs [data-baseweb="tab"] {
    height: 3rem;
    border-radius: 9px 9px 0 0;
    color: var(--text-muted);
    font-family: 'Syne', sans-serif;
    font-size: 0.82rem;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    color: #fff !important;
}

.stTabs [data-baseweb="tab-highlight"] {
    background-color: var(--accent-glow);
}

.stTextInput > div > div > input,
.stSelectbox [data-baseweb="select"] > div {
    min-height: 2.8rem;
}

.stButton > button {
    min-height: 2.8rem;
}

@media (max-width: 768px) {
    .block-container {
        padding: 1.25rem 1rem 2rem;
    }

    .app-header {
        flex-direction: column;
    }

    .card {
        padding: 1.1rem;
    }
}
</style>
""", unsafe_allow_html=True)

# ─── Session State Init ──────────────────────────────────────────────────────────
for key, default in {
    "result": None,
    "chat_history": [],
    "processing": False,
    "pipeline_done": False,
    "pipeline_steps": {},
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─── Helpers ────────────────────────────────────────────────────────────────────
def step_status(steps: dict, key: str) -> str:
    s = steps.get(key, "pending")
    if s == "active":  return "dot-active"
    if s == "done":    return "dot-done"
    return "dot-pending"

def render_step_bar(label: str, key: str, icon: str):
    css = step_status(st.session_state.pipeline_steps, key)
    st.markdown(f"""
    <div class="status-bar">
        <div class="status-dot {css}"></div>
        <span>{icon} {label}</span>
    </div>""", unsafe_allow_html=True)

# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="brand-icon">🎬</div>
        <div class="hero-title" style="font-size:1.35rem">VISTA<br>AI</div>
    </div>
    <div class="sidebar-caption">Your AI-powered meeting workspace</div>
    <div class="section-label">New analysis</div>
    """, unsafe_allow_html=True)

    source = st.text_input(
        "Video source",
        placeholder="Paste a YouTube URL or file path",
        help="Use a public YouTube URL or a video file accessible to this app.",
    )
    language = st.selectbox("Transcript language", ["english", "hinglish"], index=0)
    run_btn = st.button("✨  Analyse video", use_container_width=True)

    if st.session_state.pipeline_done:
        st.markdown('<div class="section-label">Analysis pipeline</div>', unsafe_allow_html=True)
        for step, icon, label in [
            ("audio",      "🔊", "Audio Processing"),
            ("transcript", "📝", "Transcription"),
            ("title",      "🏷️", "Title Generation"),
            ("summary",    "📋", "Summarisation"),
            ("extract",    "🔍", "Extraction"),
            ("rag",        "🧠", "RAG Engine"),
        ]:
            render_step_bar(label, step, icon)

# ─── Main Area ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div>
        <div class="app-eyebrow">Meeting intelligence workspace</div>
        <div class="hero-title">Turn conversations<br>into clear next steps.</div>
        <div class="hero-sub">Transcribe · Summarise · Ask your video</div>
    </div>
    <div class="header-status">● &nbsp; AI workspace</div>
</div>
""", unsafe_allow_html=True)

# ── Run Pipeline ────────────────────────────────────────────────────────────────
if run_btn:
    if not source.strip():
        st.error("Please enter a YouTube URL or file path.")
    else:
        st.session_state.pipeline_done = False
        st.session_state.result = None
        st.session_state.chat_history = []
        st.session_state.pipeline_steps = {}

        progress_placeholder = st.empty()
        progress_label = st.empty()
        progress_bar = st.progress(0)
        pipeline_stages = [
            ("audio", "Processing audio"),
            ("transcript", "Transcribing video"),
            ("title", "Generating session title"),
            ("summary", "Writing summary"),
            ("extract", "Extracting meeting insights"),
            ("rag", "Preparing Ask AI"),
        ]
        stage_positions = {
            key: position for position, (key, _) in enumerate(pipeline_stages)
        }

        def update_step(key, state):
            st.session_state.pipeline_steps[key] = state
            stage_index = stage_positions[key]
            completed = stage_index + (1 if state == "done" else 0)
            progress_bar.progress(int(completed / len(pipeline_stages) * 100))
            stage_label = pipeline_stages[stage_index][1]
            if state == "done" and completed < len(pipeline_stages):
                stage_label = pipeline_stages[completed][1]
            progress_label.caption(
                f"{stage_label} · {int(completed / len(pipeline_stages) * 100)}%"
            )

        try:
            with progress_placeholder.container():
                st.info("⚙️ Your video is being analysed. This may take a few minutes.")

            update_step("audio", "active")
            chunks = process_input(source)
            update_step("audio", "done")

            update_step("transcript", "active")
            transcript = transcribe_all(chunks, language)
            update_step("transcript", "done")

            update_step("title", "active")
            title = generate_title(transcript)
            update_step("title", "done")

            update_step("summary", "active")
            summary = summarize(transcript)
            update_step("summary", "done")

            update_step("extract", "active")
            action_items  = extract_action_items(transcript)
            decisions     = extract_key_decisions(transcript)
            questions     = extract_questions(transcript)
            update_step("extract", "done")

            update_step("rag", "active")
            rag_chain = build_rag_chain(transcript)
            update_step("rag", "done")

            st.session_state.result = {
                "title": title,
                "transcript": transcript,
                "summary": summary,
                "action_items": action_items,
                "key_decisions": decisions,
                "open_questions": questions,
                "rag_chain": rag_chain,
            }
            st.session_state.pipeline_done = True
            progress_placeholder.success("✅ Analysis complete!")
            progress_label.caption("Analysis complete · 100%")
            time.sleep(0.5)
            progress_placeholder.empty()
            progress_label.empty()
            st.rerun()

        except Exception as e:
            for k in ["audio","transcript","title","summary","extract","rag"]:
                if st.session_state.pipeline_steps.get(k) == "active":
                    st.session_state.pipeline_steps[k] = "pending"
            progress_placeholder.error(f"❌ Error: {e}")

# ── Results ──────────────────────────────────────────────────────────────────────
if st.session_state.result:
    r = st.session_state.result
    overview_tab, transcript_tab, insights_tab, chat_tab = st.tabs(
        ["✨  Overview", "📝  Transcript", "💡  Insights", "💬  Ask AI"]
    )

    with overview_tab:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">📌 Session overview</div>
            <div style="font-family:'Syne',sans-serif;font-size:1.55rem;font-weight:700;color:var(--text);margin-bottom:1.25rem">
                {r['title']}
            </div>
            <div class="card-title">📋 Summary</div>
            <div class="card-content">{r['summary']}</div>
        </div>""", unsafe_allow_html=True)

    with transcript_tab:
        st.markdown('<div class="section-label">Full transcript</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="transcript-box">{r["transcript"]}</div>',
            unsafe_allow_html=True,
        )

    with insights_tab:
        insight_columns = st.columns(3, gap="medium")
        for column, title, icon, key in [
            (insight_columns[0], "Action items", "✅", "action_items"),
            (insight_columns[1], "Key decisions", "🔑", "key_decisions"),
            (insight_columns[2], "Open questions", "❓", "open_questions"),
        ]:
            with column:
                st.markdown(f"""
                <div class="card">
                    <div class="card-title">{icon} {title}</div>
                    <div class="card-content">{r[key]}</div>
                </div>""", unsafe_allow_html=True)

    with chat_tab:
        st.markdown(
            '<div class="section-label">Ask questions about this meeting</div>',
            unsafe_allow_html=True,
        )

        if st.session_state.chat_history:
            chat_html = '<div class="chat-container">'
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    chat_html += f"""
                    <div class="chat-msg" style="align-items:flex-end">
                        <span class="chat-label user-label">You</span>
                        <div class="chat-bubble user-bubble">{msg['content']}</div>
                    </div>"""
                else:
                    chat_html += f"""
                    <div class="chat-msg" style="align-items:flex-start">
                        <span class="chat-label bot-label">🤖 Assistant</span>
                        <div class="chat-bubble bot-bubble">{msg['content']}</div>
                    </div>"""
            chat_html += '</div>'
            st.markdown(chat_html, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="card" style="text-align:center;padding:2rem">
                <div style="font-size:2rem;margin-bottom:0.5rem">💬</div>
                <div style="color:var(--text-muted);font-size:0.85rem">Ask anything about your meeting transcript</div>
            </div>""", unsafe_allow_html=True)

        chat_col1, chat_col2 = st.columns([5, 1], gap="small")
        with chat_col1:
            user_input = st.text_input(
                "Your question",
                placeholder="What were the main decisions made?",
                label_visibility="collapsed",
                key="chat_question",
            )
        with chat_col2:
            send_btn = st.button("Send →", use_container_width=True)

        if send_btn and user_input.strip():
            with st.spinner("Thinking…"):
                answer = ask_question(r["rag_chain"], user_input.strip())
            st.session_state.chat_history.append({"role": "user", "content": user_input.strip()})
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.rerun()

        if st.session_state.chat_history:
            if st.button("🗑️ Clear chat", type="secondary"):
                st.session_state.chat_history = []
                st.rerun()

else:
    # Empty state
    st.markdown("""
    <div class="card" style="text-align:center;padding:3.5rem 2rem;margin-top:1rem">
        <div style="font-size:3.25rem;margin-bottom:1rem">🎬</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.65rem;font-weight:700;color:var(--text);margin-bottom:0.65rem">
            Your next meeting insight starts here
        </div>
        <div style="color:var(--text-muted);font-size:0.85rem;max-width:460px;line-height:1.8;margin:0 auto">
            Add a YouTube link or video file in the sidebar. We’ll create a transcript,
            surface the important moments, and let you ask follow-up questions.
        </div>
        <div style="margin-top:2rem;display:flex;gap:0.75rem;flex-wrap:wrap;justify-content:center">
            <span class="badge badge-purple">01 · Transcribe</span>
            <span class="badge badge-cyan">02 · Summarise</span>
            <span class="badge badge-green">03 · Ask AI</span>
        </div>
    </div>""", unsafe_allow_html=True)