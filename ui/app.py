"""
ui/app.py — Streamlit UI for Financial LLM Assistant
Premium glassmorphism design
End-to-end financial assistant: Bytewax → QLoRA (Falcon-7B) → LangChain + Qdrant
"""
import streamlit as st
import requests
import json

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Financial LLM Assistant", page_icon="💰", layout="wide")

# ── Premium CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

.stApp {
    background: linear-gradient(135deg, #0a0e17 0%, #1a1c2e 40%, #0d1520 100%);
}
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 2rem; }
header[data-testid="stHeader"] { background: transparent; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0e17 0%, #080b12 100%) !important;
    border-right: 1px solid rgba(245,158,11,0.1);
}
section[data-testid="stSidebar"] .stMarkdown { color: #fde68a; }

.hero {
    text-align: center; padding: 40px 20px 20px 20px;
}
.hero h1 {
    font-size: 3rem; font-weight: 800;
    background: linear-gradient(135deg, #f59e0b, #ef4444, #f97316);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 8px; letter-spacing: -1px;
}
.hero p { color: #92785a; font-size: 1.15rem; font-weight: 300; }

.glass {
    background: rgba(255,255,255,0.03);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(245,158,11,0.1);
    border-radius: 16px; padding: 24px; margin: 12px 0; color: #fde68a;
}

.metric-glass {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(245,158,11,0.1);
    border-radius: 14px; padding: 20px; text-align: center;
}
.metric-glass .value {
    font-size: 1.8rem; font-weight: 700;
    background: linear-gradient(135deg, #f59e0b, #ef4444);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.metric-glass .label { color: #92785a; font-size: 0.85rem; margin-top: 4px; }

.chat-user {
    background: rgba(245,158,11,0.08);
    border: 1px solid rgba(245,158,11,0.15);
    border-radius: 16px 16px 4px 16px;
    padding: 14px 20px; margin: 10px 0 10px 40px;
    color: #fde68a; text-align: right;
}
.chat-bot {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px 16px 16px 4px;
    padding: 14px 20px; margin: 10px 40px 10px 0;
    color: #e5e7eb; line-height: 1.7;
}

.mode-badge {
    display: inline-block; padding: 4px 14px; border-radius: 20px;
    font-size: 0.8em; font-weight: 700; letter-spacing: 0.5px;
}
.mode-live {
    background: rgba(52,211,153,0.15); color: #34d399;
    border: 1px solid rgba(52,211,153,0.3);
}
.mode-demo {
    background: rgba(245,158,11,0.15); color: #fbbf24;
    border: 1px solid rgba(245,158,11,0.3);
}

.module-card {
    background: rgba(255,255,255,0.03);
    border-radius: 12px; padding: 20px; margin: 10px 0;
    border-left: 4px solid rgba(245,158,11,0.4);
    color: #e5e7eb; transition: transform 0.2s ease;
}
.module-card:hover { transform: translateX(4px); }
.module-streaming { border-left-color: #34d399; }
.module-dataset   { border-left-color: #f59e0b; }
.module-training  { border-left-color: #ef4444; }
.module-bot       { border-left-color: #8b5cf6; }

.pipeline-arrow {
    text-align: center; color: #92785a; font-size: 1.5rem;
    padding: 4px 0;
}

.example-q {
    background: rgba(245,158,11,0.05);
    border: 1px solid rgba(245,158,11,0.12);
    border-radius: 10px; padding: 10px 14px; margin: 4px 0;
    color: #fbbf24; font-size: 0.9em;
    transition: all 0.2s ease;
}
.example-q:hover {
    background: rgba(245,158,11,0.1);
    transform: translateX(4px);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.02); border-radius: 12px; padding: 4px; gap: 4px;
}
.stTabs [data-baseweb="tab"] { border-radius: 8px; color: #92785a; font-weight: 500; }
.stTabs [aria-selected="true"] {
    background: rgba(245,158,11,0.15) !important; color: #fbbf24 !important;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #f59e0b, #ef4444) !important;
    border: none !important; border-radius: 10px !important;
    color: #0a0e17 !important; font-weight: 700 !important;
    transition: all 0.3s ease !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(245,158,11,0.3) !important;
}

.stTextArea textarea, .stTextInput input {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(245,158,11,0.12) !important;
    border-radius: 10px !important; color: #fde68a !important;
}

.streamlit-expanderHeader {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 8px !important; color: #92785a !important;
}

/* Streamlit overrides */
footer, #MainMenu, .stDeployButton, div[data-testid="stDecoration"] { display: none !important; }
[data-testid="stAppViewContainer"] { background: transparent !important; }
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebarContent"] { background: transparent !important; }
[data-testid="stBottomBlockContainer"] { background: transparent !important; }
div[data-testid="stMetricValue"] > div { color: #fde68a !important; }
div[data-testid="stMetricDelta"] { color: #92785a !important; }
div[data-testid="stMetricLabel"] { color: #92785a !important; }
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>💰 Financial LLM Assistant</h1>
    <p>End-to-end real-time financial AI — Bytewax streaming → QLoRA fine-tuning → LangChain + Qdrant RAG</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 💰 System Status")
    st.caption("Financial AI pipeline monitor")
    st.divider()

    try:
        h = requests.get(f"{API_URL}/health", timeout=5).json()
        st.success("🟢 API Connected")
        mode = h.get("mode", "demo")
        if mode == "live":
            st.markdown('<span class="mode-badge mode-live">🔑 LIVE MODE</span>', unsafe_allow_html=True)
            st.caption("Using OpenAI GPT-4o-mini")
        else:
            st.markdown('<span class="mode-badge mode-demo">🎭 DEMO MODE</span>', unsafe_allow_html=True)
            st.caption("Set OPENAI_API_KEY for live mode")

        stats = requests.get(f"{API_URL}/stats", timeout=5).json()
        st.metric("Training Samples", stats.get("training_samples", 0))
        st.metric("Model", "Falcon-7B QLoRA")
        st.metric("Vector DB", stats.get("vector_db", "Qdrant"))
    except Exception:
        st.error("🔴 API Offline — start the backend first")
        st.stop()

    st.divider()
    about_me = st.text_area(
        "About Me (context for the assistant)",
        value="I am a 28 year old software engineer looking to grow my savings.",
        height=80,
    )
    st.divider()
    st.caption("Built with OpenAI · Falcon-7B · Qdrant · Bytewax")

    st.divider()
    st.markdown("""
    <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);border-radius:12px;padding:16px;margin-top:8px;">
        <div style="font-weight:700;font-size:1rem;color:#fde68a;margin-bottom:6px;">👨‍💻 Naresh</div>
        <div style="font-size:0.8rem;color:#92785a;line-height:1.6;">
            GenAI Engineer · Full-Stack ML<br>
            <b style="color:#f59e0b;">Skills:</b> LLMs · RAG · Fine-tuning · LangChain · FastAPI · Docker<br>
            <b style="color:#f59e0b;">Stack:</b> Python · OpenAI · FAISS · Qdrant · Streamlit
        </div>
        <div style="margin-top:10px;font-size:0.75rem;">
            <a href="https://github.com/Naresh1401" style="color:#f59e0b;text-decoration:none;">GitHub</a>
        </div>
        <details style="margin-top:10px;">
            <summary style="color:#92785a;font-size:0.8rem;cursor:pointer;">🚀 More Projects</summary>
            <div style="font-size:0.75rem;color:#92785a;margin-top:8px;line-height:1.8;">
                <a href="https://llm-safety-guardrails.onrender.com" style="color:#7c3aed;text-decoration:none;">LLM Safety Guardrails</a><br>
                <a href="https://text-to-sql-agent-2za9.onrender.com" style="color:#64ffda;text-decoration:none;">Text-to-SQL Agent</a><br>
                <a href="https://intelligent-document-processing-qyo8.onrender.com" style="color:#a855f7;text-decoration:none;">Intelligent Doc Processing</a><br>
                <a href="https://meeting-intelligent-platform.onrender.com" style="color:#38bdf8;text-decoration:none;">Meeting Intelligence</a><br>
                <a href="https://ai-code-review-agent-bon1.onrender.com" style="color:#10b981;text-decoration:none;">AI Code Review Agent</a><br>
                <a href="https://enterprise-rag-pipeline.onrender.com" style="color:#818cf8;text-decoration:none;">Enterprise RAG Pipeline</a>
            </div>
        </details>
    </div>
    """, unsafe_allow_html=True)

tab_chat, tab_arch, tab_data = st.tabs(["💬 Chat", "🏗️ Architecture", "📊 Training Data"])

# ── CHAT TAB ──────────────────────────────────────────────────────────────
with tab_chat:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bot">🤖 {msg["content"]}</div>', unsafe_allow_html=True)

    question = st.text_input(
        "Ask a financial question",
        placeholder="What's the best way to invest $10,000 for retirement?",
        key="chat_input",
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        send = st.button("📨 Send", type="primary", use_container_width=True)
    with col2:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

    if send and question.strip():
        st.session_state.chat_history.append({"role": "user", "content": question})

        with st.spinner("Thinking..."):
            history_payload = []
            for i in range(0, len(st.session_state.chat_history) - 1, 2):
                if i + 1 < len(st.session_state.chat_history):
                    history_payload.append({
                        "user": st.session_state.chat_history[i]["content"],
                        "assistant": st.session_state.chat_history[i + 1]["content"],
                    })

            resp = requests.post(f"{API_URL}/chat", json={
                "message": question,
                "about_me": about_me,
                "history": history_payload,
            }, timeout=60)

        if resp.ok:
            data = resp.json()
            answer = data["response"]
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.caption(f"⏱ {data['latency_ms']:.0f}ms | Model: {data['model']} | Mode: {data['mode']}")
            st.rerun()
        else:
            st.error(f"Error: {resp.text}")

    st.markdown("---")
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("#### 💡 Example Questions")
    examples = [
        "What's your opinion on investing in startup companies?",
        "How should I budget my $80k salary?",
        "Is cryptocurrency a good investment right now?",
        "How do I start saving for retirement at 25?",
        "What's the best strategy to pay off student debt?",
        "How can I optimize my taxes as a freelancer?",
    ]
    cols = st.columns(2)
    for i, ex in enumerate(examples):
        with cols[i % 2]:
            st.markdown(f'<div class="example-q">• {ex}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── ARCHITECTURE TAB ──────────────────────────────────────────────────────
with tab_arch:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("#### 🏗️ End-to-End Financial Assistant Pipeline")
    st.markdown('</div>', unsafe_allow_html=True)

    try:
        arch = requests.get(f"{API_URL}/architecture", timeout=5).json()
        modules = arch.get("modules", [])
    except Exception:
        modules = []

    colors = ["module-streaming", "module-dataset", "module-training", "module-bot"]
    icons = ["📡", "📝", "🧠", "🤖"]

    for i, mod in enumerate(modules):
        cls = colors[i] if i < len(colors) else ""
        icon = icons[i] if i < len(icons) else "📦"
        st.markdown(
            f'<div class="module-card {cls}">'
            f'<b>{icon} {mod["name"]}</b><br>'
            f'<code style="color:#f59e0b">{mod["tech"]}</code><br>'
            f'<span style="color:#9ca3af">{mod["desc"]}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if i < len(modules) - 1:
            st.markdown('<div class="pipeline-arrow">⬇️</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("#### Data Flow")
    st.code(arch.get("flow", "News → Qdrant → Q&A Generation → QLoRA Training → LangChain Bot → Gradio UI") if modules else "News → Qdrant → Q&A Generation → QLoRA Training → LangChain Bot → UI", language=None)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("#### Prompt Template (Falcon)")
    st.code(""">>INTRODUCTION<< You are a helpful assistant, with financial expertise.
>>DOMAIN<< {user_context}
{news_context}
>>SUMMARY<< {chat_history}
>>QUESTION<< {question}
>>ANSWER<< {answer}<|endoftext|>""", language=None)
    st.markdown('</div>', unsafe_allow_html=True)

# ── TRAINING DATA TAB ────────────────────────────────────────────────────
with tab_data:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("#### 📊 Training Data & Fine-tuning")
    st.markdown('</div>', unsafe_allow_html=True)

    try:
        sample = requests.get(f"{API_URL}/sample-data", timeout=5).json()
    except Exception:
        sample = {}

    if sample:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                f'<div class="metric-glass"><div class="value" style="-webkit-text-fill-color:unset;'
                f'color:#fbbf24;font-size:1.3rem">{sample.get("template", "Falcon")}</div>'
                f'<div class="label">Template</div></div>',
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                f'<div class="metric-glass"><div class="value">{sample.get("total_samples", 0)}</div>'
                f'<div class="label">Training Samples</div></div>',
                unsafe_allow_html=True
            )
        with col3:
            st.markdown(
                '<div class="metric-glass"><div class="value" style="-webkit-text-fill-color:unset;'
                'color:#ef4444;font-size:1.3rem">QLoRA 4-bit</div>'
                '<div class="label">Method</div></div>',
                unsafe_allow_html=True
            )

        s = sample.get("sample", {})
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown("**Sample Training Entry**")
        st.markdown(f"**Instruction:** {s.get('instruction', '')}")
        st.markdown(f"**User Context:** {s.get('user_context', '')}")
        st.markdown(f"**News Context:** {s.get('news_context', '')}")
        st.info(f"**Answer:** {s.get('answer', '')}")
        st.markdown('</div>', unsafe_allow_html=True)

        with st.expander("📋 Raw JSON"):
            st.json(sample)

    st.markdown("---")
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("#### Fine-tuning Configuration")
    config = {
        "base_model": "tiiuae/falcon-7b",
        "method": "QLoRA (4-bit quantization)",
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "learning_rate": 2e-4,
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "max_steps": 500,
        "warmup_steps": 50,
    }
    st.json(config)
    st.markdown('</div>', unsafe_allow_html=True)
