"""
ui/app.py — Streamlit UI for Financial LLM Assistant
End-to-end financial assistant: Bytewax → QLoRA (Falcon-7B) → LangChain + Qdrant
"""
import streamlit as st
import requests
import json

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Financial LLM Assistant", page_icon="💰", layout="wide")

st.markdown("""
<style>
.module-card {
    background: #f8f9fa; border-radius: 8px;
    padding: 16px; margin: 8px 0;
    border-left: 4px solid #007bff;
}
.module-streaming { border-left-color: #28a745; }
.module-dataset   { border-left-color: #fd7e14; }
.module-training  { border-left-color: #dc3545; }
.module-bot       { border-left-color: #6f42c1; }
.chat-user {
    background: #e3f2fd; border-radius: 12px;
    padding: 10px 16px; margin: 6px 0;
    text-align: right;
}
.chat-bot {
    background: #f1f8e9; border-radius: 12px;
    padding: 10px 16px; margin: 6px 0;
}
.mode-badge {
    display: inline-block; padding: 2px 10px; border-radius: 12px;
    font-size: 0.8em; font-weight: bold;
}
.mode-live { background: #d4edda; color: #155724; }
.mode-demo { background: #fff3cd; color: #856404; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("💰 Financial LLM")
    st.caption("End-to-end Real-time Financial Assistant")
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

tab_chat, tab_arch, tab_data = st.tabs(["💬 Chat", "🏗️ Architecture", "📊 Training Data"])

# ── CHAT TAB ──────────────────────────────────────────────────────────────
with tab_chat:
    st.header("Ask Your Financial Assistant")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
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

    # Example questions
    st.divider()
    st.markdown("**💡 Try these questions:**")
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
            st.caption(f"• {ex}")

# ── ARCHITECTURE TAB ──────────────────────────────────────────────────────
with tab_arch:
    st.header("System Architecture")
    st.markdown("### End-to-End Financial Assistant Pipeline")

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
            f'<code>{mod["tech"]}</code><br>'
            f'{mod["desc"]}'
            f'</div>',
            unsafe_allow_html=True,
        )
        if i < len(modules) - 1:
            st.markdown("<center>⬇️</center>", unsafe_allow_html=True)

    st.divider()
    st.markdown("### Data Flow")
    st.code(arch.get("flow", "News → Qdrant → Q&A Generation → QLoRA Training → LangChain Bot → Gradio UI"), language=None)

    st.divider()
    st.markdown("### Prompt Template (Falcon)")
    st.code(""">>INTRODUCTION<< You are a helpful assistant, with financial expertise.
>>DOMAIN<< {user_context}
{news_context}
>>SUMMARY<< {chat_history}
>>QUESTION<< {question}
>>ANSWER<< {answer}<|endoftext|>""", language=None)

# ── TRAINING DATA TAB ────────────────────────────────────────────────────
with tab_data:
    st.header("Training Data & Fine-tuning")

    try:
        sample = requests.get(f"{API_URL}/sample-data", timeout=5).json()
    except Exception:
        sample = {}

    if sample:
        col1, col2, col3 = st.columns(3)
        col1.metric("Template", sample.get("template", "Falcon"))
        col2.metric("Training Samples", sample.get("total_samples", 0))
        col3.metric("Method", "QLoRA (4-bit)")

        st.markdown("### Sample Training Entry")
        s = sample.get("sample", {})
        st.markdown(f"**Instruction:** {s.get('instruction', '')}")
        st.markdown(f"**User Context:** {s.get('user_context', '')}")
        st.markdown(f"**News Context:** {s.get('news_context', '')}")
        st.info(f"**Answer:** {s.get('answer', '')}")

        with st.expander("📋 Raw JSON"):
            st.json(sample)

    st.divider()
    st.markdown("### Fine-tuning Configuration")
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
