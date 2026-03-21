"""
src/api/main.py — Lightweight demo API for Financial LLM Assistant
Provides a demo chat interface using OpenAI (when key provided) or mock responses.
"""

import os
import json
import time
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

app = FastAPI(title="Financial LLM Assistant — Demo API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Sample dataset responses (mock mode) ──────────────────────────────────
SAMPLE_RESPONSES = {
    "invest": "Based on current market conditions, diversification remains key. Consider a mix of index funds (60%), bonds (25%), and alternative investments (15%). For younger investors, a higher equity allocation may be appropriate given the longer time horizon. Always consider your risk tolerance and consult with a certified financial advisor before making investment decisions.",
    "stock": "When evaluating stocks, focus on fundamental analysis: P/E ratio, revenue growth, debt-to-equity ratio, and free cash flow. The current market shows mixed signals — tech stocks have shown strong earnings while value stocks offer attractive valuations. Dollar-cost averaging is a proven strategy to reduce timing risk.",
    "crypto": "Cryptocurrency markets remain volatile. Bitcoin and Ethereum have shown institutional adoption, but regulatory uncertainty persists. If considering crypto allocation, limit it to 5-10% of your portfolio. Always use secure wallets and reputable exchanges. Remember: never invest more than you can afford to lose.",
    "budget": "A solid budgeting framework is the 50/30/20 rule: 50% for needs (rent, utilities, groceries), 30% for wants (entertainment, dining), and 20% for savings and debt repayment. Track expenses using apps or spreadsheets, and review monthly. Building a 3-6 month emergency fund should be your first priority.",
    "retire": "For retirement planning, start with your employer's 401(k) match — it's free money. Then maximize IRA contributions ($6,500/year, or $7,500 if 50+). Target a retirement savings rate of 15-20% of gross income. The earlier you start, the more compound interest works in your favor. A common target is 25x your annual expenses.",
    "save": "High-yield savings accounts currently offer 4-5% APY, making them attractive for emergency funds. For medium-term goals (3-5 years), consider CDs or Treasury bonds. Automate your savings with direct deposit splits. The key is consistency — even small amounts compound significantly over decades.",
    "debt": "Prioritize high-interest debt (credit cards at 18-25% APR) using either the avalanche method (highest interest first) or snowball method (smallest balance first). Consider balance transfer offers for credit card debt. Student loans may qualify for income-driven repayment plans. Avoid taking on new debt while paying off existing balances.",
    "tax": "Key tax optimization strategies: maximize pre-tax retirement contributions, harvest tax losses in taxable accounts, consider Roth conversions in low-income years, and use tax-advantaged accounts (HSA, 529) when applicable. Keep detailed records of deductible expenses. Consult a CPA for complex situations.",
}

DEFAULT_RESPONSE = "That's a great financial question! While I'd need more context about your specific situation to give personalized advice, here are some general principles: 1) Build an emergency fund first, 2) Pay off high-interest debt, 3) Invest consistently for the long term, 4) Diversify your portfolio, and 5) Keep expenses below your income. For personalized advice, consider consulting with a certified financial planner."


class ChatRequest(BaseModel):
    message: str
    about_me: Optional[str] = ""
    history: Optional[List[dict]] = []


class ChatResponse(BaseModel):
    response: str
    mode: str
    latency_ms: float
    model: str


@app.get("/health")
async def health():
    has_key = bool(os.getenv("OPENAI_API_KEY"))
    return {"status": "healthy", "mode": "live" if has_key else "demo", "model": "gpt-4o-mini" if has_key else "mock"}


@app.get("/stats")
async def stats():
    return {
        "model": "Falcon-7B QLoRA (demo mode)",
        "architecture": "Bytewax → QLoRA Fine-tuning → LangChain + Qdrant RAG",
        "training_samples": 125,
        "embedding_model": "all-MiniLM-L6-v2",
        "vector_db": "Qdrant",
        "template": "Falcon (>>INTRODUCTION<< / >>DOMAIN<< / >>QUESTION<< / >>ANSWER<<)",
    }


@app.get("/architecture")
async def architecture():
    return {
        "modules": [
            {"name": "Streaming Pipeline", "tech": "Bytewax + Alpaca API", "desc": "Real-time financial news ingestion"},
            {"name": "Q&A Dataset Generator", "tech": "GPT-4 + Qdrant", "desc": "Auto-generate fine-tuning Q&A pairs from financial data"},
            {"name": "Training Pipeline", "tech": "QLoRA + Falcon-7B", "desc": "Fine-tune LLM on financial domain with 4-bit quantization"},
            {"name": "Financial Bot", "tech": "LangChain + Qdrant + Gradio", "desc": "RAG-powered chat with vector retrieval"},
        ],
        "flow": "News → Qdrant → Q&A Generation → QLoRA Training → LangChain Bot → Gradio UI",
    }


def mock_response(message: str) -> str:
    msg_lower = message.lower()
    for keyword, response in SAMPLE_RESPONSES.items():
        if keyword in msg_lower:
            return response
    return DEFAULT_RESPONSE


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    t0 = time.time()

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            import openai
            client = openai.OpenAI(api_key=api_key)

            messages = [{"role": "system", "content": f"You are a helpful financial assistant with expertise in investing, budgeting, and personal finance. The user describes themselves as: {req.about_me or 'a general user'}. Give clear, actionable financial advice. Always recommend consulting a certified financial advisor for major decisions."}]
            for h in (req.history or [])[-5:]:
                messages.append({"role": "user", "content": h.get("user", "")})
                messages.append({"role": "assistant", "content": h.get("assistant", "")})
            messages.append({"role": "user", "content": req.message})

            resp = client.chat.completions.create(model="gpt-4o-mini", messages=messages, max_tokens=512, temperature=0.7)
            answer = resp.choices[0].message.content
            mode = "live"
            model = "gpt-4o-mini"
        except Exception as e:
            answer = f"API error — falling back to demo mode. Error: {str(e)[:100]}"
            mode = "demo-fallback"
            model = "mock"
    else:
        answer = mock_response(req.message)
        mode = "demo"
        model = "mock (Falcon-7B simulation)"

    return ChatResponse(response=answer, mode=mode, latency_ms=(time.time() - t0) * 1000, model=model)


@app.get("/sample-data")
async def sample_data():
    """Return sample training data format"""
    return {
        "template": "Falcon",
        "sample": {
            "instruction": "What are the best investment strategies for a 25-year-old?",
            "user_context": "I am a 25 year old software engineer with $50k savings.",
            "news_context": "S&P 500 up 15% YTD. Fed signals rate pause.",
            "answer": "At 25, you have a significant advantage: time. Consider allocating 80% to equity index funds and 20% to bonds..."
        },
        "total_samples": 125,
    }
