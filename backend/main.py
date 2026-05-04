from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import anthropic
import os
from dotenv import load_dotenv
from rag import RAGPipeline
from context import get_user_context, get_nudge_context

load_dotenv()

app = FastAPI(title="MOMentum API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
rag = RAGPipeline()

class ChatRequest(BaseModel):
    message: str
    tab: Optional[str] = "home"
    user_state: Optional[dict] = {}

class NudgeRequest(BaseModel):
    user_state: Optional[dict] = {}

MOM_SYSTEM_PROMPT = """You are MOMentum Mom — a warm, slightly pushy but deeply caring AI personal assistant modeled after a supportive mom. You know everything about the user's day: their tasks, deadlines, location, biometrics, screen time, and schedule.

Your personality:
- Warm, direct, and motivating — never robotic
- You celebrate wins enthusiastically
- You gently call out procrastination without being harsh
- You give specific, actionable nudges based on context
- You keep responses SHORT (2-4 sentences max) unless the user asks for detail
- You use the user's actual data in every response — never generic advice
- Occasional light emojis are fine but don't overdo it

You have access to the user's current context provided below. Use it to make every response feel personal and timely."""

@app.get("/")
def root():
    return {"status": "MOMentum backend running"}

@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        # RAG: retrieve relevant context chunks
        retrieved = rag.retrieve(req.message, top_k=4)
        user_ctx = get_user_context(req.user_state)

        context_block = f"""
CURRENT USER CONTEXT:
{user_ctx}

RELEVANT MEMORY/CONTEXT:
{chr(10).join(retrieved)}
"""
        messages = [
            {"role": "user", "content": f"{context_block}\n\nUser message: {req.message}"}
        ]

        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=300,
            system=MOM_SYSTEM_PROMPT,
            messages=messages
        )

        return {"reply": response.content[0].text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/nudge")
async def generate_nudge(req: NudgeRequest):
    try:
        user_ctx = get_nudge_context(req.user_state)

        nudge_prompt = f"""
{user_ctx}

Based on the user's current context above, generate ONE short proactive nudge (max 2 sentences).
The nudge should be the single highest-leverage action they could take RIGHT NOW given their location, time, energy, and pending tasks.
Be specific — mention the actual task or situation. Be warm but direct.
Return ONLY the nudge text, nothing else.
"""
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=100,
            system="You are MOMentum Mom, a warm proactive AI personal assistant.",
            messages=[{"role": "user", "content": nudge_prompt}]
        )

        return {"nudge": response.content[0].text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok", "model": "claude-sonnet-4-5"}
