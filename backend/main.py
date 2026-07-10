from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import json
from datetime import datetime
from urllib.error import URLError
from urllib.request import Request, urlopen
try:
    import anthropic
except ImportError:
    anthropic = None

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        return None

try:
    from rag import RAGPipeline
except ImportError:
    RAGPipeline = None

try:
    from context import get_user_context, get_nudge_context
except ImportError:
    def get_user_context(user_state):
        return str(user_state or {})

    def get_nudge_context(user_state):
        return str(user_state or {})

load_dotenv()

app = FastAPI(title="MOMentum API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = (
    anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    if anthropic and os.getenv("ANTHROPIC_API_KEY")
    else None
)
rag = RAGPipeline() if RAGPipeline else None

class ChatRequest(BaseModel):
    message: str
    tab: Optional[str] = "home"
    user_state: Optional[dict] = {}

class NudgeRequest(BaseModel):
    user_state: Optional[dict] = {}


def fetch_json(url: str):
    request = Request(url, headers={"User-Agent": "MOMentum/1.0"})
    with urlopen(request, timeout=8) as response:
        return json.loads(response.read().decode("utf-8"))


WEATHER_CODES = {
    0: "Clear sky",
    1: "Mostly clear",
    2: "Partly cloudy",
    3: "Cloudy",
    45: "Foggy",
    48: "Rime fog",
    51: "Light drizzle",
    61: "Light rain",
    63: "Rain",
    65: "Heavy rain",
    71: "Snow",
    80: "Rain showers",
    95: "Thunderstorm",
}

CONSTELLATIONS_BY_MONTH = {
    1: ("Orion", "Bright winter hunter vibes. Look south after sunset."),
    2: ("Canis Major", "Sirius is the show-off star tonight."),
    3: ("Leo", "Big main-character energy rising in the east."),
    4: ("Virgo", "Soft spring sky, very organized-study-session coded."),
    5: ("Boötes", "Follow the arc to Arcturus if the sky is clear."),
    6: ("Scorpius", "Low on the southern horizon, dramatic and spicy."),
    7: ("Cygnus", "The Northern Cross flies overhead in summer."),
    8: ("Lyra", "Vega is bright, pretty, and impossible to miss."),
    9: ("Pegasus", "The Great Square is your autumn sky anchor."),
    10: ("Andromeda", "Look northeast for mythological princess energy."),
    11: ("Cassiopeia", "A sparkling W in the northern sky."),
    12: ("Taurus", "Aldebaran and the Pleiades are doing the most."),
}

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


@app.get("/live/weather")
def live_weather(lat: float = 37.7749, lon: float = -122.4194):
    try:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            "&current=temperature_2m,apparent_temperature,weather_code,wind_speed_10m"
            "&daily=sunrise,sunset&temperature_unit=fahrenheit&wind_speed_unit=mph&timezone=auto"
        )
        data = fetch_json(url)
        current = data["current"]
        daily = data.get("daily", {})
        code = current.get("weather_code", 0)
        return {
            "source": "Open-Meteo",
            "temperature": round(current["temperature_2m"]),
            "feels_like": round(current["apparent_temperature"]),
            "condition": WEATHER_CODES.get(code, "Mystery weather"),
            "wind_mph": round(current.get("wind_speed_10m", 0)),
            "sunrise": (daily.get("sunrise") or ["--"])[0][-5:],
            "sunset": (daily.get("sunset") or ["--"])[0][-5:],
        }
    except (KeyError, URLError, TimeoutError, ValueError):
        return {
            "source": "fallback",
            "temperature": 64,
            "feels_like": 64,
            "condition": "Weather API resting",
            "wind_mph": 6,
            "sunrise": "06:02",
            "sunset": "20:29",
        }


@app.get("/live/news")
def live_news():
    try:
        data = fetch_json("https://hn.algolia.com/api/v1/search_by_date?tags=front_page")
        stories = []
        for story in data.get("hits", [])[:3]:
            title = story.get("title")
            if title:
                stories.append({
                    "title": title,
                    "source": "Hacker News",
                    "url": story.get("url") or f"https://news.ycombinator.com/item?id={story.get('objectID')}",
                })
        if stories:
            return {"source": "HN Algolia API", "stories": stories}
    except (KeyError, URLError, TimeoutError, ValueError):
        pass

    return {
        "source": "fallback",
        "stories": [
            {"title": "News API is napping, but your tasks are still awake.", "source": "MOMentum", "url": ""},
            {"title": "Tiny progress beats dramatic panic every time.", "source": "MOMentum", "url": ""},
            {"title": "Charge phone, check deadline, do the next obvious thing.", "source": "MOMentum", "url": ""},
        ],
    }


@app.get("/live/sky")
def live_sky():
    now = datetime.now()
    name, note = CONSTELLATIONS_BY_MONTH[now.month]
    return {
        "constellation": name,
        "note": note,
        "best_time": "After 9 PM",
        "moon": "Waxing gibbous-ish energy",
    }

@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        if not client or not rag:
            return {
                "reply": "Demo mode is on, sweetie — the AI brain is offline, but MOMentum still says: pick the smallest task and knock it out first."
            }

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
        if not client:
            return {
                "nudge": "Quick mom nudge: charge your phone, finish the overdue email, then protect that 4–6 PM CS 301 block."
            }

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
