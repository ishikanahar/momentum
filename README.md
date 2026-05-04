# MOMentum 🧠

A context-aware personal AI OS that acts like your digital mom. Built at the UC San Diego × Y Combinator Hackathon.

MOMentum integrates task data, location signals, biometrics, and screen time to generate real-time proactive nudges — because most of us are operating without a real-time decision layer.

---

## Stack

- **Frontend**: Vanilla HTML/CSS/JS — interactive mobile prototype
- **Backend**: Python FastAPI
- **AI**: Claude claude-sonnet-4-5 (Anthropic)
- **RAG**: sentence-transformers + FAISS for context retrieval
- **Deploy**: Railway / Render (free tier)

---

## Project Structure

```
momentum/
├── backend/
│   ├── main.py          # FastAPI server — /chat and /nudge endpoints
│   ├── rag.py           # RAG pipeline with FAISS vector index
│   ├── context.py       # User context data + knowledge base
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   └── index.html       # Full interactive mobile UI
└── README.md
```

---

## How It Works

1. User sends a message in the chat interface
2. Frontend sends message + live user state (tasks done, location, battery, focus score) to `/chat`
3. Backend runs RAG: embeds the query, retrieves the top-4 most relevant context chunks from the FAISS index
4. Retrieved context + user state is injected into Claude's system prompt
5. Claude generates a personalized, context-aware response as MOMentum Mom
6. On page load, `/nudge` generates a proactive nudge based on current time, location, and pending tasks

---

## Setup — Local

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/momentum.git
cd momentum
```

### 2. Backend setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Add your API key

```bash
cp .env.example .env
# Edit .env and add your Anthropic API key
```

Get your key at: https://console.anthropic.com

### 4. Run the backend

```bash
uvicorn main:app --reload --port 8000
```

### 5. Open the frontend

Open `frontend/index.html` directly in your browser, or serve it:

```bash
cd ../frontend
python -m http.server 3000
# then open http://localhost:3000
```

---

## Deploy to Railway (free)

1. Push this repo to GitHub
2. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub
3. Select the `backend/` directory as the root
4. Add environment variable: `ANTHROPIC_API_KEY=your_key`
5. Railway auto-detects FastAPI and deploys
6. Copy your Railway URL and update `BACKEND_URL` in `frontend/index.html`
7. Deploy frontend to GitHub Pages or Vercel (just the HTML file)

---

## Customizing the Knowledge Base

Edit `backend/context.py` — the `KNOWLEDGE_BASE` list is what the RAG index is built from. Add anything about the user's schedule, goals, habits, and behavioral patterns. The more specific, the better the nudges.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/chat` | Send a message, get AI reply |
| POST | `/nudge` | Generate a proactive nudge |

### /chat request body
```json
{
  "message": "I just finished my homework",
  "user_state": {
    "location": "home",
    "battery": 45,
    "tasks_done": 3,
    "tasks_total": 5,
    "focus_score": 72,
    "sleep": "7h",
    "steps": 6200,
    "screen_time": "2h 10m",
    "next_deadline": "Lab report due Friday"
  }
}
```

---

## Built by

Ishika Nahar — UC San Diego × Y Combinator Hackathon
