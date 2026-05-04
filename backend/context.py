from datetime import datetime

# ─────────────────────────────────────────────
# KNOWLEDGE BASE — chunks the RAG index is built from
# Edit these to reflect the actual user's life
# In production these would come from a DB / API integrations
# ─────────────────────────────────────────────
KNOWLEDGE_BASE = [
    # Tasks & deadlines
    "The user has a Canvas assignment due tonight at 11:59 PM for COGS 118A.",
    "The user has a gym session planned for 5 PM today.",
    "The user needs to submit a lab report by Friday.",
    "The user has a group project meeting tomorrow at 2 PM on Zoom.",
    "The user has 5 pending tasks: COGS homework, gym, read chapter 4, email professor, meal prep.",
    "The user completed 1 out of 5 tasks today — gym session was checked off.",

    # Schedule & calendar
    "The user has no classes on Friday afternoons — this is free deep work time.",
    "The user usually has lectures Monday, Wednesday, Friday from 9 AM to 12 PM.",
    "The user has office hours at 3 PM on Tuesdays.",

    # Location patterns
    "When the user is on the bus, they usually have 20-30 minutes of unstructured time.",
    "The user studies best at Geisel Library, usually between 2-6 PM.",
    "The user's gym is a 10 minute walk from their apartment.",
    "When the user is at home in the evening, distractions tend to spike.",

    # Biometrics & health
    "The user averaged 6.5 hours of sleep last night — slightly below their 7.5 hour goal.",
    "The user's step count today is 4,200 — below their 8,000 daily goal.",
    "The user's screen time today is 3 hours 42 minutes, above their 3 hour limit.",
    "The user's focus score today is 55 out of 100 based on task completion and screen time.",

    # Behavioral patterns
    "The user tends to procrastinate assignments until the last 2 hours before the deadline.",
    "The user is most productive in the morning between 9 AM and noon.",
    "The user checks their phone most frequently between 8-10 PM.",
    "The user responds better to nudges that are specific and time-bound rather than general reminders.",
    "The user has a 3-day productivity streak they are trying to maintain.",

    # Goals & preferences
    "The user's primary goal this semester is to improve their GPA and finish strong in COGS coursework.",
    "The user wants to build a stronger portfolio with real deployed projects.",
    "The user is applying for summer 2026 data science internships.",
    "The user prefers short, direct communication and doesn't like vague advice.",
    "The user values work-life balance and wants to make time for social activities without sacrificing deadlines.",
]


def get_user_context(user_state: dict = {}) -> str:
    """
    Formats live user state into a readable context block for the LLM.
    user_state comes from the frontend (current tab, time, any live data).
    """
    now = datetime.now()
    time_str = now.strftime("%I:%M %p")
    day_str = now.strftime("%A, %B %d")

    location = user_state.get("location", "unknown location")
    battery = user_state.get("battery", "unknown")
    current_tab = user_state.get("tab", "home")
    tasks_done = user_state.get("tasks_done", 1)
    tasks_total = user_state.get("tasks_total", 5)
    screen_time = user_state.get("screen_time", "3h 42m")
    focus_score = user_state.get("focus_score", 55)
    sleep = user_state.get("sleep", "6.5h")
    steps = user_state.get("steps", 4200)

    return f"""
Current time: {time_str} on {day_str}
Location: {location}
Battery: {battery}%
Current app section: {current_tab}

Tasks: {tasks_done}/{tasks_total} completed today
Focus score: {focus_score}/100
Screen time: {screen_time}
Sleep last night: {sleep}
Steps today: {steps}
""".strip()


def get_nudge_context(user_state: dict = {}) -> str:
    """Slimmer context for nudge generation — focused on actionable signals."""
    now = datetime.now()
    hour = now.hour
    time_str = now.strftime("%I:%M %p")

    location = user_state.get("location", "home")
    tasks_done = user_state.get("tasks_done", 1)
    tasks_total = user_state.get("tasks_total", 5)
    focus_score = user_state.get("focus_score", 55)
    next_deadline = user_state.get("next_deadline", "COGS 118A assignment due at 11:59 PM")
    battery = user_state.get("battery", 80)

    return f"""
Time: {time_str}
Location: {location}
Tasks completed: {tasks_done}/{tasks_total}
Focus score: {focus_score}/100
Next urgent deadline: {next_deadline}
Battery: {battery}%
Pending tasks: COGS homework, read chapter 4, email professor, meal prep
""".strip()
