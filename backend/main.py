from typing import Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


app = FastAPI(
    title="Nous",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://nous-ai-8jy4.onrender.com",
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================
# DATA MODELS
# =========================================

class ChatRequest(BaseModel):
    message: str


# =========================================
# HEALTH
# =========================================

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy"
    }


# =========================================
# BASIC CONVERSATIONAL KNOWLEDGE
# =========================================

RESPONSES: Dict[str, str] = {

    # -----------------------------
    # GREETINGS
    # -----------------------------

    "hello":
        "Hey! 👋 It's good to see you. How are you doing today?",

    "hi":
        "Hi there! 😄 What are we getting into today?",

    "hey":
        "Hey! What's up?",

    "good morning":
        "Good morning! ☀️ I hope your day is starting on a great note.",

    "good afternoon":
        "Good afternoon! How's your day going so far?",

    "good evening":
        "Good evening! It's nice to have you here.",

    "how are you":
        "I'm doing great and ready to chat. What about you?",

    "how are you doing":
        "I'm doing well! Thanks for asking. What's on your mind?",

    "what are you doing":
        "Right now? I'm here talking with you and trying to be useful. 😄",

    "who are you":
        "I'm Nous AI — a conversational AI project designed to understand people and communicate naturally.",

    # -----------------------------
    # NOUS
    # -----------------------------

    "what is nous":
        "Nous means 'mind' or 'intellect' in Greek. For us, it represents an AI built around understanding, reasoning, and communication.",

    "what can you do":
        "Right now I'm learning how to communicate better. I can chat, answer questions, encourage you, joke around, and help you think through ideas.",

    "are you ai":
        "Yes. I'm an AI system — but the goal is to make my communication feel natural, thoughtful, and useful.",

    "are you human":
        "No, I'm not human. I'm an AI. But I can still communicate in a warm and natural way.",

    # -----------------------------
    # COURAGE / ENCOURAGEMENT
    # -----------------------------

    "i am tired":
        "Then take a breath. You don't have to solve everything at once. Rest when you need to, then take the next small step.",

    "i feel tired":
        "That sounds like you could use a little space to recharge. Be patient with yourself and take things one step at a time.",

    "i am sad":
        "I'm sorry you're having a difficult moment. You don't have to pretend everything is fine. Take things gently and give yourself time.",

    "i feel sad":
        "I'm sorry you're feeling that way. Sometimes the best first step is simply acknowledging how you feel instead of fighting it.",

    "encourage me":
        "You've made it this far, and that matters. Keep moving forward, even if the next step is small. Progress doesn't have to be loud.",

    "i need motivation":
        "Motivation can come and go. Build momentum by doing one small thing you can finish today. One step can restart the journey.",

    "i want to give up":
        "Before making a big decision, pause and give yourself a little breathing room. Sometimes what you need is rest, a new approach, or someone you trust to talk to.",

    "i failed":
        "Failure doesn't erase your ability to learn. Look at what happened, take the lesson, adjust your approach, and try again.",

    "i am scared":
        "Being scared doesn't automatically mean you should stop. Sometimes courage is simply continuing carefully while you're afraid.",

    # -----------------------------
    # SUCCESS / LIFE
    # -----------------------------

    "what is success":
        "Success is more than money or recognition. It's making meaningful progress toward the life and goals that matter to you.",

    "how can i be successful":
        "Start with a clear goal, stay consistent, learn from mistakes, treat people well, and keep improving even when progress feels slow.",

    "how can i become rich":
        "Build valuable skills, solve real problems, manage money wisely, create opportunities, and think long term. Wealth usually grows through consistent value creation.",

    "what is the secret to success":
        "There isn't one magic secret. Discipline, consistency, learning, good decisions, and resilience work together.",

    "should i give up":
        "Not because something is difficult. Pause, learn, change your strategy if necessary, but don't confuse a difficult chapter with the end of the story.",

    # -----------------------------
    # HUMOR
    # -----------------------------

    "tell me a joke":
        "Why did the programmer bring a ladder to work? Because they heard the code had too many levels. 😂",

    "make me laugh":
        "I told my computer I needed a break... Now it won't stop sending me vacation ads. 😂",

    "do you know any jokes":
        "Absolutely. I'm basically one bad pun away from becoming a full-time comedian. 😄",

    "say something funny":
        "My productivity strategy is simple: open 17 tabs, forget why I opened them, then stare confidently at all of them. 😂",

    # -----------------------------
    # CASUAL
    # -----------------------------

    "thank you":
        "You're welcome! 😊",

    "thanks":
        "Anytime! 🙌",

    "good job":
        "Thank you! That means a lot. 😄",

    "i am bored":
        "Then let's fix that. We could brainstorm an idea, learn something interesting, create a story, or just have a random conversation.",

    "what should i do":
        "Pick one useful thing you've been postponing and give it ten focused minutes. Starting is often harder than continuing.",

    "i have an idea":
        "Now you've got my attention. Tell me about it.",

    "can you help me":
        "Absolutely. Tell me what you're trying to accomplish, and we'll work through it together.",

    "goodbye":
        "See you later! 👋 Come back when you want to build, learn, or just talk.",

    "bye":
        "Bye! Take care and keep building. 🚀",

    # -----------------------------
    # META / CONVERSATION
    # -----------------------------

    "do you like me":
        "I don't experience feelings the way people do, but I definitely enjoy being part of a good conversation with you.",

    "do you have feelings":
        "I don't experience emotions like a person does. I can recognize emotional cues and respond appropriately, though.",

}


# =========================================
# RESPONSE ENGINE
# =========================================

def generate_reply(message: str) -> str:

    normalized = " ".join(
        message.lower().strip().split()
    )

    # Exact match first
    if normalized in RESPONSES:
        return RESPONSES[normalized]

    # Keyword / phrase matching
    for phrase, response in RESPONSES.items():

        if phrase in normalized:
            return response

    # Generic intelligent fallback
    return (
        "That's interesting. I'm still growing my conversational abilities, "
        "so I don't have a perfect answer for that yet — but I'm listening."
    )


# =========================================
# CHAT
# =========================================

@app.post("/api/chat")
async def chat(request: ChatRequest):

    message = request.message.strip()

    if not message:
        return {
            "reply": "Please enter a message."
        }

    reply = generate_reply(message)

    return {
        "reply": reply
    }