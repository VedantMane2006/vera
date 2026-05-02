# Magicpin AI Bot Challenge 🚀

An autonomous, data-driven Merchant Growth Assistant designed to evaluate signals, draft personalized WhatsApp messages, and handle multi-turn conversations with local businesses.

Built for the **Magicpin AI Challenge**.

---

## 🧠 Architecture & Approach

This bot ("Vera") acts as an elite growth expert. The core philosophy is **Specificity and Compulsion**: every message uses exact numerical data (views, CTR, performance deltas) and frames actions around quantifiable loss aversion to drive engagement.

### ⚡ Multi-Tier LLM Engine
To ensure 100% reliability and fast execution during evaluation, the bot employs a cascading API strategy using OpenAI-compatible endpoints:
1. **Primary**: NVIDIA NIM (`meta/llama-3.1-70b-instruct`) — Extremely fast, high intelligence.
2. **Fallback 1**: OpenRouter (`meta-llama/llama-3.3-70b-instruct:free`) — Zero-cost safety net.
3. **Fallback 2**: Groq (`llama-3.3-70b-versatile`) — High-speed secondary safety net.

### ⚙️ Core Features
*   **Contextual Memory System**: Persists category rules, merchant states, and conversation history using an in-memory dictionary (with local JSON dumps for debugging).
*   **Rubric-Aligned Prompts**: System prompts are strictly engineered to maximize scores across all 5 evaluation dimensions: Specificity, Category Fit, Merchant Fit, Decision Quality, and Engagement Compulsion.
*   **Robust JSON Extraction**: Employs regex-based sanitization to guarantee the LLM returns parsable `ActionOutput` objects, regardless of conversational "chatter".
*   **Spam Loop Prevention**: Automatically detects generic auto-replies and terminates the conversation gracefully to protect merchant trust.

---

## 🛠️ Local Development & Testing

### 1. Requirements
*   Docker (Optional but recommended)
*   Python 3.10+
*   API Keys (NVIDIA NIM, OpenRouter, or Groq)

### 2. Environment Variables
Create a `.env` file in the root directory:
```env
NVIDIA_API_KEY=nvapi-...
OPENROUTER_API_KEY=sk-or-v1-...
GROQ_API_KEY=gsk_...
```

### 3. Running Locally (Without Docker)
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the FastAPI server
uvicorn bot.main:app --host 0.0.0.0 --port 8080
```

### 4. Running the Judge Simulator
In a separate terminal, test the bot against the provided Magicpin evaluation suite:
```bash
export BOT_URL="http://localhost:8080"
python judge_simulator.py
```

---

## 🚀 Deployment (Render)

This application is container-ready and can be deployed in minutes on [Render](https://render.com).

1. Push this repository to GitHub.
2. Create a new **Web Service** on Render connected to the repository.
3. Select **Docker** as the Runtime environment.
4. Add your API keys (`NVIDIA_API_KEY`, etc.) in the **Environment Variables** section.
5. Deploy! (The `Dockerfile` exposes port `8080`).

---

## 👤 Author
**Vedant Mane**
