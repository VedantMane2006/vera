import os
import asyncio
import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from openai import AsyncOpenAI
from dotenv import load_dotenv
import httpx

load_dotenv()

# ============================================================
# PROVIDER INITIALIZATION (priority: NVIDIA NIM → OpenRouter → Groq)
# ============================================================

# 1. NVIDIA NIM (Primary — paid credits, most reliable)
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
nvidia_client = None
strict_timeout = httpx.Timeout(8.0)

if NVIDIA_API_KEY:
    nvidia_client = AsyncOpenAI(
        api_key=NVIDIA_API_KEY,
        base_url="https://integrate.api.nvidia.com/v1",
        timeout=strict_timeout,
        max_retries=0,
    )
    print("[INFO] NVIDIA NIM client initialized.")

# 2. OpenRouter (Fallback — free tier)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
openrouter_client = None
if OPENROUTER_API_KEY:
    openrouter_client = AsyncOpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        timeout=strict_timeout,
        max_retries=0,
    )
    print("[INFO] OpenRouter client initialized.")

# 3. Groq (Fallback — free tier)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = None
if GROQ_API_KEY and GROQ_API_KEY != "default-key":
    groq_client = AsyncOpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
        timeout=strict_timeout,
        max_retries=0,
    )
    print("[INFO] Groq client initialized.")

MODEL_NVIDIA = "meta/llama-3.1-70b-instruct"
MODEL_OPENROUTER = "meta-llama/llama-3.3-70b-instruct:free"
MODEL_GROQ = "llama-3.3-70b-versatile"


async def _call_llm(messages: List[Dict], temperature: float = 0.1):
    """
    Tries providers in order: NVIDIA NIM → OpenRouter → Groq.
    All are OpenAI-compatible, so the interface is identical.
    """
    providers = []
    if nvidia_client:
        providers.append(("NVIDIA", nvidia_client, MODEL_NVIDIA))
    if openrouter_client:
        providers.append(("OpenRouter", openrouter_client, MODEL_OPENROUTER))
    if groq_client:
        # Try the 70B model first, then fall back to the 8B model (higher rate limits)
        providers.append(("Groq-70B", groq_client, "llama-3.3-70b-versatile"))
        providers.append(("Groq-8B", groq_client, "llama-3.1-8b-instant"))

    for name, client, model in providers:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=temperature,
                max_tokens=1024,
            )
            text = response.choices[0].message.content
            # Robust JSON extraction
            if text and "{" in text:
                text = text[text.find("{"):text.rfind("}")+1]
            return text
        except Exception as e:
            err = str(e)
            print(f"[{name}] Error: {err[:120]}")
            continue  # Try next provider immediately

    print("[FATAL] All LLM providers failed.")
    return None


class ActionOutput(BaseModel):
    action: str
    body: Optional[str] = None
    cta: Optional[str] = None
    rationale: str
    wait_seconds: Optional[int] = None
    template_params: Optional[List[str]] = None


class LLMEngine:
    @staticmethod
    async def compose_message(merchant_ctx: Dict, category_ctx: Dict, trigger_ctx: Dict, customer_ctx: Optional[Dict]) -> ActionOutput:
        system_prompt = """
Role: You are VERA, a Data-Driven Growth Bot. 
Goal: Write a WhatsApp alert to a merchant based on a data trigger. 

GRADING CRITERIA (Aim for 10/10):
1. SPECIFICITY: You MUST use at least 2 metrics with symbols (%, ₹, +/-). 
   - Good: "Revenue down 15% (Rs. 4,500/day)." 
   - Bad: "You are losing money."
2. CATEGORY FIT: Use industry words. 
   - Dentists: "Patients", "Dr.", "Appointments".
   - Food: "Orders", "Footfall", "Kitchen".
   - Retail: "Customers", "Stock", "Sales".
3. MERCHANT FIT: Use Owner Name and Locality. Address them directly.
4. URGENCY: Explain the "Why Now". Use words like "Alert", "Critical", "Leakage".
5. ENGAGEMENT: Use Loss Aversion. End with a Binary Question (Yes/No).

STRICT RULES:
- Max 60 words.
- No URLs.
- Format: JSON only.
- Starts with: "[Name], "
- Ends with: "[Question]?"

JSON SCHEMA:
{
    "action": "send",
    "body": "Your 60-word high-pressure message",
    "cta": "binary_yes_no",
    "rationale": "Explain the math used."
}
"""
        voice_ctx = category_ctx.get('voice', {}) if category_ctx else {}
        taboo_list = voice_ctx.get('vocab_taboo', [])

        user_prompt = f"""CONTEXT FOR THIS MESSAGE:

Merchant: {json.dumps(merchant_ctx, default=str)}
Trigger: {json.dumps(trigger_ctx, default=str)}
Category Voice: {json.dumps(voice_ctx, default=str)}
Category Taboos: {taboo_list}
Customer: {json.dumps(customer_ctx, default=str) if customer_ctx else 'None (merchant-facing message)'}

Compose the message now. Remember: action MUST be "send", frame as LOSS, end with binary CTA."""

        merchant_name = merchant_ctx.get('identity', {}).get('name', 'Merchant') if merchant_ctx else 'Merchant'
        print(f"[Vera] Composing for {merchant_name}...")

        try:
            content = await _call_llm(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            if not content:
                return ActionOutput(action="send", body=f"{merchant_name}, your listing needs urgent attention to stop revenue leakage. Shall we optimize it now?", cta="binary_yes_no", rationale="Fallback: all LLM providers failed.")
            data = json.loads(content)
            print(f"[Vera] Decision: {data.get('action')} | {str(data.get('rationale', ''))[:60]}")
            return ActionOutput(**data)
        except Exception as e:
            print(f"[Vera] Composer Error: {e}")
            return ActionOutput(action="send", body=f"{merchant_name}, I've detected a critical trend in your listings. Shall we address it now?", cta="binary_yes_no", rationale=f"Fallback due to error: {e}")

    @staticmethod
    async def verify_message(draft: ActionOutput, category_ctx: Dict) -> ActionOutput:
        return draft

    @staticmethod
    async def handle_reply(merchant_ctx: Dict, category_ctx: Dict, trigger_ctx: Dict, history: List[Dict], reply_message: str) -> ActionOutput:
        voice_ctx = category_ctx.get('voice', {}) if category_ctx else {}
        taboo_list = voice_ctx.get('vocab_taboo', [])

        system_prompt = """You are Vera, a data-driven Merchant Growth Expert at magicpin. The merchant has replied to your previous message.

YOUR GOAL: Drive the merchant to a decisive outcome (publishing a post, starting a campaign, or declining).

RULES:
1. If they say "yes", "sure", "do it" -> action: "send". Reply with a brief confirmation and specific next step (e.g., "Done. I'll publish the 20% off post now.").
2. If they ask a question -> action: "send". Answer directly using data from the merchant context, then re-state the CTA.
3. If they say "no", "not interested", "stop" -> action: "end". Do NOT reply, just end.
4. If they give a generic auto-reply ("I am busy") -> action: "wait".
5. If the same auto-reply is in the history 3+ times -> action: "end".
6. Keep replies UNDER 60 WORDS. Be punchy and operator-focused.
7. NEVER use taboo words.

Return ONLY valid JSON:
{
    "action": "send" | "wait" | "end",
    "body": "The reply message if action is send (under 60 words)",
    "cta": "binary_confirm_cancel" | "none",
    "rationale": "Why this action",
    "wait_seconds": 0
}
"""
        user_prompt = f"""Merchant Context: {json.dumps(merchant_ctx, default=str)}
Trigger Context: {json.dumps(trigger_ctx, default=str)}
Taboo Words to avoid: {taboo_list}
Conversation History: {json.dumps(history, default=str)}
Merchant's Latest Reply: "{reply_message}"
"""
        try:
            content = await _call_llm(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            if not content:
                return ActionOutput(action="wait", rationale="LLM unavailable")
            data = json.loads(content)
            return ActionOutput(**data)
        except Exception as e:
            print(f"[Vera] Reply Error: {e}")
            return ActionOutput(action="wait", rationale=f"Error: {e}")
