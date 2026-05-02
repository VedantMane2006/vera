import os
import asyncio
import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# PROVIDER INITIALIZATION (priority: NVIDIA NIM → OpenRouter → Groq)
# ============================================================

# 1. NVIDIA NIM (Primary — paid credits, most reliable)
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
nvidia_client = None
if NVIDIA_API_KEY:
    nvidia_client = AsyncOpenAI(
        api_key=NVIDIA_API_KEY,
        base_url="https://integrate.api.nvidia.com/v1",
    )
    print("[INFO] NVIDIA NIM client initialized.")

# 2. OpenRouter (Fallback — free tier)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
openrouter_client = None
if OPENROUTER_API_KEY:
    openrouter_client = AsyncOpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
    )
    print("[INFO] OpenRouter client initialized.")

# 3. Groq (Fallback — free tier)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = None
if GROQ_API_KEY and GROQ_API_KEY != "default-key":
    groq_client = AsyncOpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
    )
    print("[INFO] Groq client initialized.")

MODEL_NVIDIA = "meta/llama-3.1-70b-instruct"
MODEL_OPENROUTER = "meta-llama/llama-3.3-70b-instruct:free"
MODEL_GROQ = "llama-3.3-70b-versatile"


async def _call_llm(messages: List[Dict], temperature: float = 0.7):
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
        providers.append(("Groq", groq_client, MODEL_GROQ))

    for name, client, model in providers:
        for attempt in range(3):
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
                if "429" in err or "rate" in err.lower():
                    wait = (attempt + 1) * 3
                    print(f"[{name}] Rate limited. Retrying in {wait}s...")
                    await asyncio.sleep(wait)
                else:
                    print(f"[{name}] Error: {err[:120]}")
                    break  # Non-retryable error, try next provider

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
You are Vera, a Merchant Growth Expert at magicpin. Compose one WhatsApp-style message.

YOU MUST ALWAYS CHOOSE action="send". Only choose "wait" or "end" if the merchant sent the same auto-reply 3+ times.

SCORING RULES (you are graded strictly on these 5 dimensions):

1. SPECIFICITY (use EXACT numbers from context):
   - Compute and quote specific metrics: "2,410 views but only 18 calls = 0.7% CTR"
   - Include delta percentages: "calls dropped 50% vs last week"
   - Mention exact dates, prices, and counts from the trigger payload

2. CATEGORY FIT (match the business voice exactly):
   - Dentists: clinical peer tone, ALWAYS "Dr. [owner_first_name]", say "patients" not "customers"
   - Restaurants: operator tone, use "covers", "footfall", "orders", "kitchen"
   - Gyms: coaching tone, "members", "retention", "trials", "gains"
   - Salons: warm practical tone, "clients", "appointments", "bookings"
   - Pharmacies: trustworthy precise tone, "customers", "prescriptions", "stock"

3. MERCHANT FIT (personalized to THIS merchant):
   - Address them by owner_first_name (e.g., "Suresh", "Anjali")
   - Reference their specific active offer titles, locality name, and signals
   - Reference their conversation_history if they previously engaged

4. DECISION QUALITY (connect to the trigger's WHY NOW):
   - Name the trigger kind explicitly: "Your calls dipped 50% this week"
   - Quote trigger payload fields: delta_pct, deadline, days_remaining, metric values
   - Show you understand the urgency level and acted accordingly

5. ENGAGEMENT COMPULSION (they MUST reply — this is the hardest dimension):
   - Frame as LOSS: "You're losing X every day without this" > "You could gain Y"
   - Quantify the loss: "That's ~X in missed revenue" or "X patients going to competitors"
   - End with ONE specific binary CTA: "Should I publish it now? Yes/No"
   - Create time pressure: "before the weekend", "deadline is [date]"
   - Keep under 120 words — punchy, not verbose

HARD RULES:
- Do NOT use any URLs or taboo words from the category context
- Do NOT fabricate data — only use facts from the provided context
- Do NOT introduce yourself ("Hi I'm Vera") — go straight to value
- Return ONLY valid JSON, nothing else

JSON SCHEMA:
{
    "action": "send",
    "body": "The WhatsApp message body (under 120 words, loss-framed, ends with binary CTA)",
    "cta": "binary_yes_no",
    "rationale": "Which trigger payload fields you used and how you framed the loss",
    "wait_seconds": 0
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
                return ActionOutput(action="send", body="Your listing needs attention — reply Yes to get started.", cta="binary_yes_no", rationale="Fallback: all LLM providers failed.")
            data = json.loads(content)
            print(f"[Vera] Decision: {data.get('action')} | {str(data.get('rationale', ''))[:60]}")
            return ActionOutput(**data)
        except Exception as e:
            print(f"[Vera] Composer Error: {e}")
            return ActionOutput(action="send", body="Your listing needs attention — reply Yes to get started.", cta="binary_yes_no", rationale=f"Fallback due to error: {e}")

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
