import time
import uuid
import os
import asyncio
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from datetime import datetime

from .memory import memory
from .llm_engine import LLMEngine

# Automatically wipe memory on server startup for clean testing
if os.path.exists("memory.json"):
    try:
        os.remove("memory.json")
        memory.data = {
            "category": {}, "merchant": {}, "trigger": {},
            "customer": {}, "conversations": {}
        }
    except Exception as e:
        print(f"Failed to wipe memory.json on startup: {e}")

app = FastAPI(title="Magicpin AI Challenge Bot")
START_TIME = time.time()

# Models
class ContextPayload(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: Dict[str, Any]
    delivered_at: Optional[str] = None

class TickPayload(BaseModel):
    now: str
    available_triggers: List[str]

class ReplyPayload(BaseModel):
    conversation_id: str
    from_role: str
    message: str
    turn_number: int
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None

# /v1/healthz
@app.get("/v1/healthz")
async def healthz():
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": memory.get_counts()
    }

# /v1/metadata
@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": "Vedant Mane",
        "team_members": ["Vedant Mane"],
        "model": "meta/llama-3.1-70b-instruct",
        "approach": "Multi-provider LLM (NVIDIA NIM + OpenRouter + Groq) with rubric-aligned prompt engineering",
        "version": "2.0.0"
    }

# /v1/context
@app.post("/v1/context")
async def receive_context(req: ContextPayload):
    updated = memory.set_context(
        scope=req.scope,
        context_id=req.context_id,
        version=req.version,
        payload=req.payload
    )
    
    if not updated:
        return {
            "accepted": False,
            "reason": "stale_version",
            "current_version": memory.get_version(req.scope, req.context_id)
        }
        
    return {
        "accepted": True,
        "ack_id": f"ack_{req.context_id}_v{req.version}",
        "stored_at": datetime.utcnow().isoformat() + "Z"
    }

# /v1/tick
@app.post("/v1/tick")
async def handle_tick(req: TickPayload):
    async def process_trigger(trigger_id):
        trigger_ctx = memory.get_context("trigger", trigger_id)
        if not trigger_ctx:
            return None
            
        merchant_id = trigger_ctx.get("merchant_id")
        if not merchant_id:
            return None
            
        merchant_ctx = memory.get_context("merchant", merchant_id)
        if not merchant_ctx:
            return None
            
        category_slug = merchant_ctx.get("category_slug", "unknown")
        category_ctx = memory.get_context("category", category_slug)
        
        customer_id = trigger_ctx.get("customer_id")
        customer_ctx = memory.get_context("customer", customer_id) if customer_id else None

        conv_id = f"conv_{merchant_id}_{trigger_id}"
        if memory.is_suppressed(conv_id):
            return None

        # Compose message via LLM
        draft_action = await LLMEngine.compose_message(merchant_ctx, category_ctx, trigger_ctx, customer_ctx)
        
        if draft_action.action == "send":
            return {
                "conversation_id": conv_id,
                "merchant_id": merchant_id,
                "customer_id": customer_id,
                "send_as": "merchant_on_behalf" if customer_id else "vera",
                "trigger_id": trigger_id,
                "body": draft_action.body,
                "cta": draft_action.cta,
                "suppression_key": trigger_ctx.get("suppression_key", f"sup_{trigger_id}"),
                "rationale": draft_action.rationale
            }
        elif draft_action.action == "end":
            memory.suppress_conversation(conv_id)
            
        return None

    results = []
    batch_triggers = req.available_triggers[:20]
    
    # Process sequentially with a strict 25-second time budget
    tick_start = time.time()
    for i, t in enumerate(batch_triggers):
        if time.time() - tick_start > 25.0:
            print(f"[WARN] Tick time budget exceeded ({time.time() - tick_start:.1f}s). Returning {len(results)} actions early.")
            break
            
        try:
            remaining_time = max(0.1, 25.0 - (time.time() - tick_start))
            res = await asyncio.wait_for(process_trigger(t), timeout=remaining_time)
            if res:
                results.append(res)
        except asyncio.TimeoutError:
            print(f"[WARN] Trigger {t} timed out. Returning {len(results)} actions early.")
            break
        except Exception as e:
            print(f"Error processing trigger {t}: {e}")
            
    return {"actions": results}

# /v1/reply
@app.post("/v1/reply")
async def handle_reply(req: ReplyPayload):
    conv_id = req.conversation_id
    
    if memory.is_suppressed(conv_id):
        return {"action": "end", "rationale": "Conversation suppressed."}
    
    # Store the incoming merchant message
    memory.add_message(conv_id, req.from_role, req.message)
    
    # Retrieve full history
    history = memory.get_history(conv_id)
    
    merchant_ctx = memory.get_context("merchant", req.merchant_id) if req.merchant_id else {}
    
    # Extract trigger context from conversation_id (format: conv_{merchant_id}_{trigger_id})
    trigger_ctx = {}
    try:
        parts = conv_id.split("_", 2)
        if len(parts) >= 3:
            trigger_id = parts[2]
            trigger_ctx = memory.get_context("trigger", trigger_id) or {}
    except Exception as e:
        print(f"Failed to parse trigger_id from {conv_id}: {e}")
    
    # Get category for voice/taboos
    category_slug = merchant_ctx.get("category_slug", "unknown")
    category_ctx = memory.get_context("category", category_slug) or {}

    action = await LLMEngine.handle_reply(merchant_ctx, category_ctx, trigger_ctx, history, req.message)
    
    if action.action == "send" and action.body:
        memory.add_message(conv_id, "assistant", action.body)
        
    if action.action == "end":
        memory.suppress_conversation(conv_id)
        
    return {
        "action": action.action,
        "body": action.body if action.action == "send" else None,
        "cta": action.cta if action.action == "send" else None,
        "rationale": action.rationale,
        "wait_seconds": action.wait_seconds if action.action == "wait" else None
    }
