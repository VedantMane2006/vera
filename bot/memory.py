import json
import os
import threading
from typing import Dict, Any

class MemoryStore:
    def __init__(self, filename="memory.json"):
        self.filename = filename
        self.lock = threading.Lock()
        self.data = {
            "category": {},
            "merchant": {},
            "trigger": {},
            "customer": {},
            "conversations": {}
        }
        self.load()

    def load(self):
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content.strip():
                        loaded_data = json.loads(content)
                        # Ensure all keys exist
                        for key in self.data.keys():
                            if key not in loaded_data:
                                loaded_data[key] = {}
                        self.data = loaded_data
            except Exception as e:
                print(f"Warning: Failed to load {self.filename}: {e}")

    def save(self):
        try:
            with open(self.filename, 'w', encoding='utf-8') as f:
                json.dump(self.data, f)
        except Exception as e:
            print(f"Warning: Failed to save {self.filename}: {e}")

    def get_version(self, scope: str, context_id: str) -> int:
        with self.lock:
            current = self.data.get(scope, {}).get(context_id)
            if current:
                return current.get('version', 0)
            return 0

    def set_context(self, scope: str, context_id: str, version: int, payload: Dict[str, Any]) -> bool:
        """Returns True if updated, False if stale version"""
        with self.lock:
            scope_data = self.data.get(scope, {})
            current = scope_data.get(context_id)
            
            if current and current.get('version', 0) >= version:
                return False # stale version
            
            scope_data[context_id] = {
                "version": version,
                "payload": payload
            }
            self.data[scope] = scope_data
            self.save()
            return True

    def get_context(self, scope: str, context_id: str) -> Dict[str, Any]:
        with self.lock:
            return self.data.get(scope, {}).get(context_id, {}).get('payload', {})

    def get_all(self, scope: str) -> Dict[str, Dict[str, Any]]:
        with self.lock:
            return self.data.get(scope, {})

    def get_counts(self) -> Dict[str, int]:
        with self.lock:
            return {
                "category": len(self.data.get("category", {})),
                "merchant": len(self.data.get("merchant", {})),
                "customer": len(self.data.get("customer", {})),
                "trigger": len(self.data.get("trigger", {}))
            }

    def suppress_conversation(self, conv_id: str):
        with self.lock:
            if conv_id not in self.data["conversations"]:
                self.data["conversations"][conv_id] = {}
            self.data["conversations"][conv_id]["suppressed"] = True
            self.save()
            
    def is_suppressed(self, conv_id: str) -> bool:
        with self.lock:
            return self.data.get("conversations", {}).get(conv_id, {}).get("suppressed", False)

    def add_message(self, conv_id: str, role: str, message: str):
        with self.lock:
            if conv_id not in self.data["conversations"]:
                self.data["conversations"][conv_id] = {"history": [], "suppressed": False}
            if "history" not in self.data["conversations"][conv_id]:
                self.data["conversations"][conv_id]["history"] = []
            
            self.data["conversations"][conv_id]["history"].append({
                "role": role,
                "content": message
            })
            self.save()

    def get_history(self, conv_id: str) -> list:
        with self.lock:
            return self.data.get("conversations", {}).get(conv_id, {}).get("history", [])

# Global singleton
memory = MemoryStore()
