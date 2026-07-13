import inspect
import json
import logging
from typing import Callable, Any, Dict, List

import requests
import os
import re
import sqlite3

logger = logging.getLogger(__name__)

class ToolExecutor:
    def __init__(self):
        self.tools: Dict[str, Callable] = {}
        self.schemas: List[Dict[str, Any]] = []

    def register(self, func: Callable, description: str = None):
        """Register a Python function as a tool."""
        name = func.__name__
        self.tools[name] = func
        
        sig = inspect.signature(func)
        
        properties = {}
        required = []
        
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
                
            param_type = "string" # Default
            if param.annotation != inspect.Parameter.empty:
                if param.annotation == int:
                    param_type = "integer"
                elif param.annotation == float:
                    param_type = "number"
                elif param.annotation == bool:
                    param_type = "boolean"
                elif param.annotation == list or param.annotation == List:
                    param_type = "array"
            
            properties[param_name] = {
                "type": param_type,
                "description": f"Parameter {param_name}"
            }
            
            if param.default == inspect.Parameter.empty:
                required.append(param_name)
                
        schema = {
            "type": "function",
            "function": {
                "name": name,
                "description": description or func.__doc__ or f"Executes the {name} function.",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }
        self.schemas.append(schema)
        return func

    def execute(self, name: str, kwargs: Dict[str, Any]) -> str:
        """Execute a registered tool by name with given kwargs."""
        if name not in self.tools:
            return json.dumps({"error": f"Tool '{name}' not found."})
        
        try:
            func = self.tools[name]
            result = func(**kwargs)
            # Ensure result is serializable
            if isinstance(result, (dict, list, str, int, float, bool, type(None))):
                return json.dumps(result)
            return str(result)
        except Exception as e:
            return json.dumps({"error": str(e)})


def get_available_models(host: str = "http://localhost:11434") -> List[Dict[str, Any]]:
    """Retrieve a list of generative models and their capability profiles."""
    try:
        resp = requests.get(f"{host}/api/tags", timeout=5)
        resp.raise_for_status()
        raw_models = resp.json().get("models", [])
        
        profiles = []
        for m in raw_models:
            details = m.get("details", {})
            family = details.get("family", "").lower()
            name = m.get("name", "").lower()
            
            # Filter out embedding models
            if "bert" in family or "nomic" in family or "embed" in name or "minilm" in name:
                continue
                
            # Profile capability based on size and name
            size_gb = m.get("size", 0) / (1024**3)
            is_coder = "coder" in name or "deepseek" in name
            
            # Rough difficulty capability scale 1-10 based on param size (e.g. 7B is ~6, 1.5B is ~2)
            capability_score = min(10, max(1, int(size_gb * 1.5)))
            if is_coder:
                capability_score = min(10, capability_score + 2) # bump for specialized reasoning
                
            profiles.append({
                "name": m["name"],
                "size_gb": round(size_gb, 2),
                "family": family,
                "is_coder": is_coder,
                "capability_score": capability_score
            })
            
        # Sort by capability score (ascending) so the router can easily pick the fastest first
        profiles.sort(key=lambda x: x["capability_score"])
        return profiles
    except Exception as e:
        logger.error(f"Failed to fetch models from {host}: {e}")
        return []

def smart_chat_with_tools(
    messages: List[Dict[str, Any]], 
    tool_executor: ToolExecutor = None,
    host: str = "http://localhost:11434",
    max_loops: int = 25
) -> Dict[str, Any]:
    """
    Intelligently routes the user's prompt to the most appropriate local model
    based on intent and graded difficulty, then executes the tool-calling loop.
    """
    profiles = get_available_models(host)
    if not profiles:
        return {"error": "No generative models found on host.", "status": "error"}
        
    # The router model is the fastest one (lowest capability score)
    router_model = profiles[0]["name"]
    
    # Format user messages for the router
    user_prompt_summary = "\n".join([m["content"] for m in messages if m.get("role") == "user"])
    
    # Check Ledger for cross-file complexity
    ledger_complexity = ""
    try:
        db_path = os.path.join(os.getcwd(), ".architect_ledger.db")
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            c = conn.cursor()
            # Simple heuristic: if the prompt mentions multiple files or functions that exist in the ledger
            c.execute("SELECT file_path, func_name FROM functions")
            functions = c.fetchall()
            mentioned_funcs = []
            for filepath, funcname in functions:
                if funcname in user_prompt_summary and len(funcname) > 3:
                    mentioned_funcs.append((filepath, funcname))
            
            if len(mentioned_funcs) > 2:
                # Get unique files involved
                unique_files = set([f[0] for f in mentioned_funcs])
                if len(unique_files) > 1:
                    ledger_complexity = f"\n[LEDGER DATA]: The prompt involves {len(mentioned_funcs)} AST functions spanning {len(unique_files)} different files. This is a HIGH COMPLEXITY cross-file task. Artificially bump the difficulty to 9 or 10."
    except Exception as e:
        pass

    system_prompt = f"""You are a Compositional Generalist-Specialist smart model router. Your job is to read the user's prompt, grade its difficulty on a scale of 1-10, and select the BEST model from the available list to handle it.
Do NOT solve the user's prompt. ONLY output valid JSON.

Available Models:
{json.dumps(profiles, indent=2)}

Rules for selection:
1. If the task is simple (greetings, simple facts), pick a model with a low capability score (1-3) to save compute.
2. If the task requires tools, logic, or coding, pick a model with a higher capability score (5+).
3. If the task is explicitly about coding, strongly prefer a model where `is_coder` is true.
4. If Ledger Data indicates high complexity cross-file dependencies, you MUST output a difficulty of 9 or 10.
{ledger_complexity}

4. Select the model whose `capability_score` is closest to (or slightly above) your graded `difficulty`.
5. If the task involves massive context (e.g. merging huge sub-tasks or doing deep system architecture review), explicitly select a model with high VRAM requirements (30B+ parameters) if available, falling back to the highest capability score otherwise.

Output ONLY a JSON object in this exact format:
{{
  "intent": "brief summary of what the user wants",
  "difficulty": 5,
  "recommended_model": "model_name_here",
  "reasoning": "why you picked this model"
}}
"""
    
    target_model = profiles[-1]["name"] # fallback default
    
    # Query the router model
    try:
        router_payload = {
            "model": router_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"User Prompt:\n{user_prompt_summary}"}
            ],
            "stream": False,
            "format": "json" # Force JSON output
        }
        
        print(f"--> [Smart Router] Asking {router_model} to grade and route...", flush=True)
        resp = requests.post(f"{host}/api/chat", json=router_payload, timeout=30)
        resp.raise_for_status()
        
        router_response = resp.json()["message"]["content"]
        routing_decision = json.loads(router_response)
        
        target_model = routing_decision.get("recommended_model", target_model)
        difficulty = routing_decision.get("difficulty", 5)
        
        # Calculate the dynamic ceiling of local capabilities
        max_cap = max((p["capability_score"] for p in profiles), default=0)
        
        if difficulty > max_cap:
            print(f"--> [Smart Router] WARNING: Task difficulty ({difficulty}) exceeds max local capability ({max_cap}). BYPASSING for testing.", flush=True)
            # return {
            #     "content": "",
            #     "status": "rejected",
            #     "error": f"Task difficulty ({difficulty}/10) exceeds the maximum capability score ({max_cap}/10) of available local models. Request rejected. The main agent should handle this task directly."
            # }
        
        # Fallback if the router hallucinated a model name
        if not any(p["name"] == target_model for p in profiles):
            print(f"--> [Smart Router] Model '{target_model}' invalid. Falling back to largest.", flush=True)
            target_model = profiles[-1]["name"]
            
        print(f"--> [Smart Router] Decision: Difficulty {routing_decision.get('difficulty')}/10 -> Selected {target_model}", flush=True)
        print(f"--> [Smart Router] Reasoning: {routing_decision.get('reasoning')}", flush=True)
        
    except Exception as e:
        print(f"--> [Smart Router] Routing failed: {e}. Falling back to largest model {target_model}.", flush=True)
        
    # Execute the actual chat loop with the selected model
    result = chat_with_tools(target_model, messages, tool_executor, host, max_loops)
    
    # If the model throws a 400 Bad Request, it almost certainly lacks native tool calling support
    if result.get("status") == "error" and "400 Client Error" in result.get("error", ""):
        print(f"--> [Smart Router] ERROR: {target_model} rejected the payload (likely missing native tool support).", flush=True)
        fallback_model = "qwen2.5-coder:7b"
        for p in profiles:
            if "gemma4" in p["name"]:
                fallback_model = p["name"]
                break
        print(f"--> [Smart Router] Falling back to known tool-capable model '{fallback_model}'...", flush=True)
        return chat_with_tools(fallback_model, messages, tool_executor, host, max_loops)
        
    return result

def chat_with_tools(
    model: str, 
    messages: List[Dict[str, Any]], 
    tool_executor: ToolExecutor = None,
    host: str = "http://localhost:11434",
    max_loops: int = 25
) -> Dict[str, Any]:
    """
    Sends a chat request to Ollama with optional tools.
    If the model decides to call a tool, it automatically executes it and loops back
    to the model to get the final answer.
    """
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"num_ctx": 8192, "num_predict": 4096}
    }
    
    if tool_executor and tool_executor.schemas:
        payload["tools"] = tool_executor.schemas

    loops = 0
    consecutive_duplicates = 0
    last_tool_call_str = None
    
    for loop_count in range(max_loops):
        loops += 1
        
        try:
            response = requests.post(f"{host}/api/chat", json=payload, timeout=3600)
            response.raise_for_status()
            data = response.json()
            message = data.get("message", {})
            
            # Append model's response to history
            messages.append(message)
            
            # Check for tool calls
            tool_calls = message.get("tool_calls")
            
            if not tool_calls and message.get("content"):
                # Fallback: check if the model output the tool call as raw JSON in content
                content = message.get("content", "").strip()
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
                raw_json_match = re.search(r'(\{[\s\S]*"name"\s*:\s*"[^"]+"[\s\S]*"arguments"\s*:[\s\S]*\})', content)
                json_str = None
                
                if json_match:
                    json_str = json_match.group(1)
                elif raw_json_match:
                    json_str = raw_json_match.group(1)
                elif content.startswith("{") and content.endswith("}"):
                    json_str = content
                    
                if json_str:
                    try:
                        parsed = json.loads(json_str)
                        if "name" in parsed and "arguments" in parsed:
                            tool_calls = [{"function": parsed}]
                    except json.JSONDecodeError:
                        pass
            if not tool_calls:
                # No tool calls, model provided a normal text response
                return {
                    "content": message.get("content", ""),
                    "messages": messages,
                    "status": "success"
                }
            
            # Execute each tool call
            for tool_call in tool_calls:
                func_name = tool_call["function"]["name"]
                arguments = tool_call["function"]["arguments"]
                
                current_call_str = f"{func_name}:{arguments}"
                if current_call_str == last_tool_call_str:
                    consecutive_duplicates += 1
                else:
                    consecutive_duplicates = 0
                    last_tool_call_str = current_call_str
                
                print(f"--> [Model Invoking Tool]: {func_name}({arguments})", flush=True)
                
                if consecutive_duplicates >= 2:
                    result_str = "[SYSTEM ERROR] Loop detected. You have called this exact tool with these exact arguments multiple times in a row. The tool execution was blocked to prevent an infinite loop. You MUST change your strategy. Either call a different tool, use different arguments, or output a final textual response to explain you are stuck."
                else:
                    # Execute it
                    result_str = tool_executor.execute(func_name, arguments)
                
                # Append tool result to messages
                messages.append({
                    "role": "tool",
                    "content": result_str,
                    "name": func_name
                })
            
            # Update payload messages for the next loop iteration
            payload["messages"] = messages
            
        except Exception as e:
            logger.error(f"Error during chat interaction: {e}")
            try:
                with open(os.path.join(os.getcwd(), "error_payload.json"), "w", encoding="utf-8") as dump_f:
                    json.dump(payload, dump_f, indent=2)
            except Exception: pass
            return {
                "content": "",
                "messages": messages,
                "status": "error",
                "error": str(e)
            }
            
    return {
        "content": "",
        "messages": messages,
        "status": "error",
        "error": "Max tool execution loops reached."
    }
