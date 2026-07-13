import argparse
import json
import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(threadName)s] %(message)s')
logger = logging.getLogger(__name__)

# Try to import ollama-tool-orchestrator
ORCHESTRATOR_PATH = os.path.expanduser(r'~/.gemini/config/skills/ollama-tool-orchestrator/scripts')
if ORCHESTRATOR_PATH not in sys.path:
    sys.path.append(ORCHESTRATOR_PATH)

try:
    from orchestrator import ToolExecutor, chat_with_tools
except ImportError as e:
    logger.error(f"Failed to import orchestrator from {ORCHESTRATOR_PATH}. Ensure ollama-tool-orchestrator is installed.")
    sys.exit(1)

# Define some basic tools for the local models
def read_file(filepath: str) -> str:
    """Reads a file from the local filesystem."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"

def list_dir(directory: str) -> str:
    """Lists files in a directory."""
    try:
        return str(os.listdir(directory))
    except Exception as e:
        return f"Error listing directory: {e}"

def create_tool_executor() -> ToolExecutor:
    tool_exec = ToolExecutor()
    tool_exec.register(read_file, "Reads the contents of a file from the local filesystem. Provide an absolute filepath.")
    tool_exec.register(list_dir, "Lists the contents of a directory. Provide an absolute directory path.")
    return tool_exec

def process_task(task: dict, model: str, host: str, tool_exec: ToolExecutor) -> dict:
    task_id = task.get("id", "unknown_id")
    prompt = task.get("prompt", "")
    
    if not prompt:
        return {"id": task_id, "status": "error", "error": "No prompt provided."}
    
    logger.info(f"Starting task '{task_id}' with model {model}...")
    messages = [{"role": "user", "content": prompt}]
    
    try:
        result = chat_with_tools(
            model=model,
            messages=messages,
            tool_executor=tool_exec,
            host=host,
            max_loops=15
        )
        
        if result.get("status") == "success":
            logger.info(f"Task '{task_id}' completed successfully.")
            return {
                "id": task_id,
                "status": "success",
                "result": result.get("content", ""),
                "tool_calls": len(result.get("messages", [])) - 2 # rough count of tool interactions
            }
        else:
            logger.error(f"Task '{task_id}' failed: {result.get('error')}")
            return {
                "id": task_id,
                "status": "error",
                "error": result.get("error", "Unknown error in orchestrator"),
                "result": result.get("content", "")
            }
    except Exception as e:
        logger.error(f"Exception during task '{task_id}': {e}")
        return {"id": task_id, "status": "error", "error": str(e)}

def main():
    parser = argparse.ArgumentParser(description="Local Task Farm - Parallelize simple tasks using local Ollama models.")
    parser.add_argument("--tasks", required=True, help="Path to input tasks JSON file.")
    parser.add_argument("--output", required=True, help="Path to output results JSON file.")
    parser.add_argument("--model", default="qwen2.5-coder:7b", help="Ollama model to use. Defaults to qwen2.5-coder:7b.")
    parser.add_argument("--concurrency", type=int, default=2, help="Number of concurrent tasks to run. Defaults to 2.")
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama host URL.")
    
    args = parser.parse_args()
    
    # Load tasks
    if not os.path.exists(args.tasks):
        logger.error(f"Tasks file not found: {args.tasks}")
        sys.exit(1)
        
    try:
        with open(args.tasks, 'r', encoding='utf-8') as f:
            tasks = json.load(f)
    except Exception as e:
        logger.error(f"Failed to parse tasks JSON: {e}")
        sys.exit(1)
        
    if not isinstance(tasks, list):
        logger.error("Tasks JSON must be a list of task objects.")
        sys.exit(1)
        
    logger.info(f"Loaded {len(tasks)} tasks. Starting execution with concurrency={args.concurrency} and model={args.model}")
    
    # Prepare results container and thread lock for thread-safe writing if needed
    results = []
    
    # Tool executor instance can be shared if stateless, but to be safe we'll pass a new one or reuse safely
    # ToolExecutor just holds references to functions, so it's thread-safe for reading schemas/executing.
    tool_exec = create_tool_executor()
    
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        future_to_task = {executor.submit(process_task, t, args.model, args.host, tool_exec): t for t in tasks}
        
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                res = future.result()
                results.append(res)
            except Exception as e:
                logger.error(f"Task '{task.get('id')}' raised an unhandled exception: {e}")
                results.append({"id": task.get("id"), "status": "error", "error": str(e)})
                
    # Write output
    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Successfully wrote results for {len(results)} tasks to {args.output}")
    except Exception as e:
        logger.error(f"Failed to write output JSON: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
