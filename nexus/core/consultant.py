import argparse
import json
import os
import sys

# Add the ollama-tool-orchestrator to path
CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
ORCHESTRATOR_PATH = os.path.join(CONFIG_DIR, "skills", "ollama-tool-orchestrator")
sys.path.append(ORCHESTRATOR_PATH)

try:
    from scripts.orchestrator import smart_chat_with_tools
except ImportError:
    print("Error: Could not import ollama-tool-orchestrator. Ensure it is installed at", ORCHESTRATOR_PATH)
    sys.exit(1)

def get_session_file(session_id):
    # Store sessions in the .gemini/scratch/local_consults directory
    scratch_dir = os.path.join(CONFIG_DIR, "..", "scratch", "local_consults")
    os.makedirs(scratch_dir, exist_ok=True)
    return os.path.join(scratch_dir, f"{session_id}.json")

def load_session(session_id):
    file_path = get_session_file(session_id)
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Session file {file_path} is corrupted. Starting fresh.")
    return []

def save_session(session_id, messages):
    file_path = get_session_file(session_id)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(messages, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Consult a local LLM via Ollama")
    parser.add_argument("--prompt", required=True, help="The question or context you want to ask the model")
    parser.add_argument("--session", default=None, help="Optional session ID to maintain conversation history")
    
    args = parser.parse_args()
    
    messages = []
    if args.session:
        messages = load_session(args.session)
        
    messages.append({"role": "user", "content": args.prompt})
    
    print(f"Consulting local model...")
    # Call the orchestrator (no tool executor provided, just text chat)
    result = smart_chat_with_tools(messages)
    
    status = result.get("status")
    if status == "rejected":
        print("\n--- Consultation Rejected ---")
        print(result.get("error", "Task too complex for local models."))
        sys.exit(1)
    elif status == "error":
        print("\n--- Consultation Error ---")
        print(result.get("error", "Unknown error occurred."))
        sys.exit(1)
        
    response_content = result.get("content", "")
    print("\n--- Local Model Response ---")
    print(response_content)
    print("----------------------------\n")
    
    # Save the updated history if session is used
    if args.session:
        # The result includes the updated messages history (including the model's response)
        updated_messages = result.get("messages", messages)
        save_session(args.session, updated_messages)
        print(f"(Conversation history saved to session '{args.session}')")

if __name__ == "__main__":
    main()
