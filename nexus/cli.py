import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Nexus Hub: Monolithic Local Intelligence Server")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    server_parser = subparsers.add_parser("serve", help="Launch the Monolithic API Server (Skillbrary/Ollama Orchestrator)")
    server_parser.add_argument("--port", type=int, default=8080, help="Port to run the API Server")
    
    farm_parser = subparsers.add_parser("farm", help="Dispatch a queue to the Local Task Farm")
    farm_parser.add_argument("--queue", type=str, required=True, help="Path to JSON task queue")
    
    map_parser = subparsers.add_parser("map", help="Update the Persistent ChromaDB System Mapper")
    map_parser.add_argument("--dir", type=str, required=True, help="Project root to map")
    
    consult_parser = subparsers.add_parser("consult", help="Consult the Local Model via CLI")
    consult_parser.add_argument("--prompt", type=str, required=True, help="The prompt to consult")
    
    args = parser.parse_args()
    
    if args.command == "serve":
        print(f"[*] Starting Nexus Hub API Server on port {args.port}...")
        try:
            from nexus.core.orchestrator import start_server
            start_server(args.port)
        except ImportError:
            print("[*] (Mock) API Server is now listening. Waiting for Ollama Tool requests...")
    elif args.command == "farm":
        print(f"[*] Dispatching task queue '{args.queue}' to Synchronous Threading Farm...")
        try:
            from nexus.core.farm import execute_farm
            execute_farm(args.queue)
        except ImportError:
            print("[*] (Mock) Task Farm execution complete.")
    elif args.command == "map":
        print(f"[*] Updating ChromaDB vector store for directory: {args.dir}...")
        try:
            from nexus.core.mapper import update_database
            update_database(args.dir)
        except ImportError:
            print("[*] (Mock) Persistent Map updated successfully.")
    elif args.command == "consult":
        print(f"[*] Consulting local model on prompt: {args.prompt}")
        try:
            from nexus.core.consultant import consult_model
            consult_model(args.prompt)
        except ImportError:
            print("[*] (Mock) The model believes this is highly viable.")
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
