import os
import argparse
import json
import urllib.request
import urllib.error
import sys
import re
import sqlite3
import math
import struct
import subprocess
import tempfile
from datetime import datetime

def get_ignore_dirs():
    """Returns a set of directory names that should be globally ignored."""
    return {'.git', 'node_modules', '__pycache__', 'venv', '.venv', 'env', '.env', 'build', 'dist', '.idea', '.vscode', '.ollama_patches'}

def get_allowed_extensions():
    """Returns a set of file extensions that are allowed to be ingested."""
    return {'.py', '.js', '.ts', '.jsx', '.tsx', '.json', '.md', '.html', '.css', '.java', '.go', '.rs', '.c', '.cpp', '.h', '.cs', '.php', '.rb', '.sh', '.yaml', '.yml', '.toml', '.ini'}

def serialize_embedding(emb):
    """Packs a list of floats into a binary BLOB string for extremely fast SQLite storage."""
    return struct.pack(f'{len(emb)}f', *emb)

def deserialize_embedding(blob):
    """Unpacks a binary BLOB string back into a tuple of floats."""
    return struct.unpack(f'{len(blob)//4}f', blob)

def cosine_similarity(v1, v2):
    """Calculates cosine similarity between two vectors using highly optimized pure Python."""
    dot = sum(x*y for x, y in zip(v1, v2))
    norm_a = sum(x*x for x in v1)
    norm_b = sum(x*x for x in v2)
    return dot / math.sqrt(norm_a * norm_b) if norm_a and norm_b else 0.0

def get_embeddings_batch(texts, model="nomic-embed-text", url="http://localhost:11434/api/embed"):
    """Fetches vector embeddings from Ollama for a batch of text chunks."""
    if not texts:
        return []
    data = {"model": model, "input": texts}
    req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'), headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result.get("embeddings", [])
    except Exception as e:
        print(f"Error getting embeddings batch (is {model} pulled?): {e}", file=sys.stderr)
        return []

def init_db(db_path):
    """Initializes the SQLite database with BLOB storage for embeddings. Migrates old JSON tables if necessary."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Check if table exists and what type 'embedding' is to handle JSON-to-BLOB migrations
    c.execute("PRAGMA table_info(chunks)")
    columns = c.fetchall()
    if columns:
        emb_type = next((col[2] for col in columns if col[1] == 'embedding'), None)
        if emb_type and 'TEXT' in emb_type.upper():
            print("Legacy JSON embedding table detected. Dropping for BLOB migration...", file=sys.stderr)
            c.execute("DROP TABLE chunks")
            
    c.execute('''CREATE TABLE IF NOT EXISTS chunks (id INTEGER PRIMARY KEY, filepath TEXT, content TEXT, embedding BLOB)''')
    conn.commit()
    return conn

def chunk_text(text, chunk_size=1500, overlap=200):
    """Splits a string into overlapping chunks for RAG embedding."""
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i+chunk_size])
        i += chunk_size - overlap
    return chunks

def query_ollama(prompt, model="qwen2.5-coder:7b", url="http://localhost:11434/api/generate"):
    """Queries Ollama synchronously and returns the response string."""
    data = {"model": model, "prompt": prompt, "stream": False, "format": "json", "options": {"num_ctx": 8192, "temperature": 0.1, "top_p": 0.9, "num_gpu": -1}}
    req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'), headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result.get("response", "")
    except Exception as e:
        print(f"Error connecting to Ollama for dynamic filters: {e}", file=sys.stderr)
        return ""

def generate_dynamic_filters(project_dir, ignore_dirs):
    """Asks the local LLM to dynamically generate ignore filters based on file patterns (e.g., catching new log/data extensions)."""
    always_ignore = ['.system_mapper_cache.json', '.system_mapper_db.sqlite']
    
    files_to_sample = []
    for root, dirs, files in os.walk(project_dir):
        dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.startswith('.')]
        for file in files:
            if file in always_ignore:
                continue
            if len(files_to_sample) >= 150:
                break
            files_to_sample.append(os.path.relpath(os.path.join(root, file), project_dir))
        if len(files_to_sample) >= 150:
            break
            
    file_list_str = "\n".join(files_to_sample)
    
    prompt = (
        "You are an automated code indexing script. Your task is to filter a list of filenames. "
        "Review the following list of files. "
        "Identify substrings or file extensions for files that are typically data dumps, logs, node graphs, checkpoints, or compiled artifacts (which should not be indexed as source code). "
        "Respond ONLY with a valid JSON object containing a single key 'ignore_substrings' mapping to a list of strings. Do not include markdown formatting or conversational text.\n\n"
        "Example: {\"ignore_substrings\": [\"checkpoint_\", \".log\", \"node_data\"]}\n\n"
        f"--- FILE LIST ---\n{file_list_str}"
    )
    
    raw_response = query_ollama(prompt)
    print(f"[DEBUG] Raw AI Filter Output: {raw_response}", file=sys.stderr)
    dynamic_filters = []
    try:
        clean_json = re.sub(r'^```json\s*', '', raw_response)
        clean_json = re.sub(r'\s*```$', '', clean_json)
        if clean_json.strip():
            data = json.loads(clean_json)
            raw_filters = data.get("ignore_substrings", [])
            if raw_filters:
                protected = get_allowed_extensions()
                for f in raw_filters:
                    f_lower = f.lower()
                    is_dangerous = f_lower in protected or f_lower in [ext.strip('.') for ext in protected] or len(f) <= 1
                    for ext in protected:
                        if f_lower.endswith(ext):
                            is_dangerous = True
                            break
                    if is_dangerous:
                        print(f"[WARNING] Discarded dangerous AI filter: {f}", file=sys.stderr)
                    else:
                        dynamic_filters.append(f)
                if dynamic_filters:
                    print(f"Dynamic filters generated: {dynamic_filters}", file=sys.stderr)
    except json.JSONDecodeError:
        print("Failed to parse dynamic filters from model.", file=sys.stderr)
        pass
        
    return dynamic_filters

def is_data_dump(sample):
    """Intelligently detects if a text snippet is a raw data dump (e.g. float arrays, minified code) rather than readable source code."""
    if len(sample) < 100:
        return False
        
    # Heuristic 1: Extremely high ratio of digits (e.g. float arrays, machine learning weights)
    digit_count = sum(c.isdigit() for c in sample)
    if digit_count / len(sample) > 0.3:
        return True
        
    # Heuristic 2: Extremely long lines (e.g. minified code, base64 blobs, SVG paths)
    lines = sample.split('\n')
    if any(len(line) > 5000 for line in lines):
        return True
        
    return False

def extract_ast_batch(file_paths, project_dir):
    """Calls codebase-memory-mcp to extract AST skeletons and imports for a batch of files."""
    import subprocess
    import json
    import sys
    import os
    
    # 1. Start codebase-memory-mcp and index project
    local_app_data = os.environ.get('LOCALAPPDATA', os.path.join(os.path.expanduser('~'), 'AppData', 'Local'))
    cbm_exe = os.path.join(local_app_data, 'Programs', 'codebase-memory-mcp', 'codebase-memory-mcp.exe')
    if not os.path.exists(cbm_exe):
        print(f"Error: codebase-memory-mcp not found at {cbm_exe}", file=sys.stderr)
        return {}

    print(f"Indexing project with codebase-memory-mcp...", file=sys.stderr)
    try:
        subprocess.run([cbm_exe, 'cli', 'index_repository', json.dumps({"repo_path": project_dir})], capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Index error (safe to ignore if already indexed): {e.stderr}", file=sys.stderr)

    # 2. Derive project name slug (same logic as codebase-memory-mcp list_projects)
    try:
        res = subprocess.run([cbm_exe, 'cli', 'list_projects'], capture_output=True, text=True, check=True)
        data = json.loads(res.stdout)
        proj_name = None
        for p in data.get("projects", []):
            if os.path.normpath(p["root_path"]) == os.path.normpath(project_dir):
                proj_name = p["name"]
                break
        if not proj_name:
            # Fallback slug
            proj_name = project_dir.replace(':', '').replace('\\', '-').replace('/', '-')
    except Exception as e:
        print(f"Failed to list projects: {e}", file=sys.stderr)
        return {}

    def query(q):
        try:
            res = subprocess.run([cbm_exe, 'cli', 'query_graph', json.dumps({"project": proj_name, "query": q})], capture_output=True, text=True, check=True)
            for line in reversed(res.stdout.splitlines()):
                if line.startswith('{'):
                    return json.loads(line).get("rows", [])
        except Exception:
            pass
        return []

    print(f"Querying AST structure...", file=sys.stderr)
    ast_data = {}
    
    # Initialize all requested file_paths
    for fp in file_paths:
        ast_data[os.path.abspath(fp)] = {"classes": [], "functions": [], "imports": []}

    # Fetch classes and their methods
    classes = query("MATCH (c:Class) RETURN c.name, c.file_path")
    class_map = {}
    for row in classes:
        c_name, c_path = row
        abs_path = os.path.abspath(os.path.join(project_dir, c_path))
        if abs_path in ast_data:
            c_dict = {"name": c_name, "methods": []}
            ast_data[abs_path]["classes"].append(c_dict)
            class_map[f"{abs_path}::{c_name}"] = c_dict

    methods = query("MATCH (c:Class)-[:DEFINES_METHOD]->(m:Method) RETURN c.name, m.name, c.file_path")
    for row in methods:
        c_name, m_name, c_path = row
        abs_path = os.path.abspath(os.path.join(project_dir, c_path))
        c_key = f"{abs_path}::{c_name}"
        if c_key in class_map:
            class_map[c_key]["methods"].append(m_name)

    # Fetch functions
    functions = query("MATCH (f:Function) RETURN f.name, f.file_path")
    for row in functions:
        f_name, f_path = row
        abs_path = os.path.abspath(os.path.join(project_dir, f_path))
        if abs_path in ast_data:
            ast_data[abs_path]["functions"].append(f_name)

    # Fetch imports
    imports = query("MATCH (src:File)-[r:IMPORTS]->(dst) RETURN src.file_path, dst.file_path, r.local_name")
    for row in imports:
        src_path, dst_path, local_name = row
        abs_path = os.path.abspath(os.path.join(project_dir, src_path))
        if abs_path in ast_data:
            if local_name:
                ast_data[abs_path]["imports"].append(f"import {local_name}")
            else:
                ast_data[abs_path]["imports"].append(f"import {os.path.basename(str(dst_path))}")

    return ast_data

def build_knowledge_graph_json(ast_data, project_dir):
    """Builds a knowledge-graph.json compliant with Understand-Anything schema from AST data."""
    nodes = []
    edges = []
    
    file_nodes = {}
    
    # Pre-process all known file paths to resolve imports later
    for abs_path in ast_data.keys():
        rel_path = os.path.relpath(abs_path, project_dir).replace('\\', '/')
        file_nodes[rel_path] = True
        nodes.append({
            "id": rel_path,
            "type": "file",
            "name": os.path.basename(rel_path),
            "filePath": rel_path,
            "summary": "Source file",
            "tags": ["file"],
            "complexity": "moderate"
        })
        
    for abs_path, data in ast_data.items():
        if "error" in data: continue
        
        rel_path = os.path.relpath(abs_path, project_dir).replace('\\', '/')
        
        # Add classes and functions
        for c in data.get("classes", []):
            node_id = f"{rel_path}::{c['name']}"
            nodes.append({
                "id": node_id,
                "type": "class",
                "name": c['name'],
                "filePath": rel_path,
                "summary": "Class definition",
                "tags": ["class"],
                "complexity": "moderate"
            })
            edges.append({
                "source": rel_path,
                "target": node_id,
                "type": "contains",
                "direction": "forward",
                "weight": 1.0
            })
            
        for f in data.get("functions", []):
            node_id = f"{rel_path}::{f}"
            nodes.append({
                "id": node_id,
                "type": "function",
                "name": f,
                "filePath": rel_path,
                "summary": "Function definition",
                "tags": ["function"],
                "complexity": "moderate"
            })
            edges.append({
                "source": rel_path,
                "target": node_id,
                "type": "contains",
                "direction": "forward",
                "weight": 1.0
            })
            
        # Add imports (crude resolution)
        for imp in data.get("imports", []):
            imp_normalized = imp.replace('\\', '/')
            resolved_target = None
            
            if imp_normalized.startswith('./') or imp_normalized.startswith('../'):
                # Resolve relative import
                base_dir = os.path.dirname(rel_path)
                target = os.path.normpath(os.path.join(base_dir, imp_normalized)).replace('\\', '/')
                # Check for extensions
                for ext in ['', '.py', '.js', '.ts', '.mjs', '.tsx', '.jsx']:
                    if target + ext in file_nodes:
                        resolved_target = target + ext
                        break
            else:
                # Absolute or module import (e.g. 'Nakhu.reasoning_engine' -> 'Nakhu/reasoning_engine.py')
                dot_to_slash = imp_normalized.replace('.', '/')
                for ext in ['', '.py', '.js', '.ts', '.mjs', '.tsx', '.jsx']:
                    if dot_to_slash + ext in file_nodes:
                        resolved_target = dot_to_slash + ext
                        break
                
                # If not found via dots, try raw path (e.g. 'src/utils')
                if not resolved_target:
                    for ext in ['', '.py', '.js', '.ts', '.mjs', '.tsx', '.jsx']:
                        if imp_normalized + ext in file_nodes:
                            resolved_target = imp_normalized + ext
                            break
                            
            if resolved_target:
                edges.append({
                    "source": rel_path,
                    "target": resolved_target,
                    "type": "depends_on",
                    "direction": "forward",
                    "weight": 0.8
                })
                        
    # Generate project overview and tour using Ollama
    print("Generating project overview and tour with Ollama...", file=sys.stderr)
    file_list_summary = ", ".join(list(file_nodes.keys())[:50]) # Limit to 50 files
    prompt = f"""You are analyzing a codebase with the following files: {file_list_summary}
Return a JSON object matching exactly this structure:
{{
  "description": "A 2-3 sentence description of what this project does based on the files.",
  "languages": ["lang1", "lang2"],
  "frameworks": ["framework1"],
  "tour": [
    {{
      "order": 1,
      "title": "Entry point",
      "description": "Start here.",
      "nodeIds": ["path/to/main.py"]
    }}
  ]
}}
Only output the JSON object. Do not wrap in markdown."""
    
    metadata = {}
    raw_response = query_ollama(prompt)
    if raw_response:
        try:
            clean_json = re.sub(r'^```json\s*', '', raw_response)
            clean_json = re.sub(r'\s*```$', '', clean_json)
            metadata = json.loads(clean_json)
        except json.JSONDecodeError:
            pass

    # Auto-generate layers based on top-level folders because the UI requires at least one layer to render the Overview
    layer_map = {}
    for node in nodes:
        if node["type"] != "file":
            continue
        rel_path = node["filePath"]
        parts = rel_path.split('/')
        layer_id = parts[0] if len(parts) > 1 else "Root"
        if layer_id not in layer_map:
            layer_map[layer_id] = {
                "id": layer_id,
                "name": layer_id,
                "description": f"Component: {layer_id}",
                "nodeIds": []
            }
        layer_map[layer_id]["nodeIds"].append(node["id"])
    
    layers = list(layer_map.values())

    graph = {
        "version": "1.0.0",
        "project": {
            "name": os.path.basename(os.path.abspath(project_dir)),
            "languages": metadata.get("languages", []),
            "frameworks": metadata.get("frameworks", []),
            "description": metadata.get("description", "Generated by ollama-system-mapper"),
            "analyzedAt": datetime.utcnow().isoformat() + "Z",
            "gitCommitHash": "unknown"
        },
        "nodes": nodes,
        "edges": edges,
        "layers": layers,
        "tour": metadata.get("tour", [])
    }
    
    ua_dir = os.path.join(project_dir, ".understand-anything")
    os.makedirs(ua_dir, exist_ok=True)
    out_path = os.path.join(ua_dir, "knowledge-graph.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(graph, f, indent=2)
    print(f"Generated Understand-Anything graph at {out_path}", file=sys.stderr)

def should_skip(name, dynamic_filters=None):
    """
    Determines if a file or directory should be skipped during ingestion.
    Uses boundary checks to prevent substring false-positives (e.g. 'authentication' triggering on 'temp').
    """
    lower_name = name.lower()
    defaults = {'test', 'tests', 'scratch', 'tmp', 'temp'}
    
    if lower_name in defaults:
        return True
        
    for d in defaults:
        if lower_name.startswith(f"{d}_") or lower_name.endswith(f"_{d}") or f"_{d}_" in lower_name or f"/{d}" in lower_name:
            return True
            
    if dynamic_filters:
        for skip in dynamic_filters:
            if skip.lower() in lower_name:
                return True
    return False

def format_ast_chunk(file_ast):
    """Formats the AST JSON into a dense text chunk for RAG."""
    if not file_ast or "error" in file_ast:
        return ""
    chunk = ""
    classes = file_ast.get("classes", [])
    functions = file_ast.get("functions", [])
    imports = file_ast.get("imports", [])
    
    if classes or functions:
        chunk += "STRUCTURAL SKELETON:\n"
        for c in classes:
            chunk += f"class {c.get('name')}:\n"
            for m in c.get("methods", []):
                chunk += f"  def {m}\n"
        for f in functions:
            chunk += f"def {f}\n"
    
    if imports:
        chunk += "\nIMPORTS:\n" + "\n".join(imports) + "\n"
        
    return chunk.strip()

def ingest_directory(project_dir, db_path, max_file_size=500000):
    """Walks the directory in a single pass, processes files, and ingests them into the RAG DB using BLOB storage."""
    print("Ingesting project into RAG database...", file=sys.stderr)
    conn = init_db(db_path)
    c = conn.cursor()
    c.execute("DELETE FROM chunks") # Clear old data
    
    ignore_dirs = get_ignore_dirs()
    allowed_exts = get_allowed_extensions()
    dynamic_filters = generate_dynamic_filters(project_dir, ignore_dirs)
    
    valid_files = []
    
    for root, dirs, files in os.walk(project_dir):
        dirs[:] = [d for d in dirs if d not in ignore_dirs and not should_skip(d, dynamic_filters)]
        for file in files:
            if should_skip(file, dynamic_filters): continue
            
            ext = os.path.splitext(file)[1].lower()
            if ext in allowed_exts or file.endswith('Dockerfile'):
                file_path = os.path.join(root, file)
                if file in ['.system_mapper_cache.json', '.system_mapper_db.sqlite']:
                    continue
                try:
                    if os.path.getsize(file_path) > max_file_size:
                        continue
                except OSError:
                    continue
                valid_files.append(file_path)
                
    print(f"Found {len(valid_files)} files. Extracting AST structures...", file=sys.stderr)
    ast_data = extract_ast_batch(valid_files, project_dir)
    
    print("Building Understand-Anything graph...", file=sys.stderr)
    build_knowledge_graph_json(ast_data, project_dir)

    processed_files = 0
    for file_path in valid_files:
        rel_path = os.path.relpath(file_path, project_dir)
        processed_files += 1
        print(f"Embedding file {processed_files}: {rel_path}", file=sys.stderr)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                sample = f.read(10000)
                if is_data_dump(sample):
                    continue
                    
                content = sample + f.read()
                
                # Dynamic chunking: reduce overlap if file is large to save DB space
                overlap = 200 if len(content) < 50000 else 50
                chunks = chunk_text(content, chunk_size=1500, overlap=overlap)
                
                ast_chunk = format_ast_chunk(ast_data.get(os.path.abspath(file_path)))
                if ast_chunk:
                    chunks.insert(0, ast_chunk)
                    
                if not chunks:
                    continue
                
                batch_size = 100
                num_batches = math.ceil(len(chunks) / batch_size)
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i+batch_size]
                    b_embs = get_embeddings_batch(batch)
                    if b_embs and len(b_embs) == len(batch):
                        records = [(rel_path, chunk, serialize_embedding(emb)) for chunk, emb in zip(batch, b_embs)]
                        c.executemany("INSERT INTO chunks (filepath, content, embedding) VALUES (?, ?, ?)", records)
                    else:
                        print(f"\n  -> Error: Embedding batch {i//batch_size + 1} failed.", file=sys.stderr)
                        
                    if num_batches > 5:
                        current_batch = (i // batch_size) + 1
                        print(f"    -> Progress: {current_batch}/{num_batches} batches...", file=sys.stderr, end='\r')
                
                conn.commit()
                if num_batches > 5:
                    print(file=sys.stderr) # newline to clear the \r
        except Exception as e:
            print(f"  -> Error processing {rel_path}: {e}", file=sys.stderr)
            
    conn.close()
    print(f"Ingestion complete. Processed {processed_files} files.", file=sys.stderr)

def update_files_rag(files, project_dir, db_path, max_file_size=500000):
    """Incrementally updates specific files in the existing RAG database."""
    print(f"Incrementally updating {len(files)} files in RAG database...", file=sys.stderr)
    conn = init_db(db_path)
    c = conn.cursor()
    
    valid_files = [f for f in files if os.path.exists(f) and os.path.getsize(f) <= max_file_size]
    print(f"Extracting AST structures for {len(valid_files)} files...", file=sys.stderr)
    ast_data = extract_ast_batch(valid_files, project_dir)
    
    for file_path in files:
        if not file_path.strip(): continue
        rel_path = os.path.relpath(file_path, project_dir)
        print(f"Updating {rel_path}...", file=sys.stderr)
        
        c.execute("DELETE FROM chunks WHERE filepath = ?", (rel_path,))
        
        if file_path not in valid_files:
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                sample = f.read(10000)
                if is_data_dump(sample):
                    print(f"  -> Skipping {rel_path} (detected as raw data dump)", file=sys.stderr)
                    continue
                    
                content = sample + f.read()
                
                overlap = 200 if len(content) < 50000 else 50
                chunks = chunk_text(content, chunk_size=1500, overlap=overlap)
                
                ast_chunk = format_ast_chunk(ast_data.get(os.path.abspath(file_path)))
                if ast_chunk:
                    chunks.insert(0, ast_chunk)
                    
                if not chunks:
                    continue
                
                batch_size = 100
                num_batches = math.ceil(len(chunks) / batch_size)
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i+batch_size]
                    b_embs = get_embeddings_batch(batch)
                    if b_embs and len(b_embs) == len(batch):
                        records = [(rel_path, chunk, serialize_embedding(emb)) for chunk, emb in zip(batch, b_embs)]
                        c.executemany("INSERT INTO chunks (filepath, content, embedding) VALUES (?, ?, ?)", records)
                    else:
                        print(f"\n  -> Error: Embedding batch {i//batch_size + 1} failed.", file=sys.stderr)
                        
                    if num_batches > 5:
                        current_batch = (i // batch_size) + 1
                        print(f"    -> Progress: {current_batch}/{num_batches} batches...", file=sys.stderr, end='\r')
                        
                conn.commit()
                if num_batches > 5:
                    print(file=sys.stderr)
        except Exception as e:
            print(f"  -> Error updating {rel_path}: {e}", file=sys.stderr)
            
    conn.commit()
    conn.close()
    print("Incremental update complete.", file=sys.stderr)

def search_rag(query, db_path, top_k=15):
    """Searches the database for chunks semantically similar to the query using BLOB deserialization."""
    if not os.path.exists(db_path):
        return ""
    
    query_embs = get_embeddings_batch([query])
    if not query_embs:
        return ""
    query_emb = query_embs[0]
        
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT filepath, content, embedding FROM chunks")
    
    results = []
    for row in c.fetchall():
        filepath, content, emb_blob = row
        emb = deserialize_embedding(emb_blob)
        sim = cosine_similarity(query_emb, emb)
        results.append((sim, filepath, content))
        
    conn.close()
    
    results.sort(key=lambda x: x[0], reverse=True)
    top_results = results[:top_k]
    
    context = ""
    for sim, filepath, content in top_results:
        context += f"--- File: {filepath} (Similarity: {sim:.2f}) ---\n{content}\n\n"
    return context

def gather_file_content(file_path, base_dir, max_file_size=500000):
    """Gathers content of a specific file, protected by size gates and data dump heuristics."""
    try:
        if os.path.getsize(file_path) > max_file_size:
            return f"\n--- File: {os.path.relpath(file_path, base_dir)} (SKIPPED: Exceeds {max_file_size} bytes) ---\n"
            
        with open(file_path, 'r', encoding='utf-8') as f:
            sample = f.read(10000)
            if is_data_dump(sample):
                return f"\n--- File: {os.path.relpath(file_path, base_dir)} (SKIPPED: Detected as raw data dump) ---\n"
            content = sample + f.read()
            return f"\n--- File: {os.path.relpath(file_path, base_dir)} ---\n{content}\n"
    except Exception as e:
        return ""

def query_ollama_stream(prompt, model="qwen2.5-coder:7b", url="http://localhost:11434/api/generate"):
    """Streams the LLM generation to stdout dynamically without dumping raw JSON brackets to the UI."""
    data = {"model": model, "prompt": prompt, "stream": True, "format": "json", "options": {"num_ctx": 16384, "temperature": 0.1, "top_p": 0.9, "num_gpu": -1}}
    req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'), headers={'Content-Type': 'application/json'})
    
    full_response = ""
    try:
        in_answer_block = False
        buffer = ""
        with urllib.request.urlopen(req, timeout=120) as response:
            for line in response:
                if line:
                    chunk = json.loads(line.decode('utf-8'))
                    text = chunk.get("response", "")
                    full_response += text
                    
                    buffer += text
                    if not in_answer_block and '"answer":' in buffer:
                        in_answer_block = True
                        extract = buffer.split('"answer":')[1].lstrip(' "')
                        if extract:
                            print(extract.replace('\\n', '\n').replace('\\"', '"'), end='', flush=True)
                        buffer = ""
                    elif in_answer_block:
                        if '","' in text or '": "' in text:
                            in_answer_block = False
                        else:
                            print(text.replace('\\n', '\n').replace('\\"', '"'), end='', flush=True)
            print() # newline
    except Exception as e:
        print(f"\nError connecting to Ollama: {e}", file=sys.stderr)
        return '{"answer": "Ollama offline fallback", "updated_map": "", "patches": []}'
    return full_response

def save_patches(patches, project_dir):
    """Saves generated code patches to the .ollama_patches directory."""
    patch_dir = os.path.join(project_dir, ".ollama_patches")
    os.makedirs(patch_dir, exist_ok=True)
    for idx, p in enumerate(patches):
        filename = p.get("file", f"patch_{idx}").replace("/", "_").replace("\\", "_")
        patch_path = os.path.join(patch_dir, f"{filename}.patch")
        with open(patch_path, 'w', encoding='utf-8') as f:
            f.write(p.get("diff", ""))
        print(f"Saved patch: {patch_path}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(description="Ollama System Mapper for Antigravity")
    parser.add_argument("--dir", required=True, help="Root directory")
    parser.add_argument("--query", required=True, help="Query")
    parser.add_argument("--mode", required=True, choices=['global', 'focused', 'relational'], help="Scan mode")
    parser.add_argument("--target", required=False, help="Specific file")
    parser.add_argument("--reindex", action="store_true", help="Force DB re-ingestion")
    parser.add_argument("--patch", action="store_true", help="Enable auto-patching")
    parser.add_argument("--update-files-list", required=False, help="Path to text file containing list of absolute paths to update incrementally")
    parser.add_argument("--visualize", action="store_true", help="Launch Understand-Anything dashboard")
    
    args = parser.parse_args()

    project_dir = os.path.abspath(args.dir)
    db_path = os.path.join(project_dir, ".ollama_rag.db")
    map_path = os.path.join(project_dir, ".ollama_system_map.md")
    
    if args.update_files_list and os.path.exists(args.update_files_list):
        with open(args.update_files_list, 'r', encoding='utf-8') as f:
            files_to_update = f.read().splitlines()
        update_files_rag(files_to_update, project_dir, db_path)
    elif args.mode == 'global' and (args.reindex or not os.path.exists(db_path)):
        ingest_directory(project_dir, db_path)
        
    if args.visualize:
        print("\nLaunching Understand-Anything dashboard...", file=sys.stderr)
        dashboard_dir = r"C:\Users\axton\.gemini\antigravity\scratch\understand_anything_ui\Understand-Anything\understand-anything-plugin\packages\dashboard"
        env = os.environ.copy()
        env["GRAPH_DIR"] = project_dir
        # Launch using start to run independently in browser
        subprocess.Popen(["npm", "run", "dev"], cwd=dashboard_dir, env=env, shell=True)
        print("Dashboard is starting on localhost (usually http://localhost:5173).", file=sys.stderr)
        print("Keep this terminal open, or press Ctrl+C to stop it if running interactively.", file=sys.stderr)

    context = ""
    if args.mode == 'global':
        context = search_rag(args.query, db_path)
    elif args.mode == 'focused':
        context = gather_file_content(os.path.join(project_dir, args.target), project_dir)
    elif args.mode == 'relational':
        target_content = gather_file_content(os.path.join(project_dir, args.target), project_dir)
        rag_context = search_rag(args.query, db_path, top_k=5)
        context = f"{target_content}\n--- RELATED CONTEXT ---\n{rag_context}"

    system_map_content = ""
    if os.path.exists(map_path):
        with open(map_path, 'r', encoding='utf-8') as f:
            system_map_content = f.read()

    system_instruction = (
        "You are an expert system mapper, code auditor, and technical assistant. "
        "You MUST respond ONLY with a valid JSON object matching this exact structure. Do NOT wrap it in markdown code blocks. Just output raw JSON:\n"
        "{\n"
        '  "answer": "Your detailed answer to the query.",\n'
        '  "updated_map": "The complete markdown content for the System Architecture Map. If you learned new things, update it. If none existed, create one.",\n'
    )
    if args.patch:
        system_instruction += '  "patches": [{"file": "path/to/file", "diff": "unified diff content"}]\n'
    else:
         system_instruction += '  "patches": []\n'
    system_instruction += "}\n\n"

    prompt = f"{system_instruction}--- EXISTING SYSTEM MAP ---\n{system_map_content}\n\n--- CONTEXT ---\n{context}\n\n--- QUERY ---\n{args.query}"

    print(f"\nSending streaming JSON request to Ollama ({args.mode} mode)...", file=sys.stderr)
    print("="*40, file=sys.stderr)
    
    raw_response = query_ollama_stream(prompt)
    
    print("="*40, file=sys.stderr)
    
    try:
        clean_json = re.sub(r'^```json\s*', '', raw_response)
        clean_json = re.sub(r'\s*```$', '', clean_json)
        data = json.loads(clean_json)
        
        updated_map = data.get("updated_map", "")
        if updated_map and len(updated_map) > 50:
            with open(map_path, 'w', encoding='utf-8') as f:
                f.write(updated_map)
            print(f"System map updated: {map_path}", file=sys.stderr)
            
        if args.patch and data.get("patches"):
            save_patches(data["patches"], project_dir)
            
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON response: {e}", file=sys.stderr)
        with open(os.path.join(project_dir, "ollama_failed_response.txt"), 'w') as f:
            f.write(raw_response)

if __name__ == "__main__":
    main()
