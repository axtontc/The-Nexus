---
name: local-intelligence-hub
description: >-
  The Local Intelligence Hub (Nexus). A Monolithic API Server for Ollama that unifies System Mapping, Task Farming, Local Consultation, and Tool Orchestration.
license: MIT
metadata:
  version: v1.0.0
  publisher: Axton Carroll
---

# Local Intelligence Hub (Nexus)

The Local Intelligence Hub is a Monolithic API Server for Ollama that unifies System Mapping, Task Farming, Local Consultation, and Tool Orchestration into a single, high-performance architecture.

## Modes of Operation

When the user asks you to map the codebase, start a task farm, consult a local model, or orchestrate tools, you invoke the `nexus` CLI:

### 1. Nexus Serve
Launch the Monolithic API Server (Skillbrary/Ollama Orchestrator) in the background.
**Command:** `nexus serve --port 8080`

### 2. Nexus Farm
Dispatch a queue of simple, independent tasks to a parallel pool of local Ollama models using Synchronous Threading.
**Command:** `nexus farm --queue <path_to_json>`

### 3. Nexus Map
Update the Persistent ChromaDB System Mapper.
**Command:** `nexus map --dir <project_directory>`

### 4. Nexus Consult
Consult the local LLM via Ollama to get a "second opinion", brainstorm, or bounce ideas back and forth.
**Command:** `nexus consult --prompt "Your question here"`
