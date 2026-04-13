# ClinicalRAG — Architecture Deep Dive

## Part 3 — The Agent and MCP Server

This part explains the MCP server and the private agent — what they are, how they work, why they exist, and the tradeoffs involved.

---

### What MCP is

MCP (Model Context Protocol) is an open standard created by Anthropic that defines how AI models connect to external tools and data sources.

Before MCP, every AI application built custom integrations with every tool it needed. A company building an AI assistant for their HR system would write custom code to connect the LLM to their database, their ticketing system, their document store — all different, all proprietary.

MCP standardizes this. It defines a protocol that any AI client and any tool can speak. The result:

- Build one MCP server for your RAG pipeline → any MCP client (Claude Desktop, Claude Code, custom agents) can use it automatically
- The AI client does not need to know how RAG works internally — it just sees a tool called `query_rag` with a description
- Switch from one AI model to another → the MCP server does not change

Think of it like USB. Before USB, every device had its own connector. MCP is the USB standard for AI tools.

---

### The two sides of MCP

**MCP server** — what you built. A program that:
- Exposes tools with names and descriptions
- Handles tool calls when an AI client invokes them
- Returns results in a standardized format

**MCP client** — the AI application that uses your tools. Examples:
- Claude Desktop (what you demoed)
- Claude Code
- A custom Python agent (what you also built)
- Any application built on the MCP SDK

---

### Why MCP matters for enterprise deployments

Enterprise organizations deal with many internal systems — ERPs, HR platforms, document stores, ticketing systems. Without a standard, every AI integration requires custom code for each system.

With MCP, each system gets one MCP server. The same AI agent connects to all of them automatically. Adding a new system means building one MCP server — the agent does not change.

```
Without MCP:
AI agent → custom code for ERP system
AI agent → custom code for HR platform
AI agent → custom code for document store
Each integration different, each requires separate maintenance

With MCP:
AI agent → MCP client → ERP MCP server
                      → HR MCP server
                      → Document RAG MCP server
One standard, all systems speak the same protocol
```

Your RAG MCP server is one example of this pattern. In a production deployment there would be many MCP servers — one per enterprise system the AI needs to access.

---

### `mcp_server/server.py` — walkthrough

```python
import httpx
from mcp.server.fastmcp import FastMCP
from pathlib import Path

mcp = FastMCP("ClinicalRAG")
API_URL = "http://localhost:8000"
```

**`FastMCP("ClinicalRAG")`:** creates the MCP server with a name. Claude Desktop shows this name when listing connected servers.

**`API_URL = "http://localhost:8000"`:** the MCP server calls your FastAPI app over HTTP. It does not import from `core/` directly. This keeps the MCP server independent — it only knows that there is an API at this address, not how it works internally.

---

### The `query_rag` tool

```python
@mcp.tool()
async def query_rag(question: str) -> str:
    """Ask a question over indexed clinical documents."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{API_URL}/chat",
            json={"question": question},
            timeout=60.0
        )
        response.raise_for_status()
        data = response.json()

    answer = data["answer"]
    sources = data["sources"]

    source_lines = []
    for source in sources:
        doc = source["document"]
        preview = source["text"][:100]
        source_lines.append(f"{doc}: {preview}")

    sources_text = "\n".join(source_lines)
    return f"{answer}\nSources:\n{sources_text}"
```

**`@mcp.tool()`:** registers this function as an MCP tool. The MCP SDK reads:
- The function name → tool name (`query_rag`)
- The docstring → tool description (what the AI reads to decide when to use it)
- The type hints → input and output schema

The AI client never sees your code. It sees:
```json
{
  "name": "query_rag",
  "description": "Ask a question over indexed clinical documents.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "question": {"type": "string"}
    },
    "required": ["question"]
  }
}
```

**The docstring is critical.** It is what the AI reads to decide whether to call this tool. A bad description means the AI might not use the tool when it should. A good description is specific — "ask a question over indexed clinical documents" tells the AI exactly when this tool is relevant.

**`-> str`:** MCP tools return strings. Not JSON, not structured data — plain text. The AI client reads the string and incorporates it into its response. This is why you format the output clearly:

```
The sponsor must retain documents for at least 15 years.
Sources:
gcp_basics.txt: ICH E6 Good Clinical Practice Guidelines...
```

**The function calls FastAPI, not Qdrant directly.** The MCP server has no knowledge of Qdrant, Mistral, or the RAG pipeline. It just calls `POST /chat`. This is the right level of abstraction — the MCP server is a connector, not an AI system.

---

### The `ingest_document` tool

```python
@mcp.tool()
async def ingest_document(file_path: str) -> str:
    """Ingest a text file into the clinical RAG system."""
    path = Path(file_path)

    async with httpx.AsyncClient() as client:
        with open(path, "rb") as f:
            upload = (path.name, f, "text/plain")
            response = await client.post(
                f"{API_URL}/documents/ingest",
                files={"file": upload},
                timeout=60.0
            )
        response.raise_for_status()
        data = response.json()

    return f"Indexed {data['filename']} into {data['chunks']} chunks."
```

**`Path(file_path)`:** uses `pathlib.Path` instead of string manipulation. `path.name` extracts just the filename from a full path — works correctly on Windows and Unix.

**`open(path, "rb")`:** binary mode for HTTP file uploads.

**`upload = (path.name, f, "text/plain")`:** the multipart upload tuple that `httpx` expects — filename, file object, content type. Extracted to a named variable for readability.

**This tool is NOT exposed to the private agent.** The agent filters tools to only `query_rag`. Llama3.1 8B sometimes calls the wrong tool when multiple tools are available. Since the agent is designed for Q&A only, hiding `ingest_document` prevents incorrect tool calls. Ingestion is handled by `scripts/ingest_all.py` instead.

---

### How Claude Desktop connects

When Claude Desktop starts, it reads its config file and launches your MCP server as a subprocess:

```json
{
  "mcpServers": {
    "ClinicalRAG": {
      "command": "C:\\path\\to\\.venv\\Scripts\\python.exe",
      "args": ["C:\\path\\to\\mcp_server\\server.py"]
    }
  }
}
```

Claude Desktop runs `python mcp_server/server.py` and communicates with it over stdio (standard input/output). Claude writes MCP protocol messages to the process's stdin; the server writes responses to stdout.

This is why running `python mcp_server/server.py` directly in a terminal produces an error — there is no MCP client on the other end sending properly formatted messages. The error is expected behavior when running without a client.

When Claude Desktop connects successfully:
1. It sends an `initialize` message
2. The server responds with its capabilities
3. Claude Desktop sends `list_tools` — the server responds with the tool definitions
4. Claude now knows what tools are available and can call them

---

### Privacy analysis of the MCP + Claude Desktop setup

When Claude Desktop calls `query_rag`:

```
Your documents → stay on your machine (Qdrant) ✅
Embeddings → generated on your GPU (Ollama) ✅
Vector search → runs in Qdrant on your machine ✅
Mistral generation → runs on your GPU ✅
The answer → sent to Claude Desktop → Anthropic's servers ❌
```

The raw documents never leave your machine. But the generated answer passes through Anthropic when displayed in Claude Desktop.

For a fully private deployment, Claude Desktop would be replaced with:
- A custom internal chat UI calling `/chat` directly, OR
- A private MCP client running on-premises (which is what `agent/agent.py` implements)

The MCP server itself does not change — only the client changes. This is the value of the MCP standard.

---

### `agent/agent.py` — walkthrough

The agent is a Python program that uses Llama3.1 as its reasoning model and connects to your MCP server to call tools. No Claude Desktop, no external services — fully on-premises.

```python
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.1"
MAX_ITERATIONS = 5
```

**Two models, two jobs:**
- `mistral` — runs inside FastAPI, generates answers from retrieved context
- `llama3.1` — runs in the agent, decides which tools to call and synthesizes the final answer

This is a multi-model architecture. Each model does what it is best at.

---

### Connecting to the MCP server

```python
server_params = StdioServerParameters(
    command="python",
    args=["mcp_server/server.py"]
)

async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
```

**`StdioServerParameters`:** tells the MCP client how to start the server — run `python mcp_server/server.py` as a subprocess. The client communicates over stdio, the same way Claude Desktop does.

**`stdio_client`:** starts the MCP server subprocess and creates read/write streams.

**`ClientSession`:** wraps the streams into a proper MCP session with methods like `list_tools` and `call_tool`.

**`session.initialize()`:** performs the MCP handshake — client and server agree on protocol version and capabilities.

---

### Tool discovery

```python
tools_response = await session.list_tools()
tools = []
for tool in tools_response.tools:
    if tool.name == "query_rag":
        tools.append({
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.inputSchema
        })
        print(f"MCP tool registered: {tool.name}")
```

**`session.list_tools()`:** asks the MCP server what tools are available. This is automatic tool discovery — the agent does not have tools hardcoded, it finds them at runtime.

**Why filter to `query_rag` only:** Llama3.1 8B sometimes calls the wrong tool when multiple tools are available. In testing, it sometimes called `ingest_document` with a `question` argument instead of a `file_path` — wrong tool, wrong arguments. Filtering to one tool eliminates this failure mode.

---

### The agent loop

```python
messages = [
    {
        "role": "system",
        "content": "You are a clinical research assistant. Use the query_rag tool to answer questions. Never answer from your own knowledge."
    },
    {
        "role": "user",
        "content": user_goal
    }
]

iteration = 0

while True:
    iteration += 1
    force_answer = iteration > MAX_ITERATIONS
    ...
```

**The system prompt** sets the agent's behavior for the entire conversation. "Never answer from your own knowledge" is the key instruction — without it, Llama might answer clinical questions from its training data, which could be wrong or outdated.

**`messages` list:** the full conversation history sent to Llama on every iteration. It grows with each loop:

```
Iteration 1:  [system, user_goal]
Iteration 2:  [system, user_goal, assistant(tool_call), tool(result)]
Iteration 3:  [system, user_goal, assistant(tool_call), tool(result), assistant(answer)]
```

Llama reads the entire history on each iteration. It sees the tool results and uses them to formulate its answer.

**`temperature: 0.1`:** controls randomness. 0 = completely deterministic, 1 = very random. Low temperature makes tool calling more consistent.

**`force_answer`:** after MAX_ITERATIONS, the agent stops giving Llama tools and asks it to summarize what it found. This prevents infinite loops. Without tools, Llama cannot call anything — it must produce a text answer.

---

### Tool call handling

```python
if reply.get("tool_calls"):
    for tool_call in reply["tool_calls"]:
        index = tool_call["function"]["index"]

        # llama3.1 sometimes sends empty tool name, use position instead
        name = tool_call["function"]["name"] or (tools[index]["name"] if index < len(tools) else None)

        if not name:
            continue

        args = tool_call["function"]["arguments"]
        print(f"\nCalling tool: {name}")

        try:
            result = await session.call_tool(name, args)
            tool_result = result.content[0].text
        except Exception as e:
            tool_result = f"Tool unavailable: {e}"

        messages.append({
            "role": "tool",
            "content": tool_result
        })
```

**`reply.get("tool_calls")`:** checks if Llama wants to call a tool. If it has enough information to answer directly, `tool_calls` will be `None` and the `else` branch fires.

**The empty name workaround:** Llama3.1 8B has a known bug where it sends tool calls with an empty `name` field. The `index` field is still populated. The workaround uses the index to look up the tool name:

```python
name = tool_call["function"]["name"] or (tools[index]["name"] if index < len(tools) else None)
```

Read as: "use the name if it exists, otherwise look up the tool by its position in the list, unless the index is out of range."

This is a real-world example of working around model limitations pragmatically.

**`session.call_tool(name, args)`:** sends the tool call to the MCP server via the stdio connection. The MCP server receives it, calls FastAPI, gets the result, and returns it.

**`try/except`:** tool calls can fail — FastAPI might be down, Qdrant might be unavailable. Instead of crashing the agent, you catch the exception, record the failure as the tool result, and let Llama decide what to do next.

**Adding the tool result to `messages`:** this feeds Llama the information it retrieved. The tool result becomes a `"tool"` role message. On the next iteration, Llama reads it and decides whether it has enough to answer.

---

### Why the agent sometimes loops

Llama3.1 8B has inconsistent tool calling behavior. Common failure modes:

**Calls the tool multiple times:** Llama found the answer but keeps exploring. MAX_ITERATIONS stops this.

**Calls the tool then ignores the result:** Llama does not incorporate the tool result into its reasoning. The force_answer mechanism helps.

**Answers from training data:** the system prompt says "never answer from your own knowledge" but Llama sometimes ignores this.

These are fundamental limitations of smaller models for agentic tasks. Larger models (70B+) are significantly more reliable. Models specifically trained for tool use rarely exhibit these behaviors.

This is an honest engineering tradeoff: smaller model = cheaper, faster, fully private, but less reliable. Larger model = more reliable, but requires more hardware or external API.

---

### Comparing the two interfaces

| | Claude Desktop + MCP | Private Agent |
|--|--|--|
| Agent model | Claude (Anthropic's cloud) | Llama3.1 (local GPU) |
| Privacy | Answer passes through Anthropic | Fully on-prem |
| Reliability | Very reliable tool calling | Inconsistent (8B model) |
| Setup | Config file + restart | `python agent/agent.py` |
| UI | Chat interface | Terminal |
| Best for | Demo, development | Production with larger model |

---

### The full data flow — private agent

When you type "How long must trial records be retained?":

```
1. agent.py sends to Llama3.1 via Ollama:
   messages=[system, "How long must trial records be retained?"]
   tools=[query_rag definition]

2. Llama3.1 responds:
   tool_calls=[{name: "query_rag", arguments: {question: "..."}}]

3. agent.py calls MCP server via stdio:
   session.call_tool("query_rag", {"question": "..."})

4. MCP server calls FastAPI:
   POST http://localhost:8000/chat
   {"question": "how long must records be retained?"}

5. FastAPI calls Ollama to embed the question:
   POST http://ollama:11434/api/embed
   → [0.21, -0.85, ...]

6. FastAPI calls Qdrant to search:
   query_points(vector=[0.21, ...], limit=3)
   → top 3 chunks from indexed documents

7. FastAPI calls Ollama (Mistral) to generate:
   POST http://ollama:11434/api/generate
   → "The sponsor must retain documents for at least 15 years."

8. FastAPI returns to MCP server:
   {"answer": "15 years...", "sources": [...]}

9. MCP server formats and returns to agent:
   "The sponsor must retain... Sources: gcp_basics.txt: ..."

10. agent.py adds result to messages, sends to Llama3.1 again:
    messages=[system, user_goal, tool_call, tool_result]

11. Llama3.1 reads the tool result and responds:
    "According to ICH E6 guidelines, the sponsor must retain
     all sponsor-specific essential documents for at least 15 years."

12. agent.py prints the answer
```

Every step runs on your machine. Nothing leaves your network.

---

### What makes this architecture extensible

**Adding a new tool to the MCP server:**

```python
@mcp.tool()
async def search_internal_database(query: str) -> str:
    """Search the internal database."""
    # call your internal API
    ...
```

One function with a decorator. The agent discovers it automatically on next startup. No changes to the agent loop.

**Adding new documents:**

Add more `.txt` or `.pdf` files to `data/samples/`, run `python scripts/ingest_all.py`. The new documents are searchable immediately. No code changes.

**Switching models:**

Change `LLM_MODEL=mistral` to `LLM_MODEL=llama3.2` in `.env`. Restart FastAPI. Done.

**Scaling to multiple users:**

FastAPI is stateless — every request is independent. Run multiple FastAPI instances behind a load balancer. Qdrant and Ollama are shared services. The architecture scales horizontally without code changes.

---

*Continue to Part 4 — Design Decisions*