import asyncio
import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.1"
MAX_ITERATIONS = 5
DEBUG = False  # set to True to see full responses

def debug(msg):
    if DEBUG:
        print(f"[debug] {msg}")

async def run_agent(user_goal: str):
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server/server.py"]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("Connected to MCP server")

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

                if force_answer:
                    messages.append({
                        "role": "user",
                        "content": "Based on the information retrieved, give your final answer now."
                    })

                request = {
                    "model": OLLAMA_MODEL,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": 0.1}
                }

                if not force_answer:
                    request["tools"] = tools

                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{OLLAMA_URL}/api/chat",
                        json=request,
                        timeout=60.0
                    )

                reply = response.json()["message"]
                messages.append(reply)

                debug(f"full reply: {reply}")
                debug(f"tool calls: {reply.get('tool_calls')}")

                if force_answer:
                    print(f"\nAnswer: {reply['content']}")
                    break

                if reply.get("tool_calls"):
                    for tool_call in reply["tool_calls"]:
                        index = tool_call["function"]["index"]

                        # llama3.1 sometimes sends empty tool name, use position instead
                        name = tool_call["function"]["name"] or (tools[index]["name"] if index < len(tools) else None)

                        if not name:
                            continue

                        args = tool_call["function"]["arguments"]
                        print(f"\nCalling tool: {name}")
                        debug(f"tool call object: {tool_call}")

                        try:
                            result = await session.call_tool(name, args)
                            tool_result = result.content[0].text
                            debug(f"raw result: {result}")
                            debug(f"tool result: {tool_result}")
                        except Exception as e:
                            tool_result = f"Tool unavailable: {e}"

                        messages.append({
                            "role": "tool",
                            "content": tool_result
                        })
                else:
                    print(f"\nAnswer: {reply['content']}")
                    break

if __name__ == "__main__":
    goal = input("Enter goal: ")
    asyncio.run(run_agent(goal))