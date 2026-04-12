import asyncio
import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "Llama3.1"

async def run_agent(user_goal:str):
    # start the MCP server as a subprocess and connect to it
    server_params = StdioServerParameters(
        command='Python',
        args=['mcp_server/server.py']
    )
    # connect to mcp server
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print('Connected to MCP server')
            
            # ask the MCP server what tools are available
            # this way the agent discovers tools automatically
            tools_response = await session.list_tools()
            tools = []
            for tool in tools_response.tools:
                tools.append(
                    {
                        'name': tool.name,
                        'description': tool.description,
                        'parameters': tool.inputSchema
                    }
                )    
                print(f"Tool found: {tool.name}")
             
            # start conversation - tell the model to use tools, not its own knowledge     
            messages = [
                {
                    "role": "system",
                    "content": """You are a clinical research assistant.
                            You have one tool available: query_rag
                            To use it you must call it with exactly this format:
                            {"name": "query_rag", "arguments": {"question": "your question here"}}
                            Always use this tool to answer questions. Never answer from your own knowledge."""                                         
                },
                {
                    "role": "user",
                    "content": user_goal
                }
            ]
            
            # agent loop - keeps running until model gives a final answer
            while True:
                # send full conversation history + available tools to the model
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f'{OLLAMA_URL}/api/chat',
                        json={
                            'model': OLLAMA_MODEL,
                            'messages': messages,
                            "tools": tools,
                            'stream': False
                        },
                        timeout=60.0
                    )          
                    
                # get the model's reply and add it to history
                reply = response.json()['message']                            
                messages.append(reply)
                
                print({reply['content']})
                print(response.json()) 
                print(response.json()["message"])
                print(f"Full reply: {reply}")
                print(f"Tool calls: {reply.get('tool_calls')}")
                
                if reply.get('tool_calls'):
                    #model decided to call one or more tools
                    for tool_call in reply['tool_calls']:
                        print(f"Full tool call object: {tool_call}")
                        index = tool_call["function"]["index"]
                        name = tool_call["function"]["name"] or tools[index]["name"]
                        args = tool_call['function']['arguments']
                        print(f'\nCalling tool: {name}')
                        
                        # call the tool via MCP and get the result
                        result = await session.call_tool(name, args)
                        print(f"Raw result: {result}")
                        print(f"Content: {result.content}")
                        tool_result = result.content[0].text
                        print(f"Tool result: {tool_result}")
                        
                        
                        tool_result = result.content[0].text
                        
                        # add tool result to history so model can use it next         
                        messages.append(
                            {
                                'role': 'tool',
                                'content': tool_result
                            }
                        )
                else:
                    # no tool calls means model has a final answer
                    print(f"Answer: {reply['content']}")        
                    break                     
            
if __name__ == '__main__':
    goal = input('enter goal:')
    asyncio.run(run_agent(goal))     