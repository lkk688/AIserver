import openai
import json

def test_tool_call(base_url, model_name, api_key="sk-dummy"):
    print(f"\n--- Testing Tool Call with {model_name} at {base_url} ---")
    client = openai.OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "google_search",
                "description": "Search Google for real-time information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query."
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ]

    messages = [
        {"role": "system", "content": "You are a helpful assistant with access to tools. If a user asks a question that requires external information, use the available tools. Only use the tools provided."},
        {"role": "user", "content": "What is the release date of Qwen3?"}
    ]
    
    print(f"User: {messages[1]['content']}")

    try:
        # Step 1: Send request with tools
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        
        message = response.choices[0].message
        
        # Check if tool was called
        if message.tool_calls:
            print("Model requested tool call:")
            for tool_call in message.tool_calls:
                print(f"  Function: {tool_call.function.name}")
                print(f"  Arguments: {tool_call.function.arguments}")
                
                # Mock execution
                if tool_call.function.name == "google_search":
                    args = json.loads(tool_call.function.arguments)
                    query = args.get("query")
                    # Mock result
                    tool_result = f"Search result for '{query}': Qwen3 was released in April 2025."
                    
                    messages.append(message)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result
                    })
                    print(f"  > Executed tool. Result: {tool_result}")

            # Step 2: Get final response
            final_response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=tools
            )
            print("Final Response:", final_response.choices[0].message.content)
        else:
            print("Model did not call tool. Response:", message.content)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Test via LiteLLM
    test_tool_call("http://localhost:4000", "qwen3")
    
    # Test via vLLM directly
    test_tool_call("http://localhost:8001/v1", "Qwen/Qwen2.5-14B-Instruct-AWQ")
