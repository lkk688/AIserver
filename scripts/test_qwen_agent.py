import os
from qwen_agent.agents import Assistant

# 1. 记得去 serper.dev 免费注册并填入真实的 Key 哦！
#export SERPER_API_KEY="你的真实KEY写在这里"
#os.environ['SERPER_API_KEY'] = '请在这里填入真实的Key'

llm_cfg = {
    'model': 'qwen3.5-9b',
    'model_server': 'http://100.110.236.127:8000/v1',
    'api_key': 'EMPTY',
}

bot = Assistant(
    llm=llm_cfg,
    name='WebSearchAgent',
    description='一个拥有联网搜索能力的智能助手',
    function_list=['web_search'] 
)

messages = [{'role': 'user', 'content': '请帮我搜索一下2026年最近一周关于 SpaceX 的大事件。'}]

print("🤖 Agent 正在思考和执行...\n")

# 将所有的中间状态收集起来
responses = list(bot.run(messages))

# 拿到最后一步的最终结果
final_messages = responses[-1]

# 遍历整个对话过程，以人类可读的方式打印
for msg in final_messages:
    role = msg['role']
    content = msg['content']
    
    if role == 'user':
        print(f"🧑 提问: {content}\n")
    elif role == 'assistant':
        # 如果模型调用了工具
        if 'function_call' in msg and msg['function_call']:
            tool_name = msg['function_call']['name']
            tool_args = msg['function_call']['arguments']
            print(f"🛠️ [调用工具]: 决定使用 `{tool_name}`，搜索参数: {tool_args}\n")
        # 打印模型的回答（包括它的思考过程 <think>）
        if content:
            print(f"🤖 助手: {content}\n")
    elif role == 'function':
        # 打印工具返回的结果
        print(f"🌐 [搜索结果返回]: {content[:200]}... (已省略超长内容)\n")

print("-" * 50)
print("✅ 测试结束")