import tiktoken

def estimate_tokens(text: str) -> int:
    return len(text) // 4

def compress_messages(messages, max_allowed_tokens):
    import copy
    msgs = copy.deepcopy(messages)
    
    iters = 0
    while True:
        iters += 1
        current_tokens = sum(estimate_tokens(m.get("content", "")) for m in msgs)
        if current_tokens <= max_allowed_tokens:
            break
            
        longest_idx = -1
        longest_len = 0
        for i, m in enumerate(msgs):
            if i in (0, 1):
                continue
            content_len = len(m.get("content", ""))
            if content_len > longest_len:
                longest_len = content_len
                longest_idx = i
                
        if longest_idx == -1 or longest_len < 200:
            break
            
        content = msgs[longest_idx]["content"]
        keep_chars = int(longest_len * 0.45) 
        
        msgs[longest_idx]["content"] = content[:keep_chars] + "\n...[TRUNCATED TO FIT CONTEXT]...\n" + content[-keep_chars:]
        if iters % 1000 == 0:
            print(f"Iter {iters}, len: {len(msgs[longest_idx]['content'])}")
            
    return msgs

msgs = [
    {"role": "system", "content": "A" * 5000},
    {"role": "user", "content": "B" * 5000},
    {"role": "assistant", "content": "C" * 300},
]
compress_messages(msgs, 2000)
print("Done")
