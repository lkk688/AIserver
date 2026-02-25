import os
import json
import re
from pathlib import Path
from typing import Dict, List, Any

# ==========================================
# 配置路径 (请根据你的实际路径调整)
# ==========================================
SESSIONS_DIR = Path(".agent/sessions")
OUTPUT_DIR = Path("output")

SFT_OUT_FILE = "sft_dataset.jsonl"
DPO_OUT_FILE = "dpo_dataset.jsonl"

def parse_task_id(raw_task_id: str) -> str:
    """
    清理 task_id。
    如果你 JSONL 里的 task_id 是 '2026-02-23_133309_subtask_0'
    而 output 里的文件夹只有 '2026-02-23_133309'，可以在这里做裁剪匹配。
    """
    # 移除 _subtask_x 后缀，只保留基础 Task ID 以匹配 output 文件夹
    return re.sub(r'_subtask_\d+', '', raw_task_id)

def format_final_code_as_assistant(code: str) -> str:
    """将裸代码包装回模型习惯的 Format B (WRITE_FILE) 格式"""
    return (
        "## Reasoning\n"
        "Here is the complete and corrected implementation.\n\n"
        "## Action\n"
        "WRITE_FILE: task.py\n"
        "<<<CONTENT\n"
        f"{code.strip()}\n"
        "CONTENT>>>"
    )

def main():
    tasks_db: Dict[str, Dict[str, Any]] = {}

    print("1. 正在扫描 Agent Sessions 数据...")
    # 扫描所有 rl_trajectory.jsonl
    for jsonl_path in SESSIONS_DIR.rglob("*.jsonl"):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    data = json.loads(line)
                    raw_task_id = data.get("task_id", "")
                    reward = data.get("reward", 0.0)
                    messages = data.get("messages", [])
                    
                    if not raw_task_id or not messages:
                        continue
                        
                    task_id = parse_task_id(raw_task_id)
                    
                    if task_id not in tasks_db:
                        tasks_db[task_id] = {
                            "initial_prompt": messages[:2], # [System, User(Goal)]
                            "success_trajectories": [],
                            "fail_trajectories": [],
                            "final_code": None
                        }
                    
                    if reward > 0:
                        tasks_db[task_id]["success_trajectories"].append(messages)
                    else:
                        tasks_db[task_id]["fail_trajectories"].append(messages)
                except Exception as e:
                    print(f"解析错误 {jsonl_path}: {e}")

    print(f"共发现 {len(tasks_db)} 个独立任务。")

    print("2. 正在扫描 Output 目录中的 Final Code...")
    code_found_count = 0
    for task_id in tasks_db.keys():
        # 假设最终代码路径是 output/task_id/task.py
        code_path = OUTPUT_DIR / task_id / "task.py"
        if code_path.exists():
            tasks_db[task_id]["final_code"] = code_path.read_text(encoding="utf-8")
            code_found_count += 1
    print(f"匹配到 {code_found_count} 份人工/AI审查后的 Final Code。")

    print("3. 正在构建 SFT 和 DPO 数据集...")
    sft_records = []
    dpo_records = []

    for task_id, task_data in tasks_db.items():
        initial_prompt = task_data["initial_prompt"]
        final_code = task_data["final_code"]
        success_traj = task_data["success_trajectories"]
        fail_traj = task_data["fail_trajectories"]

        # ---------------------------------------------------------
        # 制作 SFT 数据 (高质量正向教学)
        # ---------------------------------------------------------
        if final_code:
            # 策略 A：如果有你审查过的 final code，它是最高优先级的金标准！
            # 我们抛弃中间的 Debug 挣扎，直接让模型学 "一击必中"
            perfect_assistant_msg = {
                "role": "assistant", 
                "content": format_final_code_as_assistant(final_code)
            }
            sft_records.append({
                "task_id": task_id,
                "type": "distilled_final_code",
                "messages": initial_prompt + [perfect_assistant_msg]
            })
        elif success_traj:
            # 策略 B：如果没有 final code，但 Agent 自己跑通了，取回合数最少的那条
            best_traj = min(success_traj, key=len)
            sft_records.append({
                "task_id": task_id,
                "type": "agent_success",
                "messages": best_traj
            })

        # ---------------------------------------------------------
        # 制作 DPO 数据 (错题本：正确对比错误)
        # ---------------------------------------------------------
        # 只有当我们有一个明确的“正确答案”(final_code)，且模型曾经犯过错(fail_traj)时，才能组装 DPO
        if final_code and fail_traj:
            # 提取模型犯错的第一轮回复作为 Rejected
            # fail_traj 中的每一条都是 [System, User, Assistant(错误), User(报错)...]
            for fail in fail_traj:
                if len(fail) >= 3:
                    rejected_assistant_msg = fail[2] # 模型最初的错误输出
                    chosen_assistant_msg = {
                        "role": "assistant", 
                        "content": format_final_code_as_assistant(final_code)
                    }
                    
                    dpo_records.append({
                        "task_id": task_id,
                        "system": initial_prompt[0]["content"],
                        "prompt": initial_prompt[1]["content"],
                        "chosen": chosen_assistant_msg["content"],
                        "rejected": rejected_assistant_msg["content"]
                    })
                    break # 一个任务取一对 DPO 即可

    # 保存 SFT 数据
    with open(SFT_OUT_FILE, "w", encoding="utf-8") as f:
        for rec in sft_records:
            f.write(json.dumps(rec) + "\n")

    # 保存 DPO 数据
    with open(DPO_OUT_FILE, "w", encoding="utf-8") as f:
        for rec in dpo_records:
            f.write(json.dumps(rec) + "\n")

    print(f"\n✅ 数据集构建完成！")
    print(f"👉 产出 SFT 数据: {len(sft_records)} 条 (保存至 {SFT_OUT_FILE})")
    print(f"👉 产出 DPO 数据: {len(dpo_records)} 对 (保存至 {DPO_OUT_FILE})")

if __name__ == "__main__":
    main()