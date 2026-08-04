"""
本地化因子挖掘项目 - 双程序守护脚本

同时运行两个挖掘程序(optimize F001 + new), 任一进程异常退出时自动重启续传:
  - optimize 模式跑满 max_rounds 正常结束(输出最终报告图片)后不再重启;
  - new 模式挖到入库因子或跑满轮数正常结束时不再重启;
  - 仅当进程以非零退出码(LLM故障/网络错误)结束时自动重启续传。

用法:
  python -m llm_mining.local_miner.run_both_watchdog
  或直接 python llm_mining/local_miner/run_both_watchdog.py
"""

import os
import subprocess
import sys
import time

QUANT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TASKS = [
    {
        "name": "optimize F001",
        "args": ["python", "-m", "llm_mining.local_miner.run_mining",
                 "--mode", "optimize", "--series", "F001"],
    },
    {
        "name": "new 挖掘",
        "args": ["python", "-m", "llm_mining.local_miner.run_mining", "--mode", "new"],
    },
]


def run_task(task: dict):
    """启动一个任务, 返回是否应继续守护(True=异常退出需重启, False=正常结束)"""
    name = task["name"]
    print(f"[守护] 启动 {name}: {' '.join(task['args'])}", flush=True)
    proc = subprocess.Popen(
        task["args"],
        cwd=QUANT_ROOT,
        stdout=None,   # 继承守护进程的终端输出, 便于统一观察
        stderr=None,
    )
    code = proc.wait()
    if code == 0:
        print(f"[守护] {name} 正常结束(exit=0), 不再守护。", flush=True)
        return False
    print(f"[守护] {name} 异常退出(exit={code}), 等待 30 秒后自动重启续传...", flush=True)
    time.sleep(30)
    return True


def main():
    print("=" * 60, flush=True)
    print("双程序守护启动: optimize F001 + new 挖掘", flush=True)
    print(f"工作目录: {QUANT_ROOT}", flush=True)
    print("Ctrl+C 停止全部守护。", flush=True)
    print("=" * 60, flush=True)
    for task in TASKS:
        while run_task(task):
            pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[守护] 收到中断, 停止守护。", flush=True)
        sys.exit(0)
