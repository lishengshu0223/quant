"""
本地化因子挖掘项目 - 断点续传(对应需求5)

状态保存在 workspace/checkpoint.json:
- 模型输出因子后(评价前)立即保存 pending -> 若评价阶段中断, 恢复时跳过模型调用直接评价
- 每轮评价完成后保存 history -> 恢复时从下一轮继续
"""

import json
import os

from . import console
from .config import CHECKPOINT_PATH, ensure_workspace


def save(state: dict):
    """原子写入检查点"""
    ensure_workspace()
    tmp = CHECKPOINT_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, CHECKPOINT_PATH)


def load() -> dict | None:
    if not os.path.exists(CHECKPOINT_PATH):
        return None
    try:
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        console.log(f"    [警告] 检查点读取失败({e}), 将重新开始")
        return None


def clear():
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)


def new_state(cfg) -> dict:
    return {
        "config": cfg.to_dict(),
        "round": 0,          # 已完成的轮数
        "stage": "init",     # init / model_called / round_done
        "history": [],       # 已完成轮次记录
        "pending": None,     # 模型已输出但尚未评价的内容
        "best": None,        # 最佳合格因子
    }
