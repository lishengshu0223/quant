"""
本地化因子挖掘项目 - 断点续传(对应需求5)

状态保存在按模式隔离的检查点文件(见 config.checkpoint_path):
- new 模式: workspace/checkpoint_new.json
- optimize 模式: workspace/checkpoint_optimize_<series_id>.json
规则:
- 模型输出因子后(评价前)立即保存 pending -> 若评价阶段中断, 恢复时跳过模型调用直接评价
- 每轮评价完成后保存 history -> 恢复时从下一轮继续
"""

import json
import os

from . import console
from .config import ensure_workspace


def save(state: dict, path: str):
    """原子写入检查点"""
    ensure_workspace()
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def load(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        console.log(f"    [警告] 检查点读取失败({e}), 将重新开始")
        return None


def clear(path: str):
    if os.path.exists(path):
        os.remove(path)


def new_state(cfg, mode: str = "new", series_id: str = "") -> dict:
    return {
        "config": cfg.to_dict(),
        "mode": mode,            # new / optimize
        "series_id": series_id,  # optimize 模式绑定的因子系列ID
        "round": 0,          # 已完成的轮数
        "stage": "init",     # init / model_called / round_done
        "history": [],       # 已完成轮次记录(仅含正常反馈的因子)
        "pending": None,     # 模型已输出但尚未评价的内容
        "best": None,        # 本次运行发现的最佳合格因子
        "hidden_combos": [], # 组合类因子(最后手段)隐藏缓存: {round,name,expr,eval,surpassed}
        "combo_exhausted": False,  # 是否已判定"组合=穷途末路"
    }
