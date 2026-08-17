"""
本地化因子挖掘项目 - 多进程并行评价 worker（run_mining 拆分模块）

职责: 进程池 worker 的全局数据注入(_init_worker)与单因子"计算+评价+出图"任务(_eval_factor_worker)。
每个 worker 进程持有全市场数据副本与配置, 跨轮复用; 因子评价互不影响。
"""

import os
import re
import time
from dataclasses import dataclass

from . import factor_eval, factor_plot
from .expr_engine import ExprError, compute_factor

# ---- 多进程并行评价: worker 进程内的全局数据/配置(由 initializer 注入) ----
_WORKER_DATA = None
_WORKER_CFG = None
_WORKER_OUT_DIR = None   # 每轮因子评价图输出目录(worker 进程内直接出图, 满足"每轮每因子回测图")


@dataclass
class FactorTask:
    """跨进程任务包: 待评价因子(模型原始输出 meta) + 轮次/序号。
    仅用于进程池提交边界(可 pickle), worker 内部读字段, 不参与下游 entry 链路。"""
    meta: dict
    round_no: int = 0
    idx: int = 0


def _init_worker(data, cfg, out_dir=""):
    """进程池 worker 初始化: 注入全市场数据副本与配置(含每轮评价图输出目录)"""
    global _WORKER_DATA, _WORKER_CFG, _WORKER_OUT_DIR
    _WORKER_DATA = data
    _WORKER_CFG = cfg
    _WORKER_OUT_DIR = out_dir


def _eval_factor_worker(task: FactorTask) -> dict:
    """worker 进程内执行: 计算因子宽表并评价, 返回带 eval 的 entry(多核并行, 互不影响)。
    并行路径下也在 worker 内直接生成每轮因子回测图(与串行路径等价)。"""
    data = _WORKER_DATA
    cfg = _WORKER_CFG
    meta = task.meta
    name = str(meta.get("名称", "factor"))
    desc = str(meta.get("描述", ""))
    expr = str(meta.get("公式", "")).strip()
    style = str(meta.get("风格", ""))
    round_no = task.round_no
    idx = task.idx
    entry = {"name": name, "desc": desc, "expr": expr, "style": style, "eval": None}
    t0 = time.time()
    try:
        factor_wide = compute_factor(expr, data, cfg)
        ev = factor_eval.evaluate_factor(factor_wide, data, cfg, name=name)
        entry["eval"] = factor_eval.to_serializable(ev)
        entry["_t_calc"] = round(time.time() - t0, 1)
        # 并行路径: 在 worker 内直接生成每轮评价图(避免跨进程传大宽表; 失败不中断)
        if _WORKER_OUT_DIR and not entry["eval"].get("error"):
            safe = re.sub(r"[^0-9A-Za-z_]", "", name)[:40] or "factor"
            png_rel = f"round{round_no:03d}_f{idx:02d}_{safe}.png"
            if factor_plot.plot_round_factor(entry, factor_wide, data, cfg,
                                             os.path.join(_WORKER_OUT_DIR, png_rel),
                                             round_no=round_no, idx=idx):
                entry["_png_rel"] = png_rel
    except ExprError as e:
        entry["eval"] = {"name": name, "error": f"公式校验失败: {e}", "qualified": False}
    except Exception as e:
        entry["eval"] = {"name": name, "error": f"{type(e).__name__}: {e}", "qualified": False}
    return entry
