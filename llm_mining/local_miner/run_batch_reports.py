"""
本地化因子挖掘项目 - 批量因子评价与报告生成(应用交易资格遮盖, 支持多进程并行)

对指定因子系列(path 中的每一个迭代因子)重新计算、评价并生成完整 tear sheet 图片。
评价口径与主流程一致, 并应用每日可交易状态遮盖(data.tradable):
剔除 ST / 停牌 / 上市未满一年 / 涨跌停 股票, 避免因子收益虚高。

多进程并行(--workers N): 每个因子(计算+评价+出图)在独立进程执行, 互不影响;
每 worker 持有一份全市场数据副本, 进程池跨任务复用。

用法:
  python -m llm_mining.local_miner.run_batch_reports --series F001 --workers 4
"""

import argparse
import concurrent.futures
import os
import re
import sys

from . import console, factor_eval, factor_library
from .config import MiningConfig, WORKSPACE_DIR
from .data_loader import MarketData
from .expr_engine import compute_factor

# ---- 多进程并行: worker 全局(由 initializer 注入) ----
_WORKER_DATA = None
_WORKER_CFG = None


def _init_worker(data, cfg):
    global _WORKER_DATA, _WORKER_CFG
    _WORKER_DATA = data
    _WORKER_CFG = cfg


def _process_one(rec: dict) -> dict:
    """worker 进程内: 计算 + 评价 + 出图, 返回汇总记录"""
    data = _WORKER_DATA
    cfg = _WORKER_CFG
    name = rec.get("name", "factor")
    expr = rec.get("expr", "")
    r = rec.get("round", "?")
    png_path = rec.get("_png", "")
    try:
        factor_wide = compute_factor(expr, data, cfg)
    except Exception as e:
        return {"round": r, "name": name, "ok": False,
                "reason": f"公式计算失败: {type(e).__name__}: {e}", "png": None}
    try:
        ev = factor_eval.to_serializable(
            factor_eval.evaluate_factor(factor_wide, data, cfg, name=name))
    except Exception as e:
        return {"round": r, "name": name, "ok": False,
                "reason": f"评价异常: {type(e).__name__}: {e}", "png": None}
    if ev.get("error"):
        return {"round": r, "name": name, "ok": False, "reason": ev["error"], "png": None}
    try:
        from .report import generate_report
        report_factor = {
            "name": name, "expr": expr,
            "描述": rec.get("desc", ""), "hypothesis": rec.get("hypothesis", ""),
            "eval": ev,
        }
        generate_report(report_factor, data, cfg, png_path=png_path)
    except Exception as e:
        return {"round": r, "name": name, "ok": False,
                "reason": f"出图失败: {type(e).__name__}: {e}", "png": png_path}
    return {
        "round": r, "name": name, "ok": True, "png": png_path,
        "qualified": ev.get("qualified"), "flipped": ev.get("flipped"),
        "ic_mean": ev.get("ic_mean"), "icir": ev.get("icir"),
        "long_annual": ev.get("long_annual"),
        "mono": (ev.get("monotonicity_grade") or {}).get("grade"),
    }


def main():
    p = argparse.ArgumentParser(description="批量因子评价与报告(交易资格遮盖 + 多进程并行)")
    p.add_argument("--series", type=str, default="F001", help="因子系列ID")
    p.add_argument("--out-dir", type=str, default=os.path.join(WORKSPACE_DIR, "batch_reports"),
                   help="报告图片输出目录")
    p.add_argument("--workers", type=int, default=4,
                   help="并行进程数(1=串行; 每进程一份全市场数据副本, 注意内存)")
    args = p.parse_args()

    series = factor_library.load_series(args.series)
    if series is None:
        print(f"错误: 因子库中找不到系列 {args.series}")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = MiningConfig()
    cfg.n_eval_workers = max(1, args.workers)
    console.log("加载全市场数据(含可交易状态遮盖)...")
    data = MarketData(cfg)

    factors = series.get("path", [])
    # 预先为每个因子计算图片路径, 注入任务
    tasks = []
    for rec in factors:
        name = rec.get("name", "factor")
        safe = re.sub(r"[^0-9A-Za-z_]", "", name)[:40] or "factor"
        rec2 = dict(rec)
        rec2["_png"] = os.path.join(
            args.out_dir, f"factor_report_{args.series}_R{rec.get('round', '?')}_{safe}.png")
        tasks.append(rec2)

    print(f"\n共 {len(tasks)} 个迭代因子待评价(交易资格遮盖 + "
          f"{'多进程并行' if args.workers > 1 else '串行'}: 剔除 ST/停牌/次新/涨跌停)...\n")

    results = []
    if args.workers > 1:
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=args.workers, initializer=_init_worker, initargs=(data, cfg)) as ex:
            futures = {ex.submit(_process_one, t): i for i, t in enumerate(tasks)}
            done = {}
            for fut in concurrent.futures.as_completed(futures):
                idx = futures[fut]
                try:
                    done[idx] = fut.result()
                except Exception as e:
                    t = tasks[idx]
                    done[idx] = {"round": t.get("round"), "name": t.get("name"), "ok": False,
                                 "reason": f"进程异常: {type(e).__name__}: {e}", "png": None}
            results = [done[i] for i in sorted(done)]
    else:
        for t in tasks:
            results.append(_process_one(t))

    print("\n========== 汇总(交易资格遮盖后) ==========")
    for r in results:
        if r["ok"]:
            ic = f"{r['ic_mean']*100:+.3f}%" if r["ic_mean"] is not None else "NA"
            icir = f"{r['icir']:.3f}" if r["icir"] is not None else "NA"
            la = f"{r['long_annual']*100:+.2f}%" if r["long_annual"] is not None else "NA"
            print(f"R{r['round']:>2} {r['name']:<42} qualified={r['qualified']} "
                  f"ic={ic} icir={icir} long_ann={la} mono={r['mono']} -> {os.path.basename(r['png'])}")
        else:
            print(f"R{r['round']:>2} {r['name']:<42} [失败] {r['reason']}")


if __name__ == "__main__":
    main()
