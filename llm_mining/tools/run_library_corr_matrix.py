"""
本地化因子挖掘项目 - 因子库合格因子两两截面秩相关矩阵

对库内所有 status=qualified 系列的最佳因子(按评价时方向, flipped 取反),
计算评价区间内抽样交易日的截面 Spearman 秩相关均值, 输出 N×N 矩阵。
口径与 factor_library.check_library_correlation 一致(每日截面 rank 后 Pearson, 取均值)。

用法:
  python -m llm_mining.tools.run_library_corr_matrix [--sample-dates 200]
"""

import argparse

import numpy as np
import pandas as pd

from llm_mining.local_miner import console, factor_library
from llm_mining.local_miner.config import MiningConfig
from llm_mining.local_miner.data_loader import MarketData
from llm_mining.local_miner.expr_engine import compute_factor


def main():
    p = argparse.ArgumentParser(description="因子库合格因子秩相关矩阵")
    p.add_argument("--sample-dates", type=int, default=200, help="抽样交易日数")
    args = p.parse_args()

    cfg = MiningConfig()
    console.log("加载全市场量价数据...")
    data = MarketData(cfg)

    items = []
    for s in factor_library.load_library():
        if s.get("status") != "qualified":
            continue
        best = s.get("best") or {}
        if not best.get("expr"):
            continue
        wide = compute_factor(best["expr"], data, cfg)
        if (best.get("eval") or {}).get("flipped"):
            wide = -wide
        wide = wide[wide.index >= pd.Timestamp(cfg.eval_start_date)]
        items.append({"series_id": s["series_id"], "name": best.get("name"),
                      "ranked": wide.rank(axis=1)})
        console.log(f"    已计算: {s['series_id']} {best.get('name')}")

    n = len(items)
    if n == 0:
        raise SystemExit("库内无合格因子")

    common_dates = items[0]["ranked"].index
    for it in items[1:]:
        common_dates = common_dates.intersection(it["ranked"].index)
    common_dates = common_dates.sort_values()
    step = max(1, len(common_dates) // args.sample_dates)
    sample = common_dates[::step]
    console.log(f"抽样交易日: {len(sample)} 天 ({sample[0].date()} ~ {sample[-1].date()})")

    mat = np.full((n, n), 1.0)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = items[i]["ranked"], items[j]["ranked"]
            corrs = []
            for d in sample:
                x, y = a.loc[d], b.loc[d]
                m = x.notna() & y.notna()
                if m.sum() < 30:
                    continue
                c = np.corrcoef(x[m].to_numpy(), y[m].to_numpy())[0, 1]
                if np.isfinite(c):
                    corrs.append(c)
            mat[i, j] = mat[j, i] = float(np.mean(corrs)) if corrs else np.nan

    console.log("")
    console.log("=" * 70)
    console.log(f"合格因子截面秩相关矩阵 ({len(sample)} 个抽样交易日均值):")
    header = " " * 10 + " | ".join(f"{it['series_id']:>10}" for it in items)
    console.log(header)
    for i, it in enumerate(items):
        row = " | ".join(f"{mat[i, j]:>10.3f}" for j in range(n))
        console.log(f"{it['series_id']:<10} {row}")
    console.log("")
    for i, it in enumerate(items):
        console.log(f"  {it['series_id']}: {it['name']}")
    console.log("=" * 70)


if __name__ == "__main__":
    main()
