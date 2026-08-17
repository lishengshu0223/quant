"""
本地化因子挖掘项目 - 因子 Barra 行业/市值中性化重评

对指定因子(默认 D003 AmountStability_20D)做行业市值中性化后重新评价:
- 市值: Barra v1 暴露数据的 size 风格暴露值
- 行业: Barra v1 暴露数据中的申万一级行业哑变量(31个)
- 方法: 每日截面回归 factor ~ 行业哑变量 + size, 取残差作为中性化因子值
  (FWL 等价向量化实现: 先在 日期×行业 内对 factor 与 size 各自 demean,
   再做无截距回归 beta = Σ(f_dm·s_dm)/Σ(s_dm²), resid = f_dm - beta·s_dm)

评价口径与主流程完全一致(factor_eval.evaluate_factor, 含交易资格遮盖),
并输出原始 vs 中性化的指标对比与完整 tear sheet 图片。

用法:
  python -m llm_mining.tools.run_neutral_eval --series D003
"""

import argparse
import os
import time

import numpy as np
import pandas as pd

from llm_mining.local_miner import barra_neutralize, console, factor_eval, factor_library
from llm_mining.local_miner.config import MiningConfig, WORKSPACE_DIR
from llm_mining.local_miner.data_loader import MarketData
from llm_mining.local_miner.expr_engine import compute_factor


def _fmt_pct(v, nd=2):
    return "NA" if v is None or not np.isfinite(v) else f"{v*100:+.{nd}f}%"


def print_compare(ev_raw: dict, ev_neu: dict, name: str):
    """打印原始 vs 中性化关键指标对比"""
    rows = [
        ("IC均值(5日RankIC)", _fmt_pct(ev_raw.get("ic_mean"), 3), _fmt_pct(ev_neu.get("ic_mean"), 3)),
        ("ICIR", f"{ev_raw.get('icir') or 0:.3f}", f"{ev_neu.get('icir') or 0:.3f}"),
        ("多头累计超额", _fmt_pct(ev_raw.get("long_total")), _fmt_pct(ev_neu.get("long_total"))),
        ("多头年化超额", _fmt_pct(ev_raw.get("long_annual")), _fmt_pct(ev_neu.get("long_annual"))),
        ("月度IC正占比", _fmt_pct(ev_raw.get("monthly_pos_ratio"), 1), _fmt_pct(ev_neu.get("monthly_pos_ratio"), 1)),
        ("单调性评级", (ev_raw.get("monotonicity_grade") or {}).get("grade", "NA"),
         (ev_neu.get("monotonicity_grade") or {}).get("grade", "NA")),
        ("年度IR(多头稳定性)", f"{(ev_raw.get('long_stability') or {}).get('yearly_ir') or 0:.2f}",
         f"{(ev_neu.get('long_stability') or {}).get('yearly_ir') or 0:.2f}"),
        ("是否合格", str(ev_raw.get("qualified")), str(ev_neu.get("qualified"))),
    ]
    console.log("")
    console.log("=" * 78)
    console.log(f"原始 vs Barra行业市值中性化 对比: {name}")
    console.log(f"{'指标':<22}{'原始':>24}{'中性化':>24}")
    console.log("-" * 78)
    for label, a, b in rows:
        console.log(f"{label:<22}{a:>24}{b:>24}")
    # 分年度多头超额
    y_raw = ev_raw.get("yearly_long") or {}
    y_neu = ev_neu.get("yearly_long") or {}
    console.log("-" * 78)
    console.log("分年度多头超额收益:")
    for y in sorted(set(y_raw) | set(y_neu)):
        console.log(f"  {y}: 原始 {_fmt_pct(y_raw.get(y))}  ->  中性化 {_fmt_pct(y_neu.get(y))}")
    # 单调性评级细节(定位评级变化原因)
    console.log("-" * 78)
    console.log("单调性评级细节 (drc=每日秩相关均值, doo=越序惩罚日均, peak=年度越序峰值):")
    for tag, ev in (("原始", ev_raw), ("中性化", ev_neu)):
        mg = ev.get("monotonicity_grade") or {}
        console.log(f"  {tag}: {mg.get('grade')}级 drc={mg.get('drc'):.3f} "
                    f"doo={((mg.get('doo') or 0)*100):.3f}% peak={((mg.get('yearly_peak') or 0)*100):.2f}% "
                    f"分档(秩/oos/peak)={mg.get('grade_rank')}/{mg.get('grade_oos')}/{mg.get('grade_peak')} "
                    f"坏年份={mg.get('bad_years')}")
    console.log("=" * 78)


def main():
    p = argparse.ArgumentParser(description="因子 Barra 行业/市值中性化重评")
    p.add_argument("--series", type=str, default="D003", help="因子系列ID")
    p.add_argument("--no-report", action="store_true", help="只评价对比, 不生成 tear sheet 图片")
    args = p.parse_args()

    series = factor_library.load_series(args.series)
    if series is None:
        raise SystemExit(f"错误: 因子库中找不到系列 {args.series}")
    best = series.get("best") or {}
    name = best.get("name", series.get("name", "factor"))
    expr = best.get("expr", "")
    desc = best.get("desc", "")
    console.log(f"目标因子: {args.series} {name}")
    console.log(f"公式: {expr}")

    cfg = MiningConfig()
    console.log("加载全市场量价数据(含可交易状态遮盖)...")
    data = MarketData(cfg)

    console.log("计算原始因子全区间值...")
    factor_wide = compute_factor(expr, data, cfg)

    t0 = time.time()
    console.log(f"    读取 Barra v1 因子暴露: {cfg.eval_start_date} 至今...")
    exp, ind_cols = barra_neutralize.load_barra_exposure(start_date=cfg.eval_start_date)
    console.log(f"    Barra 暴露加载完成: {len(exp)} 条记录, {len(ind_cols)} 个行业哑变量, "
                f"耗时 {time.time()-t0:.1f} 秒")
    t0 = time.time()
    neutral_wide = barra_neutralize.neutralize_factor(factor_wide, exp, ind_cols)
    del exp
    console.log(f"    中性化完成: {neutral_wide.shape[0]} 个交易日 × {neutral_wide.shape[1]} 只股票, "
                f"耗时 {time.time()-t0:.1f} 秒")

    # 原始与中性化因子的截面秩相关(衡量中性化改变幅度)
    eval_mask = factor_wide.index >= pd.Timestamp(cfg.eval_start_date)
    with np.errstate(invalid="ignore"):
        rank_corr = factor_wide[eval_mask].corrwith(
            neutral_wide.reindex(factor_wide.index[eval_mask]),
            axis=1, method="spearman")
    console.log(f"原始 vs 中性化因子 日度截面秩相关均值: {rank_corr.mean():.3f} "
                f"(1=几乎未变, 越低说明行业市值结构影响越大)")

    console.log("评价原始因子(复核)...")
    ev_raw = factor_eval.evaluate_factor(factor_wide, data, cfg, name=name)
    console.log("评价中性化因子...")
    ev_neu = factor_eval.evaluate_factor(neutral_wide, data, cfg, name=f"{name}_Neutral")

    ev_raw = factor_eval.to_serializable(ev_raw)
    ev_neu = factor_eval.to_serializable(ev_neu)
    print_compare(ev_raw, ev_neu, name)

    if args.no_report:
        console.log("(--no-report) 跳过图片生成")
        return

    # 生成中性化因子的完整 tear sheet 图片
    neutral_name = f"{name} (Barra行业市值中性化)"
    png_path = os.path.join(WORKSPACE_DIR,
                            f"factor_report_{args.series}_{name}_neutral.png")
    report_factor = {
        "name": neutral_name,
        "expr": expr,
        "描述": (f"Barra行业市值中性化残差: 每日截面回归 因子 ~ {len(ind_cols)}个申万一级行业哑变量"
                 f" + Barra size风格暴露, 取残差。原始描述: {desc}"),
        "hypothesis": "",
        "eval": ev_neu,
    }
    from llm_mining.local_miner.report import generate_report
    console.log("生成中性化因子 tear sheet 图片...")
    png = generate_report(report_factor, data, cfg, png_path=png_path,
                          factor_wide=neutral_wide)
    console.log(f"报告图片已保存: {png}")


if __name__ == "__main__":
    main()
