"""
本地化因子挖掘项目 - 最终评价与图片报告（run_mining 拆分模块）

职责: 战役收尾的"阶段6": 选定用于报告的因子(入库/最佳/方向正确候选),
生成完整 tear sheet 图片(按系列隔离路径, 保留历史)。
"""

import re
import sys
import traceback

from . import console
from .config import report_png_path


def finalize(mode, series_id, opt_series, qualified_factor, state, data, cfg, ckpt_path):
    """最终评价与图片报告(按系列隔离路径)"""
    # 组合枯竭提示
    if state.get("combo_exhausted"):
        console.banner(
            "⚠ 本次运行已判定该系列'组合=穷途末路': 模型多次产出组合类因子, "
            "且期间没有更好的单逻辑因子超越此前组合。"
            "建议重新审视该系列的经济逻辑方向, 或开启新因子系列。", "!")
    # 确定用于报告的因子
    report_factor = qualified_factor
    final_series_id = series_id
    if report_factor is None and mode == "optimize" and opt_series:
        # optimize 未产生改进: 用系列当前最佳做报告
        best = opt_series.get("best") or {}
        report_factor = {**best, "series_id": series_id}
        final_series_id = series_id
    if report_factor is None:
        # new 模式未挖到入库因子: 选方向正确的最佳候选(标注未达标)
        candidates = []
        for rec in state["history"]:
            for f in rec["factors"]:
                ev = f.get("eval") or {}
                if ev.get("direction_ok"):
                    candidates.append({**f, "round": rec["round"],
                                       "hypothesis": rec.get("hypothesis", "")})
        if candidates:
            candidates.sort(key=lambda x: x["eval"].get("ic_mean") or -1, reverse=True)
            report_factor = candidates[0]
            console.log(f"    未挖到入库因子, 选取方向正确的最佳候选 "
                        f"{report_factor['name']} 做全面评价(标注未完全达标)。")
        else:
            console.log("    没有任何方向正确的因子, 无法生成报告。请增加轮数或调整方向后重试。")
            sys.exit(1)

    if not final_series_id:
        final_series_id = report_factor.get("series_id") or "candidate"
    # 最终图片带因子名, 保留历史(同系列多次运行不互相覆盖)
    safe = re.sub(r"[^0-9A-Za-z_]", "", report_factor.get("name", "factor"))[:40] or "factor"
    png_path = report_png_path(f"{final_series_id}_{safe}")

    from .report import generate_report
    try:
        png = generate_report(report_factor, data, cfg, png_path=png_path)
        console.banner(f"挖掘结束! 因子报告图片已生成: {png}", "=")
    except Exception as e:
        console.log(f"    [错误] 报告生成失败: {e}")
        traceback.print_exc()
        sys.exit(1)
