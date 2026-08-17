"""
本地化因子挖掘项目 - new 模式合格因子入库处理（run_mining 拆分模块）

职责: 对合格因子执行"入库双防线"（LLM 语义评审 + 数值截面相关性检验），
通过后分配新系列 ID 入库、采纳出图、成功因子库归档。
"""

import os

from . import console, diagnostics, factor_archive, factor_library, review


def handle_new_mode_qualified(round_factors, round_no, hypothesis, data, cfg, state):
    """new模式: 对合格因子做库相关性检查, 通过则分配新系列ID入库。返回入库因子或None"""
    # 延迟导入: 本模块被 run_mining 顶层导入, 避免循环依赖(与 minute_data 中 _BatchRunner 同法)
    from .run_mining import recompute_series, save_adopted_report

    for f in round_factors:
        ev = f.get("eval") or {}
        if not ev.get("qualified"):
            continue
        style = f.get("style", "")
        console.log(f"    [库相关性检查] 因子 {f['name']} 合格, 检查是否与库内因子撞车...")

        # 第一道关: LLM语义相关性评审(独立对话窗口, 防"换皮造因子")
        library = factor_library.load_library()
        if library:
            rev = review.review_factor(cfg, f, library)
            if rev.get("error"):
                console.log(f"    [警告] {rev['reason']}")
            elif rev.get("reject"):
                console.log(f"    [×] 语义评审拒绝: 与 {rev.get('match_series')} 语义"
                            f"{'重复' if rev.get('similarity') is not None and rev.get('similarity') >= 0.999 else '高度相似'}"
                            f"(相似度={rev.get('similarity')}), 理由: {rev.get('reason')}。"
                            "该因子不入库, 继续挖掘。")
                f["eval"]["review_rejected"] = True
                f["eval"]["review_reason"] = rev.get("reason")
                continue
            console.log(f"    [√] 语义评审通过(相似度={rev.get('similarity')}, "
                        f"最相似: {rev.get('match_series') or '无'}, {rev.get('reason')})")
        else:
            console.log("    因子库为空, 跳过语义评审。")

        # 第二道关: 数值截面相关性检验
        try:
            factor_wide, series = recompute_series(f["expr"], ev.get("flipped"), data, cfg)
            corr_res = factor_library.check_library_correlation(factor_wide, data, cfg,
                                                                exclude_id="")
        except Exception as e:
            console.log(f"    [警告] 相关性检查异常({e}), 跳过该因子入库。")
            continue
        if corr_res["n_compared"] > 0:
            detail = ", ".join(f"{d['series_id']}|ρ|={abs(d['corr']):.2f}"
                               for d in corr_res["details"] if d.get("corr") is not None)
            console.log(f"    与库内因子截面相关: {detail} (上限{cfg.max_library_corr})")
        if corr_res["flag"]:
            console.log(f"    [×] 撞车拒绝: 与库内因子最大|ρ|={corr_res['max_abs_corr']:.2f} "
                        f"> {cfg.max_library_corr}, 该因子不入库, 继续挖掘。")
            f["eval"]["library_rejected"] = True
            f["eval"]["library_max_corr"] = corr_res["max_abs_corr"]
            continue

        # 通过 -> 计算诊断并入库(系列前缀按数据频率: 分钟→M, 日频含切割日频→D, 共用D计数序列)
        console.log(f"    [√] 相关性检查通过, 计算诊断并入库...")
        diag = diagnostics.compute_diagnostics(series, data, cfg)
        prefix = "M" if cfg.data_frequency == "minute" else "D"
        new_id = factor_library.next_series_id(prefix)
        f_entry = {**f, "round": round_no}
        series_obj = factor_library.create_series_from_history(
            new_id, state["history"], f_entry, hypothesis, style, diag)
        factor_library.save_series(series_obj)
        # 战役入库追踪(供战役结束信息压缩摘要使用)
        state.setdefault("campaign_adopted", []).append(new_id)
        console.log(f"    [入库] 新因子系列 {new_id} 已保存: "
                    f"{factor_library.series_path(new_id, f['name'])}")
        console.log("    诊断摘要:")
        for line in factor_library._diagnostics_brief(diag).splitlines():
            console.log(f"        {line}")
        # 采纳合格因子即时出图(带轮次与因子名, 不覆盖历史; 失败不中断)
        save_adopted_report({**f_entry, "hypothesis": hypothesis}, data, cfg, new_id, round_no)
        # 成功因子库归档: h5完整回测数据 + 集中评价图(与json系列文件构成三件套; 失败不中断)
        try:
            arch = factor_archive.archive_success_series(series_obj, factor_wide, data,
                                                         cfg, round_no)
            console.log(f"    [归档] 成功因子 {new_id} 回测数据已归档: "
                        f"{os.path.basename(arch['h5'])}"
                        + (f", 评价图: {os.path.basename(arch['png'])}" if arch.get("png") else ""))
        except Exception as e:
            console.log(f"    [归档失败] {type(e).__name__}: {e}, 不中断挖掘。")
        return {**f_entry, "hypothesis": hypothesis, "series_id": new_id}
    return None
