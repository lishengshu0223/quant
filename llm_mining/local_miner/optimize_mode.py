"""
本地化因子挖掘项目 - optimize 模式因子路径追加与最佳更新（run_mining 拆分模块）

职责: 把本轮因子追加进系列路径; 对合格因子按"多头收益年度稳定性"择优,
优于当前最佳则更新系列最佳并出图(采纳标准是稳定性而非IC, 因IC常由空头贡献)。
"""

from . import console, diagnostics, factor_library


def stability_better(new_stab: dict, cur_stab: dict, tol: float = 1e-6) -> bool:
    """
    判断新因子的多头收益年度稳定性是否优于当前最佳(优化模式采纳标准)。
    主比较量: 年度多头收益信息比率(score=yearly_ir, 越高=每年收益越平均稳定);
    平手时以最差完整年份收益(min_year, 越高=底部越稳)决胜。
    缺失值按最差(-inf)处理。
    """
    ns = new_stab.get("score")
    cs = cur_stab.get("score")
    ns = ns if ns is not None else float("-inf")
    cs = cs if cs is not None else float("-inf")
    if ns > cs + tol:
        return True
    if abs(ns - cs) <= tol:
        nm = new_stab.get("min_year")
        cm = cur_stab.get("min_year")
        nm = nm if nm is not None else float("-inf")
        cm = cm if cm is not None else float("-inf")
        return nm > cm + tol
    return False


def handle_optimize_mode_qualified(round_factors, round_no, hypothesis, opt_series,
                                   series_id, data, cfg, state):
    """optimize模式: 把本轮所有因子追加进系列路径; 合格且多头收益年度稳定性优于
    当前最佳则更新最佳(采纳标准是稳定性而非IC, 因IC常由空头贡献, 我们只要多头)。
    返回更新后的最佳因子(若本轮有采纳)或None"""
    # 延迟导入: 本模块被 run_mining 顶层导入, 避免循环依赖
    from .run_mining import recompute_series, save_adopted_report

    # 本轮风格优先取首个因子的风格
    round_style = next((f.get("style", "") for f in round_factors if f.get("style")), "")
    factor_library.append_round(opt_series, round_no, round_factors, hypothesis, round_style)

    cur_best_ev = (opt_series.get("best") or {}).get("eval") or {}
    cur_stab = cur_best_ev.get("long_stability") or {}
    adopted = None
    for f in round_factors:
        ev = f.get("eval") or {}
        if not ev.get("qualified"):
            continue
        new_stab = ev.get("long_stability") or {}
        if stability_better(new_stab, cur_stab):
            cur_ir = cur_stab.get("score")
            new_ir = new_stab.get("score")
            console.log(
                f"    [√] 优化采纳: {f['name']} 合格且多头收益更稳定 "
                f"(年度信息比率 {cur_ir if cur_ir is not None else float('nan'):.2f}"
                f" -> {new_ir if new_ir is not None else float('nan'):.2f}, "
                f"最差年 {new_stab.get('min_year', 0)*100:+.2f}%), 重算诊断并更新系列最佳。")
            try:
                _, series = recompute_series(f["expr"], ev.get("flipped"), data, cfg)
                diag = diagnostics.compute_diagnostics(series, data, cfg)
            except Exception as e:
                console.log(f"    [警告] 诊断计算失败({e}), 仍更新最佳但不附诊断。")
                diag = None
            f_entry = {**f, "round": round_no}
            factor_library.update_best(opt_series, f_entry, hypothesis,
                                       f.get("style", "") or round_style, diag)
            cur_stab = new_stab
            adopted = {**f_entry, "hypothesis": hypothesis, "series_id": series_id}
            # 采纳新最佳即出图(带轮次命名, 不覆盖历史图片, 供随时观看)
            save_adopted_report(adopted, data, cfg, series_id, round_no)
        else:
            console.log(
                f"    [·] {f['name']} 合格但多头收益稳定性未超越当前最佳 "
                f"(年度信息比率 {new_stab.get('score')} vs {cur_stab.get('score')}), "
                "记录路径不更新最佳。")

    factor_library.save_series(opt_series)
    if adopted:
        state["best"] = adopted
    return adopted
