"""
本地化因子挖掘项目 - 主程序(双模式)

两种模式(对应需求三):
  --mode new      新因子挖掘: 注入因子库摘要防撞车, 合格因子做库相关性检查,
                  通过则分配新系列ID(F00x)入库; 入库后不提前结束, 连续挖掘到最大轮数。
                  每轮为每个因子生成评价图(公式数学化渲染), 入库后更新全库相关性矩阵,
                  每轮末尾刷新静态HTML进度报告(output/<YYYYMMDD>_<任务名>/ 下)。
  --mode optimize --series F001  现有因子迭代优化: 注入当前因子全况(评价+诊断+整条路径)
                  与条件判断式优化建议, 每轮把改进因子追加进该系列路径,
                  若合格且优于当前最佳则更新系列最佳; 跑满轮数持续迭代。

流程(对应需求4.4的阶段衔接展示):
  阶段1 配置初始化 -> 阶段2 数据加载 -> 阶段3 系统提示词
  -> 循环[ 模型输入 -> 模型输出 -> 因子计算 -> 因子评价 -> 反馈构建 ]
  -> 最终阶段: 合格因子全面评价 + 生成图片报告(按系列隔离)

用法:
  python -m llm_mining.local_miner.run_mining --mode new
  python -m llm_mining.local_miner.run_mining --mode optimize --series F001
  中断后再次运行同样命令即可断点续传; 加 --fresh 强制重新开始。
"""

import argparse
import concurrent.futures
import datetime
import os
import re
import sys
import time
import traceback

import pandas as pd

from . import (checkpoint, combo, console, diagnostics, factor_archive, factor_eval,
               factor_library, factor_plot, failed_library, prompts, review)
from .config import (
    MiningConfig, WORKSPACE_DIR, QUANT_ROOT, ensure_workspace, checkpoint_path, mining_log_path,
    report_png_path,
)
from .data_loader import MarketData
from .expr_engine import ExprError, compute_factor
from .llm_client import LLMError, call_llm, extract_json

# ---- 多进程并行评价: worker 进程内的全局数据/配置(由 initializer 注入) ----
_WORKER_DATA = None
_WORKER_CFG = None


def _init_worker(data, cfg):
    """进程池 worker 初始化: 注入全市场数据副本与配置"""
    global _WORKER_DATA, _WORKER_CFG
    _WORKER_DATA = data
    _WORKER_CFG = cfg


def _eval_factor_worker(meta: dict) -> dict:
    """worker 进程内执行: 计算因子宽表并评价, 返回带 eval 的 entry(多核并行, 互不影响)"""
    data = _WORKER_DATA
    cfg = _WORKER_CFG
    name = str(meta.get("名称", "factor"))
    desc = str(meta.get("描述", ""))
    expr = str(meta.get("公式", "")).strip()
    style = str(meta.get("风格", ""))
    entry = {"name": name, "desc": desc, "expr": expr, "style": style, "eval": None}
    t0 = time.time()
    try:
        factor_wide = compute_factor(expr, data, cfg)
        ev = factor_eval.evaluate_factor(factor_wide, data, cfg, name=name)
        entry["eval"] = factor_eval.to_serializable(ev)
        entry["_t_calc"] = round(time.time() - t0, 1)
    except ExprError as e:
        entry["eval"] = {"name": name, "error": f"公式校验失败: {e}", "qualified": False}
    except Exception as e:
        entry["eval"] = {"name": name, "error": f"{type(e).__name__}: {e}", "qualified": False}
    return entry


def parse_args():
    p = argparse.ArgumentParser(description="本地化LLM因子挖掘(双模式)")
    p.add_argument("--mode", type=str, default="new", choices=["new", "optimize"],
                   help="new=挖掘新因子; optimize=迭代优化已有因子系列")
    p.add_argument("--series", type=str, default="", help="optimize模式绑定的因子系列ID(如 F001)")
    p.add_argument("--max-rounds", type=int, default=12, help="最大迭代轮数")
    p.add_argument("--min-library-target", type=int, default=0,
                   help="new模式: 库中对应前缀系列数达到该值即提前结束挖掘(0=不启用; 分钟挖掘建议10)")
    p.add_argument("--max-depth", type=int, default=7, help="公式语法树最大嵌套深度")
    p.add_argument("--factors-per-round", type=int, default=2, help="每轮输出因子个数")
    p.add_argument("--direction", type=str, default=None, help="初始挖掘方向(new模式)")
    p.add_argument("--eval-start", type=str, default="2018-01-01", help="因子评价起始日期")
    p.add_argument("--thinking-budget", type=int, default=4096, help="模型思考token预算")
    p.add_argument("--workers", type=int, default=2,
                   help="因子评价并行进程数(1=串行; 每进程一份全市场数据副本, 注意内存)")
    p.add_argument("--fresh", action="store_true", help="忽略检查点, 重新开始")
    p.add_argument("--frequency", type=str, default="daily", choices=["daily", "minute"],
                   help="数据频率: daily=日频挖掘(默认); minute=分钟频率挖掘(最终因子仍为日频, 分钟算子聚合出日频特征)")
    p.add_argument("--minute-frequency", type=str, default="1m", choices=["1m", "5m", "15m", "30m", "60m"],
                   help="分钟基础频率(本地数据为1分钟线, 预留更高频率)")
    p.add_argument("--minute-batch-size", type=int, default=300,
                   help="分钟模式: 分批处理的股票数(内存控制; 移除冗余排序后批次峰值≈股票数×6MB×字段数, 96GB内存可调到500-1000)")
    p.add_argument("--minute-memory-fields", type=str, default="close,volume,amount",
                   help="分钟模式: 常驻内存的分钟字段(逗号分隔; 每字段稠密矩阵约13.7GB, 其余字段用时读盘)")
    p.add_argument("--minute-dense-batch", type=int, default=1000,
                   help="分钟稠密路径: 无截面聚合按股票分批的窗口大小(中间内存≈天数×240×批数×4B)")
    p.add_argument("--minute-dense-chunk-days", type=int, default=200,
                   help="分钟稠密路径: 含截面聚合按日期分块的块天数(每块加载全部股票使截面等价全市场)")
    p.add_argument("--minute-max-depth", type=int, default=14,
                   help="分钟模式允许的公式嵌套深度(相对日频上限翻倍, 分钟因子构成更复杂)")
    p.add_argument("--llm-provider", type=str, default="auto", choices=["auto", "opencode", "deepseek", "dashscope"],
                   help="LLM路由: opencode=主模型走OpenCode Go网关(含deepseek-v4-flash); auto=按模型名前缀")
    p.add_argument("--model", type=str, default="",
                   help="主模型名(如 deepseek-v4-flash; 留空使用配置默认)")
    p.add_argument("--model-fallback", type=str, default="",
                   help="备用模型名(留空使用配置默认)")
    p.add_argument("--output-dir", type=str, default="",
                   help="评价图/相关性/HTML报告输出目录(默认 output/<YYYYMMDD>_<任务名>)")
    return p.parse_args()


def call_model_with_retry(system_prompt: str, user_prompt: str, cfg, max_parse_retry: int = 2):
    """调用模型并解析JSON; 解析失败时把错误反馈给模型重试"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    last_raw = ""
    last_thinking = ""
    last_model = ""
    for attempt in range(max_parse_retry + 1):
        result = call_llm(messages, cfg)
        raw = result["content"]
        last_raw, last_thinking, last_model = raw, result["thinking"], result["model"]
        console.show_model_output(raw, thinking=last_thinking, model=last_model)
        try:
            data = extract_json(raw)
            if not isinstance(data, dict) or "因子列表" not in data:
                raise ValueError("JSON中缺少'因子列表'字段")
            return data, last_raw, last_thinking, last_model
        except Exception as e:
            console.log(f"    [解析失败] {e}")
            if attempt < max_parse_retry:
                console.log(f"    [解析失败] 要求模型修正输出(第{attempt+2}次)...")
                messages = messages + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": (
                        f"你上一次的输出无法被解析为符合要求的JSON: {e}。"
                        "请严格只输出一个符合schema的JSON对象, 不要包含任何代码块标记或解释文字。"
                    )},
                ]
    raise LLMError(f"模型输出连续{max_parse_retry+1}次无法解析为JSON。最后原文前300字: {last_raw[:300]}")


def run_round_factors(factors_meta: list, data, cfg, round_no: int,
                      executor: concurrent.futures.Executor | None = None) -> list:
    """计算并评价一轮中的所有因子, 返回带eval结果与风格的因子列表。
    executor 非空时多进程并行评价(每因子一个任务, 互不影响), 否则串行。"""
    if executor is not None and len(factors_meta) > 1:
        # ---- 并行分支: 全部提交到进程池, 完成后按原顺序展示 ----
        for idx, meta in enumerate(factors_meta, 1):
            console.show_factor_eval_header(
                round_no, idx, str(meta.get("名称", f"factor_{round_no}_{idx}")),
                str(meta.get("公式", "")), str(meta.get("描述", "")))
        futures = {executor.submit(_eval_factor_worker, meta): idx
                   for idx, meta in enumerate(factors_meta, 1)}
        done = {}
        for fut in concurrent.futures.as_completed(futures):
            idx = futures[fut]
            try:
                done[idx] = fut.result()
            except Exception as e:  # 进程级意外异常(不应发生, 兜底)
                meta = factors_meta[idx - 1]
                name = str(meta.get("名称", f"factor_{round_no}_{idx}"))
                done[idx] = {"name": name, "desc": "", "expr": str(meta.get("公式", "")),
                             "style": "", "eval": {"name": name,
                                                   "error": f"并行评价异常: {type(e).__name__}: {e}",
                                                   "qualified": False}}
        round_factors = []
        for idx in sorted(done):
            entry = done[idx]
            t_calc = entry.pop("_t_calc", None)
            if t_calc is not None:
                console.log(f"    公式校验与计算成功, 耗时 {t_calc} 秒, 开始因子评价...")
            console.show_factor_eval_result(entry["eval"])
            round_factors.append(entry)
        return round_factors

    # ---- 串行分支(原逻辑) ----
    round_factors = []
    for idx, meta in enumerate(factors_meta, 1):
        name = str(meta.get("名称", f"factor_{round_no}_{idx}"))
        desc = str(meta.get("描述", ""))
        expr = str(meta.get("公式", "")).strip()
        style = str(meta.get("风格", ""))
        console.show_factor_eval_header(round_no, idx, name, expr, desc)

        entry = {"name": name, "desc": desc, "expr": expr, "style": style, "eval": None}
        t0 = time.time()
        try:
            factor_wide = compute_factor(expr, data, cfg)
            console.log(f"    公式校验与计算成功, 耗时 {time.time()-t0:.1f} 秒, 开始因子评价...")
            t1 = time.time()
            ev = factor_eval.evaluate_factor(factor_wide, data, cfg, name=name)
            console.log(f"    因子评价完成, 耗时 {time.time()-t1:.1f} 秒")
            entry["eval"] = factor_eval.to_serializable(ev)
            # 串行路径: 回传宽表供主进程生成评价图(并行路径在 worker 内, 不跨进程传大数组)
            entry["_factor_wide"] = factor_wide
        except ExprError as e:
            console.log(f"    [×] 公式未通过运行前校验: {e}")
            entry["eval"] = {"name": name, "error": f"公式校验失败: {e}", "qualified": False}
        except Exception as e:
            console.log(f"    [×] 因子计算异常: {type(e).__name__}: {e}")
            entry["eval"] = {"name": name, "error": f"{type(e).__name__}: {e}", "qualified": False}

        console.show_factor_eval_result(entry["eval"])
        round_factors.append(entry)
    return round_factors


def recompute_series(expr: str, flipped: bool, data, cfg):
    """重算因子宽表与长表Series(评价/诊断/相关性检查共用), 按eval方向翻转"""
    factor_wide = compute_factor(expr, data, cfg)
    if flipped:
        factor_wide = -factor_wide
    f = factor_wide[factor_wide.index >= pd.Timestamp(cfg.eval_start_date)]
    series = f.stack(future_stack=True).dropna()
    series.index.names = ["date", "code"]
    series.name = "factor"
    return factor_wide, series


def save_adopted_report(factor: dict, data, cfg, series_id: str, round_no: int) -> str | None:
    """采纳合格因子后即时出图保存(带轮次与因子名, 不覆盖历史), 失败不中断主流程"""
    from .report import generate_report
    name = factor.get("name", "factor")
    safe = re.sub(r"[^0-9A-Za-z_]", "", name)[:40] or "factor"
    png_path = os.path.join(WORKSPACE_DIR, f"factor_report_{series_id}_R{round_no}_{safe}.png")
    console.log(f"    [出图] 采纳合格因子 {name}, 正在生成回测图片(完整tear sheet)...")
    try:
        png = generate_report(factor, data, cfg, png_path=png_path)
        console.log(f"    [出图完成] 已保存: {png}")
        return png
    except Exception as e:
        console.log(f"    [出图失败] {type(e).__name__}: {e}, 不中断挖掘。")
        return None


def main():
    args = parse_args()
    if args.mode == "optimize" and not args.series:
        print("错误: optimize 模式必须通过 --series 指定因子系列ID(如 --series F001)")
        sys.exit(2)
    ensure_workspace()

    mode = args.mode
    series_id = args.series
    freq = args.frequency
    ckpt_path = checkpoint_path(mode, series_id, freq)
    log_path = mining_log_path(mode, series_id, freq)
    console.init_console(log_path)

    # ---------------- 阶段1: 配置初始化 ----------------
    cfg = MiningConfig(
        max_rounds=args.max_rounds,
        max_depth=args.max_depth,
        factors_per_round=args.factors_per_round,
        eval_start_date=args.eval_start,
        thinking_budget=args.thinking_budget,
        data_frequency=freq,
        minute_frequency=args.minute_frequency,
        minute_batch_size=args.minute_batch_size,
        minute_max_depth=args.minute_max_depth,
    )
    cfg.min_library_target = args.min_library_target
    if args.model:
        cfg.model_primary = args.model
    if args.model_fallback:
        cfg.model_fallback = args.model_fallback
    if args.direction:
        cfg.direction = args.direction
    cfg.n_eval_workers = args.workers
    if args.llm_provider != "auto":
        cfg.llm_provider = args.llm_provider
    if args.minute_memory_fields:
        cfg.minute_memory_fields = args.minute_memory_fields
    cfg.minute_dense_batch = args.minute_dense_batch
    cfg.minute_dense_chunk_days = args.minute_dense_chunk_days

    # optimize 模式: 预加载目标系列
    opt_series = None
    if mode == "optimize":
        opt_series = factor_library.load_series(series_id)
        if opt_series is None:
            console.banner(f"错误: 因子库中找不到系列 {series_id}, 无法优化", "=")
            sys.exit(1)

    is_minute = cfg.data_frequency == "minute"
    # 成果输出目录: 评价图/全库相关性/静态HTML报告
    if args.output_dir:
        out_dir = args.output_dir
    else:
        task_name = "分钟因子挖掘" if is_minute else "因子挖掘"
        out_dir = os.path.join(QUANT_ROOT, "output",
                               f"{datetime.date.today().strftime('%Y%m%d')}_{task_name}")
    os.makedirs(out_dir, exist_ok=True)
    console.stage("1/6", f"配置初始化 · 模式={'新因子挖掘' if mode=='new' else '现有因子优化'+series_id}")
    console.log(f"    [输出] 评价图/全库相关性/HTML报告目录: {out_dir}")
    show_kv = {
        "运行模式": "new(新因子挖掘)" if mode == "new" else f"optimize(优化系列 {series_id})",
        "主模型": cfg.model_primary,
        "备用模型": cfg.model_fallback,
        "深度思考": f"开启, 预算 {cfg.thinking_budget} tokens",
        "数据频率": "日频(原始逻辑)" if not is_minute else f"分钟挖掘(基础频率 {cfg.minute_frequency}, 聚合出日频因子)",
        "最大迭代轮数": cfg.max_rounds,
        "每轮因子个数": cfg.factors_per_round,
        "公式最大嵌套深度": cfg.formula_max_depth,
        "IC口径": f"{cfg.ic_period}日 RankIC",
        "分组数": f"{cfg.n_quantiles} 组(最高组为多头)",
        "月度IC为正占比要求": f"≥{cfg.monthly_ic_pos_ratio*100:.0f}%",
        "库相关性上限": f"|ρ|≤{cfg.max_library_corr}(new模式防撞车)",
        "挖掘方向": cfg.direction,
    }
    if is_minute:
        show_kv["分钟分批股票数"] = f"{cfg.minute_batch_size}(全市场分钟数据约51GB, 按批次流式计算)"
        show_kv["分钟常驻内存字段"] = cfg.minute_memory_fields
        if cfg.minute_memory_fields and cfg.n_eval_workers > 1:
            console.log("    [警告] 分钟常驻内存字段已开启(每字段约13.7GB/进程), "
                        "建议 --workers 1 以避免多进程内存叠加。")
    console.show_kv_table("运行配置(外部可变参数)", show_kv)
    if is_minute and cfg.n_eval_workers > 1:
        console.log("    [提示] 分钟模式每个因子计算较慢(需按批次遍历分钟数据), "
                    f"当前 {cfg.n_eval_workers} 个进程并行(每进程独立加载分钟数据, 注意内存)。")

    # ---------------- 断点续传 ----------------
    state = None
    if args.fresh:
        checkpoint.clear(ckpt_path)
        state = None
    else:
        state = checkpoint.load(ckpt_path)
    if state is None:
        state = checkpoint.new_state(cfg, mode=mode, series_id=series_id)
        console.log("\n    [断点续传] 未发现检查点, 全新开始。")
    else:
        console.log(f"\n    [断点续传] 恢复检查点: 已完成 {state['round']} 轮, "
                    f"阶段={state['stage']}, "
                    f"待评价因子={'有' if state.get('pending') else '无'}")

    # 长期黑名单记忆(已否决公式) 与 战役入库追踪(供战役结束压缩摘要) 初始化
    state.setdefault("rejected_mem", {"items": [], "max_items": 60, "dropped": 0})
    state.setdefault("campaign_adopted", [])
    # 回填: 已有历史轮次的被拒教训一次性沉淀进黑名单(重启后新机制立即可用)
    if not state.get("rejected_mem", {}).get("items") and state.get("history"):
        _mem = state["rejected_mem"]
        for _rec in state["history"]:
            prompts.append_rejection_memory(
                _mem, prompts.extract_rejection_memory(_rec.get("factors") or [],
                                                       _rec.get("round")))
        if _mem.get("items"):
            console.log(f"    [黑名单] 已回填历史 {len(state['history'])} 轮被拒教训, "
                        f"累计 {len(_mem['items'])} 条。")
    campaign_summary_text = prompts.load_campaign_summaries()
    if campaign_summary_text:
        console.log("    [战役记忆] 已加载历史战役信息压缩摘要, 将注入首轮提示词。")

    # 从检查点历史恢复 HTML 报告的历史轮次(旧轮次无评价图, 仅展示公式与指标)
    plot_rounds = []
    for rec in state.get("history", []):
        plot_rounds.append({"round": rec.get("round"),
                            "hypothesis": rec.get("hypothesis", ""),
                            "reflection": rec.get("reflection", ""),
                            "factors": rec.get("factors", [])})

    # ---------------- 阶段2: 数据加载 ----------------
    console.stage("2/6", "本地量价数据加载(全A股后复权, 内存化)")
    if cfg.data_frequency == "minute":
        console.log("    [分钟模式] 日频数据(宽表)照常加载; 分钟数据(约51GB)不整体载入, "
                    "因子计算时按股票批次流式读取(每批约"
                    f"{cfg.minute_batch_size}只)。")
    data = MarketData(cfg)

    # 多进程并行评价进程池(每 worker 一份全市场数据副本, 跨轮复用)
    eval_executor = None
    if cfg.n_eval_workers > 1:
        eval_executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=cfg.n_eval_workers, initializer=_init_worker, initargs=(data, cfg))
        console.log(f"    [并行] 因子评价启用 {cfg.n_eval_workers} 个进程并行"
                    f"(每进程一份全市场数据副本, 注意内存占用)。")

    # ---------------- 阶段3: 系统提示词 ----------------
    console.stage("3/6", "构建系统提示词并输入模型(固定部分略显示, 变量部分全显示)")
    output_format = prompts.OUTPUT_FORMAT_OPTIMIZE if mode == "optimize" \
        else prompts.OUTPUT_FORMAT_INITIAL
    system_prompt, fixed_parts, variables = prompts.build_system_prompt(cfg, output_format)
    console.show_prompt_parts("系统提示词", fixed_parts, variables)

    # new模式: 预构建因子库摘要(防撞车)
    library_text = factor_library.library_summary_text() if mode == "new" else ""

    # ---------------- 阶段4/5: 挖掘循环 ----------------
    qualified_factor = None
    corr_res = None       # 最近一次全库相关性矩阵(入库后刷新)
    try:
        while state["round"] < cfg.max_rounds:
            round_no = state["round"] + 1
            console.round_banner(round_no, cfg.max_rounds)

            # ---- 步骤A: 模型输出因子(或断点恢复) ----
            pending = state.get("pending")
            if pending and pending.get("round") == round_no:
                console.log(f"    [断点续传] 检测到第{round_no}轮模型输出已保存但未评价, "
                            f"跳过模型调用, 直接进入因子计算与评价。")
                parsed = pending["parsed"]
            else:
                console.stage(f"4/6 · 第{round_no}轮", "模型输入(提示词构建)")
                if mode == "optimize":
                    series_text = factor_library.series_current_text(opt_series)
                    advice_text = diagnostics.format_diagnostics_advice(
                        (opt_series.get("best") or {}).get("diagnostics") or {}, cfg)
                    if state["history"]:
                        feedback = factor_eval.build_feedback_text(
                            state["history"][-1]["factors"], cfg)
                    else:
                        feedback = factor_eval.build_feedback_text([opt_series["best"]], cfg)
                    ensemble_warning = factor_library.detect_ensemble_tendency(opt_series)
                    if ensemble_warning:
                        console.log("    [!] 检测到近期'同函数不同参数堆叠'倾向, 已向模型注入禁止告诫。")
                    user_prompt, user_vars = prompts.build_optimize_user_prompt(
                        cfg, round_no, series_text, advice_text, feedback, ensemble_warning)
                    prompt_title = f"第{round_no}轮因子优化提示词"
                elif not state["history"]:
                    failed_text = failed_library.injection_text()
                    user_prompt, user_vars = prompts.build_new_initial_user_prompt(
                        cfg, library_text, campaign_summary_text, failed_text)
                    prompt_title = "首轮新因子生成提示词(含因子库避让+战役压缩摘要+失败经验)"
                else:
                    # 仅注入最近 max_history_rounds 轮历史摘要, 控制请求体积
                    # (DeepSeek 长请求易触发空内容故障, 短请求稳定)
                    max_history_rounds = 3
                    history_summary = prompts.summarize_history(state["history"][-max_history_rounds:])
                    feedback = factor_eval.build_feedback_text(
                        state["history"][-1]["factors"], cfg)
                    blacklist_text = prompts.format_rejection_memory(
                        state.get("rejected_mem") or {})
                    user_prompt, user_vars = prompts.build_iteration_user_prompt(
                        cfg, round_no, history_summary, feedback, blacklist_text)
                    prompt_title = f"第{round_no}轮迭代优化提示词"
                console.show_prompt_parts(prompt_title, fixed_parts={}, variables=user_vars)

                console.stage(f"5/6 · 第{round_no}轮", "调用模型挖掘因子")
                parsed, raw, thinking, model = call_model_with_retry(
                    system_prompt, user_prompt, cfg)

                state["pending"] = {
                    "round": round_no, "parsed": parsed, "raw": raw, "model": model,
                }
                state["stage"] = "model_called"
                checkpoint.save(state, ckpt_path)
                console.log(f"    [断点续传] 模型输出已存档到检查点。")

            # ---- 步骤B: 因子计算与评价 ----
            console.stage(f"5/6 · 第{round_no}轮", "因子计算与评价(本地全A股)")
            hypothesis = str(parsed.get("因子假设") or parsed.get("新假设") or "")
            reflection = str(parsed.get("上轮反思") or parsed.get("优化思路") or "")
            if reflection:
                console.log(f"    模型思路: {reflection[:200]}{'...' if len(reflection)>200 else ''}")
            console.log(f"    本轮因子假设: {hypothesis}")
            factors_meta = parsed.get("因子列表") or []
            if not isinstance(factors_meta, list) or not factors_meta:
                console.log("    [警告] 模型未输出有效因子列表, 本轮跳过。")
                factors_meta = []

            round_factors = run_round_factors(factors_meta, data, cfg, round_no, eval_executor)

            # ---- 每轮评价图: 为每个因子生成数学化公式评价图(串行路径有 _factor_wide) ----
            for idx, f in enumerate(round_factors, 1):
                fw = f.pop("_factor_wide", None)
                if fw is None:
                    # 并行路径: 宽表在 worker 进程内, 不跨进程回传, 跳过出图
                    continue
                if (f.get("eval") or {}).get("error"):
                    continue
                safe = re.sub(r"[^0-9A-Za-z_]", "", f.get("name", "factor"))[:40] or "factor"
                png_rel = f"round{round_no:03d}_f{idx:02d}_{safe}.png"
                if factor_plot.plot_round_factor(f, fw, data, cfg,
                                                 os.path.join(out_dir, png_rel),
                                                 round_no=round_no, idx=idx):
                    f["_png_rel"] = png_rel

            # ---- 步骤C: 反馈构建与状态更新 ----
            console.stage(f"5/6 · 第{round_no}轮", "构建评价反馈(将用于下一轮模型输入)")

            # 组合类因子(最后手段)处理: 评价已完成, 但隐藏——不反馈、不写入对话记录
            hidden_combos = state.setdefault("hidden_combos", [])
            single_factors, combo_factors = [], []
            for f in round_factors:
                ev = f.get("eval") or {}
                if not ev.get("error") and combo.detect_combo(f.get("expr", "")):
                    combo_factors.append(f)
                else:
                    single_factors.append(f)

            # 1) 单逻辑因子与已隐藏组合比较: 仅记录"被超越"(不立即反馈, 避免过早泄露隐藏机制)
            for f in single_factors:
                ev = f.get("eval") or {}
                for c in hidden_combos:
                    if not c.get("surpassed") and combo.combo_better(ev, c.get("eval") or {}):
                        c["surpassed"] = f.get("name")

            # 2) 本轮组合因子: 缓存隐藏 + 枯竭判定 + (再次组合时)引导反馈
            for f in combo_factors:
                entry = {"round": round_no, "name": f.get("name"), "expr": f.get("expr"),
                         "eval": f.get("eval"), "surpassed": None}
                if not state.get("combo_exhausted"):
                    unsup = [c for c in hidden_combos if not c.get("surpassed")]
                    if unsup:
                        state["combo_exhausted"] = True
                        console.banner(
                            f"⚠ 穷途末路判定: 模型再次产出组合类因子({f['name']}), "
                            f"且两次之间没有单逻辑因子超越此前组合({unsup[0].get('name')})"
                            f"——该系列的单一经济逻辑已挖掘到尽头。", "!")
                hidden_combos.append(entry)
                console.log(f"    [隐藏] 组合类因子 {f['name']} 已评价但隐藏(最后手段, 不反馈): {f['expr']}")

            # 3) 反馈文本: 仅基于单逻辑因子(组合因子从对话记录中抹去)
            if single_factors:
                feedback_text = factor_eval.build_feedback_text(single_factors, cfg)
            else:
                feedback_text = (
                    "本轮输出的因子均为组合类(多个已有项相加/平均, 系统已隐藏评估)或存在错误, "
                    "未产生可评价的单逻辑因子。请重新提出基于单一经济逻辑的结构创新因子, "
                    "不要将已有项相加/平均; 组合只应在创新确实枯竭时才作为最后手段。")
            # 组合引导: 仅当模型本轮又产出组合 且 此前组合已被更好的单逻辑因子超越
            if combo_factors and not state.get("combo_exhausted"):
                sup = [c for c in hidden_combos if c.get("surpassed")]
                if sup:
                    names = "、".join(f"{c['name']}(已被{c['surpassed']}超越)" for c in sup)
                    feedback_text += (
                        f"\n【提醒】你本轮再次提出组合类因子(系统已隐藏其评价), 而此前组合 {names} "
                        "已被表现更好的单逻辑因子超越——请沿单一经济逻辑继续优化, "
                        "不要走组合捷径(组合是最后手段)。")
            console.log("    反馈文本摘要:")
            for line in feedback_text.splitlines():
                if line.startswith("[") or line.startswith("━") or line.startswith("综合") \
                        or line.startswith("【提醒"):
                    console.log(f"        {line}")

            record = {
                "round": round_no,
                "hypothesis": hypothesis,
                "reflection": reflection,
                "factors": single_factors,
            }
            state["history"].append(record)
            state["round"] = round_no
            state["pending"] = None
            state["stage"] = "round_done"

            # ---- 失败因子库沉淀: 每轮全部因子(含不合格) + 该轮一句话总结 ----
            try:
                failed_library.append_round(failed_library.cfg_campaign_key(cfg),
                                            round_no, hypothesis, reflection, round_factors)
            except Exception as e:
                console.log(f"    [失败库沉淀失败] {type(e).__name__}: {e}")

            # ---- 步骤D: 按模式处理合格因子(仅针对单逻辑因子) ----
            if mode == "new":
                adopted = handle_new_mode_qualified(
                    single_factors, round_no, hypothesis, data, cfg, state)
                if adopted is not None:
                    qualified_factor = adopted
                    console.banner(
                        f"第{round_no}轮新因子入库: {adopted['name']} "
                        f"(系列 {adopted['series_id']}), 继续下一轮挖掘...", "=")
                    # 入库后顺带更新全库相关性矩阵(热图+CSV 随 HTML 报告落盘)
                    try:
                        corr_res = factor_plot.compute_library_corr_matrix(data, cfg)
                        console.log(f"    [相关性] 全库相关性矩阵已更新: "
                                    f"{corr_res.get('n', 0)} 个因子, "
                                    f"|ρ|max={corr_res.get('max_abs', 0):.2f} "
                                    f"(上限 {cfg.max_library_corr})")
                    except Exception as e:
                        console.log(f"    [相关性更新失败] {type(e).__name__}: {e}")
                        corr_res = None
                else:
                    console.log(f"\n    第{round_no}轮无可入库新因子, 继续下一轮迭代...")
                # 长期黑名单记忆: 在语义评审/相关性检查(步骤D)之后沉淀本轮被拒因子,
                # 保证 review_rejected/library_rejected 已写入 eval 后再提取
                mem = state.setdefault("rejected_mem",
                                       {"items": [], "max_items": 60, "dropped": 0})
                entries = prompts.extract_rejection_memory(single_factors, round_no)
                if entries:
                    prompts.append_rejection_memory(mem, entries)
                    console.log(f"    [黑名单] 本轮新增被拒记忆 {len(entries)} 条, "
                                f"累计 {len(mem.get('items') or [])} 条。")
                checkpoint.save(state, ckpt_path)  # 每轮落盘, 支持长时间运行断点续传
            else:
                improved = handle_optimize_mode_qualified(
                    single_factors, round_no, hypothesis, opt_series, series_id,
                    data, cfg, state)
                if improved is not None:
                    qualified_factor = improved
                checkpoint.save(state, ckpt_path)
                console.log(f"\n    第{round_no}轮优化完成, 继续下一轮迭代...")

            # ---- 步骤E: 汇总本轮记录并刷新静态HTML报告(KaTeX公式+评价图+相关性) ----
            plot_rounds.append({"round": round_no, "hypothesis": hypothesis,
                                "reflection": reflection, "factors": round_factors})
            try:
                factor_plot.build_html_report(out_dir, plot_rounds, cfg, corr_res=corr_res)
                console.log(f"    [HTML报告] 已刷新: {os.path.join(out_dir, 'index.html')}")
            except Exception as e:
                console.log(f"    [HTML报告失败] {type(e).__name__}: {e}")

            # ---- 步骤F: 入库目标检查(new模式, 达到目标即提前结束) ----
            if mode == "new" and cfg.min_library_target > 0:
                prefix = "M" if is_minute else "F"
                n_in = factor_library.count_series(prefix)
                if n_in >= cfg.min_library_target:
                    console.banner(
                        f"已达成入库目标: 库中 {prefix} 系列 {n_in} 个 ≥ "
                        f"{cfg.min_library_target}, 提前结束挖掘", "=")
                    break

    except KeyboardInterrupt:
        checkpoint.save(state, ckpt_path)
        console.log("\n    [中断] 已保存检查点, 下次运行相同命令即可续传。")
        sys.exit(1)
    except LLMError as e:
        checkpoint.save(state, ckpt_path)
        console.log(f"\n    [错误] LLM调用失败: {e}")
        console.log("    已保存检查点, 修复网络/密钥后重新运行即可续传。")
        sys.exit(1)
    finally:
        if eval_executor is not None:
            eval_executor.shutdown(cancel_futures=True)

    # ---------------- 战役结束信息压缩 ----------------
    # 跑满 max_rounds 且本战役有因子入库时, 将整体历史压缩为概括性摘要并持久化,
    # 供下一个战役首轮注入(避免长期逐轮记忆导致上下文过长、稀释模型注意力)
    if state["round"] >= cfg.max_rounds and state.get("campaign_adopted"):
        try:
            lib_now = factor_library.load_library()
            summary = prompts.summarize_campaign(state, cfg, lib_now)
            if prompts.save_campaign_summary(summary):
                console.log(f"    [战役压缩] 战役完成, 已生成信息压缩摘要"
                            f"(入库 {len(state['campaign_adopted'])} 个系列, "
                            f"供下次挖掘首轮注入)。")
        except Exception as e:
            console.log(f"    [战役压缩失败] {type(e).__name__}: {e}")

    # 失败因子库战役总结: 跑满轮数后, 汇总全部轮次失败明细, 规则统计+LLM概括生成战役总结
    if state["round"] >= cfg.max_rounds:
        try:
            cs = failed_library.finalize_campaign(cfg, state["round"],
                                                  state.get("campaign_adopted"))
            if cs:
                console.log(f"    [失败库战役总结] 战役 {cs['campaign']} 已完成, "
                            f"LLM概括{'成功' if cs.get('llm_ok') else '失败(已降级为规则统计)'}。")
        except Exception as e:
            console.log(f"    [失败库战役总结失败] {type(e).__name__}: {e}")

    # ---------------- 阶段6: 最终评价与图片 ----------------
    console.stage("6/6", "最终因子全面评价与图片报告")
    finalize(mode, series_id, opt_series, qualified_factor, state, data, cfg, ckpt_path)


def handle_new_mode_qualified(round_factors, round_no, hypothesis, data, cfg, state):
    """new模式: 对合格因子做库相关性检查, 通过则分配新系列ID入库。返回入库因子或None"""
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

        # 通过 -> 计算诊断并入库(分钟因子用 M 系列编号, 与日频 F 系列独立计数)
        console.log(f"    [√] 相关性检查通过, 计算诊断并入库...")
        diag = diagnostics.compute_diagnostics(series, data, cfg)
        prefix = "M" if cfg.data_frequency == "minute" else "F"
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


if __name__ == "__main__":
    main()
