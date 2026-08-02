"""
本地化因子挖掘项目 - 主程序

流程(对应需求4.4的阶段衔接展示):
  阶段1 配置初始化 -> 阶段2 数据加载 -> 阶段3 系统提示词
  -> 循环[ 模型输入 -> 模型输出 -> 因子计算 -> 因子评价 -> 反馈构建 ]
  -> 最终阶段: 合格因子全面评价 + 生成图片报告

用法:
  python -m llm_mining.local_miner.run_mining --max-rounds 12 --max-depth 7
  中断后再次运行同样命令即可断点续传; 加 --fresh 强制重新开始。
"""

import argparse
import sys
import time
import traceback

from . import checkpoint, console, factor_eval, prompts
from .config import (
    MiningConfig, ensure_workspace, BEST_FACTOR_PATH, LOG_PATH, REPORT_PNG_PATH,
)
from .data_loader import MarketData
from .expr_engine import ExprError, compute_factor
from .llm_client import LLMError, call_llm, extract_json


def parse_args():
    p = argparse.ArgumentParser(description="本地化LLM因子挖掘")
    p.add_argument("--max-rounds", type=int, default=12, help="最大迭代轮数")
    p.add_argument("--max-depth", type=int, default=7, help="公式语法树最大嵌套深度")
    p.add_argument("--factors-per-round", type=int, default=2, help="每轮输出因子个数")
    p.add_argument("--direction", type=str, default=None, help="初始挖掘方向")
    p.add_argument("--eval-start", type=str, default="2018-01-01", help="因子评价起始日期")
    p.add_argument("--thinking-budget", type=int, default=4096, help="模型思考token预算")
    p.add_argument("--fresh", action="store_true", help="忽略检查点, 重新开始")
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


def run_round_factors(factors_meta: list, data, cfg, round_no: int) -> list:
    """计算并评价一轮中的所有因子, 返回带eval结果的因子列表"""
    round_factors = []
    for idx, meta in enumerate(factors_meta, 1):
        name = str(meta.get("名称", f"factor_{round_no}_{idx}"))
        desc = str(meta.get("描述", ""))
        expr = str(meta.get("公式", "")).strip()
        console.show_factor_eval_header(round_no, idx, name, expr, desc)

        entry = {"name": name, "desc": desc, "expr": expr, "eval": None}
        t0 = time.time()
        try:
            factor_wide = compute_factor(expr, data, cfg)
            console.log(f"    公式校验与计算成功, 耗时 {time.time()-t0:.1f} 秒, 开始因子评价...")
            t1 = time.time()
            ev = factor_eval.evaluate_factor(factor_wide, data, cfg, name=name)
            console.log(f"    因子评价完成, 耗时 {time.time()-t1:.1f} 秒")
            entry["eval"] = factor_eval.to_serializable(ev)
        except ExprError as e:
            console.log(f"    [×] 公式未通过运行前校验: {e}")
            entry["eval"] = {"name": name, "error": f"公式校验失败: {e}", "qualified": False}
        except Exception as e:
            console.log(f"    [×] 因子计算异常: {type(e).__name__}: {e}")
            entry["eval"] = {"name": name, "error": f"{type(e).__name__}: {e}", "qualified": False}

        console.show_factor_eval_result(entry["eval"])
        round_factors.append(entry)
    return round_factors


def main():
    args = parse_args()
    ensure_workspace()
    console.init_console(LOG_PATH)

    # ---------------- 阶段1: 配置初始化 ----------------
    cfg = MiningConfig(
        max_rounds=args.max_rounds,
        max_depth=args.max_depth,
        factors_per_round=args.factors_per_round,
        eval_start_date=args.eval_start,
        thinking_budget=args.thinking_budget,
    )
    if args.direction:
        cfg.direction = args.direction

    console.stage("1/6", "配置初始化")
    console.show_kv_table("运行配置(外部可变参数)", {
        "主模型": cfg.model_primary,
        "备用模型": cfg.model_fallback,
        "深度思考": f"开启, 预算 {cfg.thinking_budget} tokens",
        "最大迭代轮数": cfg.max_rounds,
        "每轮因子个数": cfg.factors_per_round,
        "公式最大嵌套深度": cfg.max_depth,
        "公式最大长度": cfg.max_symbol_length,
        "数据加载起始日": cfg.data_start_date,
        "因子评价起始日": cfg.eval_start_date,
        "IC口径": f"{cfg.ic_period}日 RankIC",
        "分组数": f"{cfg.n_quantiles} 组(最高组为多头)",
        "月度IC为正占比要求": f"≥{cfg.monthly_ic_pos_ratio*100:.0f}%",
        "挖掘方向": cfg.direction,
    })

    # ---------------- 断点续传 ----------------
    state = None
    if not args.fresh:
        state = checkpoint.load()
    if args.fresh:
        checkpoint.clear()
        state = None
    if state is None:
        state = checkpoint.new_state(cfg)
        console.log("\n    [断点续传] 未发现检查点, 全新开始。")
    else:
        console.log(f"\n    [断点续传] 恢复检查点: 已完成 {state['round']} 轮, "
                    f"阶段={state['stage']}, "
                    f"待评价因子={'有' if state.get('pending') else '无'}")

    # ---------------- 阶段2: 数据加载 ----------------
    console.stage("2/6", "本地量价数据加载(全A股后复权, 内存化)")
    data = MarketData(cfg)

    # ---------------- 阶段3: 系统提示词 ----------------
    console.stage("3/6", "构建系统提示词并输入模型(固定部分略显示, 变量部分全显示)")
    system_prompt, fixed_parts, variables = prompts.build_system_prompt(cfg)
    console.show_prompt_parts("系统提示词", fixed_parts, variables)

    # ---------------- 阶段4/5: 挖掘循环 ----------------
    qualified_factor = None
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
                if not state["history"]:
                    user_prompt, user_vars = prompts.build_initial_user_prompt(cfg)
                    prompt_title = "首轮因子生成提示词"
                else:
                    history_summary = prompts.summarize_history(state["history"])
                    feedback = factor_eval.build_feedback_text(
                        state["history"][-1]["factors"], cfg)
                    user_prompt, user_vars = prompts.build_iteration_user_prompt(
                        cfg, round_no, history_summary, feedback)
                    prompt_title = f"第{round_no}轮迭代优化提示词"
                console.show_prompt_parts(prompt_title, fixed_parts={}, variables=user_vars)

                console.stage(f"5/6 · 第{round_no}轮", "调用模型挖掘因子")
                parsed, raw, thinking, model = call_model_with_retry(
                    system_prompt, user_prompt, cfg)

                # 模型输出到手后立即存档(断点: 不浪费模型思考结果)
                state["pending"] = {
                    "round": round_no,
                    "parsed": parsed,
                    "raw": raw,
                    "model": model,
                }
                state["stage"] = "model_called"
                checkpoint.save(state)
                console.log(f"    [断点续传] 模型输出已存档到检查点。")

            # ---- 步骤B: 因子计算与评价 ----
            console.stage(f"5/6 · 第{round_no}轮", "因子计算与评价(本地全A股)")
            hypothesis = str(parsed.get("因子假设") or parsed.get("新假设") or "")
            reflection = str(parsed.get("上轮反思") or "")
            if reflection:
                console.log(f"    模型上轮反思: {reflection[:200]}{'...' if len(reflection)>200 else ''}")
            console.log(f"    本轮因子假设: {hypothesis}")
            factors_meta = parsed.get("因子列表") or []
            if not isinstance(factors_meta, list) or not factors_meta:
                console.log("    [警告] 模型未输出有效因子列表, 本轮跳过。")
                factors_meta = []

            round_factors = run_round_factors(factors_meta, data, cfg, round_no)

            # ---- 步骤C: 反馈构建与状态更新 ----
            console.stage(f"5/6 · 第{round_no}轮", "构建评价反馈(将用于下一轮模型输入)")
            feedback_text = factor_eval.build_feedback_text(round_factors, cfg)
            console.log("    反馈文本摘要:")
            for line in feedback_text.splitlines():
                if line.startswith("[") or line.startswith("━") or line.startswith("综合"):
                    console.log(f"        {line}")

            record = {
                "round": round_no,
                "hypothesis": hypothesis,
                "reflection": reflection,
                "factors": round_factors,
            }
            state["history"].append(record)
            state["round"] = round_no
            state["pending"] = None
            state["stage"] = "round_done"

            # 记录最佳合格因子
            for f in round_factors:
                ev = f.get("eval") or {}
                if ev.get("qualified"):
                    if qualified_factor is None or \
                            (ev.get("ic_mean") or -1) > (qualified_factor["eval"].get("ic_mean") or -1):
                        qualified_factor = {**f, "round": round_no, "hypothesis": hypothesis}
                        state["best"] = qualified_factor
            checkpoint.save(state)

            if qualified_factor is not None:
                console.banner(f"第{round_no}轮发现合格因子: {qualified_factor['name']}, 提前结束挖掘", "=")
                break
            else:
                console.log(f"\n    第{round_no}轮无合格因子, 继续下一轮迭代...")

    except KeyboardInterrupt:
        checkpoint.save(state)
        console.log("\n    [中断] 已保存检查点, 下次运行相同命令即可续传。")
        sys.exit(1)
    except LLMError as e:
        checkpoint.save(state)
        console.log(f"\n    [错误] LLM调用失败: {e}")
        console.log("    已保存检查点, 修复网络/密钥后重新运行即可续传。")
        sys.exit(1)

    # ---------------- 阶段6: 最终评价与图片 ----------------
    console.stage("6/6", "最终因子全面评价与图片报告")
    if qualified_factor is None:
        # 未挖到完全合格因子: 选方向正确的最佳因子做报告(标注未完全达标)
        candidates = []
        for rec in state["history"]:
            for f in rec["factors"]:
                ev = f.get("eval") or {}
                if ev.get("direction_ok"):
                    candidates.append({**f, "round": rec["round"],
                                       "hypothesis": rec.get("hypothesis", "")})
        if candidates:
            candidates.sort(key=lambda x: x["eval"].get("ic_mean") or -1, reverse=True)
            qualified_factor = candidates[0]
            state["best"] = qualified_factor
            checkpoint.save(state)
            console.log(f"    未挖到完全合格因子, 选取方向正确的最佳因子 "
                        f"{qualified_factor['name']} (IC均值="
                        f"{qualified_factor['eval'].get('ic_mean', 0)*100:+.3f}%) 做全面评价。")
        else:
            console.log("    没有任何方向正确的因子, 无法生成报告。请增加轮数或调整方向后重试。")
            sys.exit(1)

    import json
    with open(BEST_FACTOR_PATH, "w", encoding="utf-8") as f:
        json.dump(qualified_factor, f, ensure_ascii=False, indent=2)

    from .report import generate_report
    try:
        png_path = generate_report(qualified_factor, data, cfg)
        console.banner(f"挖掘结束! 因子报告图片已生成: {png_path}", "=")
    except Exception as e:
        console.log(f"    [错误] 报告生成失败: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
