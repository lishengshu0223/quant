"""
本地化因子挖掘项目 - 终端输出工具

要求(对应需求4):
- 固定提示词粗略显示, 外部变量完全显示
- 模型输出完整打印
- 因子评价用 [√]/[×] 清单展示达标情况
- 所有输出同时写入 workspace/mining.log
"""

import sys
import datetime

_LOG_FILE = None


def init_console(log_path: str):
    """初始化终端编码并绑定日志文件"""
    global _LOG_FILE
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    _LOG_FILE = open(log_path, "a", encoding="utf-8")
    log("")
    log(f"########## 因子挖掘会话启动: {datetime.datetime.now():%Y-%m-%d %H:%M:%S} ##########")


def log(msg: str = ""):
    """同时输出到终端和日志文件"""
    print(msg, flush=True)
    if _LOG_FILE is not None:
        _LOG_FILE.write(msg + "\n")
        _LOG_FILE.flush()


def banner(title: str, char: str = "="):
    line = char * 78
    log("")
    log(line)
    log(f"  {title}")
    log(line)


def stage(no: str, title: str):
    banner(f"[阶段 {no}] {title}", char="=")


def round_banner(round_no: int, total: int):
    banner(f"第 {round_no}/{total} 轮迭代", char="-")


def show_prompt_parts(title: str, fixed_parts: dict, variables: dict):
    """
    展示给模型的输入(需求4.1)
    - fixed_parts: {名称: 固定文本} -> 只显示长度和前60字摘要
    - variables: {名称: 变量文本} -> 完整显示
    """
    banner(f"[模型输入] {title}", char="-")
    log(">>> 固定部分(已写好, 仅显示摘要):")
    for name, text in fixed_parts.items():
        snippet = text.strip().replace("\n", " ")[:60]
        log(f"    [{name}] 共{len(text)}字符 | 摘要: {snippet}...")
    log(">>> 外部变量部分(完整显示):")
    for name, text in variables.items():
        log(f"    <<{name}>> =")
        for line in str(text).splitlines() or [""]:
            log(f"        {line}")
    log("")


def show_model_output(raw: str, thinking: str = "", model: str = ""):
    """展示模型输出(需求4.2): 终端完整打印"""
    banner(f"[模型输出] 模型: {model}", char="-")
    if thinking:
        log(">>> 模型思考过程(前800字):")
        log(thinking[:800] + ("..." if len(thinking) > 800 else ""))
        log("")
    log(">>> 模型正式回复(完整):")
    log(raw)
    log("")


def show_factor_eval_header(round_no: int, idx: int, name: str, expr: str, desc: str):
    banner(f"第{round_no}轮 · 因子{idx}: {name}", char="-")
    log(f"    公式: {expr}")
    log(f"    描述: {desc}")


def show_factor_eval_result(res: dict):
    """展示单个因子的评价结果(需求4.3)"""
    if res.get("error"):
        log(f"    [×] 因子计算/评价失败: {res['error']}")
        return
    mark = lambda ok: "[√]" if ok else "[×]"
    log(f"    {mark(res['direction_ok'])} 方向检验: "
        f"IC均值={res['ic_mean']*100:+.3f}%, 多头累计超额={res['long_total']*100:+.2f}%"
        + (f" (因子已取相反数翻转方向)" if res["flipped"] else ""))
    if not res["direction_ok"]:
        log(f"        -> IC与多头收益方向矛盾或为负, 属于无效因子, 必须剔除")
    log(f"    {mark(res['monthly_ok'])} 月度IC为正占比: {res['monthly_pos_ratio']*100:.1f}% "
        f"(要求≥{res['monthly_require']*100:.0f}%, 共{res['n_months']}个月)")
    if res.get("neg_months"):
        log(f"        -> IC为负的月份示例: {', '.join(res['neg_months'])}"
            + (f" 等共{res['n_neg_months']}个" if res['n_neg_months'] > len(res['neg_months']) else ""))
    log(f"    {mark(res['yearly_ok'])} 分年度多头超额收益: "
        + ("历史完整年份全部为正" if res["yearly_ok"] else "存在历史完整年份不为正"))
    for y, v in res["yearly_long"].items():
        tag = "(最新不完整年份, 豁免)" if y == res["latest_year"] else ""
        flag = "√" if (v > 0 or y == res["latest_year"]) else "×"
        log(f"        [{flag}] {y}年: {v*100:+.2f}% {tag}")
    log(f"    附加: ICIR={res['icir']:.3f}, 多头年化超额={res['long_annual']*100:.2f}%")
    verdict = "★ 合格因子 ★" if res["qualified"] else "不合格"
    log(f"    >>> 综合判定: {verdict}")
    log("")


def show_kv_table(title: str, kv: dict):
    banner(title, char="-")
    width = max((len(str(k)) for k in kv), default=10)
    for k, v in kv.items():
        log(f"    {str(k).ljust(width)} : {v}")
