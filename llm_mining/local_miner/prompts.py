"""
本地化因子挖掘项目 - 中文提示词

流程: 系统提示词(角色+函数库+约束+评价口径+输出格式)
      -> 首轮: 初始因子生成提示词(外部变量: 挖掘方向)
      -> 迭代: 反馈优化提示词(外部变量: 轮次/历史摘要/上轮评价反馈)
"""

# =============================================================================
# 固定部分
# =============================================================================

SYSTEM_ROLE = """你是一位资深的量化金融研究员, 专注于中国A股市场的日频量价因子挖掘。
你的任务是: 提出有经济逻辑的因子假设, 并用规定的公式语法构造出可计算的量价因子。
我会把你构造的因子在全部A股上进行严格的因子评价, 并把详细的评价结果反馈给你, 你需要基于反馈不断反思和迭代优化, 直到因子通过全部合格标准。"""

SYSTEM_VARIABLES = """可用的数据变量(全部A股, 日频, 后复权):
- $open: 当日开盘价
- $close: 当日收盘价
- $high: 当日最高价
- $low: 当日最低价
- $volume: 当日成交量
- $amount: 当日成交额
- $return: 当日收益率(收盘价日涨跌幅)"""

FUNCTION_LIB = """可用的函数库(严格区分大小写不敏感, 但建议大写):

【时序函数】对每只股票沿时间轴滚动计算, n 为正整数时间窗口:
- DELAY(x, n): n日前的值
- DELTA(x, n): x 减去 n日前的值
- TS_MEAN(x, n): n日滚动均值
- TS_SUM(x, n): n日滚动求和
- TS_STD(x, n): n日滚动标准差
- TS_MAX(x, n): n日滚动最大值
- TS_MIN(x, n): n日滚动最小值
- TS_MEDIAN(x, n): n日滚动中位数
- TS_RANK(x, n): 当前值在过去n日中的百分位排名(0~1)
- TS_ARGMAX(x, n): 距离过去n日内最大值已过去的天数
- TS_ARGMIN(x, n): 距离过去n日内最小值已过去的天数
- HIGHDAY(x, n): 同 TS_ARGMAX
- LOWDAY(x, n): 同 TS_ARGMIN
- TS_CORR(x, y, n): x与y的n日滚动相关系数
- TS_COV(x, y, n): x与y的n日滚动协方差
- TS_ZSCORE(x, n): (x - n日均值) / n日标准差
- TS_QUANTILE(x, n, q): 过去n日的q分位数, q为0~1小数
- EMA(x, n): 指数移动平均
- DECAYLINEAR(x, n): 线性衰减加权移动平均(近期权重大)
- COUNT(cond, n): 过去n日中条件cond成立的天数
- PROD(x, n): n日滚动连乘

【截面函数】对每个交易日的所有股票横截面计算:
- RANK(x): 横截面百分位排名(0~1)
- ZSCORE(x): 横截面标准化(减均值除标准差)
- SCALE(x): 横截面归一化(绝对值之和为1)
- MEAN(x): 横截面均值(广播到所有股票)

【元素级数学函数】
- ABS(x): 绝对值
- SIGN(x): 符号函数
- EXP(x): 指数
- SQRT(x): 平方根(负数按0处理)
- LOG(x): log(x+1)
- INV(x): 1/x
- POW(x, n): x的n次幂

【运算符】
- 四则运算: + - * /
- 比较运算: > < >= <= == !=
- 逻辑运算: && (与), || (或)
- 条件运算: cond ? a : b  等价于 WHERE(cond, a, b)

公式示例:
- RANK(DELTA($close, 5) / ($close + 1e-8))
- TS_CORR($close, $volume, 10)
- ($close - TS_MIN($low, 20)) / (TS_MAX($high, 20) - TS_MIN($low, 20) + 1e-8)
- TS_ZSCORE($amount, 20) * RANK($return)"""

OUTPUT_FORMAT_INITIAL = """输出格式要求: 只输出一个JSON对象, 不要输出任何解释性文字, 不要用代码块包裹。结构如下:
{
  "因子假设": "一句话描述因子背后的市场假设与经济逻辑",
  "因子列表": [
    {
      "名称": "因子的英文名称(如 ShortTermReversal_5D)",
      "描述": "因子的经济含义与构造思路",
      "公式": "基于上述函数库和变量的公式字符串"
    }
  ]
}"""

OUTPUT_FORMAT_ITERATION = """输出格式要求: 只输出一个JSON对象, 不要输出任何解释性文字, 不要用代码块包裹。结构如下:
{
  "上轮反思": "结合上一轮的评价反馈, 分析失败原因或成功经验的思考",
  "新假设": "本轮改进后的因子假设",
  "因子列表": [
    {
      "名称": "因子的英文名称",
      "描述": "因子的经济含义与构造思路",
      "公式": "公式字符串"
    }
  ]
}"""

# =============================================================================
# 构造函数
# =============================================================================

def build_system_prompt(cfg):
    """返回 (完整系统提示词, 固定部分dict, 外部变量dict)"""
    constraints = (
        f"1. 公式的括号嵌套深度(语法树深度)不得超过 {cfg.max_depth} 层, 这是硬性要求, 宁可简单也不要超深度;\n"
        f"2. 公式长度不超过 {cfg.max_symbol_length} 个字符, 使用的基础变量不超过 {cfg.max_base_features} 个;\n"
        f"3. 时间窗口 n 必须是不超过 {cfg.max_window} 的正整数;\n"
        "4. 只能使用上面列出的函数和变量, 禁止使用任何未声明的标识符;\n"
        "5. 每个因子必须独立, 公式中不得引用其他因子;\n"
        "6. 除法必须加小量防止除零, 例如除以 (x + 1e-8);\n"
        "7. 因子要有清晰的经济学假设, 避免无逻辑的随机数学堆砌, 避免过拟合。"
    )
    criteria = (
        f"1. 因子IC为 {cfg.ic_period}日 RankIC(因子值与未来{cfg.ic_period}个交易日收益的截面Spearman相关);\n"
        f"2. 按因子值分成 {cfg.n_quantiles} 组, 因子值最高的一组为多头组; 多头收益 = 多头组收益相对全市场收益均值的超额收益;\n"
        f"   (收益统计采用日度等效口径: {cfg.ic_period}日前向收益÷{cfg.ic_period}, 消除重叠收益的放大效应, 反馈中的累计/年化收益均为日度等效值);\n"
        "3. 合格标准(必须全部满足):\n"
        "   (a) 因子IC均值与多头累计收益必须同向且为正(若两者均为负, 系统会自动将因子取相反数翻转方向; 若一正一负则直接判定为差因子剔除);\n"
        f"   (b) IC按月度取均值后, 必须有超过 {cfg.monthly_ic_pos_ratio*100:.0f}% 的月份月均IC为正;\n"
        "   (c) 多头超额收益分年度看, 除最新的不完整年份外, 每一个历史完整年份都必须为正;\n"
        "4. 在满足合格标准的前提下, IC均值和多头收益越高越好。"
    )
    n_factors = f"每轮请输出 {cfg.factors_per_round} 个相互独立的因子。"

    full = (
        f"{SYSTEM_ROLE}\n\n【可用数据变量】\n{SYSTEM_VARIABLES}\n\n"
        f"【可用函数库】\n{FUNCTION_LIB}\n\n"
        f"【公式约束】\n{constraints}\n\n"
        f"【因子评价口径与合格标准】\n{criteria}\n\n"
        f"【数量要求】\n{n_factors}\n\n"
        f"【输出格式】\n{OUTPUT_FORMAT_INITIAL}"
    )
    fixed_parts = {
        "角色与任务": SYSTEM_ROLE,
        "数据变量说明": SYSTEM_VARIABLES,
        "函数库": FUNCTION_LIB,
        "输出格式": OUTPUT_FORMAT_INITIAL,
    }
    variables = {
        "公式约束": constraints,
        "评价口径与合格标准": criteria,
        "数量要求": n_factors,
    }
    return full, fixed_parts, variables


def build_initial_user_prompt(cfg):
    """首轮用户提示词。返回 (完整文本, 外部变量dict)"""
    text = (
        f"这是第 1 轮因子挖掘。请围绕以下挖掘方向, 提出创新的因子假设并构造 {cfg.factors_per_round} 个因子:\n"
        f"【挖掘方向】{cfg.direction}\n\n"
        f"请再次确认: 每个公式的括号嵌套深度不得超过 {cfg.max_depth} 层, 公式要简洁且有经济逻辑。"
        "现在请输出JSON。"
    )
    variables = {
        "挖掘方向": cfg.direction,
        "因子个数": cfg.factors_per_round,
        "最大嵌套深度": cfg.max_depth,
    }
    return text, variables


def build_iteration_user_prompt(cfg, round_no: int, history_summary: str, feedback: str):
    """迭代轮用户提示词。返回 (完整文本, 外部变量dict)"""
    text = (
        f"这是第 {round_no} 轮因子挖掘。以下是之前所有轮次的摘要:\n"
        f"【历史轮次摘要】\n{history_summary}\n\n"
        f"以下是上一轮因子在全部A股上的详细评价反馈:\n"
        f"【上一轮评价反馈】\n{feedback}\n\n"
        f"请先在\"上轮反思\"中认真分析评价结果(哪些指标达标、哪些没达标、原因可能是什么), "
        f"然后提出改进后的新假设, 并构造 {cfg.factors_per_round} 个新因子。"
        f"公式括号嵌套深度不得超过 {cfg.max_depth} 层。现在请输出JSON。"
    )
    variables = {
        "当前轮次": round_no,
        "历史轮次摘要": history_summary,
        "上一轮评价反馈": feedback,
        "因子个数": cfg.factors_per_round,
    }
    return text, variables


def summarize_history(history: list) -> str:
    """把历史轮次压缩成紧凑摘要"""
    if not history:
        return "(暂无历史)"
    lines = []
    for rec in history:
        r = rec.get("round", "?")
        hyp = (rec.get("hypothesis") or "")[:100]
        lines.append(f"第{r}轮 | 假设: {hyp}")
        for f in rec.get("factors", []):
            ev = f.get("eval") or {}
            if ev.get("error"):
                lines.append(f"    - {f.get('name')}: 计算失败({str(ev['error'])[:60]})")
                continue
            if not ev:
                lines.append(f"    - {f.get('name')}: 未评价")
                continue
            verdict = "合格" if ev.get("qualified") else "不合格"
            lines.append(
                f"    - {f.get('name')}: 公式 {f.get('expr')} | "
                f"IC均值={ev.get('ic_mean', 0)*100:+.3f}%, "
                f"多头累计超额={ev.get('long_total', 0)*100:+.2f}%, "
                f"月度IC为正占比={ev.get('monthly_pos_ratio', 0)*100:.0f}%, "
                f"判定={verdict}"
            )
    return "\n".join(lines)
