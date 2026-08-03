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
- -TS_CORR($close, $volume, 10)
- TS_MEAN(RANK($return), 5) * -1
- ($close - TS_MIN($low, 20)) / (TS_MAX($high, 20) - TS_MIN($low, 20) + 1e-8)
- TS_ZSCORE($amount, 20) * TS_MEAN($return, 5)"""

OUTPUT_FORMAT_INITIAL = """输出格式要求: 只输出一个JSON对象, 不要输出任何解释性文字, 不要用代码块包裹。结构如下:
{
  "因子假设": "一句话描述因子背后的市场假设与经济逻辑",
  "因子列表": [
    {
      "名称": "因子的英文名称(如 ShortTermReversal_5D)",
      "描述": "因子的经济含义与构造思路",
      "风格": "因子大类风格标签, 如: 短期反转/量价背离/动量/波动率/流动性/趋势/振幅 等, 可叠加周期(短期/长期)",
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
      "风格": "因子大类风格标签(短期反转/动量/波动率/流动性/量价背离等)",
      "公式": "公式字符串"
    }
  ]
}"""

OUTPUT_FORMAT_OPTIMIZE = """输出格式要求: 只输出一个JSON对象, 不要输出任何解释性文字, 不要用代码块包裹。结构如下:
{
  "优化思路": "针对当前因子的诊断问题, 说明本轮改进的着力点与预期效果",
  "新假设": "改进后的因子假设",
  "因子列表": [
    {
      "名称": "因子的英文名称",
      "描述": "改进了什么、保留了什么核心逻辑",
      "风格": "因子大类风格标签(应与原因子同族)",
      "公式": "公式字符串"
    }
  ]
}"""

# =============================================================================
# 构造函数
# =============================================================================

def build_system_prompt(cfg, output_format: str | None = None):
    """返回 (完整系统提示词, 固定部分dict, 外部变量dict)
    output_format: 指定输出格式块, 默认首轮格式; 迭代/优化模式传入对应格式。"""
    if output_format is None:
        output_format = OUTPUT_FORMAT_INITIAL
    constraints = (
        f"1. 公式的括号嵌套深度(语法树深度)不得超过 {cfg.max_depth} 层, 这是硬性要求, 宁可简单也不要超深度;\n"
        f"2. 公式长度不超过 {cfg.max_symbol_length} 个字符, 使用的基础变量不超过 {cfg.max_base_features} 个;\n"
        f"3. 时间窗口 n 必须是不超过 {cfg.max_window} 的正整数;\n"
        "4. 只能使用上面列出的函数和变量, 禁止使用任何未声明的标识符;\n"
        "5. 每个因子必须独立, 公式中不得引用其他因子;\n"
        "6. 除法必须加小量防止除零, 例如除以 (x + 1e-8);\n"
        "7. 因子要有清晰的经济学假设, 避免无逻辑的随机数学堆砌, 避免过拟合;\n"
        "8. 【严格禁止】公式最外层不得是截面RANK, 包括 RANK(...)、-RANK(...)、RANK(...)*常数 等一切仿射变体。"
        "原因: 因子后续要做行业/市值中性化和标准化, 最外层rank会把因子值压缩成均匀分布, 彻底破坏这些线性操作的有效性。"
        "RANK只允许作为公式的中间步骤(例如 TS_MEAN(RANK($return), 5)、TS_CORR(RANK($close), $volume, 10)), "
        "最终输出必须保留因子值的连续线性特征。系统会在代码端强制校验, 违反者直接拒绝。\n"
        "9. 【严格禁止·换皮造因子】仅更换价格字段($close/$open/$high/$low/$vwap 互换)、仅微调回看窗口(把n改成m)、"
        "或对已有核心项做平滑/中值/加权/取反包装, 均视为与原来同一个因子, 不得当作新因子提交。"
        "例如已出现过 TS_CORR($close,$volume,10), 则 TS_CORR($low,$volume,10)、TS_CORR($close,$volume,20)、"
        "TS_MEAN(TS_CORR($close,$volume,10),3) 等都算同一因子。要出新因子必须换经济逻辑或核心函数结构。\n"
        "10. 【组合=最后手段】将两个及以上经济逻辑相近/相同的项相加、平均或加权组合"
        "(如 TS_CORR(...,10)+TS_CORR(...,20) 取均值、量价相关与量额相关叠加、多路信号加权和), "
        "是【最后手段】: 仅当你已连续多轮尝试结构创新仍无法改进时才允许使用。"
        "一旦使用组合, 通常意味着该因子优化已接近尽头——系统会将这类因子隐藏评估、不向你展示结果。"
        "请优先产出基于单一经济逻辑的结构创新(状态门控/非线性变换/全新经济维度), "
        "而不是把已有项拼凑起来。"
    )
    criteria = (
        f"1. 因子IC为 {cfg.ic_period}日 RankIC(因子值与未来{cfg.ic_period}个交易日收益的截面Spearman相关);\n"
        f"2. 按因子值分成 {cfg.n_quantiles} 组, 因子值最高的一组为多头组; 多头收益 = 多头组收益相对全市场收益均值的超额收益;\n"
        f"   (收益统计采用日度等效口径: {cfg.ic_period}日前向收益÷{cfg.ic_period}, 消除重叠收益的放大效应, 反馈中的累计/年化收益均为日度等效值);\n"
        "3. 合格标准(必须全部满足):\n"
        "   (a) 因子IC均值与多头累计收益必须同向且为正(若两者均为负, 系统会自动将因子取相反数翻转方向; 若一正一负则直接判定为差因子剔除);\n"
        f"   (b) IC按月度取均值后, 必须有超过 {cfg.monthly_ic_pos_ratio*100:.0f}% 的月份月均IC为正;\n"
        "   (c) 多头超额收益分年度看, 除最新的不完整年份外, 每一个历史完整年份都必须为正;\n"
        "   (d) 分组单调性评级不得为C级(定义见第5条);\n"
        "4. 在满足合格标准的前提下, IC均值和多头收益越高越好;\n"
        "5. 分组单调性评级(衡量'因子值越高→当日未来收益越高'的排序质量):\n"
        f"   三项指标——①每日秩相关: 每个交易日把'分组编号1..{cfg.n_quantiles}'与'当日各组收益'做Spearman秩相关, "
        "取其日度序列的总均值(1=完全单调递增, 0=无关);\n"
        "   ②越序惩罚日均: Σmax(0, R_i-R_(i+1)) 的日度均值, 度量更低组收益高于更高组的程度(0=完全无越序);\n"
        "   ③年度聚合越序峰值: 每年对各组年内累计收益算一次越序惩罚, 取各年最大值;\n"
        f"   四级标准(取三项最低档判定)——S优秀: 秩相关≥{cfg.mono_rank_s:.2f} 且 越序日均<{cfg.mono_oos_s*100:.2f}% 且 峰值<{cfg.mono_peak_s*100:.0f}%;\n"
        f"   A良好: 秩相关≥{cfg.mono_rank_a:.2f} 且 越序日均<{cfg.mono_oos_a*100:.2f}% 且 峰值<{cfg.mono_peak_a*100:.0f}%;\n"
        f"   B可入库但需优化: 秩相关≥{cfg.mono_rank_b:.2f} 且 越序日均<{cfg.mono_oos_b*100:.2f}% 且 峰值<{cfg.mono_peak_b*100:.0f}%;\n"
        "   C丢弃: 任何一项达不到B档, 或存在秩相关为负的年份;\n"
        "   处置: C级直接判为不合格(不入库不采纳, 必须更换核心逻辑重新设计); B级可入库但须针对性优化; "
        "A级可接受; S级应保持结构。评价反馈中会给出每项指标的分年度明细, "
        "请据此找出秩相关偏低或越序偏高的年份, 诊断其市场风格共性并针对性修正。"
    )
    n_factors = f"每轮请输出 {cfg.factors_per_round} 个相互独立的因子。"

    full = (
        f"{SYSTEM_ROLE}\n\n【可用数据变量】\n{SYSTEM_VARIABLES}\n\n"
        f"【可用函数库】\n{FUNCTION_LIB}\n\n"
        f"【公式约束】\n{constraints}\n\n"
        f"【因子评价口径与合格标准】\n{criteria}\n\n"
        f"【数量要求】\n{n_factors}\n\n"
        f"【输出格式】\n{output_format}"
    )
    fixed_parts = {
        "角色与任务": SYSTEM_ROLE,
        "数据变量说明": SYSTEM_VARIABLES,
        "函数库": FUNCTION_LIB,
        "输出格式": output_format,
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


def build_new_initial_user_prompt(cfg, library_text: str):
    """新因子挖掘模式·首轮用户提示词。注入已有因子库摘要以防撞车。
    返回 (完整文本, 外部变量dict)"""
    text = (
        f"这是【新因子挖掘】第 1 轮。请围绕以下挖掘方向, 提出创新的因子假设并构造 "
        f"{cfg.factors_per_round} 个因子:\n"
        f"【挖掘方向】{cfg.direction}\n\n"
        f"【已有因子库(必须避让)】\n{library_text}\n\n"
        f"【防撞车硬性要求】新因子必须与库内因子在经济逻辑上本质不同, 判定从严:\n"
        f"(1) 禁止仅靠'更换价格字段'造新因子: 如库内已有 TS_CORR($close,$volume,n), "
        f"则 -TS_CORR($low,$volume,m)、-TS_CORR($open,$volume,m)、-TS_CORR($high,$volume,m)、"
        f"-TS_CORR($vwap,$volume,m) 等一律视为同一因子(价格-成交量相关性), 不得提交;\n"
        f"(2) 禁止仅靠'微调回看窗口'造新因子: 把 n 改成 m 的同构公式视为同一因子;\n"
        f"(3) 禁止对库内因子核心项做平滑/中值/加权包装后当作新因子;\n"
        f"(4) 新因子应来自不同的经济逻辑大类(如动量/反转/波动/流动性/资金流/微观结构等), "
        f"并使用不同的核心函数与字段组合。系统会以截面相关性 |ρ|≤{cfg.max_library_corr} 做强校验, "
        f"同类因子必然超限被拒, 请勿浪费轮次。\n\n"
        f"请再次确认: 每个公式的括号嵌套深度不得超过 {cfg.max_depth} 层, 公式最外层禁止为截面RANK, "
        f"公式要简洁且有经济逻辑。现在请输出JSON。"
    )
    variables = {
        "挖掘方向": cfg.direction,
        "已有因子库摘要": library_text,
        "因子个数": cfg.factors_per_round,
        "最大嵌套深度": cfg.max_depth,
    }
    return text, variables


def build_optimize_user_prompt(cfg, round_no: int, series_text: str,
                               advice_text: str, feedback: str,
                               ensemble_warning: str = ""):
    """现有因子优化模式·用户提示词。注入当前因子全况+条件诊断建议+参数堆叠告诫(若触发)+上轮反馈。
    返回 (完整文本, 外部变量dict)"""
    warn_block = f"\n\n{ensemble_warning}" if ensemble_warning else ""
    text = (
        f"这是【现有因子优化】第 {round_no} 轮。目标: 在保留该因子核心经济逻辑的前提下, "
        f"针对诊断暴露的问题做有针对性的改进。\n\n"
        f"【核心评判标准(务必牢记)】判断改进是否更优的静态标准是\"多头收益的年度稳定性\", "
        f"而不是单纯抬高IC。原因: IC很多时候是由空头端贡献的, 而我们只要多头收益。"
        f"我们追求: (1)每个历史完整年份都有为正的多头超额收益; "
        f"(2)多头收益在年份间分布平均, 不集中于某一两年。"
        f"该稳定性用\"年度多头收益信息比率=各年多头收益均值/各年标准差\"衡量, 越高越好"
        f"(等价于变异系数越低、收益越不集中)。请优先朝这个方向改进, 其次再兼顾诊断问题。\n\n"
        f"【当前因子全况(评价+稳定性+诊断+整条迭代路径)】\n{series_text}\n\n"
        f"【诊断结论与优化方向(条件判断)】\n{advice_text}\n\n"
        f"{warn_block}"
        f"【上一轮评价反馈】\n{feedback}\n\n"
        f"请先在\"优化思路\"中说明本轮如何提升多头收益的年度稳定性(让每年都为正且更平均)、"
        f"以及针对哪个诊断问题, 然后构造 {cfg.factors_per_round} 个改进后的同族因子。"
        f"公式括号嵌套深度不得超过 {cfg.max_depth} 层, 最外层禁止为截面RANK。现在请输出JSON。"
    )
    variables = {
        "当前轮次": round_no,
        "当前因子全况": series_text,
        "诊断结论与优化方向": advice_text,
        "上一轮评价反馈": feedback,
        "因子个数": cfg.factors_per_round,
    }
    if ensemble_warning:
        variables["参数堆叠告诫"] = ensemble_warning
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
