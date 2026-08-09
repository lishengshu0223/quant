"""
本地化因子挖掘项目 - 中文提示词

流程: 系统提示词(角色+函数库+约束+评价口径+输出格式)
      -> 首轮: 初始因子生成提示词(外部变量: 挖掘方向)
      -> 迭代: 反馈优化提示词(外部变量: 轮次/历史摘要/上轮评价反馈)
"""

import json
import os
from collections import Counter

# =============================================================================
# 固定部分
# =============================================================================

SYSTEM_ROLE = """你是一位资深的量化金融研究员, 专注于中国A股市场的日频量价因子挖掘。
你的任务是: 提出有经济逻辑的因子假设, 并用规定的公式语法构造出可计算的量价因子。
我会把你构造的因子在全部A股上进行严格的因子评价, 并把详细的评价结果反馈给你, 你需要基于反馈不断反思和迭代优化, 直到因子通过全部合格标准。"""

# ---------------- 分钟模式(数据频率=minute 时启用) ----------------

MINUTE_SYSTEM_ROLE = """你是一位资深的量化金融研究员, 专注于中国A股市场的日内分钟级量价行为因子挖掘。
你的任务: 提出有经济逻辑的因子假设, 并用规定的公式语法构造出可计算的量价因子。
分钟频率的特征是 股票|日期|分钟 三维数据, 必须通过"聚合算子"降维成 股票|日期 的日频特征,
再与日频算子组合, 最终生成【日频因子】。系统会在全部A股上对该日频因子做严格评价并反馈给你,
你需要基于反馈不断反思和迭代优化, 直到因子通过全部合格标准。"""

SYSTEM_VARIABLES = """可用的数据变量(全部A股, 日频, 后复权):
- $open: 当日开盘价
- $close: 当日收盘价
- $high: 当日最高价
- $low: 当日最低价
- $volume: 当日成交量
- $amount: 当日成交额
- $return: 当日收益率(收盘价日涨跌幅)"""

MINUTE_SYSTEM_VARIABLES = """可用的数据变量:
【日频变量】(全部A股, 后复权, 仅在日频算子的参数中使用):
- $open/$close/$high/$low: 当日开/收/高/低价
- $volume: 当日成交量; $amount: 当日成交额
- $return: 当日收益率(收盘价日涨跌幅)

【分钟变量】(1分钟频率K线, 后复权, 仅在分钟算子的参数中使用):
- $open/$close/$high/$low: 该分钟的开/收/高/低价
- $volume: 该分钟成交量; $amount: 该分钟成交额
- $return: 该分钟收益率(每分钟收盘价环比, 当日第一根为NaN)
- $minute: 当日时间戳(自0点起的秒数, 如09:31→34260, 15:00→54000), 常与 REGRESSION_SLOPE/INTERCEPT 配合度量"价格随时间的变化"

【分钟数据组织】每只股票每个交易日约240根1分钟K线
(上午09:31~11:30, 下午13:01~15:00, 均含端点)。分钟字段除 code 与 datetime 索引外,
还带有 date(交易日) 索引, 因此可对"股票+交易日"做按天聚合。"""

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

MINUTE_FUNCTION_LIB_DAILY = """可用的函数库(严格区分大小写不敏感, 但建议大写):

【日频算子】(与日频挖掘完全相同, 全部可用; 作用于"股票|日期"二维数据, 即聚合后的分钟特征或原始日频字段):
- 时序: DELAY(x,n) / DELTA(x,n) / TS_MEAN(x,n) / TS_SUM(x,n) / TS_STD(x,n) / TS_MAX(x,n) / TS_MIN(x,n)
  / TS_MEDIAN(x,n) / TS_RANK(x,n) / TS_ARGMAX(x,n) / TS_ARGMIN(x,n) / HIGHDAY(x,n) / LOWDAY(x,n)
  / TS_CORR(x,y,n) / TS_COV(x,y,n) / TS_ZSCORE(x,n) / TS_QUANTILE(x,n,q) / EMA(x,n) / DECAYLINEAR(x,n)
  / COUNT(cond,n) / PROD(x,n)
- 截面: RANK(x) / ZSCORE(x) / SCALE(x) / MEAN(x)
- 元素: ABS / SIGN / EXP / SQRT / LOG / INV / POW(x,n)
- 运算符: + - * / ; 比较 > < >= <= == != ; 逻辑 && || ; 条件 cond ? a : b

【日频算子示例】TS_MEAN(MEAN($close), 5) 表示"每日分钟均价的5日均值"。"""

MINUTE_FUNCTION_LIB_AGG = """【分钟聚合算子(降维)】把 股票|日期|分钟(3维) 压成 股票|日期(2维), 即按股票、按天 groupby 聚合:
- SUM(x)/MEAN(x)/STD(x)/MAX(x)/MIN(x)/MEDIAN(x): 当日分钟序列的和/均值/标准差/最大/最小/中位数
- SKEW(x)/KURT(x): 当日分钟序列的偏度/峰度(须≥3根有效分钟, 否则为NaN)
- LAST(x)/FIRST(x): 当日最后/第一根有效分钟的值
- COUNT(x): 当日有效分钟数
- QUANTILE(x, q): 当日分钟序列的q分位数, q为0~1小数(如0.8=当日80%分位)
- CORR(x, y): 当日两分钟序列的Pearson相关系数(如量价相关性 CORR($return, $volume))
- TS_AUTOCORR(x, lag): 当日分钟序列的lag阶自相关系数(lag为正整数, 如1=相邻分钟)
- TS_ARGMAX(x)/TS_ARGMIN(x): 当日最大值/最小值出现的分钟位置, 归一化到[0,1](0=当天第一根, 1=当天最后一根)
- REGRESSION_SLOPE(x, y)/REGRESSION_INTERCEPT(x, y): 当日用分钟序列对y做线性回归 x~y 的斜率/截距
  (y常用 $minute, 度量日内价格随时间的趋势; 也可用另一个分钟特征)
聚合后的结果即为日频值, 可被日频算子继续调用, 例如 TS_MEAN(MEAN($close), 5)。"""

MINUTE_FUNCTION_LIB_KEEP = """【分钟维持维度算子】仍然是按股票按天处理, 但计算前后都是三维(分钟序列):
- SLICE(x, "HH:MM", "HH:MM"): 只保留该时段的分钟, 如 SLICE($volume,"09:31","10:00") 取开盘30分钟,
  SLICE($close,"14:00","15:00") 取尾盘60分钟(闭区间含端点)
- MASK(x, 比较符, 阈值): 只保留满足"该分钟的值 比较符 阈值"的分钟; 阈值为当日标量(由聚合得到),
  如 MASK($close, ">", QUANTILE($close, 0.8)) 只保留价格最高的20%分钟;
  MASK($volume, ">", MEAN($volume)) 只保留成交量高于当日均值的分钟;
  阈值也支持分钟表达式(逐分钟比较), 如 MASK($close, ">", $open) 只保留收高于开的阳线分钟

【分钟截面算子(3D->3D)】对"当日该分钟的全部股票"做截面变换, 不减少每天分钟数(仍为240根):
- RANK(x): 该分钟 x 值在全市场股票中的百分位排名(0~1)
- ZSCORE(x): 该分钟 x 值在全市场股票中的标准化(减均值除标准差)
- SCALE(x): 该分钟 x 值除以全市场股票绝对值之和
- CS_MEAN(x) / CS_STD(x): 该分钟 x 值在全市场股票中的均值/标准差(广播)
  例: MEAN(SLICE(RANK($return), "09:31", "10:00")) = 开盘30分钟每根K线收益在全市场中的相对排名的日均值。
  (系统检测到这些算子时会自动按"日期分块、每块加载全市场"计算, 保证截面是全市场的)

【日内滚动算子(3D->3D)】按股票、按天在分钟序列上滚动 n 分钟窗口, 不减少每天分钟数(仍为240根):
- INTRADAY_MEAN(x, n) / INTRADAY_STD(x, n) / INTRADAY_SUM(x, n)
- INTRADAY_MAX(x, n) / INTRADAY_MIN(x, n) / INTRADAY_MEDIAN(x, n)
  n 为正整数窗口(1~240); 每天前 n-1 根为 NaN(窗口不足)。
  例: MEAN(SLICE(INTRADAY_STD($return, 5), "09:31", "10:00")) = 开盘30分钟内5分钟滚动收益波动率的日均值。

【运算顺序(必须遵守)】先分时(slice) -> 再切割(mask) -> 最后降维(聚合):
  若先切割, 分时出的时段很可能全是被切割掉的NaN; 若先降维, 每天只剩一个数无法再做掩码。
  例: MEAN(MASK(SLICE($volume,"09:31","10:00"), ">", MEAN($volume)))
      = 开盘30分钟内、成交量高于全日均值的分钟的平均成交量。
  掩码阈值的判断基准可自由选择: 用全天的数据算(如 MEAN($volume)), 也可用切割后的数据算
  (如 MASK($close, ">", QUANTILE(SLICE($close,"09:31","10:00"), 0.8)))。

【分钟模式命名约定】
- 在分钟模式下, 单参数的 mean/max/min/ts_argmax/ts_argmin 均指【分钟聚合】(对当日分钟序列聚合);
  若要对日频数据做截面变换, 请使用 RANK/ZSCORE/SCALE, 不要使用单参数 MEAN/MAX/MIN。
- RANK/ZSCORE/SCALE 的参数为分钟序列时表示"分钟截面"(对当日该分钟的全部股票), 参数为日频值时表示日频截面。
- 日频的 MEAN/MAX/MIN 仅在参数已经是日频值(宽表)时按日频函数解析(如 MEAN(TS_MEAN($close,5)))。

【分钟公式示例】
- MEAN(SLICE($return, "09:31", "10:00")): 开盘30分钟平均收益率
- STD(SLICE($return, "13:01", "15:00")): 尾盘分钟收益率波动率
- TS_ARGMAX(SLICE($volume, "09:31", "10:30")): 早盘成交量峰值出现的分钟位置
- REGRESSION_SLOPE($close, $minute): 当日收盘价随时间的回归斜率(日内趋势强度)
- MEAN(MASK($volume, ">", MEAN($volume))): 高于日均量分钟的平均成交量(放量分钟强度)
- MEAN(SLICE(RANK($return), "09:31", "10:00")): 开盘30分钟每根K线收益的全市场相对排名(分钟截面)
- MEAN(INTRADAY_STD($return, 5)): 当日5分钟滚动收益波动率的日均值(日内微观波动结构)"""

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
    output_format: 指定输出格式块, 默认首轮格式; 迭代/优化模式传入对应格式。
    cfg.data_frequency == "minute" 时使用分钟模式的角色/变量/函数库与约束。"""
    if output_format is None:
        output_format = OUTPUT_FORMAT_INITIAL
    is_minute = getattr(cfg, "data_frequency", "daily") == "minute"
    if is_minute:
        return _build_minute_system_prompt(cfg, output_format)

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


def _build_minute_system_prompt(cfg, output_format):
    """分钟模式系统提示词: 角色/变量/函数库/约束全部切换到分钟频率"""
    constraints = (
        f"1. 公式的括号嵌套深度(语法树深度)不得超过 {cfg.formula_max_depth} 层(分钟模式允许比日频更高的复杂度, "
        "但不得导致堆栈溢出), 这是硬性要求;\n"
        f"2. 公式长度不超过 {cfg.formula_max_symbol_length} 个字符, 使用的基础变量不超过 {cfg.minute_max_base_features} 个;\n"
        f"3. 日频算子的时间窗口 n 必须是不超过 {cfg.max_window} 的正整数; 分钟算子的滞后/分位数按各自要求;\n"
        "4. 只能使用上面列出的函数和变量, 禁止使用任何未声明的标识符;\n"
        "5. 每个因子必须独立, 公式中不得引用其他因子;\n"
        "6. 除法必须加小量防止除零, 例如除以 (x + 1e-8);\n"
        "7. 因子要有清晰的经济学假设, 避免无逻辑的随机数学堆砌, 避免过拟合;\n"
        "8. 分钟维度只允许在分钟算子(SLICE/MASK/聚合)的参数中使用; 公式最外层必须是日频值(宽表);\n"
        "9. 【运算顺序】分钟公式必须遵守 先分时(SLICE) -> 再切割(MASK) -> 最后降维(聚合) 的顺序;\n"
        "10. 【严格禁止】公式最外层不得是截面RANK, 包括 RANK(...)、-RANK(...)、RANK(...)*常数 等一切仿射变体。"
        "RANK只允许作为日频层的中间步骤(如 TS_MEAN(RANK(MEAN($close)), 5)), 最终输出必须保留连续线性特征;\n"
        "11. 【严格禁止·换皮造因子】仅更换价格字段、仅微调回看窗口或分钟时段(把09:31-10:00改成09:31-10:30)、"
        "或对已有核心项做平滑/取反包装, 均视为与原来同一个因子, 不得当作新因子提交。"
        "要出新因子必须换经济逻辑或核心函数结构;\n"
        "12. 【组合=最后手段】将两个及以上经济逻辑相近/相同的项相加、平均或加权组合是最后手段, "
        "系统会将这类因子隐藏评估。请优先产出基于单一经济逻辑的结构创新。"
    )
    criteria = (
        f"1. 因子IC为 {cfg.ic_period}日 RankIC(因子值与未来{cfg.ic_period}个交易日收益的截面Spearman相关);\n"
        f"2. 按因子值分成 {cfg.n_quantiles} 组, 因子值最高的一组为多头组; 多头收益 = 多头组收益相对全市场收益均值的超额收益;\n"
        f"   (收益统计采用日度等效口径: {cfg.ic_period}日前向收益÷{cfg.ic_period});\n"
        "3. 合格标准(必须全部满足):\n"
        "   (a) 因子IC均值与多头累计收益必须同向且为正(双负自动翻转; 一正一负直接剔除);\n"
        f"   (b) IC按月度取均值后, 必须有超过 {cfg.monthly_ic_pos_ratio*100:.0f}% 的月份月均IC为正;\n"
        "   (c) 多头超额收益分年度看, 除最新不完整年份外, 每个历史完整年份都必须为正;\n"
        "   (d) 分组单调性评级不得为C级;\n"
        "4. 在满足合格标准的前提下, IC均值和多头收益越高越好;\n"
        f"5. 分组单调性评级: 每日秩相关均值(分组编号1..{cfg.n_quantiles} vs 当日各组收益的Spearman秩相关, 1=完全单调递增)、"
        "越序惩罚日均(Σmax(0,R_i-R_(i+1)))、年度聚合越序峰值, 三项取最低档判定 S/A/B/C; "
        "C级一律拒绝。"
    )
    n_factors = f"每轮请输出 {cfg.factors_per_round} 个相互独立的因子。"

    func_lib = f"{MINUTE_FUNCTION_LIB_DAILY}\n\n{MINUTE_FUNCTION_LIB_AGG}\n\n{MINUTE_FUNCTION_LIB_KEEP}"
    full = (
        f"{MINUTE_SYSTEM_ROLE}\n\n【可用数据变量】\n{MINUTE_SYSTEM_VARIABLES}\n\n"
        f"【可用函数库】\n{func_lib}\n\n"
        f"【公式约束】\n{constraints}\n\n"
        f"【因子评价口径与合格标准】\n{criteria}\n\n"
        f"【数量要求】\n{n_factors}\n\n"
        f"【输出格式】\n{output_format}"
    )
    fixed_parts = {
        "角色与任务": MINUTE_SYSTEM_ROLE,
        "数据变量说明": MINUTE_SYSTEM_VARIABLES,
        "函数库": func_lib,
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
        f"请再次确认: 每个公式的括号嵌套深度不得超过 {cfg.formula_max_depth} 层, 公式要简洁且有经济逻辑。"
        "现在请输出JSON。"
    )
    variables = {
        "挖掘方向": cfg.direction,
        "因子个数": cfg.factors_per_round,
        "最大嵌套深度": cfg.formula_max_depth,
    }
    return text, variables


def build_iteration_user_prompt(cfg, round_no: int, history_summary: str, feedback: str,
                                blacklist_text: str = ""):
    """迭代轮用户提示词。返回 (完整文本, 外部变量dict)
    blacklist_text: 已否决公式黑名单(长期记忆, 防反复提交同结构/换皮因子)"""
    black_block = f"\n{blacklist_text}\n" if blacklist_text else ""
    text = (
        f"这是第 {round_no} 轮因子挖掘。以下是之前所有轮次的摘要:\n"
        f"【历史轮次摘要】\n{history_summary}\n\n"
        f"以下是上一轮因子在全部A股上的详细评价反馈:\n"
        f"【上一轮评价反馈】\n{feedback}\n\n"
        f"{black_block}"
        f"请先在\"上轮反思\"中认真分析评价结果(哪些指标达标、哪些没达标、原因可能是什么), "
        f"然后提出改进后的新假设, 并构造 {cfg.factors_per_round} 个新因子。"
        f"公式括号嵌套深度不得超过 {cfg.formula_max_depth} 层。现在请输出JSON。"
    )
    variables = {
        "当前轮次": round_no,
        "历史轮次摘要": history_summary,
        "上一轮评价反馈": feedback,
        "因子个数": cfg.factors_per_round,
    }
    if blacklist_text:
        variables["已否决公式黑名单"] = blacklist_text
    return text, variables


def build_new_initial_user_prompt(cfg, library_text: str, campaign_summary_text: str = "",
                                  failed_text: str = ""):
    """新因子挖掘模式·首轮用户提示词。注入已有因子库摘要以防撞车。
    campaign_summary_text: 历史战役信息压缩摘要(可选, 概括以往战役的成果与失败教训)。
    failed_text: 历史挖掘失败经验(可选, 概括反复失败的公式与方向, 防止重复挖掘)。
    返回 (完整文本, 外部变量dict)"""
    camp_block = f"\n【历史战役信息压缩摘要(务必借鉴: 已挖过的方向与反复失败的教训)】\n{campaign_summary_text}\n" \
        if campaign_summary_text else ""
    fail_block = f"\n【历史挖掘失败经验(务必借鉴: 以下公式/方向已被多次尝试且未通过合格标准, 请勿再次提交同型因子)】\n{failed_text}\n" \
        if failed_text else ""
    text = (
        f"这是【新因子挖掘】第 1 轮。请围绕以下挖掘方向, 提出创新的因子假设并构造 "
        f"{cfg.factors_per_round} 个因子:\n"
        f"【挖掘方向】{cfg.direction}\n\n"
        f"【已有因子库(必须避让)】\n{library_text}\n\n"
        f"{camp_block}"
        f"{fail_block}"
        f"【防撞车硬性要求】新因子必须与库内因子在经济逻辑上本质不同, 判定从严:\n"
        f"(1) 禁止仅靠'更换价格字段'造新因子: 如库内已有 TS_CORR($close,$volume,n), "
        f"则 -TS_CORR($low,$volume,m)、-TS_CORR($open,$volume,m)、-TS_CORR($high,$volume,m)、"
        f"-TS_CORR($vwap,$volume,m) 等一律视为同一因子(价格-成交量相关性), 不得提交;\n"
        f"(2) 禁止仅靠'微调回看窗口'造新因子: 把 n 改成 m 的同构公式视为同一因子;\n"
        f"(3) 禁止对库内因子核心项做平滑/中值/加权包装后当作新因子;\n"
        f"(4) 新因子应来自不同的经济逻辑大类(如动量/反转/波动/流动性/资金流/微观结构等), "
        f"并使用不同的核心函数与字段组合。系统会以截面相关性 |ρ|≤{cfg.max_library_corr} 做强校验, "
        f"同类因子必然超限被拒, 请勿浪费轮次。\n\n"
        f"请再次确认: 每个公式的括号嵌套深度不得超过 {cfg.formula_max_depth} 层, 公式最外层禁止为截面RANK, "
        f"公式要简洁且有经济逻辑。现在请输出JSON。"
    )
    variables = {
        "挖掘方向": cfg.direction,
        "已有因子库摘要": library_text,
        "因子个数": cfg.factors_per_round,
        "最大嵌套深度": cfg.formula_max_depth,
    }
    if campaign_summary_text:
        variables["历史战役信息压缩摘要"] = campaign_summary_text
    if failed_text:
        variables["历史挖掘失败经验"] = failed_text
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
        f"公式括号嵌套深度不得超过 {cfg.formula_max_depth} 层, 最外层禁止为截面RANK。现在请输出JSON。"
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


# =============================================================================
# 长期黑名单记忆(已否决公式) —— 防模型反复提交同结构/换皮因子
# 每轮把被拒因子沉淀为一行式记忆, 迭代轮 prompt 注入; 超过上限时丢弃最旧条目
# =============================================================================

def extract_rejection_memory(round_factors: list, round_no: int) -> list:
    """从一轮已评价因子的结果中提取'被拒原因'一行式记忆。返回 list[str]"""
    mem = []
    for f in round_factors:
        ev = f.get("eval") or {}
        name = f.get("name", "?")
        if not ev:
            continue
        if ev.get("error"):
            mem.append(f"R{round_no} {name}: 计算/评价失败 -> {str(ev['error'])[:90]}")
            continue
        if ev.get("review_rejected"):
            mem.append(f"R{round_no} {name}: 语义评审拒绝(与库内因子等价包装) -> "
                       f"{str(ev.get('review_reason') or '')[:120]}")
            continue
        if ev.get("library_rejected"):
            mem.append(f"R{round_no} {name}: 相关性撞车 max|ρ|="
                       f"{float(ev.get('library_max_corr') or 0):.2f} 超上限 -> 换字段/换窗口的包装无效")
            continue
        mg = ev.get("monotonicity_grade") or {}
        if mg.get("grade") == "C":
            drc = mg.get("drc")
            drc_txt = f"{drc:+.3f}" if drc is not None else "NA"
            mem.append(f"R{round_no} {name}: 单调性C级被拒(秩相关{drc_txt}, 坏年{mg.get('bad_years')})"
                       f" -> 该核心结构不可用, 勿再围绕其微调")
    return mem


def append_rejection_memory(mem: dict, entries: list) -> dict:
    """去重追加黑名单记忆。mem: {"items":[str], "max_items":int, "dropped":int}
    同名因子只保留最新一条(覆盖旧条目); 超上限丢弃最旧条目并累计 dropped。"""
    if mem is None:
        mem = {}
    items = mem.setdefault("items", [])
    max_items = int(mem.get("max_items") or 60)
    for e in entries:
        # 键 = "Rxx 因子名", 以因子名为去重依据
        key = e.split(":", 1)[0].strip()
        parts = key.split(" ", 1)
        name = parts[-1] if len(parts) > 1 else key
        idx = next((i for i, it in enumerate(items) if f" {name}:" in it), None)
        if idx is not None:
            items[idx] = e          # 同名覆盖为最新教训
        else:
            items.append(e)
    if len(items) > max_items:
        mem["dropped"] = int(mem.get("dropped") or 0) + (len(items) - max_items)
        items = items[-max_items:]   # 保留最近 max_items 条
        mem["items"] = items
    return mem


def format_rejection_memory(mem: dict) -> str:
    """把黑名单转成注入模型的紧凑文本(一条一行)"""
    items = (mem or {}).get("items") or []
    if not items:
        return ""
    dropped = int((mem or {}).get("dropped") or 0)
    head = (f"【已否决公式黑名单(共{len(items)}条, 累计精简丢弃{dropped}条)】"
            "以下公式结构均已被评价失败或判定与库内重复, 严禁再次提交同结构/同逻辑因子, 必须换经济逻辑:")
    return head + "\n" + "\n".join(f"- {it}" for it in items)


# =============================================================================
# 战役信息压缩 —— 每100轮战役结束且有因子入库后, 将整体历史压缩为概括性摘要,
# 供下一个战役首轮注入(避免长期逐轮记忆导致上下文过长、稀释模型注意力)
# =============================================================================

CAMPAIGN_SUMMARY_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "workspace", "campaign_summaries.json")


def summarize_campaign(state: dict, cfg, library: list) -> str:
    """把已完成战役(history 全部轮次 + 黑名单)压缩为概括性摘要文本。
    library: factor_library.load_library() 结果, 用于列出本次战役入库成果。"""
    history = state.get("history") or []
    adopted = state.get("campaign_adopted") or []
    rounds = len(history)
    # 1) 入库成果
    adopted_lines = []
    for sid in adopted:
        s = next((x for x in library if x.get("series_id") == sid), None)
        if s:
            best = s.get("best") or {}
            adopted_lines.append(
                f"  - {sid} {s.get('name')}: {str(best.get('expr') or '')[:80]}")
    adopted_txt = "\n".join(adopted_lines) if adopted_lines else "  (本战役无入库因子)"
    # 2) 失败模式统计(从 history 全量 eval 统计)
    reason_cnt = Counter()          # 失败原因分类计数
    struct_cnt = Counter()          # 被拒因子核心结构(公式)高频重复
    for rec in history:
        for f in rec.get("factors", []):
            ev = f.get("eval") or {}
            if not ev:
                continue
            if ev.get("error"):
                reason_cnt["计算失败"] += 1
            elif ev.get("review_rejected"):
                reason_cnt["语义评审拒绝"] += 1
            elif ev.get("library_rejected"):
                reason_cnt["相关性撞车"] += 1
            else:
                mg = ev.get("monotonicity_grade") or {}
                if mg.get("grade") == "C":
                    reason_cnt["单调性C级"] += 1
                elif not ev.get("qualified"):
                    reason_cnt["其他未达标"] += 1
            # 公式核心结构: 去掉参数与窗口, 近似去重
            expr = str(f.get("expr") or "")
            core = expr.split(",")[0].replace("(", "").replace(")", "").strip()
            struct_cnt[core] += 1
    top_struct = "、".join(f"{k}({v}次)" for k, v in struct_cnt.most_common(8))
    # 3) 有效方向: IC>0 且多头为正 但未达合格的探索
    ok_dir = []
    for rec in history:
        for f in rec.get("factors", []):
            ev = f.get("eval") or {}
            if ev and not ev.get("qualified") and not ev.get("error") \
                    and (ev.get("ic_mean") or 0) > 0.02 and (ev.get("long_total") or 0) > 0.15:
                ok_dir.append(f"{f.get('name')}(IC{(ev['ic_mean']*100):+.1f}% 多头{(ev['long_total']*100):+.0f}%)")
    ok_dir_txt = "、".join(ok_dir[:12]) or "无"
    summary = (
        f"【战役信息压缩摘要·频率={cfg.data_frequency}】本轮战役共 {rounds} 轮, "
        f"入库 {len(adopted)} 个新因子系列。\n"
        f"1. 入库成果:\n{adopted_txt}\n"
        f"2. 失败模式分布(累计): {', '.join(f'{k}{v}次' for k, v in reason_cnt.items()) or '无'}。\n"
        f"   其中高频被拒结构(勿再探索): {top_struct}。\n"
        f"3. 已探索但未达合格的有效方向(IC与多头为正, 卡在单调性/年度): {ok_dir_txt}。\n"
        f"4. 经验教训: 换字段/换窗口的包装会被语义评审与相关性检验拦截, 新因子必须来自不同的经济逻辑; "
        f"最新不完整年份不参与单调性负年判定, 历史完整年份秩相关为负仍会直接判C级。"
    )
    return summary


def save_campaign_summary(summary_text: str):
    """把战役压缩摘要追加保存(供下个战役首轮注入)"""
    try:
        recs = []
        if os.path.exists(CAMPAIGN_SUMMARY_FILE):
            with open(CAMPAIGN_SUMMARY_FILE, "r", encoding="utf-8") as f:
                recs = json.load(f)
        recs.append({"ts": __import__("time").strftime("%Y-%m-%d %H:%M"), "summary": summary_text})
        # 只保留最近 3 次战役摘要, 防止无限增长
        recs = recs[-3:]
        os.makedirs(os.path.dirname(CAMPAIGN_SUMMARY_FILE), exist_ok=True)
        with open(CAMPAIGN_SUMMARY_FILE, "w", encoding="utf-8") as f:
            json.dump(recs, f, ensure_ascii=False, indent=1)
        return True
    except Exception:
        return False


def load_campaign_summaries() -> str:
    """读取历史战役压缩摘要(最近3次), 拼接为注入文本"""
    if not os.path.exists(CAMPAIGN_SUMMARY_FILE):
        return ""
    try:
        with open(CAMPAIGN_SUMMARY_FILE, "r", encoding="utf-8") as f:
            recs = json.load(f)
        blocks = [f"── 战役@{r.get('ts')} ──\n{r.get('summary')}" for r in recs if r.get("summary")]
        return "\n\n".join(blocks)
    except Exception:
        return ""
