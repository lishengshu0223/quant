"""
本地化因子挖掘项目 - 全局配置

所有可变参数集中在此处，并通过 run_mining.py 的命令行参数暴露在最外层。
"""

import os
import sys
from dataclasses import dataclass, asdict, field

# 保证可以导入项目根目录下的 local_api / factor_analysis
QUANT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if QUANT_ROOT not in sys.path:
    sys.path.insert(0, QUANT_ROOT)

from dotenv import load_dotenv

load_dotenv(os.path.join(QUANT_ROOT, ".env"))

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.join(PROJECT_DIR, "workspace")
FACTOR_LIBRARY_DIR = os.path.join(PROJECT_DIR, "factor_library")
# 旧版单文件路径(仅用于迁移兼容, 新流程使用因子库系列文件)
CHECKPOINT_PATH = os.path.join(WORKSPACE_DIR, "checkpoint.json")
BEST_FACTOR_PATH = os.path.join(WORKSPACE_DIR, "best_factor.json")
LOG_PATH = os.path.join(WORKSPACE_DIR, "mining.log")
REPORT_PNG_PATH = os.path.join(WORKSPACE_DIR, "factor_report.png")


def checkpoint_path(mode: str, series_id: str = "", freq: str = "daily") -> str:
    """按模式隔离的检查点路径, 保证 new/optimize 两个进程可并行互不覆盖;
    数据频率也参与隔离: 分钟挖掘与日频挖掘共用 mode=new 但检查点不同, 互不覆盖"""
    if mode == "optimize" and series_id:
        return os.path.join(WORKSPACE_DIR, f"checkpoint_optimize_{series_id}.json")
    if freq == "minute":
        return os.path.join(WORKSPACE_DIR, "checkpoint_new_minute.json")
    return os.path.join(WORKSPACE_DIR, "checkpoint_new.json")


def mining_log_path(mode: str, series_id: str = "", freq: str = "daily") -> str:
    """按模式隔离的日志路径"""
    if mode == "optimize" and series_id:
        return os.path.join(WORKSPACE_DIR, f"mining_optimize_{series_id}.log")
    if freq == "minute":
        return os.path.join(WORKSPACE_DIR, "mining_new_minute.log")
    return os.path.join(WORKSPACE_DIR, "mining_new.log")


def report_png_path(series_id: str) -> str:
    """按因子系列隔离的报告图片路径"""
    return os.path.join(WORKSPACE_DIR, f"factor_report_{series_id}.png")

# LLM 配置（参考 research/holding_increase 的阿里 token plan 用法）
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL = os.environ.get(
    "DASHSCOPE_BASE_URL",
    "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
)

# DeepSeek 配置（挖掘主模型, 固定 deepseek-v4-flash）
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# OpenCode Go 套餐网关(OpenAI兼容), 内含 deepseek-v4-flash 等模型
OPENCODE_API_KEY = os.environ.get("OPENCODE_API_KEY", "")
OPENCODE_BASE_URL = os.environ.get("OPENCODE_BASE_URL", "https://opencode.ai/zen/go/v1")

DEFAULT_DIRECTION = (
    "A股日频量价行为金融方向：可从短期反转/动量、成交量异动、量价背离、"
    "日内振幅与高低位位置、波动率压缩等经典逻辑中提出可检验的因子假设。"
)


@dataclass
class MiningConfig:
    # ---------- LLM ----------
    model_primary: str = "qwen3.8-max-preview"  # 主模型: 阿里云百炼 qwen3.8-max-preview
    model_fallback: str = "qwen3.6-flash"      # 备用模型(仅主模型完全不可用时兜底)
    llm_provider: str = "auto"                  # LLM 路由: auto=按模型名前缀(deepseek*/qwen*);
                                                # opencode=主模型走 OpenCode Go 网关(含 deepseek-v4-flash);
                                                # deepseek/dashscope=固定走对应端点
    enable_thinking: bool = True                 # 开启深度思考
    thinking_budget: int = 4096                  # 思考token预算(适当调深)
    max_tokens: int = 12000                      # 最大输出token(含思考)
    temperature: float = 0.8                     # 挖掘任务需要一定创造性
    llm_timeout: int = 600                       # 单次请求超时(秒)
    llm_max_retry: int = 6                       # 每个模型的重试次数(长请求偶发空内容, 提高韧性)
    n_eval_workers: int = 2                      # 因子评价并行进程数(1=串行; 每进程一份全市场数据副本)

    # ---------- 公式约束 ----------
    max_depth: int = 7                # 语法树最大嵌套深度(暴露为最外层接口)
    max_symbol_length: int = 120      # 公式最大字符长度
    max_base_features: int = 6        # 最大基础变量个数
    max_window: int = 120             # 时间窗口 n 的上限

    # ---------- 数据频率 ----------
    # daily=日频挖掘(原逻辑, 行为完全不变); minute=分钟频率挖掘(分钟算子聚合出日频因子)
    data_frequency: str = "daily"
    minute_frequency: str = "1m"         # 分钟基础频率(本地数据为1分钟线)
    minute_batch_size: int = 300         # 分钟数据分批处理的股票数(内存控制; 96GB内存可提到500-1000, 每批约几十GB按需) 
    minute_chunk_days: int = 100         # 全市场分钟截面模式: 按日期分块的块天数(每块加载全部股票, 内存≈天数×240×股票数)
    minute_memory_fields: str = "close,volume,amount"  # 常驻内存的分钟字段(稠密矩阵[日×240×股], 每字段约13.7GB; 其余字段用时读盘)
    minute_dense_batch: int = 1000       # 稠密路径: 无截面聚合时按股票分批的窗口大小(每批中间内存≈天数×240×批数×4B)
    minute_dense_chunk_days: int = 200   # 稠密路径: 含截面算子时按日期分块的块天数(每块加载全部股票, 控制截面中间内存)
    minute_max_depth: int = 14           # 分钟模式公式嵌套深度上限(相对日频上限翻倍: 7×2, 分钟因子构成更复杂)
    minute_max_symbol_length: int = 240  # 分钟模式公式最大字符长度(相对日频翻倍: 120×2)
    minute_max_base_features: int = 12   # 分钟模式允许的基础变量个数(相对日频翻倍: 6×2)

    @property
    def formula_max_depth(self) -> int:
        """分钟模式允许更高复杂度, 深度上限随频率切换"""
        return self.minute_max_depth if self.data_frequency == "minute" else self.max_depth

    @property
    def formula_max_symbol_length(self) -> int:
        return self.minute_max_symbol_length if self.data_frequency == "minute" else self.max_symbol_length

    # ---------- 数据 ----------
    data_start_date: str = "2016-01-01"   # 量价数据加载起始日(含rolling预热)
    eval_start_date: str = "2018-01-01"   # 因子评价起始日

    # ---------- 因子评价 ----------
    ic_period: int = 5                 # 5日 RankIC
    n_quantiles: int = 10              # 10分组, 最高组为多头
    min_stocks_per_day: int = 50       # 每日最少股票数
    monthly_ic_pos_ratio: float = 0.6  # 月度IC为正的月份占比要求

    # ---------- 分组单调性评级(四级: S优秀/A良好/B可入库需优化/C丢弃) ----------
    # 锚点(基于R2/R3/R14实测标定): R2/R3(drc≈0.22, 越序日均≈0.28%, 年度峰值4-6%) -> B级;
    # R14(drc≈0.12, 越序日均≈0.37%, 年度峰值≈16%) -> C级
    # 指标A 每日秩相关(组别vs当日各组收益, 1=完全单调递增, 0=无关): 各等级下限
    mono_rank_s: float = 0.28
    mono_rank_a: float = 0.23
    mono_rank_b: float = 0.15
    # 指标B-1 越序惩罚日均 Σmax(0,R_i-R_(i+1)) (越小越好): 各等级上限
    mono_oos_s: float = 0.0022
    mono_oos_a: float = 0.0027
    mono_oos_b: float = 0.0034
    # 指标B-2 年度聚合越序峰值 (越小越好): 各等级上限
    mono_peak_s: float = 0.03
    mono_peak_a: float = 0.05
    mono_peak_b: float = 0.10
    # 任一秩相关为负的年份 -> 直接判C级(丢弃)
    mono_neg_year_reject: bool = True

    # ---------- 因子诊断 ----------
    # 换手成本: 年化净收益 = 年化收益 - 年化双边换手率(倍) × 系数;
    # 若超过 turnover_cost_neg_ratio 的年份净收益为负, 判定换手率相对收益过高
    turnover_cost_coef: float = 0.0003     # 买卖成本系数(万分之三)
    turnover_cost_neg_ratio: float = 0.6   # 判定换手过高的年份占比阈值
    # 特殊时期压力测试窗口 (名称, 起始日, 结束日; None表示到数据最新日)
    stress_periods: list = field(default_factory=lambda: [
        ("2024年2月小盘股崩盘", "2024-01-15", "2024-02-29"),
        ("2026年4月科创板抱团", "2026-04-01", None),
    ])

    # ---------- 因子库(新因子挖掘防撞车) ----------
    max_library_corr: float = 0.5      # 新因子与库内因子截面相关系数上限(防"换价格字段/窗口"的同类因子漏网)
    corr_sample_dates: int = 60        # 相关性检查的抽样交易日数
    # 事后LLM语义评审(独立对话窗口判断新因子与库内因子是否语义重复)
    review_similar_threshold: float = 0.75  # 语义评审判"相似"时, 相似度评分达到该值即拒绝入库

    # ---------- 挖掘循环 ----------
    max_rounds: int = 12               # 最大迭代轮数
    min_library_target: int = 0        # new模式: 库中对应前缀系列数达到该值即提前结束挖掘(0=不启用)
    factors_per_round: int = 2         # 每轮要求模型输出的因子个数
    direction: str = DEFAULT_DIRECTION # 初始挖掘方向

    def to_dict(self) -> dict:
        return asdict(self)


def ensure_workspace():
    os.makedirs(WORKSPACE_DIR, exist_ok=True)
    os.makedirs(FACTOR_LIBRARY_DIR, exist_ok=True)
