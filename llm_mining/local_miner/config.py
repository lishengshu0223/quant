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
CHECKPOINT_PATH = os.path.join(WORKSPACE_DIR, "checkpoint.json")
BEST_FACTOR_PATH = os.path.join(WORKSPACE_DIR, "best_factor.json")
LOG_PATH = os.path.join(WORKSPACE_DIR, "mining.log")
REPORT_PNG_PATH = os.path.join(WORKSPACE_DIR, "factor_report.png")

# LLM 配置（参考 research/holding_increase 的阿里 token plan 用法）
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL = os.environ.get(
    "DASHSCOPE_BASE_URL",
    "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
)

DEFAULT_DIRECTION = (
    "A股日频量价行为金融方向：可从短期反转/动量、成交量异动、量价背离、"
    "日内振幅与高低位位置、波动率压缩等经典逻辑中提出可检验的因子假设。"
)


@dataclass
class MiningConfig:
    # ---------- LLM ----------
    model_primary: str = "qwen3.8-max-preview"   # 千问3.8 preview 主模型
    model_fallback: str = "qwen3.6-flash"        # 备用模型
    enable_thinking: bool = True                 # 开启深度思考
    thinking_budget: int = 4096                  # 思考token预算(适当调深)
    max_tokens: int = 12000                      # 最大输出token(含思考)
    temperature: float = 0.8                     # 挖掘任务需要一定创造性
    llm_timeout: int = 600                       # 单次请求超时(秒)
    llm_max_retry: int = 4                       # 每个模型的重试次数

    # ---------- 公式约束 ----------
    max_depth: int = 7                # 语法树最大嵌套深度(暴露为最外层接口)
    max_symbol_length: int = 120      # 公式最大字符长度
    max_base_features: int = 6        # 最大基础变量个数
    max_window: int = 120             # 时间窗口 n 的上限

    # ---------- 数据 ----------
    data_start_date: str = "2016-01-01"   # 量价数据加载起始日(含rolling预热)
    eval_start_date: str = "2018-01-01"   # 因子评价起始日

    # ---------- 因子评价 ----------
    ic_period: int = 5                 # 5日 RankIC
    n_quantiles: int = 10              # 10分组, 最高组为多头
    min_stocks_per_day: int = 50       # 每日最少股票数
    monthly_ic_pos_ratio: float = 0.6  # 月度IC为正的月份占比要求

    # ---------- 挖掘循环 ----------
    max_rounds: int = 12               # 最大迭代轮数
    factors_per_round: int = 2         # 每轮要求模型输出的因子个数
    direction: str = DEFAULT_DIRECTION # 初始挖掘方向

    def to_dict(self) -> dict:
        return asdict(self)


def ensure_workspace():
    os.makedirs(WORKSPACE_DIR, exist_ok=True)
