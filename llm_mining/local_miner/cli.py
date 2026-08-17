"""
本地化因子挖掘项目 - 命令行参数解析（run_mining 拆分模块）
"""

import argparse

from .config import MiningConfig


def parse_args():
    p = argparse.ArgumentParser(description="本地化LLM因子挖掘(双模式)")
    _cfg = MiningConfig()  # 命令行默认值统一取自配置, 避免两处硬编码不同步
    p.add_argument("--mode", type=str, default="new", choices=["new", "optimize"],
                   help="new=挖掘新因子; optimize=迭代优化已有因子系列")
    p.add_argument("--series", type=str, default="", help="optimize模式绑定的因子系列ID(如 D001)")
    p.add_argument("--max-rounds", type=int, default=12, help="最大迭代轮数")
    p.add_argument("--min-library-target", type=int, default=0,
                   help="new模式: 库中对应前缀系列数达到该值即提前结束挖掘(0=不启用; 分钟挖掘建议10)")
    p.add_argument("--max-depth", type=int, default=_cfg.max_depth, help="公式语法树最大嵌套深度")
    p.add_argument("--factors-per-round", type=int, default=_cfg.factors_per_round, help="每轮输出因子个数")
    p.add_argument("--direction", type=str, default=None, help="初始挖掘方向(new模式)")
    p.add_argument("--eval-start", type=str, default="2018-01-01", help="因子评价起始日期")
    p.add_argument("--thinking-budget", type=int, default=_cfg.thinking_budget, help="模型思考token预算")
    p.add_argument("--workers", type=int, default=2,
                   help="因子评价并行进程数(1=串行; 每进程一份全市场数据副本, 注意内存)")
    p.add_argument("--fresh", action="store_true", help="忽略检查点, 重新开始")
    p.add_argument("--frequency", type=str, default="daily", choices=["daily", "minute"],
                   help="数据频率: daily=日频挖掘(默认); minute=分钟频率挖掘(最终因子仍为日频, 分钟算子聚合出日频特征)")
    p.add_argument("--mining-theme", type=str, default="", choices=["", "cut"],
                   help="挖掘主题: 空=常规挖掘(日频/分钟原逻辑); cut=因子切割论模式(强制用 CTOP/CBOT 切割算子挖掘日频/分钟切割因子, "
                        "并放宽公式深度/长度上限)")
    p.add_argument("--cut-max-depth", type=int, default=_cfg.cut_max_depth,
                   help="切割模式: 公式嵌套深度上限(默认20, 较常规14放宽)")
    p.add_argument("--cut-max-window", type=int, default=_cfg.cut_max_window,
                   help="切割模式: 日频切割窗口上限(交易日数, 默认480)")
    p.add_argument("--minute-cut-max-window", type=int, default=_cfg.minute_cut_max_window,
                   help="切割模式: 分钟切割窗口上限(bar数, 默认4800=20个交易日)")
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
                   help="LLM路由: opencode=主模型走OpenCode Go网关(建议配--model deepseek-v4-flash); "
                        "auto=按模型名路由(deepseek-v4-flash-0731→百炼, deepseek-v4-flash→Opencode)")
    p.add_argument("--model", type=str, default="",
                   help="主模型名(如 deepseek-v4-flash-0731; 留空使用配置默认)")
    p.add_argument("--model-fallback", type=str, default="",
                   help="备用模型名(如 deepseek-v4-flash; 留空使用配置默认)")
    p.add_argument("--output-dir", type=str, default="",
                   help="评价图/相关性/HTML报告输出目录(默认 output/<YYYYMMDD>_<任务名>)")
    return p.parse_args()
