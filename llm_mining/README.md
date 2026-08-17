# LLM 因子挖掘（本地化）

基于大模型（DeepSeek / 阿里百炼）的 A 股量化因子自动挖掘系统。核心代码在 `llm_mining/local_miner/`，
全程使用**本地全 A 股数据**做因子评价（不走在线回测），支持断点续传、多进程并行评价、
静态 HTML 报告（KaTeX 公式 + 评价图 + 库相关性矩阵）。

> ## ⚠️ Agent / 大模型入口指引（务必先读）
>
> 任何新会话若任务涉及本模块（LLM 因子挖掘、因子库查询、挖掘进度查看、流程/命令咨询），
> 请遵循以下顺序快速定位文档，**不要直接翻源码**：
>
> 1. **顶层导航**：先读本文件（`llm_mining/README.md`）——模块定位、快速启动、目录结构、合格标准总览。
> 2. **架构详解**：[`docs/01_项目架构.md`](docs/01_项目架构.md) —— 全部模块职责、模块间调用关系、数据流。
> 3. **流程详解**：[`docs/02_因子挖掘流程详解.md`](docs/02_因子挖掘流程详解.md) —— 从启动到战役结束的每一步骤。
> 4. **因子库/数据格式**：[`docs/03_因子库与数据格式.md`](docs/03_因子库与数据格式.md) —— success/failed 库、检查点、日志、输出目录怎么读。
> 5. **配置与命令**：[`docs/04_配置与命令参考.md`](docs/04_配置与命令参考.md) —— 全部命令行参数、配置项、辅助脚本。
> 6. 需要看运行现场时：`local_miner/workspace/`（检查点/日志/状态文件）与
>    `local_miner/factor_library/`（成功库 + 失败库）中的 json 文件即当前实际状态。
>
> 项目文档索引（`.trae/rules/document_index.md`）与 AGENTS.md 均指向本文件作为模块入口。

---

## 1. 快速启动

```powershell
conda activate multifactor
cd f:\quant

# 新因子挖掘（分钟频率, 100 轮战役, 指定 Opencode 通道示例）
python -m llm_mining.local_miner.run_mining --mode new --frequency minute `
    --llm-provider opencode --model deepseek-v4-flash --max-rounds 100 --direction "挖掘方向描述"

# 日频新因子挖掘（默认通道: 阿里百炼 deepseek-v4-flash-0731, 备用 Opencode deepseek-v4-flash）
python -m llm_mining.local_miner.run_mining --mode new --frequency daily --max-rounds 100

# 迭代优化已有系列（如 D001）
python -m llm_mining.local_miner.run_mining --mode optimize --series D001 --frequency daily
```

- 断点续传：检查点存于 `local_miner/workspace/checkpoint_<mode>_<series>_<freq>.json`，中断后重跑相同命令即续传；`--fresh` 强制重开。
- 完整参数清单见 [docs/04_配置与命令参考.md](docs/04_配置与命令参考.md)。
- 无人值守自动挖掘（编排器）：见 [docs/04_配置与命令参考.md](docs/04_配置与命令参考.md#5-无人值守编排器)。

## 2. 目录结构

```
llm_mining/
├── README.md                  # ★ 本文件（顶层入口导航）
├── docs/                      # ★ 说明文档（架构/流程/因子库/配置），见上方指引
├── local_miner/               # ★ 核心包（挖掘引擎, python -m llm_mining.local_miner.xxx）
│   ├── run_mining.py          # 主入口（双模式: new / optimize, 主循环编排）
│   ├── cli.py                 # 命令行参数解析（parse_args, 由 run_mining 复用）
│   ├── new_mode.py            # new 模式合格因子处理（入库双防线调用链）
│   ├── optimize_mode.py       # optimize 模式合格因子处理（稳定性择优）
│   ├── workers.py             # 多进程并行评价 worker（初始化/单因子评价）
│   ├── finalize.py            # 战役收尾（最终报告因子确定等）
│   ├── prompts.py             # 提示词组装（构造函数 + 黑名单记忆）
│   ├── prompts_templates.py   # 提示词模板常量（角色/变量/函数库/输出格式）
│   ├── llm_client.py          # LLM 调用（百炼+Opencode 双通道, 无限重试）
│   ├── factor_eval.py         # 因子评价/单调性评级/反馈文本
│   ├── barra_neutralize.py    # Barra 行业/市值中性化（暴露加载 + 中性化）
│   ├── factor_library.py      # 成功因子库管理（系列读写/防撞车相关性检查）
│   ├── factor_archive.py      # 成功因子归档（h5 回测宽表 + 评价图）
│   ├── failed_library.py      # 失败因子库（每轮沉淀+战役总结+战役信息压缩摘要）
│   ├── factor_plot.py         # 评价图 / 库相关性热图
│   ├── html_report.py         # 静态 HTML 报告（KaTeX 公式 + 评价图 + 相关性）
│   ├── review.py              # 入库第一道防线: LLM 语义评审（防"换皮造因子"）
│   ├── diagnostics.py         # 入库后诊断（换手成本/单调性/压力期）
│   ├── expr_engine.py         # 日频公式解析与计算引擎
│   ├── minute_engine.py       # 分钟公式引擎薄入口（compute_factor_minute + re-export）
│   ├── minute/                # 分钟引擎实现子包（仅由 minute_engine 薄入口引用）
│   │   ├── minute_parser.py       # 分钟公式解析/类型推断/校验 + 全部规格常量
│   │   ├── minute_data.py         # 分钟数据容器/日频聚合/宽表构建
│   │   ├── minute_sparse_eval.py  # 分钟稀疏求值路径（长表 groupby 按股批式流式）
│   │   ├── minute_dense_eval.py   # 分钟稠密求值路径（numpy/numba 归约, 截面聚合按日分块）
│   │   └── minute_kernels.py      # numba 加速内核（skew/kurt/std/regression/corr 等）
│   ├── data_loader.py         # 本地行情数据加载（全A股, 后复权, 可交易状态遮盖）
│   ├── combo.py               # 组合类因子检测与比较（最后手段机制）
│   ├── report.py              # 最终因子 tear sheet 单图报告
│   ├── formula_tex.py         # 公式 -> LaTeX 渲染（评价图/HTML 报告共用）
│   ├── checkpoint.py / console.py / config.py
│   ├── factor_library/        # ★ 因子库（正式交付, 长期保留）
│   │   ├── success/           # 成功因子库: 每因子一个文件夹（三件套）
│   │   └── failed/            # 失败因子库: 每战役一个文件夹
│   └── workspace/             # 检查点/日志/状态文件（过程性, 可清）
└── tools/                     # 辅助/运维脚本（python -m llm_mining.tools.xxx）
    ├── run_mining_loop.py     # 挖掘循环编排器（自动挖到目标数量合格因子）
    ├── run_both_watchdog.py   # 双程序守护（optimize + new, 异常自动重启）
    ├── run_batch_reports.py   # 批量因子评价与报告生成（多进程并行）
    ├── run_library_corr_matrix.py  # 全库合格因子两两截面秩相关矩阵
    └── run_neutral_eval.py    # Barra 行业/市值中性化重评（原始 vs 中性化对比）
tests/                         # 单元测试（pytest: 日频/分钟公式解析与校验）
```

- 核心入口一律 `python -m llm_mining.local_miner.xxx`；辅助脚本见 [docs/04_配置与命令参考.md](docs/04_配置与命令参考.md)（`python -m llm_mining.tools.xxx`）。
- 单元测试：`conda run -n multifactor python -m pytest llm_mining/tests/ -q`（覆盖 expr_engine 与 minute_parser 的解析/校验分支）。

## 3. 挖掘流程（概览）

```
阶段1 配置初始化 → 阶段2 数据加载 → 阶段3 系统提示词
→ 循环 [ 模型产出因子 → 公式校验/计算 → 因子评价 → 反馈注入 → 入库/沉淀 ]
→ 战役结束: 信息压缩 + 失败库战役总结 + 最终报告
```

**每轮循环（详见 [docs/02_因子挖掘流程详解.md](docs/02_因子挖掘流程详解.md)）**：

1. 模型产出因子（JSON：因子假设 + 因子列表 [name/desc/expr/style]）；
2. 公式校验 → 本地全 A 股计算因子宽表（多进程并行）；
3. 因子评价：IC/ICIR/月度正占比/分组单调性（秩相关+越序+年度峰值, S/A/B/C 四级）/多头收益年度稳定性；
4. 反馈文本注入下一轮（逐轮迭代；组合类因子隐藏评估，视为"最后手段"）；
5. new 模式：合格因子过"双防线"入库；optimize 模式：稳定性更优则采纳为系列最佳；
6. 每轮结束：失败因子沉淀进失败库；刷新 HTML 报告。

**战役（默认 100 轮）结束后自动**：

- 战役级概括总结（失败库规则统计 + LLM ≤300 字概括，LLM 失败自动降级为规则文本）；
- 新战役首轮自动注入三份历史经验：**成功因子库摘要 + 历史战役压缩摘要 + 失败因子库经验**（防止重复挖掘）。

## 4. 合格标准（入库门槛）

| 条件 | 要求 |
|---|---|
| (a) 方向 | IC 均值与多头累计超额同向为正（双负自动翻转；一正一负剔除） |
| (b) 月度 | 月度 IC 正占比 ≥ 60% |
| (c) 年度 | 每个历史完整年份多头超额均为正（**最新未满一年年份豁免**） |
| (d) 单调性 | 评级非 C 级（任一历史完整年份秩相关为负 → 直接 C 级；最新年份同样豁免） |

- 评价口径：5 日 RankIC，10 分组（最高组为多头），收益采用日度等效口径（n 日前向收益÷n）。
- 复杂度上限：日频嵌套深度 14 / 公式长度 240 / 基础变量 12（分钟模式同此上限）；窗口 ≤ 240。
- 评价标准核心：**多头收益的年度稳定性**（年度信息比率 = 各年多头收益均值/标准差），而非单纯抬高 IC（IC 常由空头贡献）。

## 5. 入库双防线（new 模式）

1. **LLM 语义评审**（`review.py`）：独立对话判断新因子与库内因子是否"换皮"（换价格字段/微调窗口/加包装
   一律视为同一因子），拒绝则标记 `review_rejected`。
2. **截面相关性检验**（`factor_library.check_library_correlation`）：与库内各系列最佳因子做截面
   Spearman 相关，最大 |ρ| > `max_library_corr`（默认 **0.5**）即拒绝，标记 `library_rejected`。

## 6. 因子库格式

详见 [docs/03_因子库与数据格式.md](docs/03_因子库与数据格式.md)。要点：

### 成功因子库 `factor_library/success/<series_id>_<name>/`

每个成功因子一个独立文件夹，内含**三件套**（最少存储存最多信息）：

| 文件 | 内容 |
|---|---|
| `<id>_<name>.json` | 结构化因子信息 + 回测摘要：公式/简介/风格、整条迭代路径(每步表达式+评价)、诊断、成败经验、`best` 完整评价 |
| `<id>_<name>_backtest.h5` | 完整回测宽表(float32 + zlib 压缩) + 日期/股票索引，可随时重放任意回测与图表（读法见 `factor_archive.load_backtest_h5`） |
| `<id>_<name>_回测评价.png` | 因子评价图（基于 json 信息与 h5 数据生成，标注合格） |

系列编号：`F` = 日频，`M` = 分钟，两个前缀独立计数（`next_series_id`）。

### 失败因子库 `factor_library/failed/<campaign_key>/`

`campaign_key = <frequency>_<eval_start_date>`（如 `minute_20180101`），每战役一个文件夹：

| 文件 | 内容 |
|---|---|
| `round_XXX.json` | 该轮全部因子的精简记录：name/expr/style/**失败原因**/细节/IC/多头/月度占比/评级/负年 |
| `round_XXX_summary.json` | 该轮一句话失败总结（规则生成） |
| `campaign_summary.json` | 战役级总结：规则统计(失败原因分布/高频被拒结构/有效方向) + LLM 概括 |
| `index.json` | 注入索引（仅保留最近 4 个战役摘要，控制首轮注入体积） |

失败原因分类：计算失败 / 语义评审拒绝 / 相关性撞车 / 单调性C级 / 方向无效 / 月度不足 / 年度为负 / 其他未达标。

### 新挖掘首轮注入内容（防止重复挖掘）

1. `factor_library.library_summary_text()` —— 成功库摘要（已有因子必须避让）
2. `failed_library.load_campaign_summaries()` —— 历史战役压缩摘要（`failed_library.save_campaign_summary` 生成）
3. `failed_library.injection_text()` —— 历史失败经验（最近 4 个战役的 LLM 概括 + 规则统计）

## 7. 环境与依赖

- Python 环境：conda `multifactor`（含 h5py、pandas、numpy、scipy、matplotlib、factor_analysis、local_api 等）
- LLM：默认主模型走阿里百炼 `deepseek-v4-flash-0731`，备用走 Opencode 网关 `deepseek-v4-flash`
  （双通道统一 DeepSeek v4-flash，不再使用 qwen）；`--llm-provider opencode` 时 Opencode 为唯一通道、
  失败无限等待重试、绝不切换通道（见 `config.py` 与 [docs/04_配置与命令参考.md](docs/04_配置与命令参考.md)）
- 数据：本地全 A 股后复权行情（`F:\Trade_data\stock_price`，经 `local_api` 读取）；每日可交易状态
  （`F:\Trade_data\tradable_status`）剔除 ST/停牌/次新/涨跌停；分钟数据 `F:\Trade_data\rq_backtest_data\h5\equities`
- 内存：分钟模式 `close/volume/amount` 常驻内存（各约 13.7GB），其余字段用时读盘；96GB 机器可并行 2 worker
