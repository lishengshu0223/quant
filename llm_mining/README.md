# LLM 因子挖掘（本地化）

基于大模型（DeepSeek / 阿里百炼）的 A 股量化因子自动挖掘系统。核心代码在 `llm_mining/local_miner/`，
全程使用**本地全 A 股数据**做因子评价（不走在线回测），支持断点续传、多进程并行评价、
静态 HTML 报告（KaTeX 公式 + 评价图 + 库相关性矩阵）。

> 本文件是因子挖掘流程的总说明。任何需要"了解当前因子挖掘情况"的大模型/agent 请先读本文件，
> 再按需查看 `factor_library/`（成功库 + 失败库）与 `local_miner/` 源码。

---

## 1. 快速启动

```powershell
conda activate multifactor
cd f:\quant

# 新因子挖掘（分钟频率, 100 轮战役, Opencode 唯一通道）
python -m llm_mining.local_miner.run_mining --mode new --frequency minute `
    --llm-provider opencode --model deepseek-v4-flash --max-rounds 100 --direction "挖掘方向描述"

# 日频新因子挖掘
python -m llm_mining.local_miner.run_mining --mode new --frequency daily --max-rounds 100

# 迭代优化已有系列（如 F001）
python -m llm_mining.local_miner.run_mining --mode optimize --series F001 --frequency daily
```

- 断点续传：检查点存于 `local_miner/workspace/checkpoint_<mode>_<series>_<freq>.json`，中断后重跑相同命令即续传。
- 常用参数见 `parse_args()`（`--max-rounds / --factors-per-round / --eval-start / --min-library-target /
  --llm-provider / --model / --workers / --fresh` 等）。

## 2. 目录结构

```
llm_mining/
├── README.md                  # 本文件
└── local_miner/
    ├── run_mining.py          # 主入口(双模式: new / optimize)
    ├── prompts.py             # 全部提示词组装(首轮注入成功库摘要+战役摘要+失败经验)
    ├── llm_client.py          # LLM 调用(Opencode 唯一通道, 无限重试)
    ├── factor_eval.py         # 因子评价/单调性评级/反馈文本
    ├── factor_library.py      # 成功因子库管理(系列文件读写/防撞车相关性检查)
    ├── factor_archive.py      # 成功因子归档(h5 回测宽表 + 评价图)
    ├── failed_library.py      # 失败因子库(每轮沉淀+战役总结+注入文本)
    ├── factor_plot.py         # 评价图 / 库相关性热图 / HTML 报告
    ├── review.py              # 入库第一道防线: LLM 语义评审(防"换皮造因子")
    ├── diagnostics.py         # 入库后诊断(换手成本/单调性/压力期)
    ├── expr_engine.py         # 公式解析与计算引擎(分钟/日频算子)
    ├── data_loader.py         # 本地行情数据加载(全A股, 后复权)
    ├── combo.py / checkpoint.py / console.py / config.py
    ├── factor_library/        # ★ 因子库(正式交付, 长期保留)
    │   ├── success/           # 成功因子库: 每因子一个文件夹(三件套)
    │   └── failed/            # 失败因子库: 每战役一个文件夹
    └── workspace/             # 检查点/日志/采纳因子报告图(过程性, 可清)
```

## 3. 挖掘流程（每轮循环）

```
① 模型产出因子(JSON: 因子假设 + 因子列表[name/desc/expr/style])
② 公式校验 → 本地全A股计算因子宽表(多进程并行)
③ 因子评价: IC/ICIR/月度正占比/分组单调性(秩相关+越序+年度峰值, 4级评级)/多头收益年度稳定性
④ 反馈文本注入下一轮(逐轮迭代; 组合类因子隐藏评估, 视为"最后手段")
⑤ new模式: 合格因子过"双防线"入库; optimize模式: 稳定性更优则采纳为系列最佳
⑥ 每轮结束: 失败因子沉淀进失败库; 刷新 HTML 报告
```

**战役(100轮)结束后自动**：
- 战役级概括总结（失败库规则统计 + LLM ≤300字概括，LLM 失败自动降级为规则文本）；
- 新战役首轮自动注入三份历史经验：**成功因子库摘要 + 历史战役压缩摘要 + 失败因子库经验**（防止重复挖掘）。

## 4. 合格标准（入库门槛）

| 条件 | 要求 |
|---|---|
| (a) 方向 | IC 均值与多头累计超额同向为正 |
| (b) 月度 | 月度 IC 正占比 ≥ 60% |
| (c) 年度 | 每个历史完整年份多头超额均为正（**最新未满一年年份豁免**） |
| (d) 单调性 | 评级非 C 级（任一历史完整年份秩相关为负 → 直接 C 级；最新年份同样豁免） |

- 复杂度上限：日频嵌套深度 7 / 公式长度 120 / 基础变量 6；分钟模式翻倍（14 / 240 / 12）。
- 评价标准核心：**多头收益的年度稳定性**（年度信息比率 = 各年多头收益均值/标准差），而非单纯抬高 IC（IC 常由空头贡献）。

## 5. 入库双防线（new 模式）

1. **LLM 语义评审**（`review.py`）：独立对话判断新因子与库内因子是否"换皮"（换价格字段/微调窗口/加包装
   一律视为同一因子），拒绝则标记 `review_rejected`。
2. **截面相关性检验**（`factor_library.check_library_correlation`）：与库内各系列最佳因子做截面
   Spearman 相关，最大 |ρ| > `max_library_corr`(默认0.7) 即拒绝，标记 `library_rejected`。

## 6. 因子库格式

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
2. `prompts.load_campaign_summaries()` —— 历史战役压缩摘要（`prompts.save_campaign_summary` 生成）
3. `failed_library.injection_text()` —— 历史失败经验（最近 4 个战役的 LLM 概括 + 规则统计）

## 7. 环境与依赖

- Python 环境：conda `multifactor`（含 h5py、pandas、numpy 等）
- LLM：Opencode 网关（`deepseek-v4-flash`）为唯一通道，失败无限等待重试，绝不切换通道
- 数据：本地全 A 股后复权行情（`local_api` 或本地 parquet/csv，见 `data_loader.py`）
- 内存：分钟模式 `close/volume` 常驻内存（各约 13.7GB），其余字段用时读盘；96GB 机器可并行 2 worker
