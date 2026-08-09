# 项目说明

这是一个量化金融项目，用于分析和预测金融市场的趋势，并且下载金融数据。

## 技术栈

- Python（执行环境为 conda 的 `multifactor` 环境，运行脚本前需先 `conda activate multifactor`）
- Ricequant SDK（RQData、RQAlphaPlus、RQFactor、RQOptimizer、RQPAttr）

## 项目结构

- `download/` - 金融数据下载
- `research/` - 策略研究
- `update/` - 数据更新
- `llm_mining/` - LLM 因子挖掘（核心代码 `llm_mining/local_miner/`；涉及因子挖掘任务时，先读 `llm_mining/README.md` 了解流程与因子库结构）
- `local_api/` - 本地 API 服务
- `backtest_results_test/` - 回测结果测试

## 编码规范

- 使用中文注释和文档
- 遵循 Python PEP 8 规范
- 量化策略代码使用 RQAlphaPlus 框架的约定函数（init, handle_bar 等）
- 获取行情、指数、因子、交易状态等金融数据时，优先使用 `local_api` 读取本地数据（详见 `.trae/rules/local_api_usage.md`），仅在本地数据缺失时才调用 rqdatac 在线 API

## 临时文件管理规范（必须严格遵守）

在执行分析、研究、调试、测试类任务时，agent 产生的文件必须明确分为三类，不得混淆：

**① 临时文件（任务结束自动删除）**：测试脚本、测试数据、中间结果、调试日志等过程性文件。
- 文件名一律以 `tmp_` 开头（如 `tmp_vol_test.py`、`tmp_check_data.csv`）
- 一律存放在 `f:\quant\tmp\` 目录下（不存在则创建），禁止散落在项目根目录或 `research/`、`download/` 等正式目录中；确因脚本运行路径限制必须写在别处时，仍须保持 `tmp_` 前缀
- 任务结束时 agent 必须主动删除本次任务创建的临时文件，无需用户提醒

**② 分析成果（保留给用户查看，禁止删除）**：需要用户查看的最终图片、表格、报告等。
- 一律存放在 `f:\quant\output\<YYYYMMDD>_<任务名>\` 子目录下（如 `output\20260805_成交量因子分析\`）
- 文件名可读、带业务含义、不带 `tmp_` 前缀（如 `年度收益对比.png`、`信号明细.xlsx`）
- 任何程序代码不得依赖 `output\` 或 `tmp\` 目录下的文件，用户可随时手动清空这两个目录而不影响项目运行

**③ 正式交付（长期保留）**：用户明确要求沉淀的代码、策略、数据。
- 保存为不带 `tmp_` 前缀的正式文件，放入对应正式目录（`research/`、`download/` 等）

**其他硬性要求**：
1. 任务结束的最终回复中，必须分别列出「保留的成果文件」清单（完整路径 + 一句话说明内容）和「已删除的临时文件」清单。
2. 只允许删除本次任务自己创建的临时文件，严禁删除任何既有文件（包括遗留的 `tmp_` 开头的旧文件，由用户自行处理）。
3. 若不确定某个产出用户是否需要保留，先询问，禁止擅自删除。
