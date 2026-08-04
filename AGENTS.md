# 项目说明

这是一个量化金融项目，用于分析和预测金融市场的趋势，并且下载金融数据。

## 技术栈

- Python
- Ricequant SDK（RQData、RQAlphaPlus、RQFactor、RQOptimizer、RQPAttr）

## 项目结构

- `download/` - 金融数据下载
- `research/` - 策略研究
- `update/` - 数据更新
- `llm_mining/` - LLM 数据挖掘
- `local_api/` - 本地 API 服务
- `backtest_results_test/` - 回测结果测试

## 编码规范

- 使用中文注释和文档
- 遵循 Python PEP 8 规范
- 量化策略代码使用 RQAlphaPlus 框架的约定函数（init, handle_bar 等）
- 获取行情、指数、因子、交易状态等金融数据时，优先使用 `local_api` 读取本地数据（详见 `.trae/rules/local_api_usage.md`），仅在本地数据缺失时才调用 rqdatac 在线 API
