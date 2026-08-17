"""本地化因子挖掘项目 - 辅助/运维脚本包

与核心包 llm_mining.local_miner 分离的独立工具:
  - run_mining_loop.py        挖掘循环编排器(无人值守)
  - run_both_watchdog.py      双程序守护脚本
  - run_batch_reports.py      批量因子评价与报告生成
  - run_library_corr_matrix.py 因子库合格因子相关性矩阵
  - run_neutral_eval.py       Barra 行业/市值中性化重评

所有脚本通过 `python -m llm_mining.tools.xxx` 从项目根目录(F:\quant)调用。
"""
