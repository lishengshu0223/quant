"""
本地化因子挖掘项目 - 分钟频率引擎子包（minute_engine 拆分的实现模块）

对外仅通过 local_miner.minute_engine 薄入口访问, 本子包模块不直接对外暴露:
- minute_parser.py      解析/类型推断/校验 + 全部规格常量
- minute_kernels.py     numba 稠密归约内核
- minute_data.py        分钟数据加载 + 长表聚合 + MinuteExpr/MinuteFieldCache/MinuteMarketData
- minute_sparse_eval.py 长表路径求值(_BatchRunner) + CachedEvaluator + 3D/2D 运算辅助
- minute_dense_eval.py  稠密加速路径(DenseEvaluator) + 稠密入口

依赖方向: kernels -> (parser -> expr_engine); data -> parser; sparse_eval -> data;
dense_eval -> sparse_eval。data 与 sparse_eval 的循环依赖用函数内延迟导入打破。
"""
