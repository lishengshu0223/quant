"""
本地化因子挖掘项目 - Barra 行业/市值中性化（factor_eval.py 拆分模块）

对因子做每日截面中性化: factor ~ 申万一级行业哑变量 + Barra size, 取残差作为中性化因子值。
FWL 等价向量化实现: 先在 日期×行业 内对 factor 与 size 各自 demean
(等价于对行业哑变量回归取残差), 再无截距回归 beta = Σ(f_dm·s_dm)/Σ(s_dm²),
resid = f_dm - beta·s_dm。暴露数据来自本地 local_api(barra/{model}/exposure)。
"""

import numpy as np
import pandas as pd


# Barra v1 暴露数据中的风格/元数据列, 其余中文列即为申万一级行业哑变量
BARRA_NON_INDUSTRY_COLS = {
    "beta", "book_to_price", "earnings_yield", "growth", "leverage",
    "liquidity", "momentum", "non_linear_size", "residual_volatility",
    "size", "specific_return", "specific_risk",
}


def load_barra_exposure(start_date=None, model: str = "v1"):
    """
    通过 local_api 读取 Barra 因子暴露(全市场)。

    Args:
        start_date: 只保留该日期起的暴露(可选, str 或 Timestamp)
        model: Barra 模型版本(本地数据目录 barra/{model}/exposure)

    Returns:
        (exp, ind_cols): exp 为 (date, code) MultiIndex 的暴露 DataFrame(含 size 列);
        ind_cols 为行业哑变量列名列表(申万一级中文名)。
    """
    import local_api as la
    exp = la.get_factor_exposure([], start_date=start_date, end_date=None, model=model)
    if exp is None or exp.empty:
        raise RuntimeError(f"Barra 暴露数据为空, 请检查本地数据目录 barra/{model}/exposure")
    if start_date is not None:
        exp = exp.loc[exp.index.get_level_values(0) >= pd.Timestamp(start_date)]
    ind_cols = [c for c in exp.columns if c not in BARRA_NON_INDUSTRY_COLS]
    return exp, ind_cols


def neutralize_factor(factor_wide: pd.DataFrame, exp: pd.DataFrame,
                      ind_cols: list) -> pd.DataFrame:
    """
    每日截面中性化: factor ~ 行业哑变量 + size, 返回残差宽表(日期×股票)。

    FWL 等价向量化实现: 先在 日期×行业 内对 factor 与 size 各自 demean
    (等价于对行业哑变量回归取残差), 再无截距回归
    beta = Σ(f_dm·s_dm)/Σ(s_dm²), resid = f_dm - beta·s_dm。

    Args:
        factor_wide: 因子宽表(日期×股票)
        exp: Barra 暴露(需含 size 列与行业哑变量列, MultiIndex=(date, code))
        ind_cols: 行业哑变量列名列表
    """
    f_long = factor_wide.stack(future_stack=True).dropna()
    f_long.index.names = ["date", "code"]
    f_long = f_long.rename("factor")

    ind_cols = list(ind_cols)
    need = exp[["size"] + ind_cols]
    merged = f_long.to_frame().join(need, how="inner").sort_index()
    merged = merged.dropna(subset=["size"])

    ind_vals = merged[ind_cols].to_numpy(dtype="float32")
    has_ind = ind_vals.sum(axis=1) > 0.5        # 行业哑变量全为0(缺失行业)的样本剔除
    merged = merged[has_ind]
    ind_vals = ind_vals[has_ind]
    merged["ind"] = ind_vals.argmax(axis=1)

    # 第一步: 行业内 demean (等价于回归行业哑变量取残差)
    grp = merged.groupby([merged.index.get_level_values(0), "ind"], observed=True)
    f_dm = merged["factor"] - grp["factor"].transform("mean")
    s_dm = merged["size"] - grp["size"].transform("mean")

    # 第二步: 每日无截距回归 f_dm ~ s_dm, 取残差
    dates = merged.index.get_level_values(0)
    num = (f_dm * s_dm).groupby(dates).sum()
    den = (s_dm * s_dm).groupby(dates).sum()
    beta = num / den.replace(0.0, np.nan)
    resid = f_dm - beta.loc[dates].to_numpy() * s_dm
    resid.name = "factor"
    return resid.unstack("code").astype("float64")


def neutralize_barra(factor_wide: pd.DataFrame, start_date=None,
                     model: str = "v1") -> pd.DataFrame:
    """一站式 Barra 行业/市值中性化: 加载暴露并每日截面回归, 返回残差宽表(日期×股票)"""
    exp, ind_cols = load_barra_exposure(start_date=start_date, model=model)
    return neutralize_factor(factor_wide, exp, ind_cols)
