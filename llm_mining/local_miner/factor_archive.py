"""
本地化因子挖掘项目 - 成功因子库归档

每个成功因子在 factor_library/success/<series_id>_<name>/ 下保留"三件套"
(与失败因子库 factor_library/failed/ 平级, 最少储存占用存最多信息):
  1) <id>_<name>.json           结构化因子信息 + 回测摘要(由 factor_library 系列文件提供)
  2) <id>_<name>_backtest.h5    完整回测数据: 因子宽表(float32 + zlib 压缩) + 日期/股票索引,
                                可随时重放任意回测与图表
  3) <id>_<name>_回测评价.png   因子评价图(基于 json 信息与 h5 数据生成, 标注合格)
"""

import datetime
import os
import re

import h5py
import numpy as np
import pandas as pd

from .config import FACTOR_LIBRARY_DIR, ensure_workspace
from .factor_library import SUCCESS_DIR


def _now() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def archive_path(series_id: str, name: str, ext: str) -> str:
    """返回归档文件路径(因子专属文件夹内)。ext: 不带点, 如 'h5'/'png'/'json'"""
    safe = re.sub(r"[^0-9A-Za-z_]", "", name or "factor")[:40] or "factor"
    fdir = os.path.join(SUCCESS_DIR, f"{series_id}_{safe}")
    fname = f"{series_id}_{safe}" + {
        "h5": "_backtest.h5", "png": "_回测评价.png", "json": ".json",
    }.get(ext, f".{ext}")
    return os.path.join(fdir, fname)


def save_backtest_h5(factor_wide: pd.DataFrame, series_id: str, name: str,
                     meta: dict | None = None) -> str:
    """
    把因子宽表写入 HDF5(float32 + zlib 压缩)并返回路径。
    factor_wide: DataFrame(日期索引 × 股票列), 允许含缺失(NaN)。
    """
    ensure_workspace()
    path = archive_path(series_id, name, "h5")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = np.ascontiguousarray(factor_wide.values, dtype=np.float32)
    dates = np.ascontiguousarray(np.asarray(factor_wide.index.strftime("%Y%m%d")).astype("S10"))
    codes = np.ascontiguousarray(np.asarray(factor_wide.columns.astype(str)).astype("S16"))
    tmp = path + ".tmp"
    with h5py.File(tmp, "w") as f:
        # 因子宽表: float32 + zlib 压缩
        f.create_dataset("factor_wide", data=arr, compression="gzip",
                         compression_opts=4)
        # 字符串索引: 定长字节串, 进一步省空间
        f.create_dataset("dates", data=dates)
        f.create_dataset("codes", data=codes)
        if meta:
            import json
            f.attrs["meta"] = json.dumps(meta, ensure_ascii=False)
        f.attrs["shape"] = arr.shape
    os.replace(tmp, path)
    return path


def load_backtest_h5(path: str) -> pd.DataFrame:
    """从归档 h5 恢复因子宽表 DataFrame"""
    with h5py.File(path, "r") as f:
        arr = f["factor_wide"][:]
        dates = [d.decode() for d in f["dates"][:]]
        codes = [c.decode() for c in f["codes"][:]]
    idx = pd.to_datetime(dates, format="%Y%m%d")
    return pd.DataFrame(arr, index=idx, columns=codes)


def archive_success_series(series: dict, factor_wide: pd.DataFrame, data, cfg,
                           round_no: int = 0) -> dict:
    """
    为成功入库因子生成归档: h5 回测数据 + 集中评价图。返回 {"h5": path, "png": path}。
    评价图用因子评价管线(plot_round_factor)基于公式与宽表生成, 标注合格状态。
    """
    from . import factor_plot
    best = series.get("best") or series
    sid = series.get("series_id", "")
    # 用系列名统一三件套命名(与系列 json 文件名一致), 避免因 best 改名产生分叉
    name = series.get("name") or best.get("name") or "factor"
    # 1) h5 回测数据
    h5_path = save_backtest_h5(factor_wide, sid, name,
                               meta={"series_id": sid, "name": name,
                                     "expr": best.get("expr", ""),
                                     "created_at": _now()})
    # 2) 评价图(标注合格与否; 失败不中断)
    png_path = archive_path(sid, name, "png")
    try:
        entry = {"name": name, "desc": best.get("desc", ""),
                 "expr": best.get("expr", ""), "style": best.get("style", ""),
                 "eval": dict(best.get("eval") or {})}
        # 注意: 传入的 factor_wide 已是按评价方向翻转后的最终宽表(调用方 recompute_series 处理),
        # 出图时不能再依 eval.flipped 重复翻转, 否则双重取反导致图上方向错误。
        entry["eval"].pop("flipped", None)
        factor_plot.plot_round_factor(entry, factor_wide, data, cfg, png_path,
                                      round_no=round_no or 0, idx=1)
    except Exception:
        png_path = None
    return {"h5": h5_path, "png": png_path}
