"""
本地化因子挖掘项目 - 分钟数据加载与聚合实现（minute_engine 拆分模块③）

职责:
- MinuteExpr: 分钟表达式三维数据容器 (code_idx, date, t) 键 + 值数组
- 分钟数据加载: MinuteMarketData 分批流式读取 h5 / MinuteFieldCache 稠密内存缓存
- 长表聚合实现: _agg_series/_corr_grouped(按 code,date 分组 -> 日频宽表) 与 _ser_to_wide
- SlicedData: 批内股票日频数据视图(供日频子表达式在批内求值)

依赖: minute_parser(canonical) -> expr_engine; 不依赖稀疏/稠密求值器(循环依赖处用函数内延迟导入)。
"""

import os
import time

import h5py
import numpy as np
import pandas as pd

from .. import console
from ..expr_engine import ExprError
from .minute_parser import canonical

# 本地分钟数据目录(与 local_api.stock_minute 一致: rq_backtest_data/h5/equities)
EQ_MINUTE_DIR = r"F:\Trade_data\rq_backtest_data\h5\equities"


# =============================================================================
# 分钟表达式(三维数据)
# =============================================================================

class MinuteExpr:
    """
    分钟表达式: (code_idx, date, t) 三列整数键 + 值数组 v。
    code_idx: 当前批次内的股票编号; date: YYYYMMDD 整数; t: 当日秒数。
    """
    __slots__ = ("code", "date", "t", "v")

    def __init__(self, code, date, t, v):
        self.code = code
        self.date = date
        self.t = t
        self.v = v

    def aligned(self, other) -> bool:
        return (np.array_equal(self.code, other.code)
                and np.array_equal(self.date, other.date)
                and np.array_equal(self.t, other.t))

    def slice_time(self, s_sec: int, e_sec: int) -> "MinuteExpr":
        keep = (self.t >= s_sec) & (self.t <= e_sec)
        return MinuteExpr(self.code[keep], self.date[keep], self.t[keep], self.v[keep])

    def mask(self, op: str, threshold) -> "MinuteExpr":
        """只保留 x 满足 op 条件的分钟。threshold: 标量 | 行对齐数组 | MinuteExpr(行级比较)"""
        if isinstance(threshold, MinuteExpr):
            if not self.aligned(threshold):
                raise ExprError("MASK 的阈值分钟表达式与主表达式无法对齐(来自不同的分时/掩码结果)")
            th = threshold.v
        else:
            th = threshold
        v = self.v
        if op == ">":
            keep = v > th
        elif op == "<":
            keep = v < th
        elif op == ">=":
            keep = v >= th
        elif op == "<=":
            keep = v <= th
        elif op == "==":
            keep = v == th
        elif op == "!=":
            keep = v != th
        else:
            raise ExprError(f"MASK 比较符 {op} 不受支持")
        return MinuteExpr(self.code[keep], self.date[keep], self.t[keep], v[keep])


# =============================================================================
# 聚合实现(按 code,date 分组 -> 日频宽表)
# =============================================================================

def _mk_df(code, date, v, **cols):
    d = {"code": code, "date": date, "v": v}
    d.update(cols)
    return pd.DataFrame(d)


def _corr_grouped(code, date, x, y):
    df = _mk_df(code, date, x.astype(np.float64), y=y.astype(np.float64))
    df = df[df["v"].notna() & df["y"].notna()]
    if df.empty:
        return pd.Series(dtype=float)
    df["xx"] = df["v"] * df["v"]
    df["yy"] = df["y"] * df["y"]
    df["xy"] = df["v"] * df["y"]
    g = df.groupby(["code", "date"])
    n = g.size()
    sx, sy = g["v"].sum(), g["y"].sum()
    sxx, syy, sxy = g["xx"].sum(), g["yy"].sum(), g["xy"].sum()
    denom = np.sqrt((n * sxx - sx * sx) * (n * syy - sy * sy))
    corr = (n * sxy - sx * sy) / denom
    return corr.where((n >= 2) & (denom > 0)).replace([np.inf, -np.inf], np.nan)


def _agg_series(name, mv, extra):
    """对分钟表达式聚合, 返回 (code,date) MultiIndex Series"""
    code, date, v = mv.code, mv.date, mv.v

    if name in ("SUM", "MEAN", "STD", "MAX", "MIN", "MEDIAN", "SKEW", "QUANTILE"):
        df = _mk_df(code, date, v)
        g = df.groupby(["code", "date"])
        if name == "SUM":
            ser = g["v"].sum(min_count=1)
        elif name == "MEAN":
            ser = g["v"].mean()
        elif name == "STD":
            ser = g["v"].std()
        elif name == "MAX":
            ser = g["v"].max()
        elif name == "MIN":
            ser = g["v"].min()
        elif name == "MEDIAN":
            ser = g["v"].median()
        elif name == "SKEW":
            ser = g["v"].skew()
            ser = ser.where(g.size() >= 3)
        else:  # QUANTILE
            ser = g["v"].quantile(float(extra[0]))
        return ser

    if name == "KURT":
        df = _mk_df(code, date, v.astype(np.float64))
        df = df[df["v"].notna()]
        if df.empty:
            return pd.Series(dtype=float)
        df["mu"] = df.groupby(["code", "date"])["v"].transform("mean")
        df["d2"] = (df["v"] - df["mu"]) ** 2
        df["d4"] = df["d2"] ** 2
        g = df.groupby(["code", "date"])
        n = g.size().astype(np.float64)
        m2 = g["d2"].sum()
        m4 = g["d4"].sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            # 与 scipy.stats.kurtosis(bias=False) 一致的偏差校正超额峰度
            mu2 = m2 / n
            mu4 = m4 / n
            kurt = (n - 1.0) / ((n - 2.0) * (n - 3.0)) * ((n + 1.0) * (mu4 / mu2 ** 2.0) - 3.0 * (n - 1.0))
        return kurt.where((n >= 4) & (mu2 > 0)).replace([np.inf, -np.inf], np.nan)

    if name == "COUNT":
        df = _mk_df(code, date, np.where(np.isnan(v), 0.0, 1.0))
        return df.groupby(["code", "date"])["v"].sum()

    if name in ("LAST", "FIRST"):
        df = _mk_df(code, date, v, t=mv.t).sort_values(["code", "date", "t"])
        if name == "LAST":
            df["vv"] = df.groupby(["code", "date"])["v"].ffill()
        else:
            df["vv"] = df.groupby(["code", "date"])["v"].bfill()
        return df.groupby(["code", "date"])["vv"].last()

    if name == "CORR":
        y = extra[0]
        if not mv.aligned(y):
            raise ExprError("CORR 的两个分钟表达式无法对齐")
        return _corr_grouped(code, date, v, y.v)

    if name == "TS_AUTOCORR":
        lag = int(extra[0])
        df = _mk_df(code, date, v, t=mv.t).sort_values(["code", "date", "t"])
        df["vl"] = df.groupby(["code", "date"])["v"].shift(lag)
        return _corr_grouped(df["code"].to_numpy(), df["date"].to_numpy(),
                             df["v"].to_numpy(), df["vl"].to_numpy())

    if name in ("REGRESSION_SLOPE", "REGRESSION_INTERCEPT"):
        y = extra[0]
        if not mv.aligned(y):
            raise ExprError(f"{name} 的两个分钟表达式无法对齐")
        df = _mk_df(code, date, v.astype(np.float64), y=y.v.astype(np.float64))
        df = df[df["v"].notna() & df["y"].notna()]
        if df.empty:
            return pd.Series(dtype=float)
        df["yy"] = df["y"] * df["y"]
        df["xy"] = df["v"] * df["y"]
        g = df.groupby(["code", "date"])
        n = g.size()
        sx, sy = g["v"].sum(), g["y"].sum()
        syy, sxy = g["yy"].sum(), g["xy"].sum()
        denom = n * syy - sy * sy
        slope = (n * sxy - sx * sy) / denom
        with np.errstate(divide="ignore", invalid="ignore"):
            intercept = (sx - slope * sy) / n
        ok = (n >= 2) & (denom > 0)
        res = slope if name == "REGRESSION_SLOPE" else intercept
        return res.where(ok).replace([np.inf, -np.inf], np.nan)

    if name in ("TS_ARGMAX", "TS_ARGMIN"):
        df = _mk_df(code, date, v, t=mv.t)
        df["pos"] = df.groupby(["code", "date"]).cumcount()
        dfv = df[df["v"].notna()]
        if dfv.empty:
            return pd.Series(dtype=float)
        g = dfv.groupby(["code", "date"])
        idx = g["v"].idxmax() if name == "TS_ARGMAX" else g["v"].idxmin()
        # idx: index=(code,date) 分组键, 值为 dfv 内的行标签 -> 取出该行组内位置
        pos = dfv["pos"].reindex(idx.to_numpy())
        pos.index = idx.index
        n_all = df.groupby(["code", "date"]).size().rename("n")
        res_df = pos.rename("pos").to_frame().join(n_all, how="left")
        with np.errstate(divide="ignore", invalid="ignore"):
            res = res_df["pos"] / (res_df["n"] - 1)
        return res.where(res_df["n"] >= 2)

    raise ExprError(f"聚合算子 {name} 未实现")


def _ser_to_wide(ser: pd.Series, daily_index, batch_codes):
    """(code,date) Series -> 日频宽表(index=日期, columns=股票代码)"""
    if ser.empty:
        return pd.DataFrame(index=daily_index, columns=batch_codes, dtype="float32")
    df = ser.unstack("date")  # index=code_idx, columns=date_int
    df.columns = pd.to_datetime(df.columns.astype(str), format="%Y%m%d")
    df = df.T  # index=日期, columns=code_idx
    df.columns = [batch_codes[i] for i in df.columns]
    return df.reindex(index=daily_index, columns=batch_codes)


# =============================================================================
# 分钟字段稠密内存缓存(可选): 把常用字段一次性读入 [日×240×股] 稠密矩阵, worker 进程内跨因子复用
# =============================================================================

_MINUTE_MEMORY_CACHE = {}   # id(data) -> MinuteFieldCache


def _raw_field_name(fld: str) -> str:
    return "total_turnover" if fld == "amount" else fld


def _finalize(out: pd.DataFrame):
    """与日频引擎一致的收尾检查(公式计算结果的有效性与横截面离散度)"""
    out = out.astype("float64").replace([np.inf, -np.inf], np.nan)
    valid_ratio = float(out.notna().mean().mean())
    if valid_ratio < 0.3:
        raise ExprError(f"因子有效值比例过低({valid_ratio:.1%}), 公式可能无意义")
    cross_std = out.std(axis=1)
    if (cross_std.dropna() > 1e-12).mean() < 0.5:
        raise ExprError("因子横截面几乎无变化(所有股票取值相同), 无法分组")
    return out


def _get_memory_cache(mmd) -> "MinuteFieldCache | None":
    """按 cfg.minute_memory_fields 配置构建/复用与 data 绑定的分钟内存缓存(每进程仅构建一次)。"""
    cfg = mmd.cfg
    fields = [f.strip() for f in (getattr(cfg, "minute_memory_fields", "") or "").split(",") if f.strip()]
    if not fields:
        return None
    # turnover 是派生字段(分钟量/当日股本), 不入缓存; 其依赖的 volume 需要常驻内存
    raw_fields = frozenset({_raw_field_name(f) for f in fields if f != "turnover"})
    if "turnover" in fields:
        raw_fields = raw_fields | {"volume"}
    if not raw_fields:
        return None
    key = id(mmd.data)
    cache = _MINUTE_MEMORY_CACHE.get(key)
    if cache is not None and cache.fields == raw_fields:
        return cache
    console.log(f"    [分钟内存] 一次性读入全市场分钟字段 {sorted(raw_fields)} 到内存"
                f"(每字段约13.7GB, 约15-25分钟, 仅一次)...")
    t0 = time.time()
    cache = MinuteFieldCache(mmd, raw_fields)
    _MINUTE_MEMORY_CACHE[key] = cache
    console.log(f"    [分钟内存] 构建完成, 耗时 {time.time()-t0:.0f} 秒, "
                f"常驻字段 {sorted(raw_fields)}")
    return cache


class MinuteFieldCache:
    """分钟字段稠密缓存 [n_days × 240 × n_stocks] float32, 无成交位置为NaN。
    一个 worker 进程内共享, 后续因子计算直接从内存聚合, 不再重复读盘。"""

    def __init__(self, mmd, fields: frozenset):
        self.fields = fields
        self.day_ints = np.array([int(str(d.date()).replace("-", "")) for d in mmd.daily_index],
                                 dtype=np.int64)
        self.t_grid = np.concatenate([np.arange(34260, 41460, 60),
                                      np.arange(46860, 54060, 60)]).astype(np.int32)
        assert len(self.t_grid) == 240, "1分钟线一天应为240根"
        self.codes = mmd.codes_with_minute
        self.col_map = {c: i for i, c in enumerate(self.codes)}
        self.arrays = self._build(mmd)
        # $return 派生数组(与 close 同型, float32): _build 内用原始float64 close 逐股票构建,
        # 与长表 groupby(code,date)["close"].pct_change(fill_method=None) 数值一致(避免float32舍入放大)。
        self.ret_arr = self._ret if "close" in self.fields else None

    def _build(self, mmd) -> dict:
        n_days, n_stocks = len(self.day_ints), len(self.codes)
        arrays = {f: np.full((n_days, 240, n_stocks), np.nan, dtype=np.float32) for f in self.fields}
        self._ret = np.full((n_days, 240, n_stocks), np.nan, dtype=np.float32) if "close" in self.fields else None
        start_int, end_int = mmd.start_int, mmd.end_int
        t0 = time.time()
        for col, code in enumerate(self.codes):
            fp = os.path.join(EQ_MINUTE_DIR, f"{code}.h5")
            try:
                with h5py.File(fp, "r") as f:
                    idx = f["index"][:]
                    dset = f["data"]
                    mask = (idx["date"] >= start_int) & (idx["date"] <= end_int)
                    if not mask.any():
                        continue
                    sel = np.where(mask)[0]
                    s = int(idx["line_no"][sel[0]])
                    e = int(idx["line_no"][sel[-1] + 1]) if sel[-1] + 1 < len(idx) else dset.shape[0]
                    raw = dset[s:e]
            except Exception as exc:  # 单只股票读取失败, 跳过(不影响其它股票)
                console.log(f"    [分钟内存] 读取 {code} 失败: {exc}")
                continue
            dt = raw["datetime"]
            y = dt // 10000000000
            mo = (dt // 100000000) % 100
            d = (dt // 1000000) % 100
            hh = (dt // 10000) % 100
            mm = (dt // 100) % 100
            ss = dt % 100
            dates = y * 10000 + mo * 100 + d
            ts = hh * 3600 + mm * 60 + ss
            didx = np.searchsorted(self.day_ints, dates, side="right") - 1
            ok = (didx >= 0) & (self.day_ints[didx] == dates)
            # 上午 09:31~11:30 (t∈[34260,41400]) 为 0..119; 下午 13:01~15:00 (t∈[46860,54000]) 为 120..239
            tidx = np.where(ts < 46860, (ts - 34260) // 60, 120 + (ts - 46860) // 60)
            ok &= (tidx >= 0) & (tidx < 240) & (self.t_grid[tidx] == ts)
            didx = didx[ok]
            tidx = tidx[ok]
            if len(didx) == 0:
                continue
            for f in self.fields:
                arrays[f][didx, tidx, col] = raw[f][ok].astype(np.float32)
            if self._ret is not None:
                # 用原始float64 close 逐股票构建 $return(当日有效分钟序列内 pct_change, 跨日断):
                # 与长表 groupby(code,date)["close"].pct_change(fill_method=None) 一致
                o = np.lexsort((tidx, didx))
                s_d, s_t = didx[o], tidx[o]
                c = raw["close"][ok][o].astype(np.float64)
                r = np.full(len(c), np.nan)
                with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                    r[1:] = np.where(s_d[1:] == s_d[:-1], (c[1:] - c[:-1]) / c[:-1], np.nan)
                self._ret[s_d, s_t, col] = r.astype(np.float32)
            if (col + 1) % 1000 == 0:
                console.log(f"    [分钟内存] 已读入 {col+1}/{n_stocks} 只, 耗时 {time.time()-t0:.0f} 秒")
        return arrays


# =============================================================================
# 分钟数据加载与分批求值
# =============================================================================

class SlicedData:
    """按批内股票切片后的日频数据视图(供日频子表达式在批内求值)"""

    def __init__(self, data, codes):
        self.data = data
        self.codes = codes

    def var(self, name):
        return self.data.var(name).reindex(columns=self.codes)


class MinuteMarketData:
    """全市场分钟数据分批加载器 + 聚合节点计算"""

    def __init__(self, cfg, data, fields):
        self.cfg = cfg
        self.data = data
        # $return 由 $close 的分钟环比计算, 需要额外加载 close
        if "return" in fields and "close" not in fields:
            fields = list(fields) + ["close"]
        # $turnover(分钟换手率) 由 分钟成交量/当日总股本 派生, 需要额外加载 volume
        if "turnover" in fields and "volume" not in fields:
            fields = list(fields) + ["volume"]
        self.fields = [f for f in fields if f != "minute"]
        self.daily_index = data.close.index
        self.all_codes = list(data.close.columns)
        self.start_int = int(cfg.data_start_date.replace("-", ""))
        self.end_int = int(str(data.close.index.max().date()).replace("-", ""))
        self.batch_size = int(getattr(cfg, "minute_batch_size", 35))
        self.batch_codes = []
        # 只有存在分钟文件的股票才进入批次; 其余股票因子恒为NaN
        self.codes_with_minute = [c for c in self.all_codes
                                  if os.path.exists(os.path.join(EQ_MINUTE_DIR, f"{c}.h5"))]
        self.batches = [self.codes_with_minute[i:i + self.batch_size]
                        for i in range(0, len(self.codes_with_minute), self.batch_size)]
        # A股流通股本(股)日频宽表(列=all_codes)与交易日整数序列: 供分钟换手率派生。
        # 经验证米筐 today 换手率 = 成交量 / circulation_a * 100, 分钟换手率分母取流通股本
        self.share_full = data.circulation
        self.day_ints = np.array([int(str(d.date()).replace("-", "")) for d in self.daily_index],
                                 dtype=np.int64)

    def _calc_turnover(self, df, batch_codes) -> pd.Series:
        """分钟换手率(%) = 该分钟成交量 / 当日A股流通股本 * 100; 股本来自本地 get_shares"""
        share = self.share_full.reindex(columns=batch_codes).to_numpy(dtype=np.float64)
        day_pos = np.searchsorted(self.day_ints, df["date"].to_numpy())
        share_v = share[day_pos, df["code"].to_numpy()]
        vol = df["volume"].to_numpy(dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            return pd.Series((vol / share_v * 100.0).astype(np.float32), index=df.index)

    def _load_batch(self, batch_codes, start_int=None, end_int=None):
        """加载一批股票的分钟长表(批内 code_idx 编码, 已按 code,date,t 排序)。
        start_int/end_int: 可选, 只加载该日期整数区间(YYYYMMDD); 默认全区间。
        若公式所需字段已常驻内存(稠密缓存), 直接从内存构建, 避免重复读盘。"""
        cache = _get_memory_cache(self)
        if cache is not None:
            needed = {_raw_field_name(f) for f in self.fields if f not in ("return", "turnover")}
            if "turnover" in self.fields:
                needed.add("volume")  # 派生字段 turnover 依赖 volume
            if needed <= cache.fields:
                return self._load_batch_from_memory(batch_codes, cache, start_int, end_int)
        if start_int is None:
            start_int = self.start_int
        if end_int is None:
            end_int = self.end_int
        rows = []
        for ci, code in enumerate(batch_codes):
            fp = os.path.join(EQ_MINUTE_DIR, f"{code}.h5")
            if not os.path.exists(fp):
                continue
            try:
                with h5py.File(fp, "r") as f:
                    idx = f["index"][:]
                    dset = f["data"]
                    mask = (idx["date"] >= start_int) & (idx["date"] <= end_int)
                    if not mask.any():
                        continue
                    sel = np.where(mask)[0]
                    s = int(idx["line_no"][sel[0]])
                    e = int(idx["line_no"][sel[-1] + 1]) if sel[-1] + 1 < len(idx) else dset.shape[0]
                    raw = dset[s:e]
            except Exception as exc:  # 单只股票读取失败, 跳过(不影响其它股票)
                console.log(f"    [分钟数据] 读取 {code} 失败: {exc}")
                continue
            dt = raw["datetime"]
            y = dt // 10000000000
            mo = (dt // 100000000) % 100
            d = (dt // 1000000) % 100
            hh = (dt // 10000) % 100
            mm = (dt // 100) % 100
            ss = dt % 100
            rec = {
                "code": np.full(len(raw), ci, dtype=np.int32),
                "date": (y * 10000 + mo * 100 + d).astype(np.int32),
                "t": (hh * 3600 + mm * 60 + ss).astype(np.int32),
            }
            for fld in self.fields:
                if fld in ("return", "turnover"):
                    continue  # 派生字段在拼接后统一计算
                raw_fld = "total_turnover" if fld == "amount" else fld
                rec[raw_fld] = raw[raw_fld].astype(np.float32)
            rows.append(rec)
        if not rows:
            return None
        data = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
        # 每只股票的文件内已按(date,t)有序、批内按 batch_codes 顺序追加 -> 拼接后天然按(code,date,t)有序,
        # 无需全局 sort_values(mergesort 2.8亿行会额外占用数GB内存, 也是此前批次过大时MemoryError的来源)
        df = pd.DataFrame(data)
        if "return" in self.fields:
            if "close" not in self.fields:
                raise ExprError("分钟 $return 依赖 $close, 请同时使用 $close")
            df["return"] = df.groupby(["code", "date"])["close"].pct_change(fill_method=None)
        if "turnover" in self.fields:
            df["turnover"] = self._calc_turnover(df, batch_codes)
        return df

    def _load_batch_from_memory(self, batch_codes, cache, start_int=None, end_int=None):
        """从稠密内存缓存构建批次长表(与磁盘路径输出完全一致: code/date/t 有序 + 字段列)"""
        if start_int is None:
            start_int = self.start_int
        if end_int is None:
            end_int = self.end_int
        lo = np.searchsorted(cache.day_ints, start_int)
        hi = np.searchsorted(cache.day_ints, end_int, side="right")
        valid_field = "close" if "close" in cache.fields else sorted(cache.fields)[0]
        rows = []
        for ci, code in enumerate(batch_codes):
            col = cache.col_map.get(code)
            if col is None:
                continue
            day_idx, t_idx = np.nonzero(np.isfinite(cache.arrays[valid_field][lo:hi, :, col]))
            if len(day_idx) == 0:
                continue
            day_idx = day_idx + lo
            rec = {
                "code": np.full(len(day_idx), ci, dtype=np.int32),
                "date": cache.day_ints[day_idx].astype(np.int32),
                "t": cache.t_grid[t_idx].astype(np.int32),
            }
            for fld in self.fields:
                if fld in ("return", "turnover"):
                    continue
                raw = _raw_field_name(fld)
                rec[raw] = cache.arrays[raw][day_idx, t_idx, col].astype(np.float32)
            rows.append(rec)
        if not rows:
            return None
        data = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
        df = pd.DataFrame(data)
        if "return" in self.fields:
            df["return"] = df.groupby(["code", "date"])["close"].pct_change(fill_method=None)
        if "turnover" in self.fields:
            df["turnover"] = self._calc_turnover(df, batch_codes)
        return df

    def field_of(self, df, name):
        """从批次长表取分钟字段, 包装为 MinuteExpr"""
        code = df["code"].to_numpy()
        date = df["date"].to_numpy()
        t = df["t"].to_numpy()
        if name == "$minute":
            return MinuteExpr(code, date, t, t.astype(np.float32))
        fld = name[1:]
        if fld == "amount":
            fld = "total_turnover"
        if fld not in df.columns:
            raise ExprError(f"分钟字段 {name} 不可用(本批数据未加载)")
        return MinuteExpr(code, date, t, df[fld].to_numpy(dtype=np.float32))

    def aggregate(self, name, evals):
        """对一个批次聚合节点参数求值并聚合 -> 批内日频宽表(列=批内股票代码)"""
        ser = _agg_series(name, evals[0], evals[1:])
        return _ser_to_wide(ser, self.daily_index, self.batch_codes)

    def compute_all_aggs(self, agg_nodes, types, data, cfg):
        """单趟遍历全部批次, 一次计算所有(最大)聚合节点的全市场日频宽表。
        返回 {canonical(node): 宽表}。相比逐节点遍历, 每个批次只加载一次分钟数据,
        一次算出全部聚合节点(大幅降低重复读取)。"""
        from .minute_sparse_eval import _BatchRunner   # 延迟导入避免与稀疏求值器循环依赖
        t0 = time.time()
        keys, seen = [], set()
        for node in agg_nodes:
            k = canonical(node)
            if k not in seen:
                seen.add(k)
                keys.append(k)
        parts = {k: [] for k in keys}
        n_batch = 0
        for batch_codes in self.batches:
            df = self._load_batch(batch_codes)
            if df is None:
                continue
            self.batch_codes = batch_codes
            try:
                runner = _BatchRunner(self, data, cfg, types)
                runner.set_batch(df, batch_codes)
                done = set()   # 同规范字符串的聚合节点(公式中重复出现)只算一次, 避免宽表列重复
                for node in agg_nodes:
                    k = canonical(node)
                    if k in done:
                        continue
                    done.add(k)
                    evals = [runner.eval_arg(a) for a in node.args]
                    parts[k].append(self.aggregate(node.name, evals))
            finally:
                del df
            n_batch += 1
        if not n_batch:
            raise ExprError("分钟数据为空, 无法计算聚合节点")
        cache = {}
        for k in keys:
            cache[k] = pd.concat(parts[k], axis=1).reindex(
                index=self.daily_index, columns=self.all_codes)
        console.log(f"    [分钟聚合] {len(keys)} 个聚合节点单趟遍历完成, "
                    f"{n_batch} 个批次, 耗时 {time.time()-t0:.1f} 秒")
        return cache

    def compute_all_aggs_full_market(self, agg_nodes, types, data, cfg):
        """全市场分钟截面模式: 按日期分块, 每块加载全部股票的分钟数据, 一次计算所有聚合节点。
        块内包含全部股票, 因此分钟级截面算子(RANK/ZSCORE/SCALE/CS_*)在块内即等价于全市场截面。
        每块内存 ≈ 块天数 × 240 × 全部股票数, 用 config.minute_chunk_days 控制(默认100天)。"""
        from .minute_sparse_eval import _BatchRunner   # 延迟导入避免与稀疏求值器循环依赖
        t0 = time.time()
        chunk_days = int(getattr(cfg, "minute_chunk_days", 100))
        dates = pd.Series(self.daily_index)
        chunks = [dates.iloc[i:i + chunk_days] for i in range(0, len(dates), chunk_days)]
        keys, seen = [], set()
        for node in agg_nodes:
            k = canonical(node)
            if k not in seen:
                seen.add(k)
                keys.append(k)
        parts = {k: [] for k in keys}
        all_codes = self.codes_with_minute
        n_chunk = 0
        for chunk in chunks:
            s_int = int(str(chunk.iloc[0].date()).replace("-", ""))
            e_int = int(str(chunk.iloc[-1].date()).replace("-", ""))
            df = self._load_batch(all_codes, s_int, e_int)
            if df is None:
                continue
            self.batch_codes = all_codes
            try:
                runner = _BatchRunner(self, data, cfg, types)
                runner.set_batch(df, all_codes)
                done = set()   # 同规范字符串的聚合节点只算一次, 避免宽表列重复
                for node in agg_nodes:
                    k = canonical(node)
                    if k in done:
                        continue
                    done.add(k)
                    evals = [runner.eval_arg(a) for a in node.args]
                    wide = self.aggregate(node.name, evals)
                    parts[k].append(wide.reindex(chunk.tolist()))
            finally:
                del df
            n_chunk += 1
        if not n_chunk:
            raise ExprError("分钟数据为空, 无法计算聚合节点")
        cache = {}
        for k in keys:
            cache[k] = pd.concat(parts[k], axis=0).reindex(
                index=self.daily_index, columns=self.all_codes)
        console.log(f"    [分钟聚合·全市场] {len(keys)} 个聚合节点(含截面算子)按日期分块完成, "
                    f"{n_chunk} 个分块, 耗时 {time.time()-t0:.1f} 秒")
        return cache
