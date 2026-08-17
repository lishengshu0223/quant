"""
本地化因子挖掘项目 - 分钟稠密归约 numba 内核（minute_engine 拆分模块②）

numba 可选加速: 融合循环、零中间3D数组、多线程。未安装时自动回退 numpy 实现。
与 numpy 版算法完全一致(float32 输入 + float64 累加, NaN 按无效处理),
输出 [日, 股] float64(由调用方转 float32)。数值一致性已实测到 1e-7。
注意: 不能用 fastmath=True(LLVM 假设无 NaN 会把 np.isnan 判断优化掉)。
"""

import numpy as np

try:
    from numba import njit, prange
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False


@njit(parallel=True, cache=True)
def _nb_skew(x):
    """SKEW: Fisher-Pearson 修正偏度(与 pandas .skew() 一致)"""
    D_, M_, S_ = x.shape
    out = np.empty((D_, S_), dtype=np.float64)
    for d in prange(D_):
        n = np.zeros(S_, dtype=np.int64)
        s1 = np.zeros(S_)
        for i in range(M_):
            row = x[d, i]
            for s in range(S_):
                v = row[s]
                if not np.isnan(v):
                    n[s] += 1
                    s1[s] += v
        d2s = np.zeros(S_)
        d3s = np.zeros(S_)
        for i in range(M_):
            row = x[d, i]
            for s in range(S_):
                v = row[s]
                if not np.isnan(v):
                    dv = v - s1[s] / n[s]
                    d2s[s] += dv * dv
                    d3s[s] += dv * dv * dv
        for s in range(S_):
            n_ = n[s]
            if n_ >= 3 and d2s[s] > 0:
                out[d, s] = n_ * np.sqrt(n_ - 1) * d3s[s] / ((n_ - 2) * d2s[s] ** 1.5)
            else:
                out[d, s] = np.nan
    return out


@njit(parallel=True, cache=True)
def _nb_kurt(x):
    """KURT: 超额峰度(与 pandas .kurt() 一致)"""
    D_, M_, S_ = x.shape
    out = np.empty((D_, S_), dtype=np.float64)
    for d in prange(D_):
        n = np.zeros(S_, dtype=np.int64)
        s1 = np.zeros(S_)
        for i in range(M_):
            row = x[d, i]
            for s in range(S_):
                v = row[s]
                if not np.isnan(v):
                    n[s] += 1
                    s1[s] += v
        d2s = np.zeros(S_)
        d4s = np.zeros(S_)
        for i in range(M_):
            row = x[d, i]
            for s in range(S_):
                v = row[s]
                if not np.isnan(v):
                    dv = v - s1[s] / n[s]
                    d2s[s] += dv * dv
                    d4s[s] += dv * dv * dv * dv
        for s in range(S_):
            n_ = n[s]
            mu2 = d2s[s] / n_
            if n_ >= 4 and mu2 > 0:
                mu4 = d4s[s] / n_
                out[d, s] = (n_ - 1.0) / ((n_ - 2.0) * (n_ - 3.0)) * (
                    (n_ + 1.0) * (mu4 / mu2 ** 2.0) - 3.0 * (n_ - 1.0))
            else:
                out[d, s] = np.nan
    return out


@njit(parallel=True, cache=True)
def _nb_std(x):
    """STD: 样本标准差(与 pandas .std() 一致, ddof=1)"""
    D_, M_, S_ = x.shape
    out = np.empty((D_, S_), dtype=np.float64)
    for d in prange(D_):
        n = np.zeros(S_, dtype=np.int64)
        s1 = np.zeros(S_)
        for i in range(M_):
            row = x[d, i]
            for s in range(S_):
                v = row[s]
                if not np.isnan(v):
                    n[s] += 1
                    s1[s] += v
        ss = np.zeros(S_)
        for i in range(M_):
            row = x[d, i]
            for s in range(S_):
                v = row[s]
                if not np.isnan(v):
                    dv = v - s1[s] / n[s]
                    ss[s] += dv * dv
        for s in range(S_):
            n_ = n[s]
            if n_ >= 2:
                out[d, s] = np.sqrt(ss[s] / (n_ - 1))
            else:
                out[d, s] = np.nan
    return out


@njit(parallel=True, cache=True)
def _nb_regression(v, y, want):
    """REGRESSION_SLOPE/INTERCEPT: 一元回归 y~v, want=0 斜率, want=1 截距;
    y 为常数时方差为0 -> NaN(与 pandas 一致)"""
    D_, M_, S_ = v.shape
    slope = np.empty((D_, S_), dtype=np.float64)
    inter = np.empty((D_, S_), dtype=np.float64)
    for d in prange(D_):
        n = np.zeros(S_, dtype=np.int64)
        sv = np.zeros(S_)
        sy = np.zeros(S_)
        for i in range(M_):
            rv = v[d, i]
            ry = y[d, i]
            for s in range(S_):
                a = rv[s]
                b = ry[s]
                if not np.isnan(a) and not np.isnan(b):
                    n[s] += 1
                    sv[s] += a
                    sy[s] += b
        vary = np.zeros(S_)
        cov = np.zeros(S_)
        for i in range(M_):
            rv = v[d, i]
            ry = y[d, i]
            for s in range(S_):
                a = rv[s]
                b = ry[s]
                if not np.isnan(a) and not np.isnan(b):
                    da = a - sv[s] / n[s]
                    db = b - sy[s] / n[s]
                    vary[s] += db * db
                    cov[s] += da * db
        for s in range(S_):
            n_ = n[s]
            if n_ >= 2 and vary[s] > 0:
                sl = cov[s] / vary[s]
                slope[d, s] = sl
                inter[d, s] = sv[s] / n_ - sl * (sy[s] / n_)
            else:
                slope[d, s] = np.nan
                inter[d, s] = np.nan
    return slope if want == 0 else inter


@njit(parallel=True, cache=True)
def _nb_corr(x, y):
    """CORR: 皮尔逊相关(两趟均值中心化; 常数序列方差为0 -> NaN)"""
    D_, M_, S_ = x.shape
    out = np.empty((D_, S_), dtype=np.float64)
    for d in prange(D_):
        n = np.zeros(S_, dtype=np.int64)
        sx = np.zeros(S_)
        sy = np.zeros(S_)
        for i in range(M_):
            rx = x[d, i]
            ry = y[d, i]
            for s in range(S_):
                a = rx[s]
                b = ry[s]
                if not np.isnan(a) and not np.isnan(b):
                    n[s] += 1
                    sx[s] += a
                    sy[s] += b
        varx = np.zeros(S_)
        vary = np.zeros(S_)
        cov = np.zeros(S_)
        for i in range(M_):
            rx = x[d, i]
            ry = y[d, i]
            for s in range(S_):
                a = rx[s]
                b = ry[s]
                if not np.isnan(a) and not np.isnan(b):
                    da = a - sx[s] / n[s]
                    db = b - sy[s] / n[s]
                    varx[s] += da * da
                    vary[s] += db * db
                    cov[s] += da * db
        for s in range(S_):
            n_ = n[s]
            if n_ >= 2 and varx[s] > 0 and vary[s] > 0:
                out[d, s] = cov[s] / np.sqrt(varx[s] * vary[s])
            else:
                out[d, s] = np.nan
    return out


@njit(parallel=True, cache=True)
def _nb_autocorr(x, lag):
    """TS_AUTOCORR: 按'当日有效分钟序列'滞后 lag 的自相关
    (跨停牌分钟跳号, 与长表 groupby.shift(lag) 后 corr 一致)"""
    D_, M_, S_ = x.shape
    out = np.empty((D_, S_), dtype=np.float64)
    for d in prange(D_):
        buf = np.empty(M_, dtype=np.float64)
        for s in range(S_):
            k = 0
            for i in range(M_):
                v = x[d, i, s]
                if not np.isnan(v):
                    buf[k] = v
                    k += 1
            if k < lag + 2:
                out[d, s] = np.nan
                continue
            n_ = k - lag
            s1 = 0.0
            s2 = 0.0
            for i in range(lag, k):
                s1 += buf[i]
                s2 += buf[i - lag]
            mu1 = s1 / n_
            mu2 = s2 / n_
            vv1 = 0.0
            vv2 = 0.0
            cov = 0.0
            for i in range(lag, k):
                a = buf[i] - mu1
                b = buf[i - lag] - mu2
                vv1 += a * a
                vv2 += b * b
                cov += a * b
            if vv1 > 0 and vv2 > 0:
                out[d, s] = cov / np.sqrt(vv1 * vv2)
            else:
                out[d, s] = np.nan
    return out


@njit(parallel=True, cache=True)
def _nb_last(x):
    """LAST/FIRST: 取当日最后一个有值分钟的值(与长表一致)"""
    D_, M_, S_ = x.shape
    out = np.empty((D_, S_), dtype=np.float64)
    for d in prange(D_):
        for s in range(S_):
            val = np.nan
            for i in range(M_ - 1, -1, -1):
                v = x[d, i, s]
                if not np.isnan(v):
                    val = v
                    break
            out[d, s] = val
    return out
