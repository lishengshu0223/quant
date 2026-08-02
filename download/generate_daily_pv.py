# -*- coding: utf-8 -*-
"""
从本地 parquet 日线行情数据生成 QuantaAlpha 所需的 daily_pv.h5

数据源: F:\\Trade_data\\stock_price\\ 下按交易日存放的 parquet 文件
       （每个文件为当日全市场行情，MultiIndex 为 (date, code)）
目标格式: MultiIndex 为 (instrument, datetime)，
         列为 $open/$close/$high/$low/$volume/$return（复权价格）
"""
import os
import time

import pandas as pd

# 数据源目录与输出路径
SRC_DIR = r"F:\Trade_data\stock_price"
OUT_PATH = r"f:\quant\llm_mining\reference\QuantaAlpha\daily_pv.h5"

# 源列 -> 目标列 映射（使用复权价格）
COL_MAP = {
    "adjopen": "$open",
    "adjclose": "$close",
    "adjhigh": "$high",
    "adjlow": "$low",
    "adjvolume": "$volume",
}


def main():
    start = time.time()

    # 获取所有 parquet 文件并按日期排序
    files = sorted(f for f in os.listdir(SRC_DIR) if f.endswith(".parquet"))
    total = len(files)
    print(f"待处理 parquet 文件数量: {total}")

    # 逐个读取并拼接
    dfs = []
    for i, fname in enumerate(files, 1):
        df = pd.read_parquet(os.path.join(SRC_DIR, fname))
        # 源数据 MultiIndex 为 (date, code)，重置为普通列
        df = df.reset_index()
        df = df.rename(columns={"date": "datetime", "code": "instrument"})
        # 只保留目标列并重命名
        df = df[["instrument", "datetime"] + list(COL_MAP.keys())]
        df = df.rename(columns=COL_MAP)
        dfs.append(df)

        # 每 500 个文件打印一次进度
        if i % 500 == 0 or i == total:
            print(f"已读取 {i}/{total} 个文件")

    data = pd.concat(dfs, ignore_index=True)
    del dfs

    # 设置 MultiIndex (instrument, datetime) 并排序
    data["datetime"] = pd.to_datetime(data["datetime"])
    data = data.set_index(["instrument", "datetime"]).sort_index()

    # 按标的计算日收益率（复权收盘价的日度百分比变化）
    data["$return"] = data.groupby(level=0)["$close"].pct_change().fillna(0)

    print(f"数据形状: {data.shape}")
    print(f"日期范围: {data.index.get_level_values('datetime').min()} ~ "
          f"{data.index.get_level_values('datetime').max()}")
    print(f"标的数量: {data.index.get_level_values('instrument').nunique()}")

    # 保存为 HDF5
    data.to_hdf(OUT_PATH, key="data")
    print(f"已保存: {OUT_PATH}")

    # 读回验证
    check = pd.read_hdf(OUT_PATH)
    print("\n===== 验证读回结果 =====")
    print(f"形状: {check.shape}")
    print(f"索引: {check.index.names}")
    print(f"列: {check.columns.tolist()}")
    print("前 5 行:")
    print(check.head())
    print("后 5 行:")
    print(check.tail())

    print(f"\n总耗时: {time.time() - start:.1f} 秒")


if __name__ == "__main__":
    main()
