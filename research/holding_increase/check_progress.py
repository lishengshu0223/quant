"""
数据获取进度监控脚本
"""
import os
import json
from datetime import datetime

DATA_PATH = r"F:\quant\research\holding_increase\increase_events"

print("=" * 60)
print("数据获取进度监控")
print("=" * 60)

# 统计已有文件
if os.path.exists(DATA_PATH):
    json_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.json')]
    
    print(f"\n📁 数据目录: {DATA_PATH}")
    print(f"📊 已获取事件数: {len(json_files)}")
    
    if json_files:
        # 分析日期范围
        dates = [f[:8] for f in json_files]
        min_date = min(dates)
        max_date = max(dates)
        print(f"📅 日期范围: {min_date} ~ {max_date}")
        
        # 按年份统计
        from collections import Counter
        years = [d[:4] for d in dates]
        year_counts = Counter(years)
        print("\n📆 按年份分布:")
        for year in sorted(year_counts.keys()):
            print(f"   {year}年: {year_counts[year]} 个事件")
        
        # 显示最近获取的文件
        print("\n🕐 最近获取的10个事件:")
        sorted_files = sorted(json_files, reverse=True)[:10]
        for f in sorted_files:
            filepath = os.path.join(DATA_PATH, f)
            try:
                with open(filepath, 'r', encoding='utf-8') as fp:
                    data = json.load(fp)
                date = data.get('公告日期', 'N/A')
                code = data.get('股票代码', 'N/A')
                ret_22 = data.get('22日收益率', 'N/A')
                if isinstance(ret_22, float):
                    ret_str = f"{ret_22:.4f}"
                else:
                    ret_str = str(ret_22)
                print(f"   {f}: {date} | {code} | 22日收益: {ret_str}")
            except:
                print(f"   {f}: (读取失败)")

else:
    print("❌ 数据目录不存在")

print("\n" + "=" * 60)
