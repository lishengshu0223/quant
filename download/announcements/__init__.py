"""公告族子包：巨潮公告下载、分类映射、缓存回填与维护工具

模块分工：
    announcements.py            下载核心
    announcement_type_map.py    巨潮/证监会静态分类映射表（纯静态查表）
    announcement_field_map.py   公告类别 → 必有/可能字段映射（LLM 提示词用）
    type_category_map.py        type 段集合 → 巨潮官方 26 类静态解码
    reclassify_announcements.py 用最新映射表重算已下载文件的分类列
    apply_category_cache.py     把本地 category 缓存回填到已下载文件
    generate_category_cache.py  按月生成 category 缓存
    scan_all_codes.py           全量扫描公告类型码（映射表维护工具）
    sample_aux.py               抽样公告标题（映射表维护工具）

统一导出下载与解码接口，供 update/ 与 download/ 上层调用。
"""
from .announcements import (
    download_announcements,
    download_recent_announcements,
)

__all__ = [
    "download_announcements",
    "download_recent_announcements",
]