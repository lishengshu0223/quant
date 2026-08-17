"""
本地化因子挖掘项目 - 静态 HTML 进度报告（factor_plot.py 拆分模块）

汇总所有轮次的评价图/公式(KaTeX渲染)/全库相关性 -> 静态 index.html。
KaTeX 资源优先本地 assets/katex 目录离线渲染, 缺失时回退 CDN。
"""

import datetime
import os

from . import factor_library
from .formula_tex import to_tex

KATEX_ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "katex")
KATEX_CDN = "https://cdn.jsdelivr.net/npm/katex/dist"


def _katex_base(out_dir: str) -> str:
    """返回 KaTeX 资源 URL 前缀: 本地相对路径(离线可用) 或 CDN 回退"""
    if os.path.isdir(KATEX_ASSETS):
        rel = os.path.relpath(KATEX_ASSETS, out_dir).replace("\\", "/")
        return rel
    return KATEX_CDN


HTML_HEAD = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>分钟因子挖掘进度报告</title>
<link rel="stylesheet" href="{katex}/katex.min.css">
<script defer src="{katex}/katex.min.js"></script>
<script defer src="{katex}/contrib/auto-render.min.js"
        onload="renderMathInElement(document.body,{{delimiters:[{{left:'$$',right:'$$',display:true}}]}});">
</script>
<style>
body{{font-family:"Microsoft YaHei",sans-serif;margin:24px auto;max-width:1180px;color:#2C3E50;}}
h1{{color:#2C3E50;border-bottom:3px solid #3498DB;padding-bottom:8px;}}
h2{{color:#3498DB;margin-top:36px;}}
h3{{color:#2C3E50;}}
.round{{border:1px solid #D5DBDB;border-radius:8px;padding:14px 18px;margin:14px 0;background:#FAFBFC;}}
.formula{{background:#F4F6F7;border-left:4px solid #3498DB;padding:10px 14px;margin:8px 0;
         font-size:1.05em;border-radius:4px;}}
.eval{{color:#7F8C8D;font-size:0.92em;}}
img{{max-width:100%;border:1px solid #D5DBDB;border-radius:6px;margin-top:6px;}}
table{{border-collapse:collapse;margin:10px 0;}}
th,td{{border:1px solid #BDC3C7;padding:5px 10px;font-size:0.92em;}}
th{{background:#3498DB;color:#fff;}}
.tag{{display:inline-block;padding:2px 10px;border-radius:10px;color:#fff;font-size:0.85em;font-weight:bold;}}
.ok{{background:#27AE60;}} .fail{{background:#E74C3C;}} .warn{{background:#F39C12;}}
.note{{color:#95A5A6;font-size:0.9em;}}
</style>
</head>
<body>
"""


def _ev_tag(ev: dict) -> str:
    if ev.get("error"):
        return '<span class="tag fail">计算失败</span>'
    if ev.get("qualified"):
        return '<span class="tag ok">合格</span>'
    if ev.get("review_rejected") or ev.get("library_rejected"):
        return '<span class="tag warn">拒绝入库</span>'
    if ev.get("direction_ok"):
        return '<span class="tag warn">方向正确未达标</span>'
    return '<span class="tag fail">方向无效</span>'


def _round_section(rec: dict, out_dir: str) -> str:
    round_no = rec.get("round")
    hyp = rec.get("hypothesis", "")
    parts = [f'<div class="round"><h3>第 {round_no} 轮</h3>']
    if hyp:
        parts.append(f'<p class="eval">假设: {hyp[:200]}</p>')
    for i, f in enumerate(rec.get("factors", []), 1):
        ev = f.get("eval") or {}
        parts.append(f"<h4>因子{i}: {f.get('name', '')} {_ev_tag(ev)}</h4>")
        expr = f.get("expr", "")
        if expr:
            parts.append(f'<div class="formula">$${to_tex(expr)}$$</div>')
        if ev.get("error"):
            parts.append(f'<p class="eval">错误: {ev["error"][:160]}</p>')
        else:
            stab = ev.get("long_stability") or {}
            parts.append(
                f'<p class="eval">IC均值 {ev.get("ic_mean", 0)*100:+.3f}% · '
                f'ICIR {ev.get("icir", 0):.3f} · 多头年化 '
                f'{ev.get("long_annual", 0)*100:+.2f}% · '
                f'月度正占比 {ev.get("monthly_pos_ratio", 0)*100:.0f}% · '
                f'年度信息比率 {stab.get("score") if stab.get("score") is not None else "-"}</p>')
        png = f.get("_png_rel")
        if png and os.path.exists(os.path.join(out_dir, png)):
            parts.append(f'<img src="{png}" alt="评价图">')
    parts.append("</div>")
    return "\n".join(parts)


def build_html_report(out_dir: str, rounds_meta: list, cfg,
                      corr_res: dict | None = None, library: list | None = None,
                      title: str = "分钟因子挖掘进度报告") -> str:
    """生成/覆盖 index.html: 标题 + 全库相关性 + 每轮因子(KaTeX公式+评价图)"""
    # 延迟导入: 避免与 factor_plot 循环依赖(factor_plot 顶层不依赖本模块的时机保障)
    from .factor_plot import plot_library_corr

    parts = [HTML_HEAD.format(katex=_katex_base(out_dir)), f"<h1>{title}</h1>"]
    parts.append(f'<p class="note">更新于 {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} · '
                 f'共 {len(rounds_meta)} 轮</p>')

    # 全库相关性
    library = factor_library.load_library() if library is None else library
    parts.append("<h2>因子库总览与相关性</h2>")
    if library:
        rows = []
        for s in library:
            best = s.get("best") or {}
            ev = best.get("eval") or {}
            rows.append(
                f"<tr><td>{s.get('series_id')}</td><td>{s.get('name','')}</td>"
                f"<td>{best.get('expr','')}</td>"
                f"<td>{ev.get('ic_mean',0)*100:+.3f}%</td>"
                f"<td>{ev.get('long_annual',0)*100:+.2f}%</td></tr>")
        parts.append("<table><tr><th>系列</th><th>名称</th><th>最佳公式</th>"
                     "<th>IC均值</th><th>多头年化</th></tr>" + "".join(rows) + "</table>")
    if corr_res and corr_res.get("matrix") is not None:
        mat = corr_res["matrix"]
        png_rel = "library_corr.png"
        if plot_library_corr(corr_res, os.path.join(out_dir, png_rel)):
            parts.append(f'<img src="{png_rel}" alt="全库相关性热图" '
                         f'style="max-width:640px;">')
        parts.append(f'<p class="eval">|ρ| 最大 {corr_res.get("max_abs", 0):.2f} '
                     f'(上限 {corr_res.get("threshold", 0.5)}) · '
                     f'{"撞车超标" if corr_res.get("flag") else "未超标"}</p>')
        corr_csv = os.path.join(out_dir, "library_corr.csv")
        mat.to_csv(corr_csv, encoding="utf-8-sig")

    # 各轮次
    parts.append("<h2>各轮因子评价</h2>")
    for rec in rounds_meta:
        parts.append(_round_section(rec, out_dir))

    parts.append("</body></html>")
    html = "\n".join(parts)
    out_html = os.path.join(out_dir, "index.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    return out_html
