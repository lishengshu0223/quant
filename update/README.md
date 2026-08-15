# 每日数据更新说明

> 本文档汇总本项目**每日自动更新**的调度时间、脚本与内容，供快速了解。
> 更新日志统一写入 `F:\Trade_data\logs\download_YYYYMMDD.log`（run_update.bat 另有步骤分隔记录）。

---

## 一、总览（三个计划任务）

| 计划任务 | 触发时间 | 入口脚本 | 更新内容 | 依赖 |
| --- | --- | --- | --- | --- |
| `米筐行情下载` | 每日 **20:00** | [run_update.bat](../update/run_update.bat) | 量价数据 + 回测 bundle | rqdatac（米筐） |
| `Quant\Announcement_0030` | 每日 **00:30** | [update_announcements.py](../update/update_announcements.py) | 公告隔夜补全（头一天） | 巨潮资讯网（无 rqdatac） |
| `Quant\Announcement_0850` | 每日 **08:50** | [update_announcements.py](../update/update_announcements.py) | 公告当天早间档 | 巨潮资讯网（无 rqdatac） |

三个任务相互独立、互不阻塞；公告任务按**自然日**执行（含周末/节假日），量价任务按**交易日**生效（20:00 运行时自动定位最新交易日）。

---

## 二、量价更新（每日 20:00，`米筐行情下载`）

入口 [run_update.bat](../update/run_update.bat)，依次执行 5 步：

| 步骤 | 脚本/命令 | 内容 | 说明 |
| --- | --- | --- | --- |
| Step 1 | [update_latest.py](../update/update_latest.py) | **强制更新最新交易日**：股票日线、指数日线、指数权重、Barra 风险因子、换手率、总股本、上市状态 | 17 点后运行则以当日为最新交易日，否则回退到前一交易日 |
| Step 2 | [update_history.py](../update/update_history.py) | **补全历史缺失**：2016-01-01 至今日的日线/权重/Barra/换手率/总股本/上市状态 | 自动跳过已有数据，只补缺口 |
| Step 3 | `rqsdk update-data --base` | 回测 bundle 基础数据 | 增量更新 |
| Step 4 | `rqsdk update-data --minbar stock -c 4` | 回测 bundle 股票分钟线 | 耗时较长 |
| Step 5 | `rqsdk update-data --minbar futures -c 4` | 回测 bundle 期货分钟线 | 耗时较长 |

- **数据落盘**：`F:\Trade_data\`（股票/指数/权重/Barra/上市状态等按各自子目录分日存储）；回测 bundle 在 `F:\Trade_data\rq_backtest_data`。
- **换手率**：由 `rqdatac.get_turnover_rate` 的 `today` 字段（当日换手率，单位 %）生成，按交易日存于 `F:\Trade_data\turnover\YYYYMMDD.parquet`（索引 `(date, code)`，列 `turnover_rate`）。
- **总股本**：由 `rqdatac.get_shares` 生成，字段与米筐一致（`total` 总股本、`free_circulation` 自由流通股本等 6 列，单位股），按交易日存于 `F:\Trade_data\shares\YYYYMMDD.parquet`（索引 `(date, code)`）。
- **指数权重/价格代码口径**：`download/config.py` 的 `INDEX_FULL_CODES` 统一使用 rqdatac 合法代码——中证2000 为 `932000.INDX`（**不是** `932000.CSI`，rqdatac 不接受该后缀）；`INDEX_WEIGHT_CODES` 已含 932000，Step 1/Step 2 会随指数权重一起下载。932000 权重自中证2000 发布（2023-08）后才有，历史空窗期自动跳过；价格回溯至 2016-01-04。
- **注意**：公告下载已从本链路**拆出**（见下节），量价更新不再触碰公告。

---

## 三、公告更新（00:30 + 08:50，`Quant\Announcement_*`）

### 为什么是两个时点

A 股公告按**披露日**归档（巨潮 `seDate`），每日数据分三批到齐：

| 批次 | 披露时点 | 含义 |
| --- | --- | --- |
| 00:00 档 | 当日 0 点统一上架 | **前一交易日盘后提交、选"次日披露"**的公告（隔夜大部队，约占 87%） |
| 早间档 | 7:30–8:30 | 当日早间披露（沪市早间时段，深市 7:30–8:00 提交实时发送） |
| 盘后档 | 17:00–20:00 | 当日盘后披露（主高峰 17–19 点） |

因此：

- **00:30 任务**（每天）：强制重下最近 3 个自然日 → 主职是**补全头一天**：头一天 00:00 档（= 前两日晚上的公告）+ 头一天 17–20 点盘后公告 + 延迟补录；
- **08:50 任务**（每天）：抓**当天早间档**（7:30–8:30 披露 + 沪市非直通最晚 8:40 确认发布），并再重下最近 3 天兜底。

两趟跑完，**T 日收盘后至 T+1 日开盘前的全部公告**（= T+1 日文件）即齐，早于 9:30 开盘足够策略使用。

### 脚本流程（[update_announcements.py](../update/update_announcements.py)，幂等可重复执行）

1. 强制重下最近 3 个自然日（`download_recent_announcements(days=3)`，捕获延迟/补录）；
2. 用最新映射表重算最近 7 天的分类列（`reclassify_files`，cninfo_*/csrc_* 体系）；
3. 确保 type→category 静态映射存在（`ensure_map`，缺失时自动从已回填行学习）；
4. 按 `type_codes` 解码回填 `category` 列（`decode_apply_and_learn`，巨潮官方 26 类）；映射未覆盖的 type 串**自动调用巨潮 26 类查询补学**并更新映射（默认开启，`--no-learn-api` 关闭），保证遇到当前分类之外的代码编号也能自动解码入库。

- **category 解码原理**：巨潮公告的 `announcementType` 细码串唯一确定官方 26 类（实测单日 881/881 一致），映射表 `F:\Trade_data\announcements\type_category_map.parquet`（14,675 种），详见 [type_category_map.py](../download/announcements/type_category_map.py)；无需逐月查询 26 类接口，也不受巨潮单次查询 3000 条上限影响。
- **数据落盘**：`F:\Trade_data\announcements\YYYYMMDD.parquet`（披露日 = 文件名日期）。
- **分类体系**：详见 `.trae/rules/announcement_classification.md`。

---

## 四、日志与排障

| 内容 | 位置 |
| --- | --- |
| 每日更新日志（含公告 slot 标识、下载/失败明细） | `F:\Trade_data\logs\download_YYYYMMDD.log` |
| 公告分类映射表（新增巨潮码需人工补充） | `download/announcements/announcement_type_map.py` |
| 计划任务管理 | `schtasks /Query /TN "任务名" /V` |

排障要点：
- 公告任务失败重跑即可（脚本幂等）；量价任务失败重跑 `run_update.bat`（自动跳过已完成部分）。
- 当日公告"当日全量"数在盘后会增长，属正常（盘后档要等次日 00:30 补全）。

---

## 五、当前状态（2026-08 核查）

- [x] 公告历史数据：2016-01-01 ~ 2026-08-06 共 3871 个自然日零缺口；
- [x] 巨潮码映射表 447 个码，缺失 0；
- [x] category 列（巨潮官方 26 类）已全量回填：按 `type_codes` 静态解码补齐 2,486,079 条（占空行的 99.73%），仅剩少量 type 串映射缺失的行待 API 补学（2026-08-13 已启动串行补学，预计数小时）；
- [x] 每日更新（00:30 / 08:50）已接入自动补全：新公告的 category 由 `type_category_map` 静态解码，映射未覆盖的 type 串自动调用巨潮 26 类查询补学并更新映射（详见 `download/announcements/type_category_map.py`）。
