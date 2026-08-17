"""
本地化因子挖掘项目 - 挖掘循环编排器(自动挖到目标数量的合格因子)

职责(循环往复, 无需人工值守):
1. 目标: 因子库 status=qualified 的因子数量达到 --target-count(默认50)后,
   停止全部子进程并退出。
2. 每个因子系列有累计 --rounds-per-series(默认100)轮优化预算:
   没迭代够的继续迭代, 迭代够轮次的系列停止优化。
3. 工作槽位 --parallel(默认2):
   - 始终保持 1 个"新因子挖掘"任务在跑(战役正常结束后 --fresh 重开新战役;
     战役中途崩溃则不带 --fresh 续跑当前战役);
   - 其余槽位跑优化预算未达标系列(同一系列同时只允许一个进程, 避免检查点冲突);
   - 无优化任务时其余槽位空闲(不允许两个 new 进程并存)。
4. 子进程异常退出(exit code != 0): 自动重启(run_mining 自带断点续传);
   单任务连续崩溃超过上限则冷却暂停一段时间后再试。
5. 编排器启动时接管现场: 结束所有已在跑的 run_mining 进程(其检查点已保存, 可续跑),
   由编排器统一调度, 避免重复进程争抢同一检查点。

状态文件: workspace/mining_loop_state.json (编排器可中断后原样重启)
日志:     workspace/mining_loop.log (编排器自身)
          子进程仍写各自的 mining_*.log; 子进程 stdout/stderr 汇入 mining_loop_child.log

用法:
  python -m llm_mining.tools.run_mining_loop --target-count 50 \
      --rounds-per-series 100 --parallel 2
"""

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import time

from llm_mining.local_miner import factor_library
from llm_mining.local_miner.config import (
    WORKSPACE_DIR, QUANT_ROOT, ensure_workspace, checkpoint_path, mining_log_path,
)

PYTHON = sys.executable
STATE_PATH = os.path.join(WORKSPACE_DIR, "mining_loop_state.json")
LOOP_LOG = os.path.join(WORKSPACE_DIR, "mining_loop.log")
CHILD_LOG = os.path.join(WORKSPACE_DIR, "mining_loop_child.log")
PID_FILE = os.path.join(WORKSPACE_DIR, "mining_loop.pid")
ROUNDS_PER_CAMPAIGN = 100          # 单个战役的最大轮数(run_mining --max-rounds)
MAX_CONSECUTIVE_CRASHES = 3        # 单任务连续崩溃上限, 超过则冷却
CRASH_COOLDOWN_SEC = 3600          # 一般崩溃冷却时长(秒)
QUOTA_COOLDOWN_SEC = 5 * 3600 + 300  # 兜底: 无法解析重置时刻时的配额冷却时长
QUOTA_RESET_BUFFER_SEC = 600       # 配额重置后再等10分钟缓冲, 避免服务刚恢复不稳定
POLL_INTERVAL_SEC = 60             # 主循环轮询间隔
SUMMARY_INTERVAL_SEC = 1800        # 周期性状态汇报间隔


def log(msg: str = ""):
    line = f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    try:
        ensure_workspace()
        with open(LOOP_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ---------------- 状态文件 ----------------

def load_state() -> dict | None:
    if not os.path.exists(STATE_PATH):
        return None
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log(f"[警告] 编排状态读取失败({e}), 将重建")
        return None


def save_state(state: dict):
    ensure_workspace()
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, STATE_PATH)


def seed_state(args) -> dict:
    """初始状态: D001 已完成大量轮优化(视为达标), D002/D003 尚未做专门优化;
    new 战役按现场检查点判断: 已跑满轮次则新开战役, 否则续跑。"""
    new_done = ckpt_round("new", "")
    state = {
        "target_count": args.target_count,
        "rounds_per_series": args.rounds_per_series,
        "series_opt_rounds": {"D001": 200, "D002": 0, "D003": 0},
        "new_fresh": new_done >= ROUNDS_PER_CAMPAIGN,  # 跑满则 --fresh 新开战役
        "new_campaigns_done": 0,
        "crash_counts": {},
        "paused_until": {},
        "started_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    return state


# ---------------- 库与检查点读取 ----------------

def qualified_count() -> int:
    return sum(1 for s in factor_library.load_library()
               if s.get("status") == "qualified")


def all_series_ids() -> list:
    return [s["series_id"] for s in factor_library.load_library()]


def ckpt_round(mode: str, series_id: str = "") -> int:
    """读取指定检查点中已完成的轮数(文件不存在/损坏返回0)"""
    path = checkpoint_path(mode, series_id)
    if not os.path.exists(path):
        return 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            return int(json.load(f).get("round", 0))
    except Exception:
        return 0


def sync_series(state: dict):
    """把库中(新入库的)系列同步进优化预算表"""
    for sid in all_series_ids():
        state["series_opt_rounds"].setdefault(sid, 0)


# ---------------- 进程接管 ----------------

def pid_alive(pid: int) -> bool:
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                             capture_output=True, text=True, timeout=30).stdout
        return str(pid) in out
    except Exception:
        return False


def kill_stale_miners():
    """接管现场: 结束所有已在跑的 run_mining 子进程(检查点已保存, 续跑无损)"""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"],
            capture_output=True, text=True, timeout=60).stdout
        procs = json.loads(out or "[]")
        if isinstance(procs, dict):
            procs = [procs]
        killed = 0
        for p in procs:
            pid = int(p.get("ProcessId") or 0)
            cmd = str(p.get("CommandLine") or "")
            if pid in (0, os.getpid()):
                continue
            if "run_mining_loop" in cmd:
                continue
            if "run_mining" in cmd:
                subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                               capture_output=True, timeout=30)
                log(f"    [接管] 已结束既有挖掘进程 PID={pid}")
                killed += 1
        if killed == 0:
            log("    [接管] 未发现需要接管的既有挖掘进程")
    except Exception as e:
        log(f"    [警告] 接管清理失败: {e}")


# ---------------- 任务构建与启动 ----------------

def build_cmd(mode: str, series_id: str, max_rounds: int, fresh: bool) -> list:
    cmd = [PYTHON, "-m", "llm_mining.local_miner.run_mining",
           "--mode", mode, "--max-rounds", str(max_rounds)]
    if series_id:
        cmd += ["--series", series_id]
    if fresh:
        cmd += ["--fresh"]
    return cmd


def launch_task(children: dict, child_log_f, mode: str, series_id: str,
                fresh: bool, max_rounds_arg: int, start_round: int):
    cmd = build_cmd(mode, series_id, max_rounds_arg, fresh)
    # cwd 必须用项目根 QUANT_ROOT(F:\quant), 使 `python -m llm_mining...` 可解析到包
    proc = subprocess.Popen(cmd, stdout=child_log_f, stderr=subprocess.STDOUT,
                            cwd=QUANT_ROOT,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    key = "new" if mode == "new" else series_id
    children[key] = {
        "proc": proc, "mode": mode, "series": series_id, "fresh": fresh,
        "campaign_max": max_rounds_arg, "start_round": start_round,
        "pid": proc.pid,
    }
    desc = "新因子挖掘" if mode == "new" else f"优化 {series_id}"
    log(f"    [启动] {desc}: PID={proc.pid}, --max-rounds {max_rounds_arg}"
        f"{', --fresh' if fresh else ', 断点续跑'}")


# ---------------- 任务结束处理 ----------------

def detect_quota_exhausted(mode: str, series_id: str = "") -> bool:
    """检测子进程日志尾部是否出现 LLM 配额耗尽(HTTP 429 / insufficient_quota)"""
    path = mining_log_path(mode, series_id)
    if not os.path.exists(path):
        return False
    try:
        with open(path, "rb") as f:          # 二进制模式 seek/tell 才可靠
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 6000))
            tail = f.read().decode("utf-8", errors="ignore")
        return ("insufficient_quota" in tail) or ("HTTP 429" in tail) \
            or ("quota" in tail and "exhausted" in tail)
    except Exception:
        return False


def quota_reset_local(mode: str, series_id: str = "") -> float | None:
    """从日志解析配额重置时刻(UTC, 形如 'reset at 08-05 16:35:00 UTC'),
    返回本地时区(Asia/Shanghai)的时间戳; 解析失败返回 None。"""
    path = mining_log_path(mode, series_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 12000))
            tail = f.read().decode("utf-8", errors="ignore")
        m = re.search(r"reset at (\d{2}-\d{2} \d{2}:\d{2}:\d{2}) UTC", tail)
        if not m:
            return None
        now = datetime.datetime.now()
        utc_dt = datetime.datetime.strptime(m.group(1), "%m-%d %H:%M:%S")
        for dy in (0, -1, 1):                 # 处理可能的跨年
            cand = utc_dt.replace(year=now.year + dy)
            ts = cand.timestamp() + 8 * 3600  # UTC -> 本地(UTC+8)
            if abs(ts - time.time()) < 10 * 24 * 3600:
                return ts
        return None
    except Exception:
        return None


def on_task_exit(state: dict, ch: dict, rc: int):
    mode, sid, fresh, cmax = ch["mode"], ch["series"], ch["fresh"], ch["campaign_max"]
    tag = "new" if mode == "new" else f"optimize {sid}"
    key = tag if mode == "new" else sid

    if rc == 0:
        state["crash_counts"].pop(key, None)
        if mode == "optimize":
            done_now = ckpt_round("optimize", sid)
            delta = max(0, done_now - ch["start_round"])
            state["series_opt_rounds"][sid] = state["series_opt_rounds"].get(sid, 0) + delta
            log(f"    [完成] {tag} 战役正常结束: 本战役 {delta} 轮, "
                f"累计 {state['series_opt_rounds'][sid]}/{state['rounds_per_series']} 轮预算")
        else:
            state["new_campaigns_done"] = state.get("new_campaigns_done", 0) + 1
            state["new_fresh"] = True          # 战役结束(跑满或挖到因子入库) -> 下轮新开战役
            log(f"    [完成] 新因子战役 #{state['new_campaigns_done']} 正常结束, "
                f"当前合格因子数: {qualified_count()}")
        save_state(state)
        return

    # ---- 异常退出 ----
    cc = state["crash_counts"].get(key, 0) + 1
    state["crash_counts"][key] = cc
    if mode == "new":
        if ckpt_round("new", "") >= cmax:
            # 战役轮次已跑满(多为 finalize 阶段崩溃): 视为战役结束, 下轮新开战役
            state["new_fresh"] = True
            state["new_campaigns_done"] = state.get("new_campaigns_done", 0) + 1
            log(f"    [退出] {tag} 战役已满轮次但异常退出(rc={rc}), 下一轮新开战役")
        else:
            state["new_fresh"] = fresh         # 维持当前战役续跑策略
            log(f"    [退出] {tag} 异常退出(rc={rc}), 将从检查点续跑")
    else:
        log(f"    [退出] {tag} 异常退出(rc={rc}), 将从检查点续跑")

    if cc >= MAX_CONSECUTIVE_CRASHES:
        quota = detect_quota_exhausted(mode, sid)
        reset_ts = quota_reset_local(mode, sid) if quota else None
        if reset_ts is not None:
            # 冷却到配额重置时刻 + 缓冲, 避免无谓等待
            until = reset_ts + QUOTA_RESET_BUFFER_SEC
            if until <= time.time():
                until = time.time() + CRASH_COOLDOWN_SEC   # 重置已过(兜底)
            state["paused_until"][key] = until
            log(f"    [警告] {tag} 检测到 LLM 配额耗尽: "
                f"将冷却至 {datetime.datetime.fromtimestamp(until):%m-%d %H:%M} "
                f"(配额重置 {datetime.datetime.fromtimestamp(reset_ts):%m-%d %H:%M} + {QUOTA_RESET_BUFFER_SEC // 60} 分钟缓冲)")
        else:
            cooldown = QUOTA_COOLDOWN_SEC if quota else CRASH_COOLDOWN_SEC
            state["paused_until"][key] = time.time() + cooldown
            reason = "检测到 LLM 配额耗尽" if quota else "连续崩溃"
            log(f"    [警告] {tag} {reason}: 冷却 {cooldown // 3600} 小时后再试")
        state["crash_counts"][key] = 0
    save_state(state)


# ---------------- 调度 ----------------

def pick_optimize_series(state: dict, running_series: set) -> str | None:
    """选一个优化预算未达标且未在跑的系列(进度最慢者优先); 冷却中的跳过"""
    now = time.time()
    budget = state["rounds_per_series"]
    cands = []
    for sid, done in state["series_opt_rounds"].items():
        if sid in running_series or done >= budget:
            continue
        if now < state["paused_until"].get(sid, 0):
            continue
        cands.append((done, sid))
    if not cands:
        return None
    cands.sort()
    return cands[0][1]


def main():
    p = argparse.ArgumentParser(description="挖掘循环编排器(挖到目标数量合格因子)")
    p.add_argument("--target-count", type=int, default=50, help="目标合格因子数量")
    p.add_argument("--rounds-per-series", type=int, default=100,
                   help="每个因子系列的累计优化轮数预算")
    p.add_argument("--parallel", type=int, default=2, help="并发挖掘进程数")
    args = p.parse_args()
    ensure_workspace()

    # 单实例保护
    if os.path.exists(PID_FILE):
        try:
            old_pid = int(open(PID_FILE, "r", encoding="utf-8").read().strip())
            if pid_alive(old_pid) and old_pid != os.getpid():
                print(f"编排器已在运行(PID {old_pid}), 本次退出。")
                return
        except Exception:
            pass
    with open(PID_FILE, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))

    state = load_state() or seed_state(args)
    state["target_count"] = args.target_count
    state["rounds_per_series"] = args.rounds_per_series
    save_state(state)

    log("=" * 70)
    log(f"挖掘循环编排器启动 (PID {os.getpid()}): 目标合格因子={args.target_count}, "
        f"每系列优化预算={args.rounds_per_series}轮, 并发={args.parallel}")
    kill_stale_miners()

    children = {}
    last_summary = 0.0
    child_log_f = open(CHILD_LOG, "a", encoding="utf-8")
    try:
        while True:
            n_q = qualified_count()
            if n_q >= args.target_count:
                log(f"★ 合格因子已达 {n_q} 个 (目标 {args.target_count}), 停止全部子进程并退出。")
                break

            # ---- 回收已结束的子进程 ----
            for key in list(children):
                ch = children[key]
                rc = ch["proc"].poll()
                if rc is None:
                    continue
                del children[key]
                on_task_exit(state, ch, rc)

            sync_series(state)

            # ---- 填充空闲槽位 ----
            running_series = {ch["series"] for ch in children.values()
                              if ch["mode"] == "optimize"}
            new_running = any(ch["mode"] == "new" for ch in children.values())
            slots_free = args.parallel - len(children)

            if slots_free > 0 and not new_running:
                if time.time() >= state["paused_until"].get("new", 0):
                    fresh = state.get("new_fresh", True)
                    start = 0 if fresh else ckpt_round("new", "")
                    # 续跑时目标轮数须大于当前进度, 且不低于标准战役长度
                    max_arg = ROUNDS_PER_CAMPAIGN if fresh \
                        else max(start + 1, ROUNDS_PER_CAMPAIGN)
                    launch_task(children, child_log_f, "new", "", fresh, max_arg, start)
                    slots_free -= 1

            while slots_free > 0:
                sid = pick_optimize_series(state, running_series)
                if sid is None:
                    break
                start = ckpt_round("optimize", sid)
                remaining = state["rounds_per_series"] - state["series_opt_rounds"][sid]
                rounds = min(remaining, ROUNDS_PER_CAMPAIGN)
                launch_task(children, child_log_f, "optimize", sid, False,
                            start + rounds, start)
                running_series.add(sid)
                slots_free -= 1

            # ---- 周期性状态汇报 ----
            now = time.time()
            if now - last_summary >= SUMMARY_INTERVAL_SEC:
                last_summary = now
                running_desc = ", ".join(
                    ("new" if ch["mode"] == "new" else ch["series"])
                    + f"(PID {ch['pid']})" for ch in children.values()) or "无"
                pending = [f"{s}({d}/{state['rounds_per_series']})"
                           for s, d in sorted(state["series_opt_rounds"].items())
                           if d < state["rounds_per_series"]]
                log(f"    [状态] 合格因子 {n_q}/{args.target_count} | 在跑: {running_desc} | "
                    f"待优化系列: {', '.join(pending) or '无'}")

            time.sleep(POLL_INTERVAL_SEC)
    except KeyboardInterrupt:
        log("[中断] 编排器收到中断信号, 结束全部子进程...")
    finally:
        for ch in children.values():
            try:
                ch["proc"].terminate()
            except Exception:
                pass
        for ch in children.values():
            try:
                ch["proc"].wait(timeout=30)
            except Exception:
                pass
        child_log_f.close()
        save_state(state)
        try:
            os.remove(PID_FILE)
        except Exception:
            pass
        log("挖掘循环编排器已退出。")


if __name__ == "__main__":
    main()
