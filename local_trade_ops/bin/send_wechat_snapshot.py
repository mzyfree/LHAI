#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path


def read_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip("'\"")
    return env


def fmt_money(value: object) -> str:
    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return "-"


def fmt_pct(value: object) -> str:
    try:
        return f"{float(value):.2f}%"
    except (TypeError, ValueError):
        return "-"


def fmt_price(value: object) -> str:
    try:
        price = float(value)
        if price <= 0:
            return "-"
        return f"{price:.2f}"
    except (TypeError, ValueError):
        return "-"


def fmt_score(value: object) -> str:
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return "-"


def action_label(action: object) -> str:
    value = str(action or "").upper()
    if value == "BUY":
        return "买入"
    if value == "SELL":
        return "卖出"
    return value or "-"


def status_label(status: object) -> str:
    value = str(status or "").lower()
    if value == "filled":
        return "已成交"
    if value == "partial":
        return "部分成交"
    if value == "skipped":
        return "跳过/未成交"
    if value == "pending":
        return "待执行"
    return value or "-"


def is_short_hold_snapshot(snapshot: dict) -> bool:
    return snapshot.get("strategy") == "short_hold_single_peak"


def snapshot_label(snapshot: dict, env: dict[str, str]) -> str:
    if is_short_hold_snapshot(snapshot):
        return env.get("SHORT_HOLD_REPORT_LABEL", "短持有日报").strip() or "短持有日报"
    return env.get("DAILY_REPORT_LABEL", "交易日报").strip() or "交易日报"


def snapshot_topic(snapshot: dict, env: dict[str, str]) -> str:
    if is_short_hold_snapshot(snapshot):
        return env.get("SHORT_HOLD_PUSHPLUS_TOPIC", env.get("PUSHPLUS_TOPIC", "")).strip()
    return env.get("PUSHPLUS_TOPIC", "").strip()


def first_present(row: dict, keys: list[str], default: object = 0) -> object:
    for key in keys:
        if key in row and row.get(key) not in {None, ""}:
            return row.get(key)
    return default


def operation_line(row: dict) -> str:
    action = str(row.get("action", "")).upper()
    role = "备选" if row.get("order_role") == "backup" else "主计划"
    planned = first_present(row, ["planned_shares", "planned", "shares"], 0)
    filled = first_present(row, ["filled_shares", "filled", "fill_shares"], 0)
    ref = fmt_price(row.get("ref_price"))
    fill_price = fmt_price(row.get("fill_price"))
    note = row.get("note") or ""
    price_bits = [f"参考 {ref}"] if ref != "-" else []
    if action == "BUY":
        price_5pct = fmt_price(row.get("price_5pct"))
        if price_5pct != "-":
            price_bits.append(f"5%红线 {price_5pct}")
    if fill_price != "-":
        price_bits.append(f"成交价 {fill_price}")
    suffix = f"，{'; '.join(price_bits)}" if price_bits else ""
    note_suffix = f"，备注：{note}" if note else ""
    return (
        f"- {row.get('code', '-')}: {role}{action_label(action)} {planned}，"
        f"成交 {filled}，状态：{status_label(row.get('status'))}{suffix}{note_suffix}"
    )


def build_markdown(snapshot: dict, label: str) -> str:
    account = snapshot.get("account", {})
    orders = snapshot.get("orders", {})
    positions = snapshot.get("positions", [])
    operations = orders.get("operations", [])
    report_operations = [
        row
        for row in operations
        if int(float(first_present(row, ["filled_shares", "filled", "fill_shares"], 0) or 0)) > 0
    ]

    lines = [
        f"## {label} {snapshot.get('date', '-')}",
        f"> 生成时间：{snapshot.get('generated_at', '-')}",
        "",
        f"- NAV：{fmt_money(account.get('nav'))}",
        f"- 现金：{fmt_money(account.get('cash'))}",
        f"- 仓位：{fmt_pct(account.get('position_ratio_pct'))}",
        f"- 本次/当日盈亏：{fmt_money(account.get('daily_pnl'))}",
        f"- 本次/当日收益：{fmt_pct(account.get('daily_return_pct'))}",
        f"- 累计收益：{fmt_pct(account.get('cum_return_pct'))}",
        f"- 成交：{orders.get('filled_count', 0)} / 计划 {orders.get('planned_count', 0)}",
        f"- 跳过/未成交：{orders.get('skipped_count', 0)}",
    ]

    if report_operations:
        lines += ["", "### 当日操作"]
        for row in report_operations[:12]:
            lines.append(operation_line(row))
        if len(report_operations) > 12:
            lines.append(f"- 其余 {len(report_operations) - 12} 条略")

    if positions:
        lines += ["", "### 当前持仓"]
        for row in positions[:12]:
            rank = row.get("model_rank")
            score = row.get("score")
            signal_date = row.get("signal_date")
            code = first_present(row, ["code", "instrument"], "-")
            rank_suffix = f"，rank {rank}" if rank not in {None, ""} else ""
            score_text = fmt_score(score)
            score_suffix = f"，score {score_text}" if score_text != "-" else ""
            signal_suffix = f"，信号 {signal_date}" if signal_date not in {None, ""} else ""
            lines.append(
                f"- {code}: {row.get('shares', 0)} 股，成本 {fmt_money(row.get('cost_basis'))}"
                f"{rank_suffix}{score_suffix}{signal_suffix}"
            )
        if len(positions) > 12:
            lines.append(f"- 其余 {len(positions) - 12} 只略")

    return "\n".join(lines)


def build_title(snapshot: dict, label: str) -> str:
    account = snapshot.get("account", {})
    orders = snapshot.get("orders", {})
    date = snapshot.get("date", "-")
    return (
        f"{label} {date} "
        f"NAV {fmt_money(account.get('nav'))} "
        f"收益 {fmt_pct(account.get('daily_return_pct'))} "
        f"仓位 {fmt_pct(account.get('position_ratio_pct'))} "
        f"成交 {orders.get('filled_count', 0)}/{orders.get('planned_count', 0)}"
    )


def post_wecom_markdown(webhook_url: str, content: str) -> dict:
    payload = {
        "msgtype": "markdown",
        "markdown": {"content": content},
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        webhook_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def post_pushplus(token: str, title: str, content: str, topic: str = "") -> dict:
    payload = {
        "token": token,
        "title": title,
        "content": content,
        "template": "markdown",
        "channel": "wechat",
    }
    if topic:
        payload["topic"] = topic
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        "https://www.pushplus.plus/send",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def post_serverchan(sendkey: str, title: str, content: str) -> dict:
    data = urllib.parse.urlencode({"title": title, "desp": content}).encode("utf-8")
    req = urllib.request.Request(
        f"https://sctapi.ftqq.com/{sendkey}.send",
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def post_bottalk(sendkey: str, title: str, content: str) -> dict:
    payload = {"title": title, "desp": content}
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"https://bot-talk.com/{sendkey}.send",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Send latest daily snapshot to a WeChat-compatible channel.")
    parser.add_argument("--snapshot", default="")
    parser.add_argument("--env-file", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ops_home = Path(os.environ.get("OPS_HOME", Path(__file__).resolve().parents[1])).resolve()
    env_file = Path(args.env_file).expanduser().resolve() if args.env_file else ops_home / "config" / "env.local"
    env = {**read_env(env_file), **os.environ}
    snapshot_dir = Path(env.get("DAILY_SNAPSHOT_DIR", ops_home / "daily_snapshots")).expanduser()
    snapshot_path = Path(args.snapshot).expanduser() if args.snapshot else snapshot_dir / "daily_snapshot_latest.json"
    if not snapshot_path.exists():
        raise SystemExit(f"Snapshot not found: {snapshot_path}. Generate daily snapshot first.")

    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    label = snapshot_label(snapshot, env)
    content = build_markdown(snapshot, label)
    title = build_title(snapshot, label)
    mode = env.get("WECHAT_REPORT_MODE", "pushplus").strip()

    if args.dry_run or mode in {"", "console", "dry_run"}:
        print(content)
        return 0

    if mode == "wecom_webhook":
        webhook_url = env.get("WECHAT_REPORT_WEBHOOK_URL", "").strip()
        if not webhook_url:
            raise SystemExit("WECHAT_REPORT_WEBHOOK_URL is empty. Paste your Enterprise WeChat group robot webhook into config/env.local.")
        result = post_wecom_markdown(webhook_url, content)
        ok = result.get("errcode") == 0
    elif mode == "pushplus":
        token = env.get("PUSHPLUS_TOKEN", "").strip()
        if not token:
            raise SystemExit("PUSHPLUS_TOKEN is empty. Set WECHAT_REPORT_MODE=pushplus and paste your PushPlus token into config/env.local.")
        result = post_pushplus(token, title, content, snapshot_topic(snapshot, env))
        ok = result.get("code") == 200
    elif mode == "serverchan":
        sendkey = env.get("SERVERCHAN_SENDKEY", "").strip()
        if not sendkey:
            raise SystemExit("SERVERCHAN_SENDKEY is empty. Set WECHAT_REPORT_MODE=serverchan and paste your ServerChan SendKey into config/env.local.")
        result = post_serverchan(sendkey, title, content)
        ok = result.get("code") == 0
    elif mode == "bottalk":
        sendkey = env.get("BOTTALK_SENDKEY", "").strip() or env.get("SERVERCHAN_SENDKEY", "").strip()
        if not sendkey:
            raise SystemExit("BOTTALK_SENDKEY is empty. Set WECHAT_REPORT_MODE=bottalk and paste your BotTalk SendKey into config/env.local.")
        result = post_bottalk(sendkey, title, content)
        ok = result.get("code") == 0
    else:
        raise SystemExit(
            f"Unsupported WECHAT_REPORT_MODE={mode}. Currently supported: pushplus, serverchan, bottalk, wecom_webhook, console."
        )

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not ok:
        raise SystemExit(f"WeChat push failed via {mode}: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
