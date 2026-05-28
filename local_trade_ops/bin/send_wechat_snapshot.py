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


def build_markdown(snapshot: dict) -> str:
    account = snapshot.get("account", {})
    orders = snapshot.get("orders", {})
    positions = snapshot.get("positions", [])
    skipped = orders.get("skipped", [])

    lines = [
        f"## 交易日报 {snapshot.get('date', '-')}",
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

    if skipped:
        lines += ["", "### 跳过/未成交"]
        for row in skipped:
            reason = row.get("note") or "未填写原因"
            lines.append(
                f"- {row.get('code', '-')}: 计划 {row.get('planned_shares', 0)}，"
                f"成交 {row.get('filled_shares', 0)}，原因：{reason}"
            )

    if positions:
        lines += ["", "### 当前持仓"]
        for row in positions[:12]:
            lines.append(
                f"- {row.get('code', '-')}: {row.get('shares', 0)} 股，成本 {fmt_money(row.get('cost_basis'))}"
            )
        if len(positions) > 12:
            lines.append(f"- 其余 {len(positions) - 12} 只略")

    return "\n".join(lines)


def build_title(snapshot: dict) -> str:
    account = snapshot.get("account", {})
    orders = snapshot.get("orders", {})
    date = snapshot.get("date", "-")
    return (
        f"交易日报 {date} "
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
    content = build_markdown(snapshot)
    title = build_title(snapshot)
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
        result = post_pushplus(token, title, content, env.get("PUSHPLUS_TOPIC", "").strip())
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
