import { createServer } from "node:http";
import { spawn } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, readdirSync, statSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { URL } from "node:url";

type Env = Record<string, string>;
type Job = {
  id: string;
  name: string;
  status: "running" | "success" | "failed";
  startedAt: string;
  finishedAt?: string;
  exitCode?: number | null;
  log: string;
};

const OPS_HOME = resolve(new URL("..", import.meta.url).pathname);
const PROJECT_HOME = resolve(OPS_HOME, "..");
const PAPER_HOME = resolve(PROJECT_HOME, "paper_trading_system");
const ENV_PATH = resolve(OPS_HOME, "config", "env.local");
const SCHEDULER_CONFIG_PATH = resolve(OPS_HOME, "config", "scheduler.json");
const jobs = new Map<string, Job>();
let lastSchedulerCheck = "";

function parseEnv(path: string): Env {
  const env: Env = {};
  if (!existsSync(path)) return env;
  for (const raw of readFileSync(path, "utf8").split(/\r?\n/)) {
    const line = raw.trim();
    if (!line || line.startsWith("#") || !line.includes("=")) continue;
    const [key, ...rest] = line.split("=");
    env[key.trim()] = rest.join("=").trim().replace(/^['"]|['"]$/g, "");
  }
  return env;
}

function readCsv(path: string): Record<string, string>[] {
  if (!existsSync(path)) return [];
  const lines = readFileSync(path, "utf8").trim().split(/\r?\n/).filter(Boolean);
  if (lines.length < 2) return [];
  const headers = lines[0].split(",");
  return lines.slice(1).map((line) => {
    const cells = line.split(",");
    return Object.fromEntries(headers.map((h, i) => [h, cells[i] ?? ""]));
  });
}

function readJson(path: string | null): unknown {
  if (!path || !existsSync(path)) return null;
  const text = readFileSync(path, "utf8")
    .replace(/:\s*NaN(?=\s*[,}])/g, ": null")
    .replace(/:\s*Infinity(?=\s*[,}])/g, ": null")
    .replace(/:\s*-Infinity(?=\s*[,}])/g, ": null");
  return JSON.parse(text);
}

function csvEscape(value: unknown): string {
  const text = String(value ?? "");
  return /[",\n\r]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

function writeCsv(path: string, rows: Record<string, unknown>[]) {
  if (!rows.length) {
    writeFileSync(path, "", "utf8");
    return;
  }
  const headers = Object.keys(rows[0]);
  const lines = [headers.join(",")];
  for (const row of rows) {
    lines.push(headers.map((h) => csvEscape(row[h])).join(","));
  }
  writeFileSync(path, `${lines.join("\n")}\n`, "utf8");
}

function listFiles(dir: string, prefix = ""): string[] {
  if (!existsSync(dir)) return [];
  return readdirSync(dir)
    .filter((name) => (prefix ? name.startsWith(prefix) : true))
    .sort();
}

function latestFile(dir: string, prefix: string, suffix = ""): string | null {
  const files = listFiles(dir, prefix).filter((name) => (suffix ? name.endsWith(suffix) : true));
  return files.length ? resolve(dir, files[files.length - 1]) : null;
}

function dateFromFile(path: string | null, prefix: string, suffix = ""): string | null {
  if (!path) return null;
  const name = path.split("/").pop() || "";
  const end = suffix && name.endsWith(suffix) ? name.length - suffix.length : name.length;
  const date = name.slice(prefix.length, end);
  return /^\d{4}-\d{2}-\d{2}$/.test(date) ? date : null;
}

function tail(path: string, n: number): string[] {
  if (!existsSync(path)) return [];
  return readFileSync(path, "utf8").trim().split(/\r?\n/).slice(-n);
}

function fileInfo(path: string) {
  if (!existsSync(path)) return { exists: false };
  const stat = statSync(path);
  return {
    exists: true,
    path,
    size: stat.size,
    mtime: stat.mtime.toISOString(),
  };
}

function defaultSchedulerConfig() {
  return {
    dataUpdate: {
      autoAfterClose: true,
      enabled: true,
      time: "20:30",
      timezone: "Asia/Shanghai",
      lastRunDate: "",
      lastJobId: "",
    },
  };
}

function loadSchedulerConfig() {
  if (!existsSync(SCHEDULER_CONFIG_PATH)) {
    const config = defaultSchedulerConfig();
    mkdirSync(resolve(OPS_HOME, "config"), { recursive: true });
    writeFileSync(SCHEDULER_CONFIG_PATH, JSON.stringify(config, null, 2), "utf8");
    return config;
  }
  const defaults = defaultSchedulerConfig();
  const loaded = JSON.parse(readFileSync(SCHEDULER_CONFIG_PATH, "utf8"));
  return {
    ...defaults,
    ...loaded,
    dataUpdate: {
      ...defaults.dataUpdate,
      ...(loaded.dataUpdate || {}),
    },
  };
}

function saveSchedulerConfig(config: ReturnType<typeof defaultSchedulerConfig>) {
  mkdirSync(resolve(OPS_HOME, "config"), { recursive: true });
  writeFileSync(SCHEDULER_CONFIG_PATH, JSON.stringify(config, null, 2), "utf8");
}

function localParts(timeZone: string) {
  const parts = new Intl.DateTimeFormat("en-CA", {
    timeZone,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hourCycle: "h23",
  }).formatToParts(new Date());
  const value = (type: string) => parts.find((p) => p.type === type)?.value || "";
  return {
    date: `${value("year")}-${value("month")}-${value("day")}`,
    time: `${value("hour")}:${value("minute")}`,
  };
}

function schedulerStatus() {
  const config = loadSchedulerConfig();
  const dataUpdate = config.dataUpdate;
  return {
    configPath: SCHEDULER_CONFIG_PATH,
    enabled: Boolean(dataUpdate.enabled),
    autoAfterClose: Boolean(dataUpdate.autoAfterClose),
    schedule: `${dataUpdate.time} ${dataUpdate.timezone}`,
    lastRunDate: dataUpdate.lastRunDate,
    lastJobId: dataUpdate.lastJobId,
  };
}

function schedulerTick() {
  const config = loadSchedulerConfig();
  const dataUpdate = config.dataUpdate;
  if (!dataUpdate.enabled) return;
  const now = localParts(dataUpdate.timezone);
  const key = `${now.date} ${now.time}`;
  if (lastSchedulerCheck === key) return;
  lastSchedulerCheck = key;
  if (now.time < dataUpdate.time) return;
  if (dataUpdate.lastRunDate === now.date) return;
  const job = dataUpdate.autoAfterClose
    ? startJob("scheduled-daily-pipeline", OPS_HOME, "bash", ["bin/daily_after_data_pipeline.sh"])
    : startJob("scheduled-update-data", PAPER_HOME, "bash", ["bin/update_data.sh"]);
  dataUpdate.lastRunDate = now.date;
  dataUpdate.lastJobId = job.id;
  saveSchedulerConfig(config);
}

function loadStatus() {
  const env = parseEnv(ENV_PATH);
  const stateDir = env.PAPER_STATE_DIR || resolve(OPS_HOME, "state", "live_10w_main");
  const reportDir = env.PAPER_REPORT_DIR || resolve(OPS_HOME, "reports", "live_10w_main");
  const taskDir = env.LIVE_TASK_DIR || resolve(OPS_HOME, "live_tasks");
  const submissionDir = env.LIVE_SUBMISSION_DIR || resolve(OPS_HOME, "live_submissions");
  const fillDir = env.LIVE_FILL_DIR || resolve(OPS_HOME, "live_fills");
  const snapshotDir = env.DAILY_SNAPSHOT_DIR || resolve(OPS_HOME, "daily_snapshots");
  const qlibDir = env.QLIB_PROVIDER_URI || resolve(PAPER_HOME, "data", "current");
  const navRows = readCsv(resolve(stateDir, "paper_nav.csv"));
  const positions = readCsv(resolve(stateDir, "paper_positions.csv"));
  const latestPending = latestFile(stateDir, "pending_orders_");
  const latestApproved = latestFile(stateDir, "approved_orders_");
  const latestTask = latestFile(taskDir, "live_order_task_", ".csv");
  const latestTaskJson = latestFile(taskDir, "live_order_task_", ".json");
  const latestSubmission = latestFile(submissionDir, "live_submission_", ".json");
  const latestBrokerTicket = latestFile(submissionDir, "broker_order_ticket_", ".csv");
  const latestRealtimeDecision = latestFile(submissionDir, "realtime_decision_", ".csv");
  const latestFillTemplate = latestFile(fillDir, "manual_fill_template_", ".csv");
  const latestAppliedFills = latestFile(fillDir, "manual_fills_applied_", ".csv");
  const latestReport = latestFile(reportDir, "paper_after_close_");
  const latestSnapshot = latestFile(snapshotDir, "daily_snapshot_", ".json");
  const predPaths = [env.PRED_A, env.PRED_B, env.PRED_C].filter(Boolean);

  const calendarTail = tail(resolve(qlibDir, "calendars", "day.txt"), 8);
  const lastCalendarDate = calendarTail.at(-1) || null;
  const todayLocal = new Intl.DateTimeFormat("en-CA", {
    timeZone: "Asia/Shanghai",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).format(new Date());
  const defaultExecutionDate =
    dateFromFile(latestPending, "pending_orders_", ".csv") ||
    dateFromFile(latestApproved, "approved_orders_", ".csv") ||
    todayLocal;

  return {
    now: new Date().toISOString(),
    env: {
      qlibDir,
      stateDir,
      reportDir,
      taskDir,
      submissionDir,
      fillDir,
      snapshotDir,
      capital: env.CAPITAL,
      topk: env.TOPK,
      nDrop: env.N_DROP,
      weights: env.WEIGHTS,
      filterMode: env.FILTER_MODE,
      liveBrokerMode: env.LIVE_BROKER_MODE || "manual",
      liveTradingEnabled: env.LIVE_TRADING_ENABLED || "0",
      liveOrderHook: env.LIVE_ORDER_HOOK || "",
    },
    data: {
      currentLink: fileInfo(qlibDir),
      calendarTail,
      lastCalendarDate,
      todayLocal,
      todayReady: lastCalendarDate === todayLocal,
      note: "todayReady=true means the local Qlib calendar already includes today's trade date.",
    },
    scheduler: schedulerStatus(),
    predictions: predPaths.map((p) => fileInfo(p)),
    account: {
      latestNav: navRows.at(-1) || null,
      positions,
    },
    workflow: {
      latestPending,
      latestApproved,
      latestTask,
      latestTaskJson,
      latestSubmission,
      latestBrokerTicket,
      latestRealtimeDecision,
      latestFillTemplate,
      latestAppliedFills,
      latestReport,
      latestSnapshot,
      latestSnapshotData: readJson(latestSnapshot),
      defaultExecutionDate,
      pendingOrders: latestPending ? readCsv(latestPending) : [],
      approvedOrders: latestApproved ? readCsv(latestApproved) : [],
      liveTaskOrders: latestTask ? readCsv(latestTask) : [],
      brokerTicketOrders: latestBrokerTicket ? manualTicketRows(readCsv(latestBrokerTicket)) : [],
      realtimeDecisionRows: latestRealtimeDecision ? readCsv(latestRealtimeDecision) : [],
      fillTemplateRows: latestFillTemplate ? addGapThresholds(readCsv(latestFillTemplate)) : [],
      appliedFillRows: latestAppliedFills ? readCsv(latestAppliedFills) : [],
    },
    jobs: [...jobs.values()].slice(-20).reverse(),
  };
}

function loadShortHoldStatus() {
  const env = parseEnv(ENV_PATH);
  const stateDir = env.SHORT_HOLD_STATE_DIR || resolve(OPS_HOME, "state", "short_hold_single_peak");
  const reportDir = env.SHORT_HOLD_REPORT_DIR || resolve(OPS_HOME, "reports", "short_hold_single_peak");
  const taskDir = env.SHORT_HOLD_TASK_DIR || resolve(OPS_HOME, "short_hold_tasks");
  const submissionDir = env.SHORT_HOLD_SUBMISSION_DIR || resolve(OPS_HOME, "short_hold_submissions");
  const fillDir = env.SHORT_HOLD_FILL_DIR || resolve(OPS_HOME, "short_hold_fills");
  const snapshotDir = env.SHORT_HOLD_SNAPSHOT_DIR || resolve(OPS_HOME, "short_hold_daily_snapshots");
  const qlibDir = env.QLIB_PROVIDER_URI || resolve(PAPER_HOME, "data", "current");
  const navRows = readCsv(resolve(stateDir, "paper_nav.csv"));
  const positions = readCsv(resolve(stateDir, "paper_positions.csv"));
  const latestPending = latestFile(stateDir, "pending_orders_");
  const latestApproved = latestFile(stateDir, "approved_orders_");
  const latestTask = latestFile(taskDir, "short_hold_order_task_", ".csv");
  const latestBrokerTicket = latestFile(submissionDir, "short_hold_broker_ticket_", ".csv");
  const latestFillTemplate = latestFile(fillDir, "short_hold_fill_template_", ".csv");
  const latestAppliedFills = latestFile(fillDir, "short_hold_fills_applied_", ".csv");
  const latestCandidateReview = latestFile(stateDir, "short_hold_candidate_review_", ".csv");
  const latestModelTopReview = latestFile(stateDir, "short_hold_model_top_review_", ".csv");
  const latestSignalRankings = latestFile(stateDir, "signal_rankings_", ".csv");
  const latestReport = latestFile(reportDir, "short_hold_fill_");
  const latestSnapshot = latestFile(snapshotDir, "short_hold_snapshot_", ".json");
  const predPaths = [env.PRED_A, env.PRED_B, env.PRED_C].filter(Boolean);
  const calendarTail = tail(resolve(qlibDir, "calendars", "day.txt"), 8);
  const todayLocal = new Intl.DateTimeFormat("en-CA", {
    timeZone: "Asia/Shanghai",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).format(new Date());
  const defaultExecutionDate =
    dateFromFile(latestPending, "pending_orders_", ".csv") ||
    dateFromFile(latestApproved, "approved_orders_", ".csv") ||
    todayLocal;

  return {
    now: new Date().toISOString(),
    env: {
      qlibDir,
      stateDir,
      reportDir,
      taskDir,
      submissionDir,
      fillDir,
      snapshotDir,
      capital: env.SHORT_HOLD_CAPITAL || env.CAPITAL || "100000",
      topk: env.SHORT_HOLD_TOPK || "4",
      backupCount: env.SHORT_HOLD_BACKUP_COUNT || "3",
      excludeRestrictedMarkets: env.SHORT_HOLD_EXCLUDE_RESTRICTED_MARKETS || "1",
      minAmount: env.SHORT_HOLD_MIN_AMOUNT || "30000000",
      maxPositionPct: env.SHORT_HOLD_MAX_POSITION_PCT || "0.70",
      buyBudgetMode: env.SHORT_HOLD_BUY_BUDGET_MODE || env.BUY_BUDGET_MODE || "capital_pool",
      reserveCashPct: env.SHORT_HOLD_RESERVE_CASH_PCT || "0.02",
      scoreTemperature: env.SHORT_HOLD_SCORE_TEMPERATURE || "1.0",
      scoringMode: env.SHORT_HOLD_SCORING_MODE || "v2",
      v3Alpha: env.SHORT_HOLD_V3_ALPHA || "1.0",
      v3Beta: env.SHORT_HOLD_V3_BETA || "2.0",
      v3Gamma: env.SHORT_HOLD_V3_GAMMA || "0.0",
    },
    data: {
      currentLink: fileInfo(qlibDir),
      calendarTail,
      lastCalendarDate: calendarTail.at(-1) || null,
      todayLocal,
    },
    predictions: predPaths.map((p) => fileInfo(p)),
    account: {
      latestNav: navRows.at(-1) || null,
      positions,
    },
    workflow: {
      latestPending,
      latestApproved,
      latestTask,
      latestBrokerTicket,
      latestFillTemplate,
      latestAppliedFills,
      latestCandidateReview,
      latestModelTopReview,
      latestSignalRankings,
      latestReport,
      latestSnapshot,
      latestSnapshotData: readJson(latestSnapshot),
      defaultExecutionDate,
      pendingOrders: latestPending ? readCsv(latestPending) : [],
      approvedOrders: latestApproved ? readCsv(latestApproved) : [],
      brokerTicketOrders: latestBrokerTicket
        ? shortHoldTicketRows(readCsv(latestBrokerTicket), readCsv(latestSignalRankings || ""))
        : [],
      fillTemplateRows: latestFillTemplate ? shortHoldFillRows(readCsv(latestFillTemplate)) : [],
      appliedFillRows: latestAppliedFills ? readCsv(latestAppliedFills) : [],
      modelTopReviewRows: latestModelTopReview ? shortHoldModelTopReviewRows(readCsv(latestModelTopReview)) : [],
      candidateReviewRows: latestCandidateReview ? readCsv(latestCandidateReview) : [],
      recentFilteredTopRows: shortHoldRecentFilteredTopRows(stateDir),
    },
    jobs: [...jobs.values()].slice(-20).reverse(),
  };
}

function addGapThresholds(rows: Record<string, string>[]): Record<string, string>[] {
  return rows.map((row) => {
    const price = Number(row.estimated_price || row.price || row.limit_price || "");
    if (row.action !== "BUY" || !Number.isFinite(price) || price <= 0) {
      return {
        ...row,
        price_3pct: row.price_3pct || "",
        price_5pct: row.price_5pct || "",
        manual_price_rule: row.manual_price_rule || "卖出单无买入溢价红线，按盘口人工卖出",
      };
    }
    return {
      ...row,
      price_3pct: row.price_3pct || (price * 1.03).toFixed(2),
      price_5pct: row.price_5pct || (price * 1.05).toFixed(2),
      manual_price_rule: row.manual_price_rule || "限价=略高于卖一价且<=price_5pct",
      rule: "涨停/高开>5%跳过",
    };
  });
}

function manualTicketRows(rows: Record<string, string>[]): Record<string, string>[] {
  return addGapThresholds(rows).map((row) => ({
    role: row.order_role || "primary",
    model_rank: row.model_rank || "",
    score: row.score ? Number(row.score).toFixed(6) : "",
    signal_date: row.signal_date,
    execution_date: row.execution_date,
    code: row.instrument,
    action: row.action,
    shares: row.shares,
    ref_price: Number(row.estimated_price || "").toFixed(2),
    price_3pct: row.price_3pct,
    price_5pct: row.price_5pct,
    manual_price_rule: row.manual_price_rule,
    note: row.order_role === "backup"
      ? "备选补位：主买入跳过时才看"
      : (row.action === "BUY" ? "超过5%/涨停/买不进则跳过" : "卖出优先完成"),
  }));
}

function scoreByInstrument(rows: Record<string, string>[]): Map<string, string> {
  return new Map(rows.map((row) => [row.instrument || row.code || "", row.score || ""]));
}

function finiteNumber(value: unknown): number | null {
  const n = Number(value ?? "");
  return Number.isFinite(n) ? n : null;
}

function formatNumber(value: unknown, digits: number): string {
  const n = finiteNumber(value);
  return n === null ? "" : n.toFixed(digits);
}

function formatPercent(value: unknown, digits = 2): string {
  const n = finiteNumber(value);
  return n === null ? "" : `${(n * 100).toFixed(digits)}%`;
}

function v3ValueColumns(row: Record<string, string>): Record<string, string> {
  const hasV3Values = [row.final_score, row.buyability_risk, row.strong_prob, row.liquidity_risk].some(
    (value) => finiteNumber(value) !== null,
  );
  if (!hasV3Values) {
    return {};
  }
  return {
    return_score: formatNumber(row.return_score || row.score, 6),
    final_score: formatNumber(row.final_score, 6),
    buyability_risk: formatPercent(row.buyability_risk),
    strong_prob: formatPercent(row.strong_prob),
    liquidity_risk: formatNumber(row.liquidity_risk, 4),
    score_source: row.score_source || "v3",
  };
}

function shortHoldRecentFilteredTopRows(stateDir: string, limit = 5): Record<string, string>[] {
  const files = listFiles(stateDir, "short_hold_candidate_review_")
    .filter((name) => name.endsWith(".csv"))
    .reverse();
  const bySignalDate = new Map<string, Record<string, string>>();

  for (const name of files) {
    const rows = readCsv(resolve(stateDir, name))
      .filter((row) => ["ORDERED", "BACKUP"].includes(row.status))
      .filter((row) => finiteNumber(row.planned_shares) !== null && Number(row.planned_shares) > 0)
      .sort((a, b) => {
        const rankA = finiteNumber(a.model_rank) ?? Number.POSITIVE_INFINITY;
        const rankB = finiteNumber(b.model_rank) ?? Number.POSITIVE_INFINITY;
        if (rankA !== rankB) return rankA - rankB;
        return (
          (finiteNumber(b.final_score || b.score) ?? Number.NEGATIVE_INFINITY) -
          (finiteNumber(a.final_score || a.score) ?? Number.NEGATIVE_INFINITY)
        );
      });
    const top = rows[0];
    if (!top || !top.signal_date || bySignalDate.has(top.signal_date)) continue;

    const price = finiteNumber(top.estimated_price);
    bySignalDate.set(top.signal_date, {
      signal_date: top.signal_date,
      execution_date: top.execution_date || "",
      exit_date: top.exit_date || "",
      code: top.instrument || "",
      role: top.order_role || "",
      model_rank: top.model_rank || "",
      score: formatNumber(top.score, 6),
      ...v3ValueColumns(top),
      ref_price: formatNumber(top.estimated_price, 2),
      price_5pct: price === null ? "" : (price * 1.05).toFixed(2),
      shares: top.planned_shares || "",
      planned_value: formatNumber(top.planned_value, 0),
      status: top.status || "",
    });
    if (bySignalDate.size >= limit) break;
  }

  return [...bySignalDate.values()];
}

function shortHoldTicketRows(
  rows: Record<string, string>[],
  latestSignalRows: Record<string, string>[] = [],
): Record<string, string>[] {
  const latestSignalScore = scoreByInstrument(latestSignalRows);
  return addGapThresholds(rows).map((row) => ({
    role: row.order_role || "primary",
    signal_date: row.signal_date,
    execution_date: row.execution_date,
    exit_date: row.exit_date,
    code: row.instrument,
    action: row.action,
    shares: row.shares,
    ref_price: Number(row.estimated_price || "").toFixed(2),
    price_3pct: row.action === "BUY" ? row.price_3pct : "",
    price_5pct: row.action === "BUY" ? row.price_5pct : "",
    model_rank: row.model_rank || "",
    entry_score: row.action === "SELL" && row.score ? Number(row.score).toFixed(6) : "",
    signal_score: (latestSignalScore.get(row.instrument) || (row.action === "BUY" ? row.score : ""))
      ? Number(latestSignalScore.get(row.instrument) || row.score).toFixed(6)
      : "",
    ...v3ValueColumns(row),
    alloc_weight: row.softmax_weight ? (Number(row.softmax_weight) * 100).toFixed(2) + "%" : "",
    target_value: row.target_value ? Number(row.target_value).toFixed(0) : "",
    planned_value: row.planned_value ? Number(row.planned_value).toFixed(0) : "",
    note: row.order_role === "backup"
      ? "backup 组合：primary 买不进时使用"
      : row.action === "SELL"
      ? "到期卖出；entry_score为入仓分数，signal_score为最新模型分数"
      : "primary 主单：按日期节奏执行",
  }));
}

function shortHoldFillRows(rows: Record<string, string>[]): Record<string, string>[] {
  return addGapThresholds(rows);
}

function shortHoldModelTopReviewRows(rows: Record<string, string>[]): Record<string, string>[] {
  return rows.map((row) => ({
    model_rank: row.model_rank || "",
    code: row.instrument || "",
    score: row.score ? Number(row.score).toFixed(6) : "",
    close: row.close ? Number(row.close).toFixed(2) : "",
    amount20: row.amount20 ? (Number(row.amount20) / 10000).toFixed(0) + "万" : "",
    filter_status: row.filter_status || "",
    filter_reason: row.filter_reason || "",
  }));
}

function validateDate(date: unknown): string {
  if (typeof date !== "string") {
    throw new Error("executionDate must be YYYY-MM-DD");
  }
  const match = date.trim().match(/^(\d{4})-(\d{1,2})-(\d{1,2})$/);
  if (!match) {
    throw new Error("executionDate must be YYYY-MM-DD, for example 2026-05-26");
  }
  const [, year, month, day] = match;
  const normalized = `${year}-${month.padStart(2, "0")}-${day.padStart(2, "0")}`;
  const parsed = new Date(`${normalized}T00:00:00Z`);
  if (
    Number.isNaN(parsed.getTime()) ||
    parsed.toISOString().slice(0, 10) !== normalized
  ) {
    throw new Error(`Invalid executionDate: ${date}`);
  }
  return normalized;
}

function startJob(name: string, cwd: string, command: string, args: string[]): Job {
  const id = `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  const job: Job = { id, name, status: "running", startedAt: new Date().toISOString(), log: "" };
  jobs.set(id, job);
  const child = spawn(command, args, { cwd, env: process.env });
  child.stdout.on("data", (chunk) => {
    job.log += chunk.toString();
  });
  child.stderr.on("data", (chunk) => {
    job.log += chunk.toString();
  });
  child.on("close", (code) => {
    job.exitCode = code;
    job.finishedAt = new Date().toISOString();
    job.status = code === 0 ? "success" : "failed";
  });
  child.on("error", (err) => {
    job.status = "failed";
    job.finishedAt = new Date().toISOString();
    job.log += `\n${err.stack || err.message}\n`;
  });
  return job;
}

async function parseJsonBody(req: IncomingMessageLike): Promise<Record<string, unknown>> {
  const chunks: Buffer[] = [];
  for await (const chunk of req) chunks.push(Buffer.from(chunk));
  if (!chunks.length) return {};
  return JSON.parse(Buffer.concat(chunks).toString("utf8"));
}

type IncomingMessageLike = AsyncIterable<Buffer | string> & { method?: string; url?: string };

function html() {
  return `<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Local Trade Ops</title>
  <style>
    :root { color-scheme: light; --ink:#1d241f; --muted:#68736b; --line:#d9dfd7; --bg:#f7f3e8; --card:#fffdf6; --accent:#0e7c66; --warn:#b45309; }
    body { margin:0; font-family: ui-serif, Georgia, "Times New Roman", serif; background: radial-gradient(circle at 20% 0%, #e0f1df, transparent 28rem), var(--bg); color:var(--ink); }
    header { padding: 28px 34px 12px; }
    h1 { margin:0; font-size:32px; letter-spacing:-0.03em; }
    main { padding: 0 34px 34px; display:grid; grid-template-columns: 1.1fr .9fr; gap:18px; }
    section { background: color-mix(in srgb, var(--card) 92%, white); border:1px solid var(--line); border-radius:18px; padding:18px; box-shadow: 0 14px 40px #0000000d; }
    .span-all { grid-column:1 / -1; }
    h2 { margin:0 0 12px; font-size:18px; }
    button { border:0; border-radius:999px; background:var(--accent); color:white; padding:10px 14px; margin:4px; cursor:pointer; font-weight:700; }
    button.secondary { background:#30453f; }
    button.warn { background:var(--warn); }
    input { padding:9px 11px; border-radius:10px; border:1px solid var(--line); margin:4px; }
    details { border:1px solid var(--line); border-radius:14px; padding:10px 12px; margin-top:12px; background:#ffffff66; }
    summary { cursor:pointer; font-weight:800; }
    .flow { display:grid; gap:12px; margin-top:4px; }
    .primary-actions { display:grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap:14px; margin-top:4px; }
    .big-action { border-left:5px solid var(--accent); background:#f6fbf4; border-radius:16px; padding:14px; }
    .big-action h3 { margin:0 0 8px; font-size:18px; }
    .big-action p { margin:0 0 10px; line-height:1.55; }
    .big-action button { font-size:15px; padding:12px 16px; }
    .step { border-left:4px solid var(--accent); padding:10px 12px; background:#f6fbf4; border-radius:12px; }
    .step h3 { margin:0 0 6px; font-size:15px; }
    .step p { margin:0 0 8px; }
    .time-note { margin:8px 0; padding:9px 11px; border-radius:12px; background:#fff8df; color:#4c3b1d; line-height:1.55; }
    .time-note b { color:#2f2718; }
    .playbook { display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap:10px; margin:10px 0; }
    .play-card { background:#fffdf7; border:1px solid #dce5d7; border-radius:14px; padding:10px 12px; }
    .play-card h4 { margin:0 0 6px; font-size:14px; }
    .play-card p { margin:0; line-height:1.5; }
    .pill-row { display:flex; flex-wrap:wrap; gap:6px; margin:8px 0 2px; }
    .pill { display:inline-flex; align-items:center; border-radius:999px; padding:5px 8px; font-size:12px; font-weight:800; background:#e8f3ea; color:#173f34; }
    .pill.warn { background:#fff0d7; color:#7c2d12; }
    .pill.stop { background:#fde5dc; color:#9a3412; }
    .simple-rule { margin:10px 0 4px; padding:12px; border-radius:14px; background:#eef7ef; border:1px solid #c9dfc9; line-height:1.6; }
    .submit-row { display:flex; justify-content:flex-end; margin:14px 0 4px; }
    .discipline { margin:10px 0 4px; padding:10px 12px; border-radius:14px; background:#eef7ef; border:1px solid #c9dfc9; line-height:1.55; }
    .discipline h4 { margin:0 0 6px; font-size:14px; }
    .discipline ul { margin:0; padding-left:18px; }
    .discipline li { margin:3px 0; }
    .danger { color:#9a3412; font-weight:800; }
    .inline { display:flex; flex-wrap:wrap; align-items:center; gap:4px; }
    .fill-input { width:90px; padding:6px 8px; }
    .job-card { margin-top:12px; padding:10px 12px; border-radius:12px; background:#17231f; color:#ecf7ed; font-size:13px; }
    .job-card.failed { background:#4a1f16; }
    .job-card.success { background:#173f34; }
    .job-card.running { background:#17231f; }
    .job-card b { color:#fff7c2; }
    .job-log { margin:8px 0 0; max-height:130px; overflow:auto; white-space:pre-wrap; color:#fff7e0; }
    table { width:100%; border-collapse:collapse; font-size:13px; }
    td, th { border-bottom:1px solid var(--line); padding:7px; text-align:left; }
    pre { white-space:pre-wrap; background:#17231f; color:#ecf7ed; padding:14px; border-radius:12px; max-height:360px; overflow:auto; }
    .grid { display:grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap:12px; }
    .muted { color:var(--muted); }
    @media (max-width: 900px) { main { grid-template-columns:1fr; padding: 0 18px 24px; } header { padding:22px 18px 10px; } .primary-actions, .playbook { grid-template-columns:1fr; } }
  </style>
</head>
<body>
  <header>
    <h1>Local Trade Ops</h1>
    <p class="muted">数据更新、信号生成、人工 review、实盘任务触发、收益展示。<a href="/short-hold" style="color:#8a4f18;font-weight:800;">进入单峰短持有页面</a></p>
  </header>
  <main>
    <section>
      <h2>操作台</h2>
      <div class="primary-actions">
        <div class="big-action">
          <h3>1. 生成当天预测/下单清单</h3>
          <p class="muted">一键完成：更新最新数据、校验/生成预测、生成候选订单、自动确认、导出东方财富手工下单清单。你不需要理解中间步骤。</p>
          <button onclick="run('prepare-daily-orders')">生成当天清单</button>
        </div>
        <div class="big-action">
          <h3>2. 回填持仓/成交</h3>
          <p class="muted">09:30-09:35 人工逐只下单后，把东方财富「当日成交」里的实际成交股数和成交均价填回来，系统据此更新现金、持仓和收益。</p>
          <button onclick="run('apply-manual-fills')" class="secondary">应用成交回填</button>
          <button onclick="run('generate-and-send-wechat-snapshot')" class="secondary">生成并推送日报</button>
        </div>
      </div>
      <div class="flow">
        <div class="step">
          <h3>买入原则</h3>
          <div class="simple-rule">
            <b>09:30 前不下单；09:30-09:35 打开东方财富 App 人工逐只买。</b><br>
            程序只给候选股票和 <code>price_5pct</code> 红线；具体挂多少由你看盘口决定，但不超过红线。成交后只回填东方财富「当日成交」里的真实股数和均价。
          </div>
          <div class="time-note">
            <b>A股交易时间：</b>09:15-09:25 开盘集合竞价；09:30-11:30 上午连续竞价；13:00-14:57 下午连续竞价；14:57-15:00 收盘集合竞价。<br>
            <b>我们的窗口：</b>09:25 只观察开盘价和涨跌幅；09:30-09:35 人工买入；尾盘/收盘前处理需要卖出的单子。
          </div>
          <div class="pill-row">
            <span class="pill">≤3%：舒适区，可正常看盘口买</span>
            <span class="pill warn">3%-5%：偏高但允许，人来判断</span>
            <span class="pill stop">&gt;5%：主单跳过，可顺延看备选</span>
          </div>
        </div>
      </div>
      <div id="jobSummary" class="job-card">等待操作...</div>
      <p class="muted">默认不会真实下单；当前适合东方财富 App 人工执行。只有配置 LIVE_TRADING_ENABLED=1 和 LIVE_ORDER_HOOK 后，才会接券商 adapter。</p>
    </section>
    <section>
      <h2>账户 / 状态</h2>
      <div id="account"></div>
    </section>
    <section class="span-all">
      <h2>日报快照</h2>
      <div id="snapshot"></div>
    </section>
    <section class="span-all">
      <h2>订单</h2>
      <div id="orders"></div>
    </section>
    <section style="grid-column:1 / -1">
      <h2>任务日志</h2>
      <pre id="jobs">loading...</pre>
    </section>
  </main>
  <script>
    async function api(path, body) {
      const res = await fetch(path, body ? {method:'POST', headers:{'content-type':'application/json'}, body:JSON.stringify(body)} : {});
      if (!res.ok) throw new Error(await res.text());
      return res.json();
    }
    function table(rows) {
      if (!rows || !rows.length) return '<p class="muted">暂无</p>';
      const cols = Object.keys(rows[0]);
      return '<table><thead><tr>' + cols.map(c => '<th>'+c+'</th>').join('') + '</tr></thead><tbody>' +
        rows.map(r => '<tr>' + cols.map(c => '<td>'+(r[c] ?? '')+'</td>').join('') + '</tr>').join('') + '</tbody></table>';
    }
    function fillEditor(rows) {
      if (!rows || !rows.length) return '<p class="muted">暂无回填模板。先生成手工下单清单。</p>';
      return '<table><thead><tr><th>角色</th><th>代码</th><th>方向</th><th>委托股数</th><th>参考价</th><th>成交股数</th><th>成交价</th><th>备注</th></tr></thead><tbody>' +
        rows.map((r, i) => '<tr>' +
          '<td>' + escapeHtml(r.order_role || 'primary') + '</td>' +
          '<td>' + escapeHtml(r.instrument) + '</td>' +
          '<td>' + escapeHtml(r.action) + '</td>' +
          '<td>' + escapeHtml(r.shares) + '</td>' +
          '<td>' + escapeHtml(r.estimated_price) + '</td>' +
          '<td><input class="fill-input" data-fill-row="' + i + '" data-fill-field="fill_shares" value="' + escapeHtml(r.fill_shares || '') + '" placeholder="实际股数"></td>' +
          '<td><input class="fill-input" data-fill-row="' + i + '" data-fill-field="fill_price" value="' + escapeHtml(r.fill_price || '') + '" placeholder="成交价"></td>' +
          '<td><input class="fill-input" data-fill-row="' + i + '" data-fill-field="operator_note" value="' + escapeHtml(r.operator_note || '') + '" placeholder="可选"></td>' +
        '</tr>').join('') + '</tbody></table>';
    }
    function escapeHtml(text) {
      return String(text ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
    }
    function jobSummary(job) {
      const el = document.querySelector('#jobSummary');
      if (!job) {
        el.className = 'job-card';
        el.textContent = '等待操作...';
        return;
      }
      el.className = 'job-card ' + job.status;
      const log = (job.log || '').trim();
      const shortLog = log.split('\\n').slice(-8).join('\\n');
      el.innerHTML = '<b>最近任务</b>: ' + escapeHtml(job.name) + ' ｜ ' + escapeHtml(job.status) + ' ｜ ' + escapeHtml(job.startedAt) +
        (shortLog ? '<div class="job-log">' + escapeHtml(shortLog) + '</div>' : '');
      const logEl = el.querySelector('.job-log');
      if (logEl) logEl.scrollTop = logEl.scrollHeight;
    }
    function renderSnapshot(snapshot) {
      const el = document.querySelector('#snapshot');
      if (!snapshot) {
        el.innerHTML = '<p class="muted">暂无日报快照。回填成交后点击“生成日报快照”。</p>';
        return;
      }
      const account = snapshot.account || {};
      const orders = snapshot.orders || {};
      const positions = snapshot.positions || [];
      const operations = (orders.operations || []).map(row => ({
        code: row.code,
        action: row.action,
        planned: row.planned_shares,
        filled: row.filled_shares,
        ref_price: row.ref_price,
        price_5pct: row.price_5pct,
        fill_price: row.fill_price,
        status: row.status,
        note: row.note || '',
      }));
      const fmtPct = v => (v === null || v === undefined || v === '') ? '-' : (Number(v).toFixed(2) + '%');
      el.innerHTML =
        '<p class="muted">生成时间：' + escapeHtml(snapshot.generated_at || '-') + ' ｜ 日期：' + escapeHtml(snapshot.date || '-') + '</p>' +
        '<div class="grid">' +
          '<div><b>NAV</b><br>' + escapeHtml(account.nav ?? '-') + '</div>' +
          '<div><b>现金</b><br>' + escapeHtml(account.cash ?? '-') + '</div>' +
          '<div><b>仓位</b><br>' + fmtPct(account.position_ratio_pct) + '</div>' +
          '<div><b>累计收益</b><br>' + fmtPct(account.cum_return_pct) + '</div>' +
          '<div><b>本次/当日盈亏</b><br>' + escapeHtml(account.daily_pnl ?? '-') + '</div>' +
          '<div><b>本次/当日收益</b><br>' + fmtPct(account.daily_return_pct) + '</div>' +
          '<div><b>已成交</b><br>' + escapeHtml(orders.filled_count ?? 0) + ' / 计划 ' + escapeHtml(orders.planned_count ?? 0) + '</div>' +
          '<div><b>跳过/未成交</b><br>' + escapeHtml(orders.skipped_count ?? 0) + '</div>' +
        '</div>' +
        '<h3>当日操作</h3>' + table(operations) +
        '<h3>持仓快照</h3>' + table(positions);
    }
    function hasFillDraft() {
      return Array.from(document.querySelectorAll('.fill-input')).some(input => input.value.trim() !== '');
    }
    function isEditingFill() {
      return Boolean(document.activeElement && document.activeElement.classList && document.activeElement.classList.contains('fill-input'));
    }
    function renderOrders(s) {
      document.querySelector('#orders').innerHTML =
        '<h3>手工下单清单</h3><p class="muted">signal_date 是信号日期，表示用这天收盘后的数据/预测生成清单；execution_date 是人工下单执行日。estimated_price 是昨晚参考价；price_3pct 是舒适区提示；price_5pct 是最高买入红线。主单超过 price_5pct、涨停、买不进就跳过；role=backup 是备选补位，优先参考 model_rank/score 和盘口，由你决定是否补位。</p>' + table(s.workflow.brokerTicketOrders) +
        '<h3>成交回填</h3><p class="muted">填东方财富「当日成交」里的实际成交股数和成交均价。没成交/跳过/撤单填 0 或留空，并在备注写原因。</p>' + fillEditor(s.workflow.fillTemplateRows) +
        '<div class="submit-row"><button id="submit-fills" onclick="run(\\'apply-manual-fills\\')" class="secondary">提交成交回填</button></div>' +
        '<p class="muted">最新实盘任务: ' + (s.workflow.latestTask || '-') +
        '<br>最新提交记录: ' + (s.workflow.latestSubmission || '-') +
        '<br>最新回填模板: ' + (s.workflow.latestFillTemplate || '-') +
        '<br>最新已应用成交: ' + (s.workflow.latestAppliedFills || '-') + '</p>';
    }
    async function refresh(options = {}) {
      const s = await api('/api/status');
      const latestJob = s.jobs?.[0];
      jobSummary(latestJob);
      document.querySelector('#account').innerHTML = '<div class="grid">' +
        '<div><b>NAV</b><br>' + (s.account.latestNav?.nav ?? '-') + '</div>' +
        '<div><b>现金</b><br>' + (s.account.latestNav?.cash ?? '-') + '</div>' +
        '<div><b>持仓市值</b><br>' + (s.account.latestNav?.position_value ?? '-') + '</div>' +
        '<div><b>成本</b><br>' + (s.account.latestNav?.trade_cost ?? '-') + '</div>' +
        '</div><h3>状态</h3>' +
        '<p><b>本地最新交易日</b>: ' + (s.data.lastCalendarDate || '-') + ' ｜ 今天=' + s.data.todayLocal + '</p>' +
        '<p><b>今日下单清单</b>: ' + (s.workflow.brokerTicketOrders?.length ? '已生成 ' + s.workflow.brokerTicketOrders.length + ' 条' : '暂无') + '</p>' +
        '<p><b>数据日历</b>: ' + s.data.calendarTail.join(', ') + '</p>' +
        '<p><b>定时链路</b>: ' + (s.scheduler.enabled ? '开启' : '关闭') + ' ｜ 自动跑到Review=' + (s.scheduler.autoAfterClose ? 'YES' : 'NO') + ' ｜ ' + s.scheduler.schedule + ' ｜ lastRun=' + (s.scheduler.lastRunDate || '-') + '</p>' +
        '<p><b>策略</b>: top' + s.env.topk + '/drop' + s.env.nDrop + ', weights=' + s.env.weights + ', filter=' + s.env.filterMode + '</p>' +
        '<p><b>Live</b>: mode=' + s.env.liveBrokerMode + ', enabled=' + s.env.liveTradingEnabled + ', hook=' + (s.env.liveOrderHook || '-') + '</p>' +
        '<p><b>最新报告</b>: ' + (s.workflow.latestReport || '-') + '</p>' +
        '<h3>持仓</h3>' + table(s.account.positions);
      const dateInput = document.querySelector('#date');
      if (dateInput && !dateInput.value) dateInput.placeholder = '默认执行日 ' + (s.workflow.defaultExecutionDate || s.data.todayLocal);
      const shouldKeepOrderDraft = isEditingFill() || hasFillDraft();
      if (options.forceOrders || !shouldKeepOrderDraft) {
        renderOrders(s);
      }
      renderSnapshot(s.workflow.latestSnapshotData);
      document.querySelector('#jobs').textContent = s.jobs.map(j => '['+j.status+'] '+j.name+' '+j.startedAt+'\\n'+j.log).join('\\n\\n') || '暂无任务';
      document.querySelector('#jobs').scrollTop = document.querySelector('#jobs').scrollHeight;
    }
    async function run(action) {
      const button = action === 'apply-manual-fills' ? document.querySelector('#submit-fills') : null;
      try {
        if (button) {
          button.disabled = true;
          button.textContent = '提交中...';
        }
        const dateEl = document.querySelector('#date');
        const executionDate = dateEl ? dateEl.value.trim() : '';
        const needsDate = ['approve','export-manual-ticket','refresh-realtime-quotes','trigger-live-task','submit-live-orders','apply-manual-fills','mock-all-fills','generate-daily-snapshot','generate-and-send-wechat-snapshot','after-open'].includes(action);
        const status = await api('/api/status');
        const resolvedDate = executionDate || status.workflow.defaultExecutionDate || status.data.todayLocal;
        if (needsDate && !resolvedDate) return alert('未找到可用执行日');
        document.querySelector('#jobSummary').textContent = '已触发 ' + action + '，正在执行...';
        if (action === 'mock-all-fills') {
          if (!confirm('Mock 会把当前清单按计划股数和 estimated_price 全部视为成交并更新账本，只用于裸跑观察收益。确认继续？')) return;
        }
        if (action === 'apply-manual-fills') {
          const rows = (status.workflow.fillTemplateRows || []).map((row, i) => {
            const next = {...row};
            document.querySelectorAll('[data-fill-row="' + i + '"]').forEach(input => {
              next[input.dataset.fillField] = input.value.trim();
            });
            return next;
          });
          if (!rows.length) return alert('暂无回填模板，请先生成手工下单清单');
          const hasFill = rows.some(row => Number(row.fill_shares || 0) > 0 && Number(row.fill_price || 0) > 0);
          if (!hasFill) return alert('还没有填写任何实际成交。请先在“成交股数”和“成交价”里填东方财富「当日成交」数据；如果今天都没成交，就不用点“应用成交回填”。');
          await api('/api/save-fills', {executionDate: resolvedDate, rows});
        }
        await api('/api/run/' + action, needsDate ? {executionDate: resolvedDate} : {});
        if (button) button.textContent = '已提交，处理中...';
        setTimeout(() => refresh({forceOrders: true}), 1200);
      } catch (err) {
        const message = err && err.message ? err.message : String(err);
        document.querySelector('#jobSummary').textContent = '操作失败：' + message;
        alert('操作失败：' + message);
        if (button) {
          button.disabled = false;
          button.textContent = '提交成交回填';
        }
      }
    }
    refresh();
    setInterval(refresh, 30 * 60 * 1000);
  </script>
</body>
</html>`;
}

function shortHoldHtml() {
  return `<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Short Hold Ops</title>
  <style>
    :root { color-scheme: light; --ink:#1d241f; --muted:#68736b; --line:#d9dfd7; --bg:#f5f0e4; --card:#fffdf6; --accent:#8a4f18; --green:#0e7c66; }
    body { margin:0; font-family: ui-serif, Georgia, "Times New Roman", serif; background: radial-gradient(circle at 18% 0%, #f7d9a7, transparent 26rem), var(--bg); color:var(--ink); }
    header { padding: 28px 34px 12px; }
    h1 { margin:0; font-size:32px; letter-spacing:-0.03em; }
    main { padding: 0 34px 34px; display:grid; grid-template-columns: 1fr 1fr; gap:18px; }
    section { background: color-mix(in srgb, var(--card) 93%, white); border:1px solid var(--line); border-radius:18px; padding:18px; box-shadow: 0 14px 40px #0000000d; }
    .span-all { grid-column:1 / -1; }
    h2 { margin:0 0 12px; font-size:18px; }
    h3 { margin:12px 0 8px; font-size:16px; }
    button { border:0; border-radius:999px; background:var(--green); color:white; padding:10px 14px; margin:4px; cursor:pointer; font-weight:800; }
    button.secondary { background:#30453f; }
    button.warn { background:var(--accent); }
    .cards { display:grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap:14px; }
    .card { border-left:5px solid var(--accent); background:#fff7e8; border-radius:16px; padding:14px; }
    .card p { line-height:1.55; margin:0 0 10px; }
    .rule { padding:12px; border:1px solid #ebd2a9; border-radius:14px; background:#fff9ec; line-height:1.6; margin-top:12px; }
    .risk { padding:12px; border:1px solid #e8b4a4; border-radius:14px; background:#fff0eb; line-height:1.6; margin-top:12px; }
    .metric-row { display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap:8px; margin-top:10px; }
    .metric { border:1px solid #ead6b4; border-radius:12px; background:#fffdf8; padding:9px 10px; }
    .metric b { display:block; font-size:12px; color:#79521e; }
    .metric span { font-weight:900; font-size:17px; }
    .pill { display:inline-flex; border-radius:999px; padding:5px 8px; margin:4px 4px 0 0; background:#f7ead4; color:#683a0e; font-weight:800; font-size:12px; }
    .job-card { margin-top:12px; padding:10px 12px; border-radius:12px; background:#17231f; color:#ecf7ed; font-size:13px; }
    .job-card.failed { background:#4a1f16; }
    .job-card.success { background:#173f34; }
    .job-log { margin:8px 0 0; max-height:130px; overflow:auto; white-space:pre-wrap; color:#fff7e0; }
    .grid { display:grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap:12px; }
    table { width:100%; border-collapse:collapse; font-size:13px; }
    td, th { border-bottom:1px solid var(--line); padding:7px; text-align:left; }
    input { padding:7px 9px; border-radius:10px; border:1px solid var(--line); width:90px; }
    pre { white-space:pre-wrap; background:#17231f; color:#ecf7ed; padding:14px; border-radius:12px; max-height:320px; overflow:auto; }
    .muted { color:var(--muted); }
    .submit-row { display:flex; justify-content:flex-end; margin:14px 0 4px; }
    nav a { color:#6b3c12; font-weight:800; }
    @media (max-width: 900px) { main { grid-template-columns:1fr; padding:0 18px 24px; } header { padding:22px 18px 10px; } .cards { grid-template-columns:1fr; } }
  </style>
</head>
<body>
  <header>
    <h1>Short Hold Ops</h1>
    <p class="muted">单峰短持有策略：T 信号，T+1 人工买入，T+2 尾盘/收盘卖出。<nav><a href="/">返回原 T/T+1 页面</a></nav></p>
  </header>
  <main>
    <section>
      <h2>操作台</h2>
      <div class="cards">
        <div class="card">
          <h3>1. 生成短持有清单</h3>
          <p class="muted">复用原来的数据更新和 PKL 推理，然后生成短持有专用 BUY/SELL 清单。SELL 是到期持仓，BUY 是新一轮信号。</p>
          <button onclick="run('prepare-short-hold-orders')">生成短持有清单</button>
        </div>
        <div class="card">
          <h3>2. 回填成交 / 日报</h3>
          <p class="muted">人工按东方财富成交回填。买入会记录 exit_date；到期卖出完成后释放现金。</p>
          <button onclick="run('generate-and-send-short-hold-snapshot')" class="warn">生成并推送短持有日报</button>
        </div>
      </div>
      <div class="rule">
        <b>执行纪律</b><br>
        买入：T+1 09:30-09:35 按清单人工判断，仍遵守不追涨停、不超过 price_5pct。<br>
        卖出：T+2 尾盘/收盘前按到期 SELL 清单优先完成。<br>
        A股交易时间：09:15-09:25 开盘集合竞价；09:30-11:30 上午连续竞价；13:00-14:57 下午连续竞价；14:57-15:00 收盘集合竞价。<br>
        操作节奏：09:25 只观察；09:30-09:35 买入；14:57 前后开始检查到期 SELL，15:00 前完成能成交的卖出。<br>
        当前默认初始资金池：<b>10W</b>；生成手工清单时默认按整体资金池做 BUY 预算，不受模拟账本剩余现金影响；如需实盘口径，可设置 SHORT_HOLD_BUY_BUDGET_MODE=available_cash。默认参数：top4、min_amount=3000万、max_position_pct=70%、softmax 分配。
        <div><span class="pill">独立账本</span><span class="pill">独立回填</span><span class="pill">独立日报</span></div>
      </div>
      <div id="jobSummary" class="job-card">等待操作...</div>
    </section>
    <section>
      <h2>账户 / 状态</h2>
      <div id="account"></div>
    </section>
    <section class="span-all">
      <h2>日报快照</h2>
      <div id="snapshot"></div>
    </section>
    <section class="span-all">
      <h2>短持有订单</h2>
      <div id="orders"></div>
    </section>
    <section class="span-all">
      <h2>任务日志</h2>
      <pre id="jobs">loading...</pre>
    </section>
  </main>
  <script>
    async function api(path, body) {
      const res = await fetch(path, body ? {method:'POST', headers:{'content-type':'application/json'}, body:JSON.stringify(body)} : {});
      if (!res.ok) throw new Error(await res.text());
      return res.json();
    }
    function escapeHtml(text) {
      return String(text ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
    }
    const LABELS = {
      role: '角色',
      order_role: '角色',
      signal_date: '信号日',
      execution_date: '执行日',
      exit_date: '退出日',
      code: '代码',
      instrument: '代码',
      action: '方向',
      shares: '股数',
      ref_price: '参考价',
      price_3pct: '3%提示',
      price_5pct: '5%红线',
      model_rank: '模型排名',
      score: '原始分',
      entry_score: '入仓分',
      signal_score: '当日分',
      return_score: '收益分',
      final_score: '最终分',
      buyability_risk: '买入风险',
      strong_prob: '强势概率',
      liquidity_risk: '流动性风险',
      score_source: '打分源',
      alloc_weight: '分配权重',
      target_value: '目标金额',
      planned_value: '计划金额',
      note: '说明',
      filter_status: '过滤状态',
      filter_reason: '过滤原因',
      close: '收盘价',
      amount20: '20日成交额',
      status: '状态',
    };
    function tableColumns(rows) {
      const seen = new Set();
      const cols = [];
      rows.forEach(row => Object.keys(row || {}).forEach(col => {
        if (!seen.has(col)) { seen.add(col); cols.push(col); }
      }));
      return cols;
    }
    function table(rows) {
      if (!rows || !rows.length) return '<p class="muted">暂无</p>';
      const cols = tableColumns(rows);
      return '<table><thead><tr>' + cols.map(c => '<th>'+escapeHtml(LABELS[c] || c)+'</th>').join('') + '</tr></thead><tbody>' +
        rows.map(r => '<tr>' + cols.map(c => '<td>'+escapeHtml(r[c] ?? '')+'</td>').join('') + '</tr>').join('') + '</tbody></table>';
    }
    function fillEditor(rows) {
      if (!rows || !rows.length) return '<p class="muted">暂无回填模板。先生成短持有清单。</p>';
      return '<table><thead><tr><th>角色</th><th>代码</th><th>方向</th><th>执行日</th><th>退出日</th><th>委托股数</th><th>参考价</th><th>成交股数</th><th>成交价</th><th>备注</th></tr></thead><tbody>' +
        rows.map((r, i) => '<tr>' +
          '<td>' + escapeHtml(r.order_role || 'primary') + '</td>' +
          '<td>' + escapeHtml(r.instrument) + '</td>' +
          '<td>' + escapeHtml(r.action) + '</td>' +
          '<td>' + escapeHtml(r.execution_date) + '</td>' +
          '<td>' + escapeHtml(r.exit_date || '') + '</td>' +
          '<td>' + escapeHtml(r.shares) + '</td>' +
          '<td>' + escapeHtml(r.estimated_price) + '</td>' +
          '<td><input data-fill-row="' + i + '" data-fill-field="fill_shares" value="' + escapeHtml(r.fill_shares || '') + '" placeholder="股数"></td>' +
          '<td><input data-fill-row="' + i + '" data-fill-field="fill_price" value="' + escapeHtml(r.fill_price || '') + '" placeholder="价格"></td>' +
          '<td><input data-fill-row="' + i + '" data-fill-field="operator_note" value="' + escapeHtml(r.operator_note || '') + '" placeholder="可选"></td>' +
        '</tr>').join('') + '</tbody></table>';
    }
    function jobSummary(job) {
      const el = document.querySelector('#jobSummary');
      if (!job) { el.className = 'job-card'; el.textContent = '等待操作...'; return; }
      el.className = 'job-card ' + job.status;
      const log = (job.log || '').trim().split('\\n').slice(-8).join('\\n');
      el.innerHTML = '<b>最近任务</b>: ' + escapeHtml(job.name) + ' ｜ ' + escapeHtml(job.status) + ' ｜ ' + escapeHtml(job.startedAt) +
        (log ? '<div class="job-log">' + escapeHtml(log) + '</div>' : '');
    }
    function hasFillDraft() {
      return Array.from(document.querySelectorAll('[data-fill-row]')).some(input => input.value.trim() !== '');
    }
    function isEditingFill() {
      return Boolean(document.activeElement && document.activeElement.dataset && document.activeElement.dataset.fillRow);
    }
    function budgetModeLabel(mode) {
      if (mode === 'available_cash') return '当前剩余现金';
      if (mode === 'reserve_nav' || mode === 'available_cash_reserved') return '当前现金-NAV预留';
      if (mode === 'capital_pool_reserved') return '整体资金池-预留';
      return '整体资金池';
    }
    function renderSnapshot(snapshot) {
      if (!snapshot) {
        document.querySelector('#snapshot').innerHTML = '<p class="muted">暂无短持有日报。回填后点击“生成并推送短持有日报”。</p>';
        return;
      }
      const account = snapshot.account || {};
      document.querySelector('#snapshot').innerHTML =
        '<p class="muted">生成时间：' + escapeHtml(snapshot.generated_at || '-') + ' ｜ 日期：' + escapeHtml(snapshot.date || '-') + '</p>' +
        '<div class="grid">' +
        '<div><b>NAV</b><br>' + escapeHtml(account.nav ?? '-') + '</div>' +
        '<div><b>现金</b><br>' + escapeHtml(account.cash ?? '-') + '</div>' +
        '<div><b>仓位</b><br>' + escapeHtml(Number(account.position_ratio_pct || 0).toFixed(2)) + '%</div>' +
        '<div><b>累计收益</b><br>' + escapeHtml(Number(account.cum_return_pct || 0).toFixed(2)) + '%</div>' +
        '<div><b>本日盈亏</b><br>' + escapeHtml(account.daily_pnl ?? '-') + '</div>' +
        '<div><b>本日收益</b><br>' + escapeHtml(Number(account.daily_return_pct || 0).toFixed(2)) + '%</div>' +
        '</div><h3>当日操作</h3>' + table(snapshot.orders?.operations || []) +
        '<h3>当前持仓</h3>' + table(snapshot.positions || []);
    }
    function renderOrders(s) {
      const rows = s.workflow.brokerTicketOrders || [];
      const firstBuy = rows.find(r => r.action === 'BUY');
      const timeline = firstBuy
        ? (firstBuy.signal_date + ' 信号；' + firstBuy.execution_date + ' 开盘人工买；' + firstBuy.exit_date + ' 尾盘/收盘卖')
        : '生成清单后会显示具体 signal / execution / exit 日期';
      const scoringText = s.env.scoringMode === 'v3'
        ? '当前为 v3 重排：原始分是收益模型分；最终分 = 收益分 - 买入风险惩罚 + 强势概率加分 - 流动性惩罚，用于排序和 softmax 分配。'
        : '当前为 v2：按原始收益模型分排序和 softmax 分配。';
      document.querySelector('#orders').innerHTML =
        '<h3>近5个信号日 Filter 后 Top1</h3><p class="muted">来源：短持有 candidate review，排除 FILTERED/SKIPPED，只看已经通过过滤并可进入主单或 backup 的 BUY 候选。</p>' + table(s.workflow.recentFilteredTopRows) +
        '<h3>模型 Top5 / 过滤检查</h3><p class="muted">score 是模型原始分；FILTERED 表示不会进入主单/备选池。</p>' + table(s.workflow.modelTopReviewRows) +
        '<h3>手工下单清单</h3><p class="muted"><b>本轮节奏：</b>' + escapeHtml(timeline) + '。' + escapeHtml(scoringText) + ' role=primary 是主单；role=backup 是 primary 因涨停/高开/买不进/未成交时使用的组合补位，按释放主单资金重新 softmax 分配。入仓分是已有持仓当初买入时的分数，没有入仓则为空；当日分是最新信号日模型跑出的当天原始分。BUY 的 price_5pct 是最高买入红线；SELL 是到期持仓，优先卖出。</p>' + table(rows) +
        '<h3>成交回填</h3><p class="muted">填东方财富「当日成交」里的真实成交股数/均价；没成交填 0 或留空并备注。</p>' + fillEditor(s.workflow.fillTemplateRows) +
        '<div class="submit-row"><button onclick="run(\\'short-hold-apply-fills\\')" class="secondary">提交短持有回填</button></div>' +
        '<p class="muted">最新任务: ' + (s.workflow.latestTask || '-') +
        '<br>最新清单: ' + (s.workflow.latestBrokerTicket || '-') +
        '<br>最新回填模板: ' + (s.workflow.latestFillTemplate || '-') +
        '<br>最新已应用成交: ' + (s.workflow.latestAppliedFills || '-') + '</p>';
    }
    async function refresh(options = {}) {
      const s = await api('/api/short-hold/status');
      jobSummary(s.jobs?.[0]);
      document.querySelector('#account').innerHTML =
        '<div class="grid">' +
        '<div><b>NAV</b><br>' + (s.account.latestNav?.nav ?? '-') + '</div>' +
        '<div><b>现金</b><br>' + (s.account.latestNav?.cash ?? '-') + '</div>' +
        '<div><b>持仓市值</b><br>' + (s.account.latestNav?.position_value ?? '-') + '</div>' +
        '<div><b>成本</b><br>' + (s.account.latestNav?.trade_cost ?? '-') + '</div>' +
        '</div><h3>状态</h3>' +
        '<p><b>本地最新交易日</b>: ' + (s.data.lastCalendarDate || '-') + ' ｜ 今天=' + s.data.todayLocal + '</p>' +
        '<p><b>今日短持有清单</b>: ' + (s.workflow.brokerTicketOrders?.length ? '已生成 ' + s.workflow.brokerTicketOrders.length + ' 条' : '暂无') + '</p>' +
        '<p><b>初始资金池</b>: ' + Number(s.env.capital || 100000).toLocaleString('zh-CN') + ' ｜ <b>BUY预算</b>: ' + budgetModeLabel(s.env.buyBudgetMode) + ' ｜ <b>策略</b>: top' + s.env.topk + ' + backup' + s.env.backupCount + ', scoring=' + s.env.scoringMode + ', maxPositionPct=' + s.env.maxPositionPct + ', minAmount=' + s.env.minAmount + ', temperature=' + s.env.scoreTemperature + ', 排除创业板/科创板=' + s.env.excludeRestrictedMarkets + '</p>' +
        '<p><b>数据日历</b>: ' + s.data.calendarTail.join(', ') + '</p>' +
        '<h3>持仓</h3>' + table(s.account.positions);
      if (options.forceOrders || (!isEditingFill() && !hasFillDraft())) renderOrders(s);
      renderSnapshot(s.workflow.latestSnapshotData);
      document.querySelector('#jobs').textContent = s.jobs.map(j => '['+j.status+'] '+j.name+' '+j.startedAt+'\\n'+j.log).join('\\n\\n') || '暂无任务';
      document.querySelector('#jobs').scrollTop = document.querySelector('#jobs').scrollHeight;
    }
    async function run(action) {
      try {
        const status = await api('/api/short-hold/status');
        const executionDate = status.workflow.defaultExecutionDate || status.data.todayLocal;
        document.querySelector('#jobSummary').textContent = '已触发 ' + action + '，正在执行...';
        if (action === 'short-hold-apply-fills') {
          const rows = (status.workflow.fillTemplateRows || []).map((row, i) => {
            const next = {...row};
            document.querySelectorAll('[data-fill-row="' + i + '"]').forEach(input => {
              next[input.dataset.fillField] = input.value.trim();
            });
            return next;
          });
          if (!rows.length) return alert('暂无短持有回填模板，请先生成短持有清单');
          const hasFill = rows.some(row => Number(row.fill_shares || 0) > 0 && Number(row.fill_price || 0) > 0);
          if (!hasFill) return alert('还没有填写任何实际成交。');
          await api('/api/short-hold/save-fills', {executionDate, rows});
        }
        await api('/api/short-hold/run/' + action, {executionDate});
        setTimeout(() => refresh({forceOrders: true}), 1200);
      } catch (err) {
        const message = err && err.message ? err.message : String(err);
        document.querySelector('#jobSummary').textContent = '操作失败：' + message;
        alert('操作失败：' + message);
      }
    }
    refresh();
    setInterval(refresh, 30 * 60 * 1000);
  </script>
</body>
</html>`;
}

function send(res: import("node:http").ServerResponse, status: number, body: unknown, type = "application/json") {
  res.writeHead(status, { "content-type": `${type}; charset=utf-8` });
  res.end(type === "application/json" ? JSON.stringify(body, null, 2) : String(body));
}

const server = createServer(async (req, res) => {
  try {
    const url = new URL(req.url || "/", "http://localhost");
    if (req.method === "GET" && url.pathname === "/") return send(res, 200, html(), "text/html");
    if (req.method === "GET" && url.pathname === "/short-hold") return send(res, 200, shortHoldHtml(), "text/html");
    if (req.method === "GET" && url.pathname === "/api/status") return send(res, 200, loadStatus());
    if (req.method === "GET" && url.pathname === "/api/short-hold/status") return send(res, 200, loadShortHoldStatus());
    if (req.method === "POST" && url.pathname === "/api/save-fills") {
      const body = await parseJsonBody(req);
      const executionDate = validateDate(body.executionDate);
      const env = parseEnv(ENV_PATH);
      const fillDir = env.LIVE_FILL_DIR || resolve(OPS_HOME, "live_fills");
      mkdirSync(fillDir, { recursive: true });
      const fillPath = resolve(fillDir, `manual_fill_template_${executionDate}.csv`);
      const rows = Array.isArray(body.rows) ? body.rows : [];
      if (!rows.length) throw new Error("rows is required");
      writeCsv(fillPath, rows);
      return send(res, 200, { path: fillPath, nRows: rows.length });
    }
    if (req.method === "POST" && url.pathname === "/api/short-hold/save-fills") {
      const body = await parseJsonBody(req);
      const executionDate = validateDate(body.executionDate);
      const env = parseEnv(ENV_PATH);
      const fillDir = env.SHORT_HOLD_FILL_DIR || resolve(OPS_HOME, "short_hold_fills");
      mkdirSync(fillDir, { recursive: true });
      const fillPath = resolve(fillDir, `short_hold_fill_template_${executionDate}.csv`);
      const rows = Array.isArray(body.rows) ? body.rows : [];
      if (!rows.length) throw new Error("rows is required");
      writeCsv(fillPath, rows);
      return send(res, 200, { path: fillPath, nRows: rows.length });
    }
    if (req.method === "POST" && url.pathname.startsWith("/api/short-hold/run/")) {
      const action = url.pathname.split("/").pop() || "";
      const body = await parseJsonBody(req);
      let job: Job;
      if (action === "prepare-short-hold-orders") job = startJob(action, OPS_HOME, "bash", ["bin/prepare_short_hold_orders.sh"]);
      else if (action === "short-hold-apply-fills") job = startJob(action, OPS_HOME, "bash", ["bin/short_hold_apply_fills.sh", validateDate(body.executionDate)]);
      else if (action === "short-hold-daily-snapshot") job = startJob(action, OPS_HOME, "bash", ["bin/short_hold_daily_snapshot.sh", validateDate(body.executionDate)]);
      else if (action === "generate-and-send-short-hold-snapshot") job = startJob(action, OPS_HOME, "bash", ["bin/generate_and_send_short_hold_snapshot.sh", validateDate(body.executionDate)]);
      else return send(res, 404, { error: `unknown short-hold action: ${action}` });
      return send(res, 200, job);
    }
    if (req.method === "POST" && url.pathname.startsWith("/api/run/")) {
      const action = url.pathname.split("/").pop() || "";
      const body = await parseJsonBody(req);
      let job: Job;
      if (action === "update-data") job = startJob(action, PAPER_HOME, "bash", ["bin/update_data.sh"]);
      else if (action === "enable-data-scheduler") {
        const config = loadSchedulerConfig();
        config.dataUpdate.enabled = true;
        saveSchedulerConfig(config);
        job = { id: `config-${Date.now()}`, name: action, status: "success", startedAt: new Date().toISOString(), finishedAt: new Date().toISOString(), exitCode: 0, log: "Data scheduler enabled." };
        jobs.set(job.id, job);
      } else if (action === "disable-data-scheduler") {
        const config = loadSchedulerConfig();
        config.dataUpdate.enabled = false;
        saveSchedulerConfig(config);
        job = { id: `config-${Date.now()}`, name: action, status: "success", startedAt: new Date().toISOString(), finishedAt: new Date().toISOString(), exitCode: 0, log: "Data scheduler disabled." };
        jobs.set(job.id, job);
      }
      else if (action === "sync-predictions") job = startJob(action, OPS_HOME, "bash", ["bin/sync_predictions.sh"]);
      else if (action === "prepare-daily-orders") job = startJob(action, OPS_HOME, "bash", ["bin/prepare_daily_manual_orders.sh"]);
      else if (action === "generate-predictions") job = startJob(action, OPS_HOME, "bash", ["bin/generate_predictions.sh"]);
      else if (action === "daily-pipeline") job = startJob(action, OPS_HOME, "bash", ["bin/daily_after_data_pipeline.sh"]);
      else if (action === "after-close") job = startJob(action, OPS_HOME, "bash", ["bin/after_close_local.sh"]);
      else if (action === "approve") job = startJob(action, OPS_HOME, parseEnv(ENV_PATH).PYTHON_BIN || "python3", ["bin/approve_orders.py", "--execution-date", validateDate(body.executionDate), "--status", "approved"]);
      else if (action === "export-manual-ticket") job = startJob(action, OPS_HOME, "bash", ["bin/export_manual_order_ticket.sh", validateDate(body.executionDate), "--force"]);
      else if (action === "refresh-realtime-quotes") job = startJob(action, OPS_HOME, parseEnv(ENV_PATH).PYTHON_BIN || "python3", ["bin/fetch_realtime_quotes.py", "--execution-date", validateDate(body.executionDate)]);
      else if (action === "trigger-live-task") job = startJob(action, OPS_HOME, "bash", ["bin/trigger_live_task.sh", validateDate(body.executionDate), "--force"]);
      else if (action === "submit-live-orders") job = startJob(action, OPS_HOME, "bash", ["bin/submit_live_orders.sh", validateDate(body.executionDate), "--force"]);
      else if (action === "apply-manual-fills") job = startJob(action, OPS_HOME, "bash", ["bin/apply_manual_fills.sh", validateDate(body.executionDate)]);
      else if (action === "generate-daily-snapshot") job = startJob(action, OPS_HOME, "bash", ["bin/generate_daily_snapshot.sh", validateDate(body.executionDate)]);
      else if (action === "generate-and-send-wechat-snapshot") job = startJob(action, OPS_HOME, "bash", ["bin/generate_and_send_wechat_snapshot.sh", validateDate(body.executionDate)]);
      else if (action === "send-wechat-snapshot") job = startJob(action, OPS_HOME, "bash", ["bin/send_wechat_snapshot.sh"]);
      else if (action === "mock-all-fills") job = startJob(action, OPS_HOME, "bash", ["bin/mock_all_fills.sh", validateDate(body.executionDate)]);
      else if (action === "after-open") job = startJob(action, OPS_HOME, "bash", ["bin/after_open_local.sh", validateDate(body.executionDate)]);
      else return send(res, 404, { error: `unknown action: ${action}` });
      return send(res, 200, job);
    }
    return send(res, 404, { error: "not found" });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return send(res, 500, { error: message });
  }
});

const port = Number(process.env.PORT || 8787);
server.listen(port, "127.0.0.1", () => {
  console.log(`Local Trade Ops running at http://127.0.0.1:${port}`);
});
setInterval(schedulerTick, 30_000);
schedulerTick();
