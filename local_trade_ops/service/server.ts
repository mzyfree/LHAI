import { createServer } from "node:http";
import { spawn } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, readdirSync, statSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { URL } from "node:url";
import { loadNakedKStatus, nakedKHtml, recordNakedKFill } from "./naked-k.ts";
import { loadNakedKModelStatus, nakedKModelHtml, saveNakedKModelDecision } from "./naked-k-model.ts";
import {
  ignoreNakedKLiveDecision,
  loadNakedKLiveStatus,
  nakedKLiveHtml,
  recordNakedKLiveDecision,
  recordNakedKLiveFill,
  recordNakedKLiveOpen,
} from "./naked-k-live.ts";

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
const RECOMMENDATION_TRACKER_DIR = resolve(OPS_HOME, "recommendation_tracker");
const RECOMMENDATION_STORE_PATH = resolve(RECOMMENDATION_TRACKER_DIR, "recommendations.json");
const jobs = new Map<string, Job>();
let lastSchedulerCheck = "";

type RecommendationRecord = {
  id: string;
  source: string;
  recommendationDate: string;
  code: string;
  name: string;
  focusPrice: string;
  suggestedPosition: string;
  targetPrice: string;
  supportPrice: string;
  rationale: string;
  disclaimer: string;
  advisor: string;
  buyPrice: string;
  buyLots: string;
  buyDate: string;
  buySavedAt?: string;
  sellPrice: string;
  sellLots: string;
  sellDate: string;
  sellSavedAt?: string;
  note: string;
  createdAt: string;
  updatedAt: string;
};

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

function readRecommendationRecords(): RecommendationRecord[] {
  if (!existsSync(RECOMMENDATION_STORE_PATH)) return [];
  try {
    const records = JSON.parse(readFileSync(RECOMMENDATION_STORE_PATH, "utf8"));
    return Array.isArray(records) ? records as RecommendationRecord[] : [];
  } catch {
    return [];
  }
}

function writeRecommendationRecords(records: RecommendationRecord[]) {
  mkdirSync(RECOMMENDATION_TRACKER_DIR, { recursive: true });
  writeFileSync(RECOMMENDATION_STORE_PATH, `${JSON.stringify(records, null, 2)}\n`, "utf8");
}

function cleanText(value: unknown): string {
  return typeof value === "string" ? value.trim() : value == null ? "" : String(value).trim();
}

function optionalDate(value: unknown, field: string): string {
  const text = cleanText(value);
  if (!text) return "";
  try {
    return validateDate(text);
  } catch {
    throw new Error(`${field} must be YYYY-MM-DD`);
  }
}

function positiveNumberText(value: unknown, field: string): string {
  const text = cleanText(value);
  if (!text) return "";
  const number = Number(text);
  if (!Number.isFinite(number) || number < 0) throw new Error(`${field} must be a non-negative number`);
  return String(number);
}

function recommendationView(record: RecommendationRecord) {
  const buyLots = Number(record.buyLots || 0);
  const sellLots = Number(record.sellLots || 0);
  const buyPrice = Number(record.buyPrice || 0);
  const sellPrice = Number(record.sellPrice || 0);
  const boughtShares = buyLots * 100;
  const soldShares = sellLots * 100;
  const heldShares = Math.max(0, boughtShares - soldShares);
  const realizedPnl = buyPrice > 0 && sellPrice > 0 && soldShares > 0
    ? (sellPrice - buyPrice) * soldShares
    : null;
  const realizedReturn = buyPrice > 0 && sellPrice > 0 ? (sellPrice / buyPrice - 1) * 100 : null;
  const status = !boughtShares
    ? "待买入"
    : !soldShares
      ? "持有中"
      : heldShares > 0
        ? "部分卖出"
        : "已卖出";
  return { ...record, boughtShares, soldShares, heldShares, realizedPnl, realizedReturn, status };
}

function recommendationStatus() {
  const records = readRecommendationRecords()
    .map(recommendationView)
    .sort((a, b) => `${b.recommendationDate}${b.createdAt}`.localeCompare(`${a.recommendationDate}${a.createdAt}`));
  const today = localParts("Asia/Shanghai").date;
  return {
    today,
    records,
    stats: {
      total: records.length,
      todayCount: records.filter((record) => record.recommendationDate === today).length,
      pending: records.filter((record) => record.status === "待买入").length,
      holding: records.filter((record) => record.status === "持有中" || record.status === "部分卖出").length,
      sold: records.filter((record) => record.status === "已卖出").length,
    },
  };
}

function recommendationFromBody(body: Record<string, unknown>, existing?: RecommendationRecord): RecommendationRecord {
  const now = new Date().toISOString();
  const record: RecommendationRecord = {
    id: existing?.id || `rec-${Date.now()}-${Math.random().toString(16).slice(2)}`,
    source: cleanText(body.source ?? existing?.source),
    recommendationDate: optionalDate(body.recommendationDate ?? existing?.recommendationDate, "recommendationDate"),
    code: cleanText(body.code ?? existing?.code).toUpperCase(),
    name: cleanText(body.name ?? existing?.name),
    focusPrice: cleanText(body.focusPrice ?? existing?.focusPrice),
    suggestedPosition: cleanText(body.suggestedPosition ?? existing?.suggestedPosition),
    targetPrice: cleanText(body.targetPrice ?? existing?.targetPrice),
    supportPrice: cleanText(body.supportPrice ?? existing?.supportPrice),
    rationale: cleanText(body.rationale ?? existing?.rationale),
    disclaimer: cleanText(body.disclaimer ?? existing?.disclaimer),
    advisor: cleanText(body.advisor ?? existing?.advisor),
    buyPrice: positiveNumberText(body.buyPrice ?? existing?.buyPrice, "buyPrice"),
    buyLots: positiveNumberText(body.buyLots ?? existing?.buyLots, "buyLots"),
    buyDate: optionalDate(body.buyDate ?? existing?.buyDate, "buyDate"),
    buySavedAt: cleanText(body.buySavedAt ?? existing?.buySavedAt),
    sellPrice: positiveNumberText(body.sellPrice ?? existing?.sellPrice, "sellPrice"),
    sellLots: positiveNumberText(body.sellLots ?? existing?.sellLots, "sellLots"),
    sellDate: optionalDate(body.sellDate ?? existing?.sellDate, "sellDate"),
    sellSavedAt: cleanText(body.sellSavedAt ?? existing?.sellSavedAt),
    note: cleanText(body.note ?? existing?.note),
    createdAt: existing?.createdAt || now,
    updatedAt: now,
  };
  if (!record.source || !record.recommendationDate || !record.code || !record.name) {
    throw new Error("source, recommendationDate, code and name are required");
  }
  if (Number(record.sellLots || 0) > Number(record.buyLots || 0)) {
    throw new Error("sellLots cannot exceed buyLots");
  }
  return record;
}

function normalizeRecommendationDate(value: string): string {
  const text = value.trim().replace(/[./]/g, "-");
  if (/^\d{1,2}-\d{1,2}$/.test(text)) {
    const [month, day] = text.split("-").map(Number);
    return validateDate(`${localParts("Asia/Shanghai").date.slice(0, 4)}-${String(month).padStart(2, "0")}-${String(day).padStart(2, "0")}`);
  }
  return validateDate(text);
}

function fieldFromRecommendationText(text: string, labels: string[]): string {
  for (const label of labels) {
    const match = text.match(new RegExp(`(?:^|\\n)\\s*${label}\\s*[：:]\\s*([^\\n]+)`, "i"));
    if (match) return match[1].trim();
  }
  return "";
}

function parseRecommendationText(rawText: string): RecommendationRecord[] {
  const blocks = rawText
    .replace(/\r/g, "")
    .split(/(?=【[^\n】]+】\s*(?:教学案例|推荐|案例)?\s*\n?\s*入选时间[：:])/)
    .map((block) => block.trim())
    .filter(Boolean);
  if (!blocks.length) throw new Error("未识别到推荐文本，请从“【服务名】… 入选时间：…”开始粘贴");
  const now = new Date().toISOString();
  return blocks.map((block) => {
    const source = (block.match(/^【([^】]+)】/) || [])[1] || "第三方推荐";
    const date = normalizeRecommendationDate(fieldFromRecommendationText(block, ["入选时间", "入选日期"]));
    const code = fieldFromRecommendationText(block, ["股票代码", "代码"]).replace(/\s/g, "");
    const name = fieldFromRecommendationText(block, ["股票名称", "名称"]);
    const rationaleMatch = block.match(/入选理由\s*[：:]\s*([\s\S]*?)(?=\n\s*【风险提示】|$)/);
    const disclaimerMatch = block.match(/【风险提示】\s*([\s\S]*)$/);
    const advisorMatch = block.match(/投资顾问\s*[：:]?\s*([^\s]+).*?执业编号\s*[：:]?\s*([A-Z0-9]+)/);
    const record: RecommendationRecord = {
      id: `rec-${Date.now()}-${Math.random().toString(16).slice(2)}`,
      source,
      recommendationDate: date,
      code: code.toUpperCase(),
      name,
      focusPrice: fieldFromRecommendationText(block, ["关注价格"]),
      suggestedPosition: fieldFromRecommendationText(block, ["参考仓位", "建议仓位"]),
      targetPrice: fieldFromRecommendationText(block, ["目标价格"]) || (fieldFromRecommendationText(block, ["压力价格"]) ? `压力：${fieldFromRecommendationText(block, ["压力价格"])}` : ""),
      supportPrice: fieldFromRecommendationText(block, ["支撑价格"]),
      rationale: rationaleMatch?.[1]?.trim() || "",
      disclaimer: disclaimerMatch?.[1]?.trim() || "",
      advisor: advisorMatch ? `${advisorMatch[1]} / ${advisorMatch[2]}` : "",
      buyPrice: "", buyLots: "", buyDate: "", sellPrice: "", sellLots: "", sellDate: "", note: "",
      createdAt: now, updatedAt: now,
    };
    if (!record.code || !record.name) throw new Error(`未能识别股票代码或名称：${block.slice(0, 80)}`);
    return record;
  });
}

function updateRecommendationFill(id: string, side: "buy" | "sell", body: Record<string, unknown>) {
  const records = readRecommendationRecords();
  const index = records.findIndex((record) => record.id === id);
  if (index < 0) throw new Error("recommendation not found");
  const existing = records[index];
  const savedAt = new Date().toISOString();
  const fillDate = localParts("Asia/Shanghai").date;
  const patch = side === "buy"
    ? {
      buyPrice: positiveNumberText(body.buyPrice, "buyPrice"),
      buyLots: positiveNumberText(body.buyLots, "buyLots"),
      buyDate: fillDate,
      buySavedAt: savedAt,
    }
    : {
      sellPrice: positiveNumberText(body.sellPrice, "sellPrice"),
      sellLots: positiveNumberText(body.sellLots, "sellLots"),
      sellDate: fillDate,
      sellSavedAt: savedAt,
    };
  const record = recommendationFromBody({ ...existing, ...patch }, existing);
  records[index] = record;
  writeRecommendationRecords(records);
  return recommendationView(record);
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

function latestDirectoryWithFile(dir: string, fileName: string): string | null {
  if (!existsSync(dir)) return null;
  const candidates = readdirSync(dir, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => resolve(dir, entry.name))
    .filter((path) => existsSync(resolve(path, fileName)))
    .sort((a, b) => {
      const diff = statSync(a).mtimeMs - statSync(b).mtimeMs;
      return diff || a.localeCompare(b);
    });
  return candidates.length ? candidates[candidates.length - 1] : null;
}

function loadLatestShortHoldV3ResearchReport() {
  const env = parseEnv(ENV_PATH);
  const researchDir = env.SHORT_HOLD_V3_RESEARCH_DIR || resolve(OPS_HOME, "reports", "short_hold_v3_research");
  const latestDir = latestDirectoryWithFile(researchDir, "research_report.md");
  if (!latestDir) return { reportPath: "", markdown: "" };
  const reportPath = resolve(latestDir, "research_report.md");
  return {
    reportPath,
    markdown: readFileSync(reportPath, "utf8"),
  };
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

function fileInfoWithPath(path: string, name?: string) {
  return {
    name: name || path.split("/").pop() || path,
    path,
    ...fileInfo(path),
  };
}

function sideModelArtifacts(dir: string) {
  if (!dir) return [];
  return ["buyability_model.pkl", "strong_model.pkl", "metadata.json"].map((name) =>
    fileInfoWithPath(resolve(dir, name), name),
  );
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
    nakedK: {
      enabled: true,
      timezone: "Asia/Shanghai",
      phases: {
        preflight: { time: "09:00", lastRunDate: "", lastJobId: "", attempts: 0, lastStatus: "waiting" },
        open: { time: "09:25", lastRunDate: "", lastJobId: "", attempts: 0, lastStatus: "waiting" },
        tail: { time: "14:50", lastRunDate: "", lastJobId: "", attempts: 0, lastStatus: "waiting" },
        settle: { time: "15:10", lastRunDate: "", lastJobId: "", attempts: 0, lastStatus: "waiting" },
      },
    },
    nakedKLive: {
      enabled: true,
      timezone: "Asia/Shanghai",
      preflightTime: "09:00",
      openTime: "09:25",
      firstIntradayTime: "09:35",
      lastIntradayTime: "14:55",
      lastPreflightDate: "",
      lastOpenDate: "",
      lastIntradaySlot: "",
      lastJobId: "",
      lastStatus: "waiting",
      runningPhase: "",
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
    nakedK: {
      ...defaults.nakedK,
      ...(loaded.nakedK || {}),
      phases: {
        ...defaults.nakedK.phases,
        ...(loaded.nakedK?.phases || {}),
      },
    },
    nakedKLive: {
      ...defaults.nakedKLive,
      ...(loaded.nakedKLive || {}),
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
  const liveConfig = config.nakedKLive;
  if (liveConfig.enabled) {
    const now = localParts(liveConfig.timezone);
    const weekday = new Intl.DateTimeFormat("en-US", { timeZone: liveConfig.timezone, weekday: "short" }).format(new Date());
    const isWeekday = weekday !== "Sat" && weekday !== "Sun";
    const launchLive = (phase: "preflight" | "open" | "intraday", slot: string) => {
      const env = parseEnv(ENV_PATH);
      const args = ["bin/naked_k_llm_live_runner.py", "--phase", phase, "--date", now.date];
      if (phase === "intraday") args.push("--at", now.time);
      const job = startJob(`naked-k-live-${phase}-${slot}`, OPS_HOME, env.PYTHON_BIN || "python3", args, (code) => {
        const latest = loadSchedulerConfig();
        latest.nakedKLive.lastStatus = code === 0 ? "success" : "failed";
        latest.nakedKLive.runningPhase = "";
        if (code === 0 && phase === "preflight") latest.nakedKLive.lastPreflightDate = now.date;
        if (code === 0 && phase === "open") latest.nakedKLive.lastOpenDate = now.date;
        saveSchedulerConfig(latest);
      });
      liveConfig.lastJobId = job.id;
      liveConfig.lastStatus = "running";
      liveConfig.runningPhase = phase;
    };
    if (isWeekday && !liveConfig.runningPhase && now.time >= liveConfig.preflightTime && now.time < "15:00" && liveConfig.lastPreflightDate !== now.date) {
      launchLive("preflight", now.date);
    }
    if (isWeekday && !liveConfig.runningPhase && liveConfig.lastPreflightDate === now.date && now.time >= liveConfig.openTime && now.time < "15:00" && liveConfig.lastOpenDate !== now.date) {
      launchLive("open", now.date);
    }
    const minute = Number(now.time.split(":")[1]);
    const isFiveMinuteSlot = minute % 5 === 0;
    const inMorning = now.time >= liveConfig.firstIntradayTime && now.time <= "11:30";
    const inAfternoon = now.time >= "13:05" && now.time <= liveConfig.lastIntradayTime;
    const slot = `${now.date}T${now.time}`;
    if (isWeekday && !liveConfig.runningPhase && liveConfig.lastOpenDate === now.date && isFiveMinuteSlot && (inMorning || inAfternoon) && liveConfig.lastIntradaySlot !== slot) {
      launchLive("intraday", slot);
      liveConfig.lastIntradaySlot = slot;
    }
    saveSchedulerConfig(config);
  }
  if (config.nakedK.enabled) {
    const nakedNow = localParts(config.nakedK.timezone);
    const weekday = new Intl.DateTimeFormat("en-US", { timeZone: config.nakedK.timezone, weekday: "short" }).format(new Date());
    if (weekday !== "Sat" && weekday !== "Sun") {
      for (const [phase, phaseConfig] of Object.entries(config.nakedK.phases)) {
        if (nakedNow.time < phaseConfig.time || phaseConfig.lastRunDate === nakedNow.date) continue;
        const env = parseEnv(ENV_PATH);
        const newDay = phaseConfig.lastRunDate !== nakedNow.date;
        const attempts = newDay ? 1 : Number(phaseConfig.attempts || 0) + 1;
        if (!newDay && attempts > 3) continue;
        const job = startJob(`naked-k-${phase}`, OPS_HOME, env.PYTHON_BIN || "python3", [
          "bin/naked_k_daily_runner.py", "--phase", phase, "--date", nakedNow.date,
        ], (code) => {
          const latest = loadSchedulerConfig();
          const saved = latest.nakedK.phases[phase as keyof typeof latest.nakedK.phases];
          saved.lastStatus = code === 0 ? "success" : "failed";
          if (code !== 0 && Number(saved.attempts || 0) < 3) saved.lastRunDate = "";
          saveSchedulerConfig(latest);
        });
        phaseConfig.lastRunDate = nakedNow.date;
        phaseConfig.lastJobId = job.id;
        phaseConfig.attempts = attempts;
        phaseConfig.lastStatus = "running";
      }
      saveSchedulerConfig(config);
    }
  }
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

type ShortHoldProfile = "v2" | "v3" | "v5";

function loadShortHoldStatus(profile: ShortHoldProfile = "v2") {
  const env = parseEnv(ENV_PATH);
  const isV3 = profile === "v3";
  const isV5 = profile === "v5";
  const stateDir = isV5
    ? env.SHORT_HOLD_V5_STATE_DIR || resolve(OPS_HOME, "state", "short_hold_v5_baseline")
    : isV3
    ? env.SHORT_HOLD_V3_STATE_DIR || resolve(OPS_HOME, "state", "short_hold_v3")
    : env.SHORT_HOLD_STATE_DIR || resolve(OPS_HOME, "state", "short_hold_single_peak");
  const reportDir = isV5
    ? env.SHORT_HOLD_V5_REPORT_DIR || resolve(OPS_HOME, "reports", "short_hold_v5_baseline")
    : isV3
    ? env.SHORT_HOLD_V3_REPORT_DIR || resolve(OPS_HOME, "reports", "short_hold_v3")
    : env.SHORT_HOLD_REPORT_DIR || resolve(OPS_HOME, "reports", "short_hold_single_peak");
  const taskDir = isV5
    ? env.SHORT_HOLD_V5_TASK_DIR || resolve(OPS_HOME, "short_hold_v5_tasks")
    : isV3
    ? env.SHORT_HOLD_V3_TASK_DIR || resolve(OPS_HOME, "short_hold_v3_tasks")
    : env.SHORT_HOLD_TASK_DIR || resolve(OPS_HOME, "short_hold_tasks");
  const submissionDir = isV5
    ? env.SHORT_HOLD_V5_SUBMISSION_DIR || resolve(OPS_HOME, "short_hold_v5_submissions")
    : isV3
    ? env.SHORT_HOLD_V3_SUBMISSION_DIR || resolve(OPS_HOME, "short_hold_v3_submissions")
    : env.SHORT_HOLD_SUBMISSION_DIR || resolve(OPS_HOME, "short_hold_submissions");
  const fillDir = isV5
    ? env.SHORT_HOLD_V5_FILL_DIR || resolve(OPS_HOME, "short_hold_v5_fills")
    : isV3
    ? env.SHORT_HOLD_V3_FILL_DIR || resolve(OPS_HOME, "short_hold_v3_fills")
    : env.SHORT_HOLD_FILL_DIR || resolve(OPS_HOME, "short_hold_fills");
  const snapshotDir = isV5
    ? env.SHORT_HOLD_V5_SNAPSHOT_DIR || resolve(OPS_HOME, "short_hold_v5_daily_snapshots")
    : isV3
    ? env.SHORT_HOLD_V3_SNAPSHOT_DIR || resolve(OPS_HOME, "short_hold_v3_daily_snapshots")
    : env.SHORT_HOLD_SNAPSHOT_DIR || resolve(OPS_HOME, "short_hold_daily_snapshots");
  const predDir = isV5
    ? env.SHORT_HOLD_V5_PRED_DIR || resolve(OPS_HOME, "preds", "csi1000_short_hold_v5")
    : isV3
    ? env.SHORT_HOLD_V3_PRED_DIR || resolve(OPS_HOME, "preds", "csi1000_short_hold_v3")
    : env.SHORT_HOLD_PRED_DIR || resolve(OPS_HOME, "preds", "csi1000_short_hold_v2");
  const modelPackage = isV5
    ? env.SHORT_HOLD_V5_MODEL_PACKAGE ||
      resolve(OPS_HOME, "model_packages", "latest_csi1000_short_hold_v2_model_package.tar.gz")
    : isV3
    ? env.SHORT_HOLD_V3_MODEL_PACKAGE ||
      resolve(OPS_HOME, "model_packages", "latest_csi1000_short_hold_v3_model_package.tar.gz")
    : env.SHORT_HOLD_MODEL_PACKAGE || "";
  const modelExtractDir = isV5
    ? env.SHORT_HOLD_V5_MODEL_EXTRACT_DIR || resolve(OPS_HOME, "model_packages", "extracted", "short_hold_v5")
    : isV3
    ? env.SHORT_HOLD_V3_MODEL_EXTRACT_DIR ||
      env.SHORT_HOLD_MODEL_EXTRACT_DIR ||
      resolve(OPS_HOME, "model_packages", "extracted")
    : env.SHORT_HOLD_MODEL_EXTRACT_DIR || "";
  const autoPredict = isV5
    ? env.SHORT_HOLD_V5_AUTO_PREDICT || env.SHORT_HOLD_AUTO_PREDICT || "1"
    : isV3
    ? env.SHORT_HOLD_V3_AUTO_PREDICT || env.SHORT_HOLD_AUTO_PREDICT || "1"
    : env.SHORT_HOLD_AUTO_PREDICT || "";
  const sideModelDir = isV3
    ? env.SHORT_HOLD_V3_SIDE_MODEL_DIR || resolve(OPS_HOME, "model_packages", "short_hold_v3_side_models")
    : "";
  const sectorContextEnabled = isV3 ? env.SHORT_HOLD_V3_SECTOR_CONTEXT_ENABLED || "1" : "";
  const researchProfile = isV3 ? env.SHORT_HOLD_V3_RESEARCH_PROFILE || "baseline" : "";
  const sectorStockMeta = isV3
    ? env.SHORT_HOLD_V3_STOCK_META || resolve(PAPER_HOME, "data", "meta", "tushare_stock_basic_latest.csv")
    : "";
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
  const predPaths = isV3
    ? [
        env.SHORT_HOLD_V3_PRED_A || resolve(predDir, "xgb_csi1000_long_prod2026.pkl"),
        env.SHORT_HOLD_V3_PRED_B || resolve(predDir, "doubleensemble_csi1000_short_prod2026.pkl"),
        env.SHORT_HOLD_V3_PRED_C || resolve(predDir, "catboost_csi1000_long_prod2026.pkl"),
      ].filter(Boolean)
    : isV5
      ? [
          env.SHORT_HOLD_V5_PRED_A || resolve(predDir, "xgb_csi1000_long_prod2026.pkl"),
          env.SHORT_HOLD_V5_PRED_B || resolve(predDir, "doubleensemble_csi1000_short_prod2026.pkl"),
          env.SHORT_HOLD_V5_PRED_C || resolve(predDir, "catboost_csi1000_long_prod2026.pkl"),
        ].filter(Boolean)
    : [env.PRED_A, env.PRED_B, env.PRED_C].filter(Boolean);
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
      profile,
      qlibDir,
      stateDir,
      reportDir,
      taskDir,
      submissionDir,
      fillDir,
      snapshotDir,
      predDir,
      modelPackage,
      modelExtractDir,
      autoPredict,
      sideModelDir,
      capital: (isV5 ? env.SHORT_HOLD_V5_CAPITAL : isV3 ? env.SHORT_HOLD_V3_CAPITAL : env.SHORT_HOLD_CAPITAL) || env.SHORT_HOLD_CAPITAL || env.CAPITAL || "100000",
      topk: (isV5 ? env.SHORT_HOLD_V5_TOPK : isV3 ? env.SHORT_HOLD_V3_TOPK : env.SHORT_HOLD_TOPK) || env.SHORT_HOLD_TOPK || "4",
      backupCount: (isV5 ? env.SHORT_HOLD_V5_BACKUP_COUNT : isV3 ? env.SHORT_HOLD_V3_BACKUP_COUNT : env.SHORT_HOLD_BACKUP_COUNT) || env.SHORT_HOLD_BACKUP_COUNT || "3",
      excludeRestrictedMarkets:
        (isV5 ? env.SHORT_HOLD_V5_EXCLUDE_RESTRICTED_MARKETS : isV3 ? env.SHORT_HOLD_V3_EXCLUDE_RESTRICTED_MARKETS : env.SHORT_HOLD_EXCLUDE_RESTRICTED_MARKETS) ||
        env.SHORT_HOLD_EXCLUDE_RESTRICTED_MARKETS ||
        "1",
      minAmount: (isV5 ? env.SHORT_HOLD_V5_MIN_AMOUNT : isV3 ? env.SHORT_HOLD_V3_MIN_AMOUNT : env.SHORT_HOLD_MIN_AMOUNT) || env.SHORT_HOLD_MIN_AMOUNT || "30000000",
      maxPositionPct:
        (isV5 ? env.SHORT_HOLD_V5_MAX_POSITION_PCT : isV3 ? env.SHORT_HOLD_V3_MAX_POSITION_PCT : env.SHORT_HOLD_MAX_POSITION_PCT) ||
        env.SHORT_HOLD_MAX_POSITION_PCT ||
        "0.70",
      buyBudgetMode:
        (isV5 ? env.SHORT_HOLD_V5_BUY_BUDGET_MODE : isV3 ? env.SHORT_HOLD_V3_BUY_BUDGET_MODE : env.SHORT_HOLD_BUY_BUDGET_MODE) ||
        env.SHORT_HOLD_BUY_BUDGET_MODE ||
        env.BUY_BUDGET_MODE ||
        "capital_pool",
      reserveCashPct:
        (isV5 ? env.SHORT_HOLD_V5_RESERVE_CASH_PCT : isV3 ? env.SHORT_HOLD_V3_RESERVE_CASH_PCT : env.SHORT_HOLD_RESERVE_CASH_PCT) ||
        env.SHORT_HOLD_RESERVE_CASH_PCT ||
        "0.02",
      scoreTemperature:
        (isV5 ? env.SHORT_HOLD_V5_SCORE_TEMPERATURE : isV3 ? env.SHORT_HOLD_V3_SCORE_TEMPERATURE : env.SHORT_HOLD_SCORE_TEMPERATURE) ||
        env.SHORT_HOLD_SCORE_TEMPERATURE ||
        "1.0",
      scoringMode: isV3 ? "v3" : "v2",
      v3Alpha: env.SHORT_HOLD_V3_ALPHA || "1.0",
      v3Beta: env.SHORT_HOLD_V3_BETA || "0.0",
      v3Gamma: env.SHORT_HOLD_V3_GAMMA || "0.0",
      sectorContextEnabled,
      researchProfile,
      sectorStockMeta,
    },
    data: {
      currentLink: fileInfo(qlibDir),
      calendarTail,
      lastCalendarDate: calendarTail.at(-1) || null,
      todayLocal,
    },
    predictions: predPaths.map((p) => fileInfo(p)),
    sideModelArtifacts: isV3 ? sideModelArtifacts(sideModelDir) : [],
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

function optionalFiniteNumber(value: unknown): number | null {
  if (value === undefined || value === null || String(value).trim() === "") {
    return null;
  }
  return finiteNumber(value);
}

function optionalFormatNumber(value: unknown, digits: number): string {
  const n = optionalFiniteNumber(value);
  return n === null ? "" : n.toFixed(digits);
}

function optionalFormatPercent(value: unknown, digits = 2): string {
  const n = optionalFiniteNumber(value);
  return n === null ? "" : `${(n * 100).toFixed(digits)}%`;
}

function v3ValueColumns(row: Record<string, string>): Record<string, string> {
  const hasV3Values = [
    row.final_score,
    row.buyable_score,
    row.strong_score,
    row.buyability_risk,
    row.strong_prob,
    row.liquidity_risk,
    row.liquidity_quality_score,
    row.industry,
    row.sector_context_note,
    row.sector_heat_score,
    row.sector_return_3d,
    row.sector_breadth_1d,
  ].some(
    (value) => finiteNumber(value) !== null || String(value || "").trim() !== "",
  );
  if (!hasV3Values) {
    return {};
  }
  const buyabilityRisk = optionalFiniteNumber(row.buyability_risk);
  const buyableScore = optionalFiniteNumber(row.buyable_score) ?? (buyabilityRisk === null ? null : 1 - buyabilityRisk);
  const strongScore = optionalFiniteNumber(row.strong_score) ?? optionalFiniteNumber(row.strong_prob);
  return {
    return_score: optionalFormatNumber(row.return_score || row.score, 6),
    return_rank_score: optionalFormatNumber(row.return_rank_score, 4),
    buyable_score: buyableScore === null ? "" : buyableScore.toFixed(4),
    strong_score: strongScore === null ? "" : strongScore.toFixed(4),
    final_score: optionalFormatNumber(row.final_score, 6),
    buyability_risk: optionalFormatPercent(row.buyability_risk),
    strong_prob: optionalFormatPercent(row.strong_prob),
    liquidity_risk: optionalFormatNumber(row.liquidity_risk, 4),
    liquidity_quality_score: optionalFormatNumber(row.liquidity_quality_score, 4),
    model_agreement_score: optionalFormatNumber(row.model_agreement_score, 4),
    sector_rank_score: optionalFormatNumber(row.sector_rank_score, 4),
    relative_sector_rank_score: optionalFormatNumber(row.relative_sector_rank_score, 4),
    market_regime: row.market_regime || "",
    research_profile: row.research_profile || "baseline",
    research_profile_note: row.research_profile_note || "",
    score_source: row.score_source || "v3",
    industry: row.industry || "",
    sector_return_1d: optionalFormatPercent(row.sector_return_1d),
    sector_return_3d: optionalFormatPercent(row.sector_return_3d),
    sector_return_5d: optionalFormatPercent(row.sector_return_5d),
    sector_breadth_1d: optionalFormatPercent(row.sector_breadth_1d),
    sector_amount_ratio_5d: optionalFormatNumber(row.sector_amount_ratio_5d, 2),
    stock_vs_sector_return_3d: optionalFormatPercent(row.stock_vs_sector_return_3d),
    sector_heat_score: optionalFormatNumber(row.sector_heat_score, 4),
    sector_context_note: row.sector_context_note || "",
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

function startJob(name: string, cwd: string, command: string, args: string[], onClose?: (code: number | null) => void): Job {
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
    onClose?.(code);
  });
  child.on("error", (err) => {
    job.status = "failed";
    job.finishedAt = new Date().toISOString();
    job.log += `\n${err.stack || err.message}\n`;
  });
  return job;
}

function runJsonCommand(command: string, args: string[], cwd: string): Promise<Record<string, unknown>> {
  return new Promise((resolvePromise, reject) => {
    const child = spawn(command, args, { cwd, env: process.env });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });
    child.on("error", reject);
    child.on("close", (code) => {
      let result: Record<string, unknown> | null = null;
      try {
        result = JSON.parse(stdout.trim()) as Record<string, unknown>;
      } catch {
        reject(new Error(stderr.trim() || stdout.trim() || "查询模型分数失败"));
        return;
      }
      if (code !== 0) {
        reject(new Error(cleanText(result.error) || stderr.trim() || "查询模型分数失败"));
        return;
      }
      resolvePromise(result);
    });
  });
}

async function lookupShortHoldV2Score(code: string): Promise<Record<string, unknown>> {
  const env = parseEnv(ENV_PATH);
  const predDir = env.SHORT_HOLD_PRED_DIR || resolve(OPS_HOME, "preds", "csi1000_short_hold_v2");
  const predictions = [
    env.PRED_A || resolve(predDir, "xgb_csi1000_long_prod2026.pkl"),
    env.PRED_B || resolve(predDir, "doubleensemble_csi1000_short_prod2026.pkl"),
    env.PRED_C || resolve(predDir, "catboost_csi1000_long_prod2026.pkl"),
  ];
  const missing = predictions.filter((path) => !existsSync(path));
  if (missing.length) throw new Error(`Short Hold V2 预测文件不存在: ${missing.join(", ")}`);
  const python = env.PYTHON_BIN || "python3";
  return runJsonCommand(python, [
    resolve(OPS_HOME, "bin", "query_short_hold_v2_score.py"),
    "--pred-a", predictions[0],
    "--pred-b", predictions[1],
    "--pred-c", predictions[2],
    "--weights", env.WEIGHTS || "3,1,1",
    "--code", code,
  ], OPS_HOME);
}

async function lookupShortHoldV5Score(code: string): Promise<Record<string, unknown>> {
  const env = parseEnv(ENV_PATH);
  const predDir = env.SHORT_HOLD_V5_PRED_DIR || resolve(OPS_HOME, "preds", "csi1000_short_hold_v5");
  const predictions = [
    env.SHORT_HOLD_V5_PRED_A || resolve(predDir, "xgb_csi1000_long_prod2026.pkl"),
    env.SHORT_HOLD_V5_PRED_B || resolve(predDir, "doubleensemble_csi1000_short_prod2026.pkl"),
    env.SHORT_HOLD_V5_PRED_C || resolve(predDir, "catboost_csi1000_long_prod2026.pkl"),
  ];
  const missing = predictions.filter((path) => !existsSync(path));
  if (missing.length) throw new Error(`Short Hold V5 基线预测文件不存在: ${missing.join(", ")}`);
  const python = env.PYTHON_BIN || "python3";
  return runJsonCommand(python, [
    resolve(OPS_HOME, "bin", "query_short_hold_v2_score.py"),
    "--pred-a", predictions[0],
    "--pred-b", predictions[1],
    "--pred-c", predictions[2],
    "--weights", env.WEIGHTS || "3,1,1",
    "--code", code,
  ], OPS_HOME);
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
    <p class="muted">数据更新、信号生成、人工 review、实盘任务触发、收益展示。<a href="/short-hold" style="color:#8a4f18;font-weight:800;">进入单峰短持有页面</a> ｜ <a href="/short-hold-v5" style="color:#8a4f18;font-weight:800;">进入 V5 冻结基线页面</a></p>
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

function shortHoldHtml(profile: ShortHoldProfile = "v2") {
  const isV3 = profile === "v3";
  const isV5 = profile === "v5";
  const title = isV3 ? "Short Hold V3 Ops" : isV5 ? "Short Hold V5 Baseline" : "Short Hold Ops";
  const apiBase = isV3 ? "/api/short-hold-v3" : isV5 ? "/api/short-hold-v5" : "/api/short-hold";
  const siblingLink = isV3
    ? '<a href="/short-hold">返回 V2 短持有页面</a>'
    : isV5
      ? '<a href="/short-hold">返回 V2 短持有页面</a> ｜ <a href="/short-hold-v3">进入 V3 独立实验页面</a>'
      : '<a href="/short-hold-v3">进入 V3 独立实验页面</a> ｜ <a href="/short-hold-v5">进入 V5 冻结基线页面</a>';
  const scoringDescription = isV3
    ? "短持有 V3 独立实验：复用短持收益模型预测，再用 v3 侧模型/重排逻辑独立生成清单。"
    : isV5
      ? "短持有 V5 冻结基线：策略、融合权重和过滤规则与 V2 保持一致，但使用独立预测、订单、回填、日报和账本目录。"
      : "单峰短持有策略：T 信号，T+1 人工买入，T+2 尾盘/收盘卖出。";
  const researchSection = isV3
    ? `<section class="span-all">
      <h2>最新 V3 研究报告</h2>
      <div id="research"></div>
    </section>`
    : "";
  const v2ScoreLookup = isV3
    ? ""
    : `<div class="score-lookup">
        <b>查询单只股票当日模型分</b><br>
        <span class="muted">输入代码，查看 Short Hold ${isV5 ? "V5 基线" : "V2"} 最新信号日的融合分与排名，不生成订单。</span><br>
        <input id="v2ScoreCode" maxlength="8" placeholder="601225 或 SH601225" />
        <button type="button" id="v2ScoreButton" class="secondary">查询分数</button>
        <span id="v2ScoreResult" class="muted"></span>
      </div>`;
  return `<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>${title}</title>
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
    .decision-table-wrap { overflow-x:auto; border:1px solid var(--line); border-radius:12px; background:#fffdf8; }
    .decision-table { min-width:0; table-layout:fixed; font-size:12px; }
    .decision-table th { background:#fff1d8; color:#5d3918; white-space:normal; }
    .decision-table td { white-space:pre-line; line-height:1.4; vertical-align:top; overflow-wrap:anywhere; }
    .decision-rank { font-weight:900; color:#8a4f18; }
    .decision-guide { margin:8px 0 10px; padding:10px 12px; border-radius:12px; background:#eef7ef; border:1px solid #c9dfc9; line-height:1.55; }
    .diagnostic { margin-top:12px; border:1px solid var(--line); border-radius:12px; padding:0 12px 12px; }
    .diagnostic summary { cursor:pointer; padding:10px 0; font-weight:800; color:#5d3918; }
    input { padding:7px 9px; border-radius:10px; border:1px solid var(--line); width:90px; }
    .score-lookup { margin-top:12px; padding:11px 12px; border:1px solid #d6c39f; border-radius:12px; background:#fffaf0; line-height:1.55; }
    .score-lookup input { width:170px; margin:7px 4px 0 0; }
    .score-lookup button { padding:8px 12px; }
    #v2ScoreResult { display:inline-block; margin-left:6px; font-weight:800; }
    pre { white-space:pre-wrap; background:#17231f; color:#ecf7ed; padding:14px; border-radius:12px; max-height:320px; overflow:auto; }
    .muted { color:var(--muted); }
    .submit-row { display:flex; justify-content:flex-end; margin:14px 0 4px; }
    nav a { color:#6b3c12; font-weight:800; }
    @media (max-width: 900px) { main { grid-template-columns:1fr; padding:0 18px 24px; } header { padding:22px 18px 10px; } .cards { grid-template-columns:1fr; } }
  </style>
</head>
<body>
  <header>
    <h1>${title}</h1>
    <p class="muted">${scoringDescription}<nav><a href="/">返回原 T/T+1 页面</a> ｜ ${siblingLink}</nav></p>
  </header>
  <main>
    <section>
      <h2>操作台</h2>
      <div class="cards">
        <div class="card">
          <h3>1. 生成短持有清单</h3>
          <p class="muted">复用原来的数据更新和 PKL 推理，然后生成${isV3 ? " v3 独立" : ""}短持有专用 BUY/SELL 清单。SELL 是到期持仓，BUY 是新一轮信号。</p>
          <button onclick="run('prepare-short-hold-orders')">生成短持有清单</button>
          ${v2ScoreLookup}
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
        当前默认初始资金池：<b>10W</b>；生成手工清单时默认按整体资金池做 BUY 预算，不受模拟账本剩余现金影响；如需实盘口径，可设置 SHORT_HOLD_BUY_BUDGET_MODE=available_cash。${isV3 ? "V3 当前部署为 Top2 主推荐，行业/一致性/流动性评分只作排序与解释。" : "默认参数：top4、min_amount=3000万、max_position_pct=70%、softmax 分配。"}
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
    ${researchSection}
    <section class="span-all">
      <h2>任务日志</h2>
      <pre id="jobs">loading...</pre>
    </section>
  </main>
  <script>
    const API_BASE = '${apiBase}';
    const IS_V3 = ${isV3 ? "true" : "false"};
    const SCORE_API = '${isV5 ? "/api/short-hold-v5/v5-score" : "/api/short-hold/v2-score"}';
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
      return_rank_score: '收益排名分',
      final_score: '最终分',
      buyability_risk: '买入风险(仅展示)',
      strong_prob: '强势概率',
      liquidity_risk: '流动性风险',
      liquidity_quality_score: '流动性质量',
      model_agreement_score: '模型一致性',
      sector_rank_score: '行业热度排名',
      relative_sector_rank_score: '相对行业排名',
      market_regime: '市场状态',
      score_source: '打分源',
      industry: '行业',
      sector_return_1d: '行业1日',
      sector_return_3d: '行业3日',
      sector_return_5d: '行业5日',
      sector_breadth_1d: '行业上涨占比',
      sector_amount_ratio_5d: '行业量能',
      stock_vs_sector_return_3d: '个股超行业3日',
      sector_heat_score: '行业热度',
      sector_context_note: '行业判断',
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
    function v3DecisionTable(rows, columns) {
      if (!rows || !rows.length) return '<p class="muted">暂无</p>';
      return '<div class="decision-table-wrap"><table class="decision-table"><thead><tr>' +
        columns.map(c => '<th>' + escapeHtml(c.label) + '</th>').join('') +
        '</tr></thead><tbody>' + rows.map((row, index) => '<tr>' + columns.map(column => {
          const value = column.value ? column.value(row, index) : row[column.key];
          return '<td>' + escapeHtml(value ?? '-') + '</td>';
        }).join('') + '</tr>').join('') + '</tbody></table></div>';
    }
    function renderV3OrderBlock(rows, env) {
      const sells = rows.filter(row => row.action === 'SELL');
      const primaryBuys = rows.filter(row => row.action === 'BUY' && row.role === 'primary');
      const backups = rows.filter(row => row.action === 'BUY' && row.role === 'backup');
      const expectedTopk = Number(env.topk || 2);
      const capital = Number(env.capital || 0);
      const plannedValue = row => {
        const raw = row.planned_value;
        if (raw === undefined || raw === null || String(raw).trim() === '') return null;
        const value = Number(raw);
        return Number.isFinite(value) ? value : null;
      };
      const plannedWeight = row => {
        const value = plannedValue(row);
        return value === null || capital <= 0 ? '-' : (value / capital * 100).toFixed(1) + '%';
      };
      const twoLine = (first, second) => first + '\\n' + second;
      const outputProfile = rows.find(row => row.research_profile)?.research_profile || '';
      const profileChanged = rows.length > 0 && (!outputProfile || outputProfile !== (env.researchProfile || 'baseline'));
      const buyColumns = [
        { label: '主推荐', value: (_, index) => 'Top' + (index + 1) },
        { label: '股票 / 行业', value: row => twoLine(row.code || '-', row.industry || '行业待生成') },
        { label: '计划仓位', value: plannedWeight },
        { label: '价格指引', value: row => twoLine('参考 ' + (row.ref_price || '-'), '红线 ' + (row.price_5pct || '-')) },
        { label: '模型分', value: row => twoLine('最终 ' + (row.final_score || '-'), '收益 ' + (row.return_score || '-')) },
        { label: '行业分', value: row => twoLine('热度 ' + (row.sector_heat_score || '-'), '3日 ' + (row.sector_return_3d || '-')) },
        { label: '相对行业', key: 'stock_vs_sector_return_3d' },
        { label: '质量分', value: row => twoLine('一致 ' + (row.model_agreement_score || '-'), '流动 ' + (row.liquidity_quality_score || '-')) },
        { label: '市场状态', key: 'market_regime' },
      ];
      const sellColumns = [
        { label: '优先级', value: () => '先卖' },
        { label: '代码', key: 'code' },
        { label: '股数', key: 'shares' },
        { label: '退出日', key: 'exit_date' },
        { label: '参考价', key: 'ref_price' },
        { label: '入仓分', key: 'entry_score' },
      ];
      const backupColumns = [
        { label: '备选顺序', value: (_, index) => '备选 ' + (index + 1) },
        { label: '代码', key: 'code' },
        { label: '行业', key: 'industry' },
        { label: '股数', key: 'shares' },
        { label: '5%红线', key: 'price_5pct' },
        { label: '最终推荐分', key: 'final_score' },
        { label: '行业热度', key: 'sector_heat_score' },
        { label: '模型一致性', key: 'model_agreement_score' },
        { label: '流动性质量', key: 'liquidity_quality_score' },
      ];
      const warning = profileChanged
        ? '<p class="risk"><b>这是历史清单，不能按当前 V3 策略解读。</b>' + (outputProfile ? '其 Profile 为 ' + escapeHtml(outputProfile) + '。' : '该文件没有 V3 行业/一致性/流动性字段。') + '当前部署为 ' + escapeHtml(env.researchProfile || 'top2_full') + ' / Top' + escapeHtml(expectedTopk) + '。请点击「生成短持有清单」后再按新策略操作。</p>'
        : '';
      const sellBlock = sells.length
        ? '<h3>优先：到期卖出</h3><p class="muted">SELL 与本轮新推荐无关，按到期日先完成卖出。</p>' + v3DecisionTable(sells, sellColumns)
        : '';
      const buyBlock = '<h3>今日应买：Top' + escapeHtml(expectedTopk) + ' 主推荐</h3>' +
        '<div class="decision-guide"><b>只看本表决定是否买入。</b>按 Top1、Top2 的顺序人工检查盘口；不追涨停，委托价不超过 5% 红线。只有对应主推荐买不进/高开超线/未成交时，才看下方备选组合。</div>' +
        v3DecisionTable(primaryBuys, buyColumns);
      const backupBlock = backups.length
        ? '<details class="diagnostic"><summary>查看备选组合（主推荐无法买入时才使用）</summary><p class="muted">备选不与主推荐同时买入；主单释放的预算会在备选中重新 softmax 分配。</p>' + v3DecisionTable(backups, backupColumns) + '</details>'
        : '';
      return warning + sellBlock + buyBlock + backupBlock;
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
        ? '当前为 v3 重排：原始分是收益模型分；最终分由当前 V3 Profile 决定，用于排序和 softmax 分配；买入风险只展示，不扣分。'
        : '当前为 v2：按原始收益模型分排序和 softmax 分配。';
      const orderBlock = IS_V3 ? renderV3OrderBlock(rows, s.env) : table(rows);
      const reviewBlock = IS_V3
        ? '<details class="diagnostic"><summary>诊断：模型 Top5 / 过滤检查</summary><p class="muted">仅排查时查看。score 是模型原始分；FILTERED 表示不会进入主单/备选池。</p>' + table(s.workflow.modelTopReviewRows) + '</details>'
        : '<h3>模型 Top5 / 过滤检查</h3><p class="muted">score 是模型原始分；FILTERED 表示不会进入主单/备选池。</p>' + table(s.workflow.modelTopReviewRows);
      const historyBlock = IS_V3
        ? ''
        : '<h3>近5个信号日 Filter 后 Top1</h3><p class="muted">来源：短持有 candidate review，排除 FILTERED/SKIPPED，只看已经通过过滤并可进入主单或 backup 的 BUY 候选。</p>' + table(s.workflow.recentFilteredTopRows);
      document.querySelector('#orders').innerHTML =
        historyBlock +
        '<h3>' + (IS_V3 ? '本轮交易清单' : '手工下单清单') + '</h3><p class="muted"><b>本轮节奏：</b>' + escapeHtml(timeline) + '。' + escapeHtml(scoringText) + ' BUY 的 price_5pct 是最高买入红线；SELL 是到期持仓，优先卖出。</p>' + orderBlock +
        reviewBlock +
        '<h3>成交回填</h3><p class="muted">填东方财富「当日成交」里的真实成交股数/均价；没成交填 0 或留空并备注。</p>' + fillEditor(s.workflow.fillTemplateRows) +
        '<div class="submit-row"><button onclick="run(\\'short-hold-apply-fills\\')" class="secondary">提交短持有回填</button></div>' +
        '<p class="muted">最新任务: ' + (s.workflow.latestTask || '-') +
        '<br>最新清单: ' + (s.workflow.latestBrokerTicket || '-') +
        '<br>最新回填模板: ' + (s.workflow.latestFillTemplate || '-') +
        '<br>最新已应用成交: ' + (s.workflow.latestAppliedFills || '-') + '</p>';
    }
    async function renderShortHoldV3Research() {
      if (!IS_V3) return;
      const target = document.querySelector('#research');
      if (!target) return;
      try {
        const report = await api(API_BASE + '/research/latest');
        const markdown = report.markdown || '';
        const preview = markdown
          ? markdown.slice(0, 2000)
          : '暂无 V3 研究报告。可先运行 bin/run_short_hold_v3_candidate_lab.sh 生成 Candidate Lab 报告。';
        target.innerHTML =
          '<p class="muted"><b>报告路径</b>: ' + escapeHtml(report.reportPath || '-') + '</p>' +
          '<pre>' + escapeHtml(preview) + '</pre>';
      } catch (err) {
        const message = err && err.message ? err.message : String(err);
        target.innerHTML = '<p class="muted">读取 V3 研究报告失败：' + escapeHtml(message) + '</p>';
      }
    }
    async function refresh(options = {}) {
      const s = await api(API_BASE + '/status');
      jobSummary(s.jobs?.[0]);
      const v3ModelBlock = s.env.profile === 'v3'
        ? '<p><b>V3模型包</b>: ' + escapeHtml(s.env.modelPackage || '-') +
          '<br><b>V3预测目录</b>: ' + escapeHtml(s.env.predDir || '-') +
          '<br><b>V3自动推理</b>: ' + escapeHtml(s.env.autoPredict || '-') +
          '<br><b>V3侧模型目录</b>: ' + escapeHtml(s.env.sideModelDir || '-') +
          '<br><b>V3研究 Profile</b>: ' + escapeHtml(s.env.researchProfile || 'baseline') +
          '<br><b>V3行业上下文</b>: ' + escapeHtml(s.env.sectorContextEnabled || '-') +
          '<br><b>行业元数据</b>: ' + escapeHtml(s.env.sectorStockMeta || '-') +
          '<br><b>模型解压目录</b>: ' + escapeHtml(s.env.modelExtractDir || '-') + '</p>' +
          '<h3>V3 side model 文件</h3>' + table(s.sideModelArtifacts || [])
        : '';
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
        v3ModelBlock +
        '<p><b>数据日历</b>: ' + s.data.calendarTail.join(', ') + '</p>' +
        '<h3>持仓</h3>' + table(s.account.positions);
      if (options.forceOrders || (!isEditingFill() && !hasFillDraft())) renderOrders(s);
      await renderShortHoldV3Research();
      renderSnapshot(s.workflow.latestSnapshotData);
      document.querySelector('#jobs').textContent = s.jobs.map(j => '['+j.status+'] '+j.name+' '+j.startedAt+'\\n'+j.log).join('\\n\\n') || '暂无任务';
      document.querySelector('#jobs').scrollTop = document.querySelector('#jobs').scrollHeight;
    }
    async function run(action) {
      try {
        const status = await api(API_BASE + '/status');
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
          await api(API_BASE + '/save-fills', {executionDate, rows});
        }
        await api(API_BASE + '/run/' + action, {executionDate});
        setTimeout(() => refresh({forceOrders: true}), 1200);
      } catch (err) {
        const message = err && err.message ? err.message : String(err);
        document.querySelector('#jobSummary').textContent = '操作失败：' + message;
        alert('操作失败：' + message);
      }
    }
    const v2ScoreButton = document.querySelector('#v2ScoreButton');
    if (v2ScoreButton) {
      v2ScoreButton.addEventListener('click', async () => {
        const code = document.querySelector('#v2ScoreCode').value.trim();
        const output = document.querySelector('#v2ScoreResult');
        if (!code) return alert('请输入股票代码');
        output.textContent = '查询中...';
        try {
          const result = await api(SCORE_API, { code });
          output.textContent = result.found
            ? '信号日 ' + result.signalDate + ' ｜ 融合分 ' + Number(result.score).toFixed(6) + ' ｜ 排名 ' + result.rank + ' / ' + result.universeSize
            : '信号日 ' + result.signalDate + ' 未在候选池中找到 ' + result.code;
        } catch (err) {
          output.textContent = '查询失败：' + (err.message || String(err));
        }
      });
    }
    refresh();
    setInterval(refresh, 30 * 60 * 1000);
  </script>
</body>
</html>`;
}

function recommendationsHtml() {
  return `<!doctype html>
<html lang="zh-CN">
<head><meta charset="utf-8" /><meta name="viewport" content="width=device-width, initial-scale=1" /><title>Recommendation Journal</title>
<style>
:root{--ink:#17212b;--muted:#64748b;--line:#d8e0e8;--blue:#155e75;--orange:#b45309;--green:#166534;--red:#b42318}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:linear-gradient(135deg,#e6f1f5,#f9f5ed 50%,#eef4eb);font-family:ui-serif,Georgia,"Noto Serif SC",serif}header,main{max-width:1500px;margin:auto;padding-left:clamp(18px,4vw,54px);padding-right:clamp(18px,4vw,54px)}header{padding-top:34px;padding-bottom:20px}h1{margin:0;font-size:clamp(30px,4vw,48px);letter-spacing:-.045em}header p,.panel>p{color:var(--muted)}a{color:var(--blue);font-weight:700}.hero{display:flex;justify-content:space-between;gap:18px;align-items:end}.stats{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px}.stat,.panel{background:#ffffffd9;border:1px solid var(--line);border-radius:16px;box-shadow:0 10px 30px #18324a0b}.stat{padding:14px}.stat b{display:block;font-size:25px;margin-top:4px}.stat span{color:var(--muted);font-size:13px}main{display:grid;gap:18px;padding-bottom:42px}.panel{padding:18px}.panel h2{margin:0 0 6px;font-size:21px}form{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}label{display:grid;gap:5px;font-weight:700;font-size:14px}.wide{grid-column:span 2}.full{grid-column:1/-1}input,textarea{width:100%;border:1px solid #cbd5df;border-radius:9px;background:#fff;padding:9px 10px;font:inherit;color:var(--ink)}textarea{min-height:78px;resize:vertical}button{border:0;border-radius:999px;padding:10px 16px;background:var(--blue);color:#fff;font:inherit;font-weight:700;cursor:pointer}.save{background:var(--orange);white-space:nowrap}.table-wrap{overflow-x:auto;border:1px solid var(--line);border-radius:12px}.table-wrap table{min-width:1120px}table{width:100%;border-collapse:collapse;font-size:14px}th{background:#e8f1f5;text-align:left;white-space:nowrap}td,th{padding:10px 9px;border-bottom:1px solid var(--line);vertical-align:top}tr:last-child td{border:0}.status{display:inline-block;padding:3px 8px;border-radius:999px;background:#e5e7eb;white-space:nowrap;font-size:12px;font-weight:700}.status.持有中{background:#dcfce7;color:var(--green)}.status.已卖出{background:#dbeafe;color:#1d4ed8}.status.部分卖出{background:#fef3c7;color:#92400e}.fill{display:grid;grid-template-columns:70px 70px 110px;gap:5px}.fill input{padding:6px;min-width:0}.pnl-pos{color:var(--green);font-weight:700}.pnl-neg{color:var(--red);font-weight:700}.rationale{max-width:260px;white-space:pre-wrap;color:#475569}.empty{padding:32px;color:var(--muted);text-align:center}#message{min-height:22px;color:var(--muted);font-weight:700}@media(max-width:900px){.stats{grid-template-columns:repeat(2,1fr)}form{grid-template-columns:repeat(2,minmax(0,1fr))}.wide{grid-column:span 2}}@media(max-width:560px){.hero{display:block}form{grid-template-columns:1fr}.wide,.full{grid-column:auto}}
</style></head>
<body><header><div class="hero"><div><h1>Recommendation Journal</h1><p>独立记录第三方推荐、人工买卖回填与已实现结果。仅供交易日记使用，不构成投资建议。</p></div><a href="/">返回交易平台</a></div></header>
<main><div class="stats" id="stats"></div>
<section class="panel"><h2>录入推荐</h2><p>每天通常 2 只。完整理由、风险提示与投顾信息会原样保存，方便后续复盘。</p>
<form id="createForm"><label>来源/服务<input name="source" value="龙头掘金" required /></label><label>入选日期<input name="recommendationDate" type="date" required /></label><label>股票代码<input name="code" placeholder="601225" required /></label><label>股票名称<input name="name" placeholder="陕西煤业" required /></label><label>关注价格<input name="focusPrice" placeholder="24-25元" /></label><label>参考仓位<input name="suggestedPosition" placeholder="1成仓" /></label><label>目标价格<input name="targetPrice" placeholder="27元" /></label><label>支撑价格<input name="supportPrice" placeholder="22元" /></label><label class="wide">投顾/执业信息<input name="advisor" placeholder="姓名 / 执业编号" /></label><label class="wide">入选理由<textarea name="rationale" placeholder="粘贴完整入选理由"></textarea></label><label class="full">风险提示/原文备注<textarea name="disclaimer" placeholder="粘贴服务方风险提示或原文说明"></textarea></label><button type="submit">保存推荐</button><span id="message"></span></form></section>
<section class="panel"><h2>推荐跟踪与回填</h2><p>手数按 A 股 1 手 = 100 股。盈亏只按手工回填的买入价、卖出价和卖出手数计算，未计佣金、印花税和持仓浮盈。</p><div class="table-wrap" id="records"></div></section></main>
<script>
let status=null;
const esc=v=>String(v??'').replace(/[&<>"']/g,x=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[x]));
const num=(v,d=2)=>v==null||!Number.isFinite(Number(v))?'—':Number(v).toFixed(d);
async function api(path,body){const r=await fetch(path,body?{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(body)}:undefined);const d=await r.json();if(!r.ok)throw new Error(d.error||'请求失败');return d}
function inp(r,field,type,placeholder){return '<input data-id="'+r.id+'" data-field="'+field+'" type="'+type+'" value="'+esc(r[field])+'" placeholder="'+placeholder+'" />'}
function render(){const s=status;document.querySelector('[name=recommendationDate]').value=s.today;const list=[['总推荐',s.stats.total],['今日已录入',s.stats.todayCount+' / 2'],['待买入',s.stats.pending],['持有中',s.stats.holding],['已卖出',s.stats.sold]];document.querySelector('#stats').innerHTML=list.map(x=>'<div class="stat"><span>'+x[0]+'</span><b>'+x[1]+'</b></div>').join('');if(!s.records.length){document.querySelector('#records').innerHTML='<div class="empty">还没有推荐记录。先从上方录入今天的两只股票。</div>';return}document.querySelector('#records').innerHTML='<table><thead><tr><th>入选</th><th>股票</th><th>来源</th><th>关注/目标/支撑</th><th>建议仓位</th><th>状态</th><th>买入回填</th><th>卖出回填</th><th>已实现盈亏</th><th>理由</th><th>保存</th></tr></thead><tbody>'+s.records.map(r=>{const p=r.realizedPnl==null?'—':'<span class="'+(Number(r.realizedPnl)>=0?'pnl-pos':'pnl-neg')+'">'+num(r.realizedPnl)+' ('+num(r.realizedReturn)+'%)</span>';return '<tr><td>'+esc(r.recommendationDate)+'</td><td><b>'+esc(r.code)+'</b><br>'+esc(r.name)+'</td><td>'+esc(r.source)+'</td><td>关注 '+esc(r.focusPrice||'—')+'<br>目标 '+esc(r.targetPrice||'—')+'<br>支撑 '+esc(r.supportPrice||'—')+'</td><td>'+esc(r.suggestedPosition||'—')+'</td><td><span class="status '+esc(r.status)+'">'+esc(r.status)+'</span><br><small>持有 '+r.heldShares+' 股</small></td><td><div class="fill">'+inp(r,'buyPrice','number','价格')+inp(r,'buyLots','number','手数')+inp(r,'buyDate','date','')+'</div></td><td><div class="fill">'+inp(r,'sellPrice','number','价格')+inp(r,'sellLots','number','手数')+inp(r,'sellDate','date','')+'</div></td><td>'+p+'</td><td class="rationale">'+esc(r.rationale||'—')+'</td><td><button class="save" data-save="'+r.id+'">保存回填</button></td></tr>'}).join('')+'</tbody></table>';document.querySelectorAll('[data-save]').forEach(b=>b.addEventListener('click',async()=>{const id=b.dataset.save,fields={};document.querySelectorAll('[data-id="'+id+'"]').forEach(i=>fields[i.dataset.field]=i.value);try{b.disabled=true;b.textContent='保存中';await api('/api/recommendations/'+id,fields);await refresh('回填已保存')}catch(e){alert(e.message||String(e));b.disabled=false;b.textContent='保存回填'}}))}
async function refresh(message){status=await api('/api/recommendations');render();document.querySelector('#message').textContent=message||''}
document.querySelector('#createForm').addEventListener('submit',async e=>{e.preventDefault();try{await api('/api/recommendations',Object.fromEntries(new FormData(e.currentTarget).entries()));e.currentTarget.reset();await refresh('推荐已保存')}catch(err){alert(err.message||String(err))}});
refresh();setInterval(refresh,30*60*1000);
</script></body></html>`;
}

function recommendationsTrackerHtml() {
  return `<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Recommendation Journal</title>
<style>
:root{--ink:#17212b;--muted:#64748b;--line:#d8e0e8;--blue:#155e75;--orange:#b45309;--green:#166534;--red:#b42318}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:linear-gradient(135deg,#e6f1f5,#f9f5ed 50%,#eef4eb);font-family:ui-serif,Georgia,"Noto Serif SC",serif}header,main{max-width:1440px;margin:auto;padding-left:clamp(18px,4vw,54px);padding-right:clamp(18px,4vw,54px)}header{padding-top:34px;padding-bottom:18px}.hero{display:flex;justify-content:space-between;gap:18px;align-items:end}h1{margin:0;font-size:clamp(30px,4vw,46px);letter-spacing:-.04em}p{color:var(--muted)}a{color:var(--blue);font-weight:700}.stats{display:grid;grid-template-columns:repeat(5,1fr);gap:10px}.stat,.panel{background:#ffffffdc;border:1px solid var(--line);border-radius:16px;box-shadow:0 10px 30px #18324a0b}.stat{padding:14px}.stat b{display:block;font-size:24px;margin-top:4px}.stat span{color:var(--muted);font-size:13px}main{display:grid;gap:18px;padding-bottom:42px}.panel{padding:18px}.panel h2{margin:0 0 5px;font-size:21px}textarea{width:100%;min-height:230px;resize:vertical;border:1px solid #cbd5df;border-radius:11px;background:#fff;padding:12px;font:inherit;color:var(--ink)}button{border:0;border-radius:999px;padding:9px 13px;background:var(--blue);color:#fff;font:inherit;font-weight:700;cursor:pointer}.sell{background:var(--orange)}.table-wrap{overflow-x:auto;border:1px solid var(--line);border-radius:12px}.table-wrap table{min-width:1250px}table{width:100%;border-collapse:collapse;font-size:14px}th{background:#e8f1f5;text-align:left;white-space:nowrap}td,th{padding:9px 8px;border-bottom:1px solid var(--line);vertical-align:top}tr:last-child td{border:0}.status{display:inline-block;padding:3px 8px;border-radius:999px;background:#e5e7eb;white-space:nowrap;font-size:12px;font-weight:700}.status.持有中{background:#dcfce7;color:var(--green)}.status.已卖出{background:#dbeafe;color:#1d4ed8}.status.部分卖出{background:#fef3c7;color:#92400e}.fill{display:grid;grid-template-columns:68px 58px 110px;gap:4px}.fill input{width:100%;min-width:0;border:1px solid #cbd5df;border-radius:7px;padding:6px;font:inherit}.logged{font-size:11px;color:var(--muted);margin-top:5px}.pnl-pos{color:var(--green);font-weight:700}.pnl-neg{color:var(--red);font-weight:700}.empty{padding:32px;color:var(--muted);text-align:center}#message{margin-left:10px;color:var(--muted);font-weight:700}@media(max-width:800px){.stats{grid-template-columns:repeat(2,1fr)}.hero{display:block}}
</style></head><body>
<header><div class="hero"><div><h1>Recommendation Journal</h1><p>第三方推荐独立跟踪账本。回填仅记录实际成交，不构成投资建议。</p></div><a href="/">返回交易平台</a></div></header>
<main><div class="stats" id="stats"></div>
<section class="panel"><h2>粘贴推荐原文</h2><p>一次可粘贴当天多条“【龙头掘金】教学案例”文本，系统自动提取股票、价格区间、建议仓位和目标/支撑价格。</p><textarea id="rawText" placeholder="粘贴完整推荐文本，例如：&#10;【龙头掘金】教学案例&#10;入选时间：7.20&#10;股票代码：601225&#10;股票名称：陕西煤业&#10;..."></textarea><p><button id="importButton">解析并保存推荐</button><span id="message"></span></p></section>
<section class="panel"><h2>Short Hold V2 当日分数查询</h2><p>输入股票代码，查询现有 V2 最新信号日的三模型融合估计分与横截面排名；只读查询，不影响推荐账本或交易清单。</p><p><input id="scoreCode" maxlength="8" placeholder="例如 601225 或 SH601225"><button id="scoreButton">查询当日分数</button></p><p id="scoreResult"></p></section>
<section class="panel"><h2>推荐跟踪与回填</h2><p>1 手 = 100 股。买入、卖出各自独立保存；每次保存自动以点击当天作为成交日期并写入回填日志。</p><div class="table-wrap" id="records"></div></section>
</main><script>
let status=null;
const esc=v=>String(v??'').replace(/[&<>"']/g,x=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[x]));
const num=(v,d=2)=>v==null||!Number.isFinite(Number(v))?'—':Number(v).toFixed(d);
const saved=v=>v?'回填于 '+new Intl.DateTimeFormat('zh-CN',{timeZone:'Asia/Shanghai',year:'numeric',month:'2-digit',day:'2-digit'}).format(new Date(v)):'尚未保存';
async function api(path,body){const r=await fetch(path,body?{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(body)}:undefined);const d=await r.json();if(!r.ok)throw new Error(d.error||'请求失败');return d}
function input(r,field,type,holder){return '<input data-id="'+r.id+'" data-field="'+field+'" type="'+type+'" value="'+esc(r[field])+'" placeholder="'+holder+'">'}
function fill(r,side){const cap=side==='buy'?'买入':'卖出';return '<div class="fill">'+input(r,side+'Price','number','价格')+input(r,side+'Lots','number','手数')+'</div><div class="logged">'+saved(r[side+'SavedAt'])+'</div><button class="'+(side==='sell'?'sell':'')+'" data-save="'+r.id+'" data-side="'+side+'">保存'+cap+'回填</button>'}
function render(){const s=status;const stats=[['总推荐',s.stats.total],['今日录入',s.stats.todayCount+' / 2'],['待买入',s.stats.pending],['持有中',s.stats.holding],['已卖出',s.stats.sold]];document.querySelector('#stats').innerHTML=stats.map(x=>'<div class="stat"><span>'+x[0]+'</span><b>'+x[1]+'</b></div>').join('');if(!s.records.length){document.querySelector('#records').innerHTML='<div class="empty">还没有记录。</div>';return}document.querySelector('#records').innerHTML='<table><thead><tr><th>入选日期</th><th>股票代码</th><th>股票名称</th><th>来源</th><th>关注价格</th><th>目标价格</th><th>支撑价格</th><th>建议仓位</th><th>状态</th><th>买入回填</th><th>卖出回填</th><th>已实现盈亏</th></tr></thead><tbody>'+s.records.map(r=>{const pnl=r.realizedPnl==null?'—':'<span class="'+(Number(r.realizedPnl)>=0?'pnl-pos':'pnl-neg')+'">'+num(r.realizedPnl)+' ('+num(r.realizedReturn)+'%)</span>';return '<tr><td>'+esc(r.recommendationDate)+'</td><td><b>'+esc(r.code)+'</b></td><td>'+esc(r.name)+'</td><td>'+esc(r.source)+'</td><td>'+esc(r.focusPrice||'—')+'</td><td>'+esc(r.targetPrice||'—')+'</td><td>'+esc(r.supportPrice||'—')+'</td><td>'+esc(r.suggestedPosition||'—')+'</td><td><span class="status '+esc(r.status)+'">'+esc(r.status)+'</span><br><small>持有 '+r.heldShares+' 股</small></td><td>'+fill(r,'buy')+'</td><td>'+fill(r,'sell')+'</td><td>'+pnl+'</td></tr>'}).join('')+'</tbody></table>';document.querySelectorAll('[data-save]').forEach(b=>b.addEventListener('click',async()=>{const id=b.dataset.save,side=b.dataset.side,fields={};document.querySelectorAll('[data-id="'+id+'"]').forEach(i=>fields[i.dataset.field]=i.value);try{b.disabled=true;b.textContent='保存中';await api('/api/recommendations/'+id+'/'+side+'-fill',fields);await refresh(side==='buy'?'买入回填已保存':'卖出回填已保存')}catch(e){alert(e.message||String(e));b.disabled=false;b.textContent='保存回填'}}))}
async function refresh(message){status=await api('/api/recommendations');render();document.querySelector('#message').textContent=message||''}
document.querySelector('#importButton').addEventListener('click',async()=>{const text=document.querySelector('#rawText').value.trim();if(!text)return alert('请先粘贴推荐原文');try{await api('/api/recommendations/import',{rawText:text});document.querySelector('#rawText').value='';await refresh('推荐已解析并保存')}catch(e){alert(e.message||String(e))}});
document.querySelector('#scoreButton').addEventListener('click',async()=>{const code=document.querySelector('#scoreCode').value.trim(),out=document.querySelector('#scoreResult');if(!code)return alert('请输入股票代码');out.textContent='查询中...';try{const r=await api('/api/recommendations/short-hold-v2-score',{code});out.textContent=r.found?'信号日 '+r.signalDate+' ｜ V2 融合估计分 '+Number(r.score).toFixed(6)+' ｜ 排名 '+r.rank+' / '+r.universeSize+' ｜ 前 '+Number(r.percentile).toFixed(2)+'%':'信号日 '+r.signalDate+' 未在 V2 候选股票池中找到 '+r.code}catch(e){out.textContent='查询失败：'+(e.message||String(e))}});
refresh();setInterval(refresh,30*60*1000);
</script></body></html>`;
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
    if (req.method === "GET" && url.pathname === "/short-hold-v3") return send(res, 200, shortHoldHtml("v3"), "text/html");
    if (req.method === "GET" && url.pathname === "/short-hold-v5") return send(res, 200, shortHoldHtml("v5"), "text/html");
    if (req.method === "GET" && url.pathname === "/naked-k") return send(res, 200, nakedKHtml(), "text/html");
    if (req.method === "GET" && url.pathname === "/naked-k-model") return send(res, 200, nakedKModelHtml(), "text/html");
    if (req.method === "GET" && url.pathname === "/naked-k-live") return send(res, 200, nakedKLiveHtml(), "text/html");
    if (req.method === "GET" && url.pathname === "/recommendations") return send(res, 200, recommendationsTrackerHtml(), "text/html");
    if (req.method === "GET" && url.pathname === "/api/status") return send(res, 200, loadStatus());
    if (req.method === "GET" && url.pathname === "/api/naked-k/status") return send(res, 200, loadNakedKStatus());
    if (req.method === "GET" && url.pathname === "/api/naked-k-live/status") return send(res, 200, loadNakedKLiveStatus());
    if (req.method === "POST" && url.pathname === "/api/naked-k-live/open") {
      return send(res, 201, recordNakedKLiveOpen(await parseJsonBody(req)));
    }
    if (req.method === "POST" && url.pathname === "/api/naked-k-live/decisions") {
      return send(res, 201, recordNakedKLiveDecision(await parseJsonBody(req)));
    }
    const nakedKIgnoreMatch = url.pathname.match(/^\/api\/naked-k-live\/decisions\/([^/]+)\/ignore$/);
    if (req.method === "POST" && nakedKIgnoreMatch) {
      return send(res, 200, ignoreNakedKLiveDecision(decodeURIComponent(nakedKIgnoreMatch[1])));
    }
    if (req.method === "POST" && url.pathname === "/api/naked-k-live/fills") {
      return send(res, 201, recordNakedKLiveFill(await parseJsonBody(req)));
    }
    if (req.method === "GET" && url.pathname === "/api/naked-k-model/status") return send(res, 200, loadNakedKModelStatus());
    if (req.method === "POST" && url.pathname === "/api/naked-k-model/decisions") {
      return send(res, 201, saveNakedKModelDecision(await parseJsonBody(req)));
    }
    if (req.method === "POST" && url.pathname === "/api/naked-k/fills") {
      return send(res, 201, recordNakedKFill(await parseJsonBody(req)));
    }
    if (req.method === "GET" && url.pathname === "/api/recommendations") return send(res, 200, recommendationStatus());
    if (req.method === "POST" && url.pathname === "/api/recommendations") {
      const records = readRecommendationRecords();
      const record = recommendationFromBody(await parseJsonBody(req));
      records.push(record);
      writeRecommendationRecords(records);
      return send(res, 201, { record: recommendationView(record) });
    }
    if (req.method === "POST" && url.pathname === "/api/recommendations/import") {
      const body = await parseJsonBody(req);
      const records = parseRecommendationText(cleanText(body.rawText));
      const existing = readRecommendationRecords();
      const keys = new Set(
        existing.map((record) => `${record.source}|${record.recommendationDate}|${record.code}`),
      );
      const additions = records.filter((record) => {
        const key = `${record.source}|${record.recommendationDate}|${record.code}`;
        if (keys.has(key)) return false;
        keys.add(key);
        return true;
      });
      writeRecommendationRecords([...existing, ...additions]);
      return send(res, 201, {
        imported: additions.length,
        skippedDuplicates: records.length - additions.length,
        records: additions.map(recommendationView),
      });
    }
    if (req.method === "POST" && url.pathname === "/api/recommendations/short-hold-v2-score") {
      const body = await parseJsonBody(req);
      return send(res, 200, await lookupShortHoldV2Score(cleanText(body.code)));
    }
    if (req.method === "POST" && url.pathname.match(/^\/api\/recommendations\/[^/]+\/(buy|sell)-fill$/)) {
      const [, id, side] = url.pathname.match(/^\/api\/recommendations\/([^/]+)\/(buy|sell)-fill$/) || [];
      if (!id || (side !== "buy" && side !== "sell")) return send(res, 404, { error: "invalid fill route" });
      return send(res, 200, { record: updateRecommendationFill(id, side, await parseJsonBody(req)) });
    }
    if (req.method === "POST" && url.pathname.startsWith("/api/recommendations/")) {
      const id = url.pathname.slice("/api/recommendations/".length);
      if (!id) return send(res, 404, { error: "recommendation id is required" });
      const records = readRecommendationRecords();
      const index = records.findIndex((record) => record.id === id);
      if (index < 0) return send(res, 404, { error: "recommendation not found" });
      const record = recommendationFromBody(await parseJsonBody(req), records[index]);
      records[index] = record;
      writeRecommendationRecords(records);
      return send(res, 200, { record: recommendationView(record) });
    }
    if (req.method === "GET" && url.pathname === "/api/short-hold/status") return send(res, 200, loadShortHoldStatus());
    if (req.method === "POST" && url.pathname === "/api/short-hold/v2-score") {
      const body = await parseJsonBody(req);
      return send(res, 200, await lookupShortHoldV2Score(cleanText(body.code)));
    }
    if (req.method === "GET" && url.pathname === "/api/short-hold-v3/status") return send(res, 200, loadShortHoldStatus("v3"));
    if (req.method === "GET" && url.pathname === "/api/short-hold-v5/status") return send(res, 200, loadShortHoldStatus("v5"));
    if (req.method === "POST" && url.pathname === "/api/short-hold-v5/v5-score") {
      const body = await parseJsonBody(req);
      return send(res, 200, await lookupShortHoldV5Score(cleanText(body.code)));
    }
    if (req.method === "GET" && url.pathname === "/api/short-hold-v3/research/latest") {
      return send(res, 200, loadLatestShortHoldV3ResearchReport());
    }
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
    if (req.method === "POST" && (
      url.pathname === "/api/short-hold/save-fills" ||
      url.pathname === "/api/short-hold-v3/save-fills" ||
      url.pathname === "/api/short-hold-v5/save-fills"
    )) {
      const body = await parseJsonBody(req);
      const executionDate = validateDate(body.executionDate);
      const env = parseEnv(ENV_PATH);
      const isV3 = url.pathname.startsWith("/api/short-hold-v3/");
      const isV5 = url.pathname.startsWith("/api/short-hold-v5/");
      const fillDir = isV5
        ? env.SHORT_HOLD_V5_FILL_DIR || resolve(OPS_HOME, "short_hold_v5_fills")
        : isV3
        ? env.SHORT_HOLD_V3_FILL_DIR || resolve(OPS_HOME, "short_hold_v3_fills")
        : env.SHORT_HOLD_FILL_DIR || resolve(OPS_HOME, "short_hold_fills");
      mkdirSync(fillDir, { recursive: true });
      const fillPath = resolve(fillDir, `short_hold_fill_template_${executionDate}.csv`);
      const rows = Array.isArray(body.rows) ? body.rows : [];
      if (!rows.length) throw new Error("rows is required");
      writeCsv(fillPath, rows);
      return send(res, 200, { path: fillPath, nRows: rows.length });
    }
    if (
      req.method === "POST" &&
      (
        url.pathname.startsWith("/api/short-hold/run/") ||
        url.pathname.startsWith("/api/short-hold-v3/run/") ||
        url.pathname.startsWith("/api/short-hold-v5/run/")
      )
    ) {
      const isV3 = url.pathname.startsWith("/api/short-hold-v3/");
      const isV5 = url.pathname.startsWith("/api/short-hold-v5/");
      const action = url.pathname.split("/").pop() || "";
      const body = await parseJsonBody(req);
      let job: Job;
      if (action === "prepare-short-hold-orders") {
        job = startJob(action, OPS_HOME, "bash", [isV5 ? "bin/prepare_short_hold_v5_orders.sh" : isV3 ? "bin/prepare_short_hold_v3_orders.sh" : "bin/prepare_short_hold_orders.sh"]);
      }
      else if (action === "short-hold-apply-fills") {
        job = startJob(action, OPS_HOME, "bash", [
          isV5 ? "bin/short_hold_v5_apply_fills.sh" : isV3 ? "bin/short_hold_v3_apply_fills.sh" : "bin/short_hold_apply_fills.sh",
          validateDate(body.executionDate),
        ]);
      }
      else if (action === "short-hold-daily-snapshot") {
        job = startJob(action, OPS_HOME, "bash", [
          isV5 ? "bin/short_hold_v5_daily_snapshot.sh" : isV3 ? "bin/short_hold_v3_daily_snapshot.sh" : "bin/short_hold_daily_snapshot.sh",
          validateDate(body.executionDate),
        ]);
      }
      else if (action === "generate-and-send-short-hold-snapshot") {
        job = startJob(action, OPS_HOME, "bash", [
          isV5 ? "bin/generate_and_send_short_hold_v5_snapshot.sh" : isV3 ? "bin/generate_and_send_short_hold_v3_snapshot.sh" : "bin/generate_and_send_short_hold_snapshot.sh",
          validateDate(body.executionDate),
        ]);
      }
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
