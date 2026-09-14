import { spawn } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

type Side = "buy" | "sell";
type DecisionAction = "wait" | "buy" | "hold" | "sell" | "blocked" | "error";

type LiveConfig = {
  strategyId: string;
  strategyVersion: string;
  initialCapital: number;
  openBreadthThreshold: number;
  watchlistSize: number;
  firstIntradayDecision: string;
  lastBuyDecision: string;
  lastSellDecision: string;
  decisionIntervalMinutes: number;
  allowSameDaySwitch: boolean;
  blockSameDayRebuy: boolean;
  singlePosition: boolean;
  allowAddPosition: boolean;
  costs: { buyRate: number; sellRate: number; minimumFee: number; lotSize: number };
};

type Candidate = {
  code: string;
  name: string;
  industry?: string;
  price?: number;
  score?: number;
  evidenceCount?: number;
  context?: Record<string, unknown>;
};

type OpenDecision = {
  id: string;
  date: string;
  generatedAt: string;
  dataCutoff: string;
  openBreadth: number | null;
  entryAllowed: boolean;
  hotSectors: Array<Record<string, unknown> | string>;
  top3: Candidate[];
  reasons: string[];
  risks: string[];
  status: "success" | "failed";
};

type IntradayDecision = {
  id: string;
  date: string;
  generatedAt: string;
  dataCutoff: string;
  action: DecisionAction;
  code: string | null;
  name: string;
  referencePrice: number | null;
  confidence: number | null;
  evidenceCount: number | null;
  reasons: string[];
  risks: string[];
  hardBlock: string | null;
  status: "pending" | "informational" | "executed" | "ignored";
  resolvedAt?: string;
  linkedFillId?: string;
};

type Fill = {
  id: string;
  requestId: string;
  side: Side;
  tradeDate: string;
  createdAt: string;
  code: string;
  name: string;
  price: number;
  shares: number;
  gross: number;
  fee: number;
  linkedDecisionId: string | null;
  entryContext?: Record<string, unknown>;
};

type Ledger = { initialCapital: number; createdAt: string; fills: Fill[] };
type DayState = { date: string; open: OpenDecision | null; timeline: IntradayDecision[] };

const OPS_HOME = resolve(new URL("..", import.meta.url).pathname);
const CONFIG_PATH = process.env.NAKED_K_LIVE_CONFIG
  ? resolve(process.env.NAKED_K_LIVE_CONFIG)
  : resolve(OPS_HOME, "config", "naked_k_llm_live.json");
const STATE_DIR = process.env.NAKED_K_LIVE_STATE_DIR
  ? resolve(process.env.NAKED_K_LIVE_STATE_DIR)
  : resolve(OPS_HOME, "state", "naked_k_llm_live");
const LEDGER_PATH = resolve(STATE_DIR, "ledger.json");
const LOG_PATH = resolve(STATE_DIR, "service.jsonl");
const SYSTEM_PATH = resolve(STATE_DIR, "system.json");
const SCHEDULER_PATH = resolve(OPS_HOME, "config", "scheduler.json");

function nowIso() {
  return new Date().toISOString();
}

function shanghaiDate(date = new Date()) {
  return new Intl.DateTimeFormat("en-CA", {
    timeZone: "Asia/Shanghai", year: "numeric", month: "2-digit", day: "2-digit",
  }).format(date);
}

function shanghaiTime(date = new Date()) {
  const parts = new Intl.DateTimeFormat("en-GB", {
    timeZone: "Asia/Shanghai", hour: "2-digit", minute: "2-digit", hourCycle: "h23",
  }).formatToParts(date);
  const value = (type: string) => parts.find((part) => part.type === type)?.value || "00";
  return `${value("hour")}:${value("minute")}`;
}

function atomicJson(path: string, value: unknown) {
  mkdirSync(STATE_DIR, { recursive: true });
  const temporary = `${path}.${process.pid}.tmp`;
  writeFileSync(temporary, `${JSON.stringify(value, null, 2)}\n`, "utf8");
  renameSync(temporary, path);
}

function readJson<T>(path: string, fallback: T): T {
  if (!existsSync(path)) return fallback;
  return JSON.parse(readFileSync(path, "utf8")) as T;
}

export function loadNakedKLiveConfig(): LiveConfig {
  const config = readJson<LiveConfig | null>(CONFIG_PATH, null);
  if (!config) throw new Error(`裸K LLM配置不存在: ${CONFIG_PATH}`);
  const initialCapital = Number(process.env.INITIAL_CAPITAL || config.initialCapital);
  if (!Number.isFinite(initialCapital) || initialCapital <= 0) throw new Error("INITIAL_CAPITAL必须为正数");
  return { ...config, initialCapital };
}

function readLedger(): Ledger {
  const config = loadNakedKLiveConfig();
  return readJson(LEDGER_PATH, { initialCapital: config.initialCapital, createdAt: nowIso(), fills: [] });
}

function dayPath(date: string) {
  return resolve(STATE_DIR, "days", `${date}.json`);
}

function readDay(date: string): DayState {
  return readJson(dayPath(date), { date, open: null, timeline: [] });
}

function writeDay(day: DayState) {
  mkdirSync(resolve(STATE_DIR, "days"), { recursive: true });
  atomicJson(dayPath(day.date), day);
}

function appendLog(level: "info" | "warning" | "error", event: string, message: string, context: Record<string, unknown> = {}) {
  mkdirSync(STATE_DIR, { recursive: true });
  const line = JSON.stringify({ timestamp: nowIso(), level, event, message, context });
  const existing = existsSync(LOG_PATH) ? readFileSync(LOG_PATH, "utf8") : "";
  writeFileSync(LOG_PATH, `${existing}${line}\n`, "utf8");
}

function recentLogs(limit = 80) {
  if (!existsSync(LOG_PATH)) return [];
  return readFileSync(LOG_PATH, "utf8").trim().split(/\r?\n/).filter(Boolean).slice(-limit).reverse().map((line) => {
    try { return JSON.parse(line); } catch { return { timestamp: "", level: "error", event: "invalid_log", message: line }; }
  });
}

function normalizeCode(value: unknown) {
  const raw = String(value || "").trim().toUpperCase();
  if (/^\d{6}\.(SH|SZ|BJ)$/.test(raw)) return raw;
  const prefixed = raw.match(/^(SH|SZ|BJ)(\d{6})$/);
  if (prefixed) return `${prefixed[2]}.${prefixed[1]}`;
  if (!/^\d{6}$/.test(raw)) throw new Error("股票代码格式不正确");
  if (/^6/.test(raw)) return `${raw}.SH`;
  if (/^[03]/.test(raw)) return `${raw}.SZ`;
  return `${raw}.BJ`;
}

function feeFor(side: Side, gross: number, config: LiveConfig) {
  const rate = side === "buy" ? config.costs.buyRate : config.costs.sellRate;
  return Math.round(Math.max(gross * rate, config.costs.minimumFee) * 100) / 100;
}

function accountView(ledger: Ledger, today: DayState) {
  const positions = new Map<string, { code: string; name: string; shares: number; cost: number; lastBuyDate: string; entryContext?: Record<string, unknown> }>();
  let cash = ledger.initialCapital;
  let realizedPnl = 0;
  let totalFees = 0;
  for (const fill of ledger.fills) {
    const position = positions.get(fill.code) || { code: fill.code, name: fill.name, shares: 0, cost: 0, lastBuyDate: "" };
    totalFees += fill.fee;
    if (fill.side === "buy") {
      const previousCost = position.cost * position.shares;
      position.shares += fill.shares;
      position.cost = (previousCost + fill.gross + fill.fee) / position.shares;
      position.lastBuyDate = fill.tradeDate;
      position.entryContext = fill.entryContext;
      cash -= fill.gross + fill.fee;
    } else {
      realizedPnl += fill.gross - fill.fee - position.cost * fill.shares;
      position.shares -= fill.shares;
      cash += fill.gross - fill.fee;
    }
    positions.set(fill.code, position);
  }
  const openPositions = [...positions.values()].filter((position) => position.shares > 0);
  const prices = new Map<string, number>();
  for (const candidate of today.open?.top3 || []) if (candidate.price) prices.set(candidate.code, candidate.price);
  for (const decision of today.timeline) if (decision.code && decision.referencePrice) prices.set(decision.code, decision.referencePrice);
  const position = openPositions[0] || null;
  const markPrice = position ? prices.get(position.code) || position.cost : 0;
  const marketValue = position ? position.shares * markPrice : 0;
  const openCostValue = position ? position.shares * position.cost : 0;
  const unrealizedPnl = marketValue - openCostValue;
  const totalAssets = cash + marketValue;
  return {
    initialCapital: ledger.initialCapital,
    cash,
    position: position ? { ...position, markPrice, marketValue, unrealizedPnl, canSell: today.date > position.lastBuyDate } : null,
    marketValue,
    totalAssets,
    realizedPnl,
    unrealizedPnl,
    totalPnl: totalAssets - ledger.initialCapital,
    totalReturn: totalAssets / ledger.initialCapital - 1,
    totalFees,
    fills: [...ledger.fills].reverse(),
  };
}

function inferName(code: string, ledger: Ledger, day: DayState) {
  const candidate = day.open?.top3.find((item) => normalizeCode(item.code) === code)?.name;
  if (candidate) return candidate;
  const decision = [...day.timeline].reverse().find((item) => item.code && normalizeCode(item.code) === code)?.name;
  if (decision) return decision;
  return [...ledger.fills].reverse().find((fill) => fill.code === code)?.name || code;
}

function unresolvedDecision(day: DayState) {
  return [...day.timeline].reverse().find((item) => item.status === "pending") || null;
}

function soldToday(ledger: Ledger, date: string, code: string) {
  return ledger.fills.some((fill) => fill.tradeDate === date && fill.code === code && fill.side === "sell");
}

export function recordNakedKLiveFill(body: Record<string, unknown>) {
  const config = loadNakedKLiveConfig();
  const ledger = readLedger();
  const date = shanghaiDate();
  const day = readDay(date);
  const requestId = String(body.requestId || "").trim() || `fill-request-${Date.now()}`;
  const duplicate = ledger.fills.find((fill) => fill.requestId === requestId);
  if (duplicate) return { fill: duplicate, account: accountView(ledger, day), duplicate: true };
  const side = String(body.side || "") as Side;
  if (side !== "buy" && side !== "sell") throw new Error("请选择买入或卖出");
  const code = normalizeCode(body.code);
  const price = Number(body.price);
  const shares = Number(body.shares);
  if (!Number.isFinite(price) || price <= 0) throw new Error("成交价格必须为正数");
  if (!Number.isInteger(shares) || shares <= 0) throw new Error("股数必须为正整数");
  if (shares % config.costs.lotSize !== 0) throw new Error(`买卖数量必须是${config.costs.lotSize}股的整数倍`);
  const current = accountView(ledger, day);
  if (side === "buy") {
    if (current.position) throw new Error("单股策略已有持仓，禁止新增或加仓");
    if (config.blockSameDayRebuy && soldToday(ledger, date, code)) throw new Error("该股票今天已经卖出，禁止当日买回");
  } else {
    if (!current.position || current.position.code !== code) throw new Error("卖出代码与当前持仓不一致");
    if (!current.position.canSell) throw new Error("该持仓今日买入，A股T+1暂不可卖");
    if (shares > current.position.shares) throw new Error("卖出股数超过当前持仓");
  }
  const gross = Math.round(price * shares * 100) / 100;
  const fee = feeFor(side, gross, config);
  const pending = unresolvedDecision(day);
  const name = inferName(code, ledger, day);
  const entryContext = side === "buy"
    ? day.open?.top3.find((candidate) => normalizeCode(candidate.code) === code)?.context
    : undefined;
  const fill: Fill = {
    id: `fill-${Date.now()}-${Math.random().toString(16).slice(2)}`,
    requestId, side, tradeDate: date, createdAt: nowIso(), code, name, price, shares, gross, fee,
    linkedDecisionId: pending && pending.action === side && pending.code === code ? pending.id : null,
    entryContext,
  };
  ledger.fills.push(fill);
  atomicJson(LEDGER_PATH, ledger);
  if (fill.linkedDecisionId && pending) {
    pending.status = "executed";
    pending.resolvedAt = fill.createdAt;
    pending.linkedFillId = fill.id;
    writeDay(day);
  }
  const updated = accountView(ledger, day);
  appendLog(updated.cash < 0 ? "warning" : "info", "fill_recorded", `${name} ${side === "buy" ? "买入" : "卖出"}成交已保存`, { fillId: fill.id, code, shares, price, cash: updated.cash });
  return { fill, account: updated, duplicate: false, warning: updated.cash < 0 ? "成交后现金为负，请核对初始资金或成交记录" : null };
}

function normalizeCandidate(value: unknown): Candidate {
  const row = (value || {}) as Record<string, unknown>;
  return {
    code: normalizeCode(row.code || row.ts_code), name: String(row.name || row.code || row.ts_code || ""),
    industry: row.industry == null ? undefined : String(row.industry),
    price: Number.isFinite(Number(row.price ?? row.open)) ? Number(row.price ?? row.open) : undefined,
    score: Number.isFinite(Number(row.score)) ? Number(row.score) : undefined,
    evidenceCount: Number.isFinite(Number(row.evidenceCount)) ? Number(row.evidenceCount) : undefined,
    context: row.context && typeof row.context === "object" ? row.context as Record<string, unknown> : undefined,
  };
}

function notifyNative(title: string, message: string) {
  if (process.platform !== "darwin" || process.env.NAKED_K_NATIVE_NOTIFICATIONS === "0") return;
  const safe = (value: string) => value.replace(/\\/g, "\\\\").replace(/"/g, '\\"');
  const child = spawn("osascript", ["-e", `display notification "${safe(message)}" with title "${safe(title)}"`], { stdio: "ignore" });
  child.unref();
}

export function recordNakedKLiveOpen(body: Record<string, unknown>) {
  const config = loadNakedKLiveConfig();
  const date = String(body.date || shanghaiDate());
  const breadth = body.openBreadth == null ? null : Number(body.openBreadth);
  const status = body.status === "failed" ? "failed" : "success";
  const top3 = Array.isArray(body.top3) ? body.top3.slice(0, config.watchlistSize).map(normalizeCandidate) : [];
  const decision: OpenDecision = {
    id: String(body.id || `open-${date}`), date, generatedAt: String(body.generatedAt || nowIso()),
    dataCutoff: String(body.dataCutoff || `${date}T09:25:00+08:00`), openBreadth: Number.isFinite(breadth) ? breadth : null,
    entryAllowed: status === "success" && Number.isFinite(breadth) && Number(breadth) >= config.openBreadthThreshold,
    hotSectors: Array.isArray(body.hotSectors) ? body.hotSectors.slice(0, 5) as Array<Record<string, unknown> | string> : [],
    top3, reasons: Array.isArray(body.reasons) ? body.reasons.map(String).slice(0, 8) : [],
    risks: Array.isArray(body.risks) ? body.risks.map(String).slice(0, 8) : [], status,
  };
  const day = readDay(date);
  day.open = decision;
  writeDay(day);
  appendLog(status === "failed" ? "error" : "info", "open_decision", decision.entryAllowed ? "今日允许寻找买点" : "今日禁止新开仓", { breadth: decision.openBreadth, top3: top3.map((item) => item.code) });
  notifyNative("裸K 09:25决策", decision.entryAllowed ? `允许入场，广度${((decision.openBreadth || 0) * 100).toFixed(1)}%` : "今日禁止新开仓");
  return decision;
}

export function recordNakedKLiveDecision(body: Record<string, unknown>) {
  const date = String(body.date || shanghaiDate());
  const day = readDay(date);
  const action = String(body.action || "wait") as DecisionAction;
  if (!["wait", "buy", "hold", "sell", "blocked", "error"].includes(action)) throw new Error("无效的盘中动作");
  if (action === "buy" && !day.open?.entryAllowed) throw new Error("今日没有开仓权限，不能记录买入建议");
  const code = body.code ? normalizeCode(body.code) : null;
  if ((action === "buy" || action === "sell") && !code) throw new Error("买卖建议必须提供股票代码");
  const existing = day.timeline.find((item) => item.id === body.id);
  if (existing) return { decision: existing, duplicate: true };
  const pending = unresolvedDecision(day);
  if (pending && (action === "buy" || action === "sell")) throw new Error("已有待处理建议，请先成交回填或忽略");
  const decision: IntradayDecision = {
    id: String(body.id || `decision-${date}-${shanghaiTime().replace(":", "")}-${Math.random().toString(16).slice(2)}`),
    date, generatedAt: String(body.generatedAt || nowIso()), dataCutoff: String(body.dataCutoff || nowIso()), action, code,
    name: code ? String(body.name || inferName(code, readLedger(), day)) : "",
    referencePrice: Number.isFinite(Number(body.referencePrice)) ? Number(body.referencePrice) : null,
    confidence: Number.isFinite(Number(body.confidence)) ? Number(body.confidence) : null,
    evidenceCount: Number.isFinite(Number(body.evidenceCount)) ? Number(body.evidenceCount) : null,
    reasons: Array.isArray(body.reasons) ? body.reasons.map(String).slice(0, 8) : [],
    risks: Array.isArray(body.risks) ? body.risks.map(String).slice(0, 8) : [],
    hardBlock: body.hardBlock ? String(body.hardBlock) : null,
    status: action === "buy" || action === "sell" ? "pending" : "informational",
  };
  day.timeline.push(decision);
  writeDay(day);
  const level = action === "error" ? "error" : decision.hardBlock ? "warning" : "info";
  appendLog(level, "intraday_decision", `${decision.name || "市场"} ${action}`, { decisionId: decision.id, confidence: decision.confidence, hardBlock: decision.hardBlock });
  if (action === "buy" || action === "sell" || action === "error" || decision.hardBlock) {
    notifyNative(action === "error" ? "裸K系统故障" : `裸K${action === "buy" ? "买入" : action === "sell" ? "卖出" : "风控"}提醒`, `${decision.name || code || "系统"}${decision.referencePrice ? ` · ${decision.referencePrice}` : ""}`);
  }
  return { decision, duplicate: false };
}

export function ignoreNakedKLiveDecision(id: string) {
  const date = shanghaiDate();
  const day = readDay(date);
  const decision = day.timeline.find((item) => item.id === id);
  if (!decision) throw new Error("没有找到该决策");
  if (decision.status !== "pending") throw new Error("该决策已经处理");
  decision.status = "ignored";
  decision.resolvedAt = nowIso();
  writeDay(day);
  appendLog("info", "decision_ignored", `${decision.name || decision.code}建议已忽略`, { decisionId: id });
  return decision;
}

function derivedState(day: DayState, account: ReturnType<typeof accountView>) {
  const pending = unresolvedDecision(day);
  if (pending?.action === "buy") return "BUY_PENDING";
  if (pending?.action === "sell") return "SELL_PENDING";
  if (account.position) return account.position.canSell ? "T1_MONITORING" : "T0_LOCKED";
  if (day.open && !day.open.entryAllowed) return "ENTRY_BLOCKED";
  if (day.open?.top3.length) return "WATCHING";
  return "PREMARKET";
}

function systemView() {
  const stored = readJson<Record<string, unknown>>(SYSTEM_PATH, {});
  const schedulerConfig = readJson<Record<string, Record<string, unknown>>>(SCHEDULER_PATH, {});
  const logs = recentLogs();
  const latestError = logs.find((item) => item.level === "error") || null;
  return {
    service: "online", now: nowIso(), stateDir: STATE_DIR, configPath: CONFIG_PATH,
    scheduler: { ...(schedulerConfig.nakedKLive || { enabled: true }), ...((stored.scheduler || {}) as Record<string, unknown>) },
    marketData: stored.marketData || { status: "unknown" },
    llm: stored.llm || { status: "waiting_preflight", model: "deepseek-chat" },
    latestError, logs,
  };
}

export function loadNakedKLiveStatus() {
  const config = loadNakedKLiveConfig();
  const date = shanghaiDate();
  const day = readDay(date);
  const ledger = readLedger();
  const account = accountView(ledger, day);
  const pending = unresolvedDecision(day);
  return {
    now: nowIso(), today: date, config, day, account, pending,
    state: derivedState(day, account), system: systemView(),
  };
}

export function nakedKLiveHtml() {
  return `<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>裸K LLM 实盘台</title><style>
:root{--ink:#15231c;--paper:#f4efe1;--card:#fffdf6;--line:#c9c4b4;--red:#bd3928;--green:#126b55;--amber:#b87b19;--muted:#69736a}*{box-sizing:border-box}body{margin:0;color:var(--ink);font-family:"FZKai-Z03","Kaiti SC",serif;background:radial-gradient(circle at 8% 0,#fff6cf 0,transparent 27rem),linear-gradient(140deg,#e9efe7,#f5eddf 65%,#eadcd0);min-height:100vh}.shell{max-width:1320px;margin:auto;padding:24px 22px 60px}header{display:flex;justify-content:space-between;align-items:end;border-bottom:2px solid var(--ink);padding:18px 0 24px}.eyebrow{font:700 11px ui-monospace,monospace;letter-spacing:.2em;color:var(--red)}h1{font-size:clamp(38px,6vw,70px);line-height:.95;margin:10px 0}.sub{color:var(--muted);max-width:720px}.live{border:1px solid var(--green);padding:9px 13px;color:var(--green);font:700 12px ui-monospace,monospace}.hero{display:grid;grid-template-columns:1.25fr .75fr;gap:16px;margin:20px 0}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.card{background:#fffdf6eF;border:1px solid var(--line);box-shadow:5px 6px 0 #1f2e2412;padding:20px}.wide{grid-column:1/-1}.card h2{margin:0 0 15px;font-size:21px}.decision{border-left:6px solid var(--amber)}.decision.buy{border-color:var(--red)}.decision.sell{border-color:var(--green)}.action{font:800 clamp(30px,5vw,54px) ui-sans-serif,sans-serif;letter-spacing:-.05em}.muted{color:var(--muted)}.row{display:flex;justify-content:space-between;gap:14px;border-bottom:1px dashed var(--line);padding:10px 0;font:14px ui-sans-serif,sans-serif}.row:last-child{border:0}.pill{display:inline-block;padding:5px 9px;border-radius:999px;background:#e4ebe3;font:700 12px ui-sans-serif,sans-serif}.pill.warn{background:#f7e8c7;color:#895b10}.pill.bad{background:#f6dcd7;color:#9e2f20}.top3{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.candidate{border:1px solid var(--line);padding:12px;background:#fff}.candidate b{display:block;font:700 17px ui-sans-serif,sans-serif}.candidate small{color:var(--muted)}form{display:grid;grid-template-columns:110px 1fr 150px 150px 130px;gap:9px}input,select,button{border:1px solid var(--line);background:#fffdf8;padding:11px;font:14px ui-sans-serif,sans-serif;color:var(--ink)}button{background:var(--ink);color:#fff;font-weight:750;cursor:pointer}button.secondary{background:transparent;color:var(--ink)}button:disabled{opacity:.45;cursor:not-allowed}.message{min-height:21px;margin-top:9px;font:13px ui-sans-serif,sans-serif}.positive{color:var(--red)}.negative{color:var(--green)}table{width:100%;border-collapse:collapse;font:13px ui-sans-serif,sans-serif}th,td{text-align:left;padding:9px 7px;border-bottom:1px solid var(--line)}th{font-size:11px;color:var(--muted)}details{border-top:1px solid var(--line);padding:10px 0;font:13px ui-sans-serif,sans-serif}.log{font:12px/1.55 ui-monospace,monospace;max-height:330px;overflow:auto}.log-line{padding:6px;border-bottom:1px solid #ddd}.error{color:#a5261a}.warning{color:#986313}.empty{text-align:center;padding:22px;color:var(--muted)}.modal{display:none;position:fixed;inset:0;background:#101a15aa;z-index:10;align-items:center;justify-content:center}.modal.show{display:flex}.modal-box{width:min(520px,calc(100vw - 34px));background:var(--card);border:2px solid var(--ink);padding:24px;box-shadow:12px 14px 0 #0004}.modal-box h2{font-size:30px;margin:0 0 12px}@media(max-width:820px){header{display:block}.live{display:inline-block;margin-top:12px}.hero,.grid{grid-template-columns:1fr}.wide{grid-column:auto}.top3{grid-template-columns:1fr}form{grid-template-columns:1fr 1fr}form button{grid-column:1/-1}}
</style></head><body><main class="shell"><header><div><div class="eyebrow">NAKED K / LLM / LIVE DESK</div><h1>裸K实盘台</h1><div class="sub">09:25固定观察池，每5分钟只处理有效决策。模型建议不会改变账户，人工成交回填才是账本事实。</div></div><div class="live">LOCAL · MANUAL EXECUTION</div></header><section class="hero"><div class="card decision" id="decisionCard"><div class="muted">最新决策</div><div class="action" id="action">加载中</div><div id="decisionBody"></div><div id="decisionButtons"></div></div><div class="card"><h2>09:25 开仓许可</h2><div id="openStatus"></div><h2 style="margin-top:20px">固定 Top3</h2><div class="top3" id="top3"></div></div></section><section class="grid"><div class="card"><h2>当前持仓</h2><div id="position"></div></div><div class="card"><h2>账户</h2><div id="account"></div></div><div class="card wide"><h2>人工成交回填</h2><form id="fillForm"><select id="side"><option value="buy">买入</option><option value="sell">卖出</option></select><input id="code" placeholder="股票代码" required><input id="price" type="number" step="0.001" min="0" placeholder="成交价格" required><input id="shares" type="number" step="100" min="100" placeholder="股数" required><button id="saveFill">保存成交</button></form><div class="message" id="message">成交时间由服务器自动记录；股票名称自动识别。</div></div><div class="card wide"><h2>今日滚动决策</h2><div id="timeline"></div></div><div class="card wide"><h2>成交历史</h2><div id="fills"></div></div><div class="card"><h2>系统状态</h2><div id="system"></div></div><div class="card"><h2>运行日志</h2><div class="log" id="logs"></div></div></section></main><div class="modal" id="modal"><div class="modal-box"><h2 id="modalTitle"></h2><div id="modalBody"></div><button id="modalClose">我知道了</button></div></div><script>
let state,lastAlert=localStorage.getItem('naked-k-last-alert')||'';const q=s=>document.querySelector(s),esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])),money=v=>'¥'+Number(v||0).toLocaleString('zh-CN',{minimumFractionDigits:2,maximumFractionDigits:2}),pct=v=>(Number(v||0)*100).toFixed(2)+'%',row=(a,b)=>'<div class="row"><span>'+a+'</span><b>'+b+'</b></div>';
function actionLabel(a){return({buy:'买入',sell:'卖出',hold:'继续持有',wait:'等待',blocked:'风控拦截',error:'系统故障'})[a]||a}
function alertDecision(d){if(!d||d.id===lastAlert||!['buy','sell','error'].includes(d.action))return;lastAlert=d.id;localStorage.setItem('naked-k-last-alert',d.id);q('#modalTitle').textContent=actionLabel(d.action)+'提醒';q('#modalBody').innerHTML='<p><b>'+esc(d.name||d.code||'系统')+'</b>'+(d.referencePrice?' · '+esc(d.referencePrice):'')+'</p><p>'+esc((d.reasons||[])[0]||'请查看详细决策')+'</p>';q('#modal').classList.add('show');if(Notification.permission==='granted')new Notification('裸K '+actionLabel(d.action),{body:(d.name||d.code||'系统')+(d.referencePrice?' · '+d.referencePrice:'')})}
function render(){const s=state,d=s.day.timeline.at(-1),pending=s.pending,open=s.day.open,a=s.account,p=a.position;q('#decisionCard').className='card decision '+(d?.action||'');q('#action').textContent=d?actionLabel(d.action):(open?'等待09:35':'等待09:25');q('#decisionBody').innerHTML=d?row('股票',esc(d.name||d.code||'—'))+row('参考价',d.referencePrice??'—')+row('置信度',d.confidence==null?'—':pct(d.confidence))+row('证据',d.evidenceCount==null?'—':d.evidenceCount+'项')+row('数据截止',esc(d.dataCutoff))+((d.reasons||[]).map(x=>'<details><summary>理由</summary>'+esc(x)+'</details>').join(''))+((d.risks||[]).map(x=>'<details><summary>风险</summary>'+esc(x)+'</details>').join('')):'<div class="empty">尚无盘中决策</div>';q('#decisionButtons').innerHTML=pending?'<button class="secondary" id="ignore">忽略本次建议</button>':'';if(pending){q('#code').value=pending.code||'';q('#price').value=pending.referencePrice||'';q('#side').value=pending.action;q('#ignore').onclick=ignorePending}
q('#openStatus').innerHTML=open?row('开盘广度',open.openBreadth==null?'—':pct(open.openBreadth))+row('37%门槛',open.entryAllowed?'<span class="pill">允许入场</span>':'<span class="pill bad">禁止开仓</span>')+row('数据截止',esc(open.dataCutoff)):'<div class="empty">等待09:25任务</div>';q('#top3').innerHTML=open?.top3?.length?open.top3.map((x,i)=>'<div class="candidate"><small>TOP '+(i+1)+' · '+esc(x.industry||'—')+'</small><b>'+esc(x.name)+'</b><span>'+esc(x.code)+'</span></div>').join(''):'<div class="empty">暂无观察股</div>';
q('#position').innerHTML=p?row('股票',esc(p.name)+' '+esc(p.code))+row('状态',p.canSell?'<span class="pill">T+1可卖</span>':'<span class="pill warn">T+0锁定</span>')+row('股数',p.shares+'股')+row('成本价',Number(p.cost).toFixed(3))+row('最新价',Number(p.markPrice).toFixed(3))+row('浮动盈亏','<span class="'+(p.unrealizedPnl>=0?'positive':'negative')+'">'+money(p.unrealizedPnl)+'</span>'):'<div class="empty">当前空仓 · '+esc(s.state)+'</div>';q('#account').innerHTML=row('初始资金',money(a.initialCapital))+row('可用现金',money(a.cash))+row('持仓市值',money(a.marketValue))+row('总资产',money(a.totalAssets))+row('累计收益','<span class="'+(a.totalReturn>=0?'positive':'negative')+'">'+pct(a.totalReturn)+'</span>')+row('已实现盈亏',money(a.realizedPnl))+row('手续费',money(a.totalFees));
q('#timeline').innerHTML=s.day.timeline.length?'<table><thead><tr><th>时间</th><th>动作</th><th>股票</th><th>置信度</th><th>证据</th><th>状态</th></tr></thead><tbody>'+[...s.day.timeline].reverse().map(x=>'<tr><td>'+esc(new Date(x.generatedAt).toLocaleTimeString('zh-CN',{hour:'2-digit',minute:'2-digit'}))+'</td><td>'+actionLabel(x.action)+'</td><td>'+esc(x.name||x.code||'—')+'</td><td>'+(x.confidence==null?'—':pct(x.confidence))+'</td><td>'+(x.evidenceCount??'—')+'</td><td>'+esc(x.status)+'</td></tr>').join('')+'</tbody></table>':'<div class="empty">今日尚无决策</div>';q('#fills').innerHTML=a.fills.length?'<table><thead><tr><th>时间</th><th>方向</th><th>股票</th><th>价格</th><th>股数</th><th>费用</th></tr></thead><tbody>'+a.fills.slice(0,20).map(x=>'<tr><td>'+esc(new Date(x.createdAt).toLocaleString('zh-CN'))+'</td><td>'+(x.side==='buy'?'买入':'卖出')+'</td><td>'+esc(x.name)+' '+esc(x.code)+'</td><td>'+x.price.toFixed(3)+'</td><td>'+x.shares+'</td><td>'+money(x.fee)+'</td></tr>').join('')+'</tbody></table>':'<div class="empty">尚无成交</div>';
const sys=s.system;q('#system').innerHTML=row('服务','<span class="pill">在线</span>')+row('状态机',esc(s.state))+row('自动运行',esc(sys.scheduler.status||sys.scheduler.message||'未接入'))+row('行情',esc(sys.marketData.status||'unknown'))+row('DeepSeek',esc(sys.llm.status||'unknown'));q('#logs').innerHTML=sys.logs.length?sys.logs.map(x=>'<div class="log-line '+esc(x.level)+'">'+esc(x.timestamp)+' ['+esc(x.level)+'] '+esc(x.event)+' · '+esc(x.message)+'</div>').join(''):'<div class="empty">暂无日志</div>';alertDecision(d)}
async function api(path,options){const r=await fetch(path,options),d=await r.json();if(!r.ok)throw Error(d.error||'请求失败');return d}async function refresh(){state=await api('/api/naked-k-live/status');render()}async function ignorePending(){await api('/api/naked-k-live/decisions/'+encodeURIComponent(state.pending.id)+'/ignore',{method:'POST'});await refresh()}q('#fillForm').onsubmit=async e=>{e.preventDefault();const b=q('#saveFill');b.disabled=true;const requestId=crypto.randomUUID();try{const d=await api('/api/naked-k-live/fills',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({requestId,side:q('#side').value,code:q('#code').value,price:Number(q('#price').value),shares:Number(q('#shares').value)})});q('#message').textContent=d.warning||d.fill.name+'成交已保存';q('#fillForm').reset();await refresh()}catch(err){q('#message').textContent=err.message}finally{b.disabled=false}};q('#modalClose').onclick=()=>q('#modal').classList.remove('show');if('Notification'in window&&Notification.permission==='default')Notification.requestPermission();refresh().catch(e=>q('#action').textContent=e.message);setInterval(()=>refresh().catch(()=>{}),15000);
</script></body></html>`;
}
