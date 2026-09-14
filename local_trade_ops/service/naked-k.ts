import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

type StrategyConfig = {
  strategy_id: string;
  status: string;
  frozen_at: string;
  research_data_end: string;
  forward_observation_start: string;
  initial_capital: number;
  pool_count: number;
  topk: number;
  allow_preemption: boolean;
  leader_weights: Record<string, number>;
  entry: Record<string, number>;
  exit: Record<string, number | boolean | null>;
  costs: Record<string, number>;
  in_sample_reference: Record<string, number | string>;
};

const OPS_HOME = resolve(new URL("..", import.meta.url).pathname);
const CONFIG_PATH = resolve(OPS_HOME, "config", "auction_top1_forward_20260908.json");
const NAV_PATH = resolve(
  OPS_HOME,
  "..",
  "reports",
  "auction_parameter_research",
  "auction_parameter_risk_robustness_2026_details",
  "best_0bps_nav.csv",
);
const TRADES_PATH = resolve(
  OPS_HOME,
  "..",
  "reports",
  "auction_parameter_research",
  "auction_parameter_risk_robustness_2026_details",
  "best_0bps_trades.csv",
);
const STATE_DIR = process.env.NAKED_K_STATE_DIR
  ? resolve(process.env.NAKED_K_STATE_DIR)
  : resolve(OPS_HOME, "state", "naked_k_forward");
const LEDGER_PATH = resolve(STATE_DIR, "ledger.json");

type Fill = {
  id: string;
  side: "buy" | "sell";
  date: string;
  code: string;
  name: string;
  price: number;
  shares: number;
  gross: number;
  fee: number;
  note: string;
  createdAt: string;
};

type Ledger = { fills: Fill[] };

function readLedger(): Ledger {
  if (!existsSync(LEDGER_PATH)) return { fills: [] };
  const value = JSON.parse(readFileSync(LEDGER_PATH, "utf8")) as Ledger;
  return { fills: Array.isArray(value.fills) ? value.fills : [] };
}

function ledgerView(ledger: Ledger) {
  const positions = new Map<string, { code: string; name: string; shares: number; cost: number }>();
  let realizedPnl = 0;
  let netCashFlow = 0;
  for (const fill of ledger.fills) {
    const position = positions.get(fill.code) || { code: fill.code, name: fill.name, shares: 0, cost: 0 };
    if (fill.side === "buy") {
      const oldGross = position.shares * position.cost;
      position.shares += fill.shares;
      position.cost = position.shares ? (oldGross + fill.gross + fill.fee) / position.shares : 0;
      netCashFlow -= fill.gross + fill.fee;
    } else {
      realizedPnl += fill.gross - fill.fee - position.cost * fill.shares;
      position.shares -= fill.shares;
      netCashFlow += fill.gross - fill.fee;
    }
    positions.set(fill.code, position);
  }
  const openPositions = [...positions.values()].filter((position) => position.shares > 0);
  return {
    positions: openPositions,
    fills: [...ledger.fills].reverse(),
    openCostValue: openPositions.reduce((sum, position) => sum + position.cost * position.shares, 0),
    realizedPnl,
    netCashFlow,
  };
}

function normalizeStockCode(value: unknown): string {
  const raw = String(value || "").trim().toUpperCase();
  if (/^\d{6}\.(SH|SZ|BJ)$/.test(raw)) return raw;
  const prefixed = raw.match(/^(SH|SZ|BJ)(\d{6})$/);
  if (prefixed) return `${prefixed[2]}.${prefixed[1]}`;
  if (!/^\d{6}$/.test(raw)) throw new Error("股票代码格式不正确");
  if (/^(6|68)/.test(raw)) return `${raw}.SH`;
  if (/^[03]/.test(raw)) return `${raw}.SZ`;
  return `${raw}.BJ`;
}

function inferStockName(code: string, ledger: Ledger): string {
  const previous = [...ledger.fills].reverse().find((fill) => fill.code === code)?.name;
  if (previous) return previous;
  const today = new Intl.DateTimeFormat("en-CA", {
    timeZone: "Asia/Shanghai", year: "numeric", month: "2-digit", day: "2-digit",
  }).format(new Date());
  const signal = phaseStatus(today, "open").data as { candidates?: Array<{ ts_code: string; name: string }> } | null;
  return signal?.candidates?.find((candidate) => candidate.ts_code === code)?.name || "";
}

export function recordNakedKFill(body: Record<string, unknown>) {
  const strategy = JSON.parse(readFileSync(CONFIG_PATH, "utf8")) as StrategyConfig;
  const ledger = readLedger();
  const side = String(body.side || "") as "buy" | "sell";
  const code = normalizeStockCode(body.code);
  const name = String(body.name || "").trim() || inferStockName(code, ledger);
  const price = Number(body.price);
  const lots = Number(body.lots);
  const date = String(body.date || "").trim();
  if (!(["buy", "sell"] as string[]).includes(side)) throw new Error("side must be buy or sell");
  if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) throw new Error("date must be YYYY-MM-DD");
  if (!name) throw new Error("未能从今日候选或历史持仓中识别股票名称");
  if (!Number.isFinite(price) || price <= 0) throw new Error("price must be positive");
  if (!Number.isInteger(lots) || lots <= 0) throw new Error("lots must be a positive integer");
  const shares = lots * strategy.costs.lot_size;
  const gross = price * shares;
  const rate = side === "buy" ? strategy.costs.buy_rate : strategy.costs.sell_rate;
  const fee = Math.round(Math.max(gross * rate, strategy.costs.minimum_fee) * 100) / 100;
  const current = ledgerView(ledger);
  if (side === "buy") {
    if (current.positions.length) throw new Error("当前冻结策略为单池，已有持仓时不能新增买入");
  } else {
    const position = current.positions.find((item) => item.code === code);
    if (!position || position.shares < shares) throw new Error("卖出数量超过当前持仓");
  }
  const fill: Fill = {
    id: `fill-${Date.now()}-${Math.random().toString(16).slice(2)}`,
    side, date, code, name, price, shares, gross, fee,
    note: String(body.note || "").trim(), createdAt: new Date().toISOString(),
  };
  ledger.fills.push(fill);
  mkdirSync(STATE_DIR, { recursive: true });
  writeFileSync(LEDGER_PATH, `${JSON.stringify(ledger, null, 2)}\n`, "utf8");
  return { fill, account: ledgerView(ledger) };
}

function readCsv(path: string): Record<string, string>[] {
  if (!existsSync(path)) return [];
  const lines = readFileSync(path, "utf8").trim().split(/\r?\n/).filter(Boolean);
  if (lines.length < 2) return [];
  const headers = lines[0].split(",");
  return lines.slice(1).map((line) => {
    const cells = line.split(",");
    return Object.fromEntries(headers.map((header, index) => [header, cells[index] ?? ""]));
  });
}

function numberFrom(row: Record<string, string>, names: string[]): number | null {
  for (const name of names) {
    const value = Number(row[name]);
    if (Number.isFinite(value)) return value;
  }
  return null;
}

function dateFrom(row: Record<string, string>): string {
  return row.date || row.datetime || row.trade_date || row.exit_date || row.entry_date || "";
}

function phaseStatus(date: string, phase: string) {
  const path = resolve(STATE_DIR, "phases", `${date}_${phase}.json`);
  return { phase, path, data: existsSync(path) ? JSON.parse(readFileSync(path, "utf8")) : null };
}

function monthlyReturns(navRows: Record<string, string>[], initialCapital: number) {
  const rows = navRows
    .map((row) => ({
      date: dateFrom(row).slice(0, 10),
      nav: numberFrom(row, ["nav", "total_value", "account_value", "equity"]),
    }))
    .filter((row): row is { date: string; nav: number } => Boolean(row.date && row.nav !== null))
    .sort((a, b) => a.date.localeCompare(b.date));
  if (!rows.length) return [];
  const output: { month: string; return: number; endNav: number }[] = [];
  let previous = initialCapital;
  for (const month of [...new Set(rows.map((row) => row.date.slice(0, 7)))]) {
    const monthRows = rows.filter((row) => row.date.startsWith(month));
    const endNav = Number(monthRows.at(-1)?.nav || previous);
    output.push({ month, return: previous ? endNav / previous - 1 : 0, endNav });
    previous = endNav;
  }
  return output;
}

export function loadNakedKStatus() {
  if (!existsSync(CONFIG_PATH)) throw new Error(`裸K策略配置不存在: ${CONFIG_PATH}`);
  const strategy = JSON.parse(readFileSync(CONFIG_PATH, "utf8")) as StrategyConfig;
  const navRows = readCsv(NAV_PATH);
  const trades = readCsv(TRADES_PATH);
  const account = ledgerView(readLedger());
  const forwardNavPath = resolve(STATE_DIR, "forward_performance.json");
  const forwardNav = existsSync(forwardNavPath) ? JSON.parse(readFileSync(forwardNavPath, "utf8")) : [];
  const today = new Intl.DateTimeFormat("en-CA", {
    timeZone: "Asia/Shanghai",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).format(new Date());
  const forwardDays = Math.max(
    0,
    Math.floor((Date.parse(`${today}T00:00:00+08:00`) - Date.parse(`${strategy.forward_observation_start}T00:00:00+08:00`)) / 86_400_000) + 1,
  );
  return {
    now: new Date().toISOString(),
    today,
    strategy,
    research: {
      navPath: NAV_PATH,
      tradesPath: TRADES_PATH,
      monthlyReturns: monthlyReturns(navRows, strategy.initial_capital),
      recentTrades: trades.slice(-8).reverse(),
    },
    account: { ...account, navHistory: forwardNav, latestNav: forwardNav.at(-1) || null },
    automation: {
      phases: ["preflight", "open", "tail", "settle"].map((phase) => phaseStatus(today, phase)),
      schedule: { preflight: "09:00", open: "09:25", tail: "14:50", settle: "15:10" },
    },
    forward: {
      status: today < strategy.forward_observation_start ? "waiting" : "observing",
      calendarDays: forwardDays,
      signal: phaseStatus(today, "open").data,
      position: account.positions[0] || null,
      note: "前向信号账本尚未接入。页面不会用样本内交易冒充实时信号。",
    },
  };
}

export function nakedKHtml() {
  return `<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>裸K策略</title><style>
:root{--paper:#f2eddf;--ink:#17221b;--muted:#657067;--line:#c9c5b8;--red:#bd3c27;--green:#14705b;--gold:#c08b32;--card:#fffdf7}
*{box-sizing:border-box}body{margin:0;color:var(--ink);background:radial-gradient(circle at 12% 8%,#fff9df 0,transparent 32%),linear-gradient(145deg,#e9efe7,#f4ecdc 58%,#e8ddd0);font-family:"Songti SC","STSong",serif;min-height:100vh}
body:before{content:"";position:fixed;inset:0;pointer-events:none;opacity:.22;background-image:repeating-linear-gradient(90deg,transparent 0 47px,#8a806c22 48px),repeating-linear-gradient(0deg,transparent 0 47px,#8a806c18 48px)}
.shell{position:relative;max-width:1240px;margin:auto;padding:28px 22px 60px}.top{display:flex;align-items:flex-start;justify-content:space-between;gap:24px;padding:26px 0 30px;border-bottom:2px solid var(--ink)}
.eyebrow{font:700 12px/1.2 ui-monospace,monospace;letter-spacing:.18em;color:var(--red)}h1{font-size:clamp(40px,7vw,82px);line-height:.9;margin:13px 0 18px;letter-spacing:-.06em}.lead{max-width:690px;margin:0;color:var(--muted);font-size:16px;line-height:1.7}.stamp{border:2px solid var(--red);color:var(--red);padding:10px 13px;font:bold 13px ui-monospace,monospace;transform:rotate(3deg);white-space:nowrap}
.panel{background:color-mix(in srgb,var(--card) 92%,transparent);border:1px solid var(--line);box-shadow:5px 6px 0 #28352b14}.positive{color:var(--red)}.negative{color:var(--green)}
.grid{display:grid;grid-template-columns:1.12fr .88fr;gap:16px}.panel{padding:20px}.panel h2{font-size:22px;margin:0 0 16px}.status-row{display:flex;justify-content:space-between;gap:15px;padding:11px 0;border-bottom:1px dashed var(--line);font:14px sans-serif}.status-row:last-child{border:0}.tag{display:inline-block;border-radius:999px;padding:5px 9px;background:#dce8df;color:#28563d;font:bold 12px sans-serif}.weights{display:grid;gap:11px}.weight-head{display:flex;justify-content:space-between;font:13px sans-serif}.bar{height:9px;background:#e2dfd5}.bar i{display:block;height:100%;background:linear-gradient(90deg,var(--green),#6d9e74)}
form{display:grid;grid-template-columns:110px 1fr 140px 110px 120px;gap:9px}input,select,button{min-width:0;border:1px solid var(--line);padding:10px;background:#fffdf8;color:var(--ink);font:14px sans-serif}button{background:var(--ink);color:#fff;font-weight:700;cursor:pointer}.message{min-height:20px;margin:9px 0 0;color:var(--muted);font:13px sans-serif}
table{width:100%;border-collapse:collapse;font:13px sans-serif}th,td{text-align:left;padding:9px 7px;border-bottom:1px solid var(--line)}th{color:var(--muted);font-size:11px;text-transform:uppercase}.wide{grid-column:1/-1}.empty{padding:26px;text-align:center;background:#f0eee5;color:var(--muted);font:14px sans-serif}.foot{display:flex;justify-content:space-between;margin-top:16px;color:var(--muted);font:12px sans-serif}.foot a{color:var(--ink)}
@media(max-width:800px){.top{display:block}.stamp{display:inline-block;margin-top:18px}.grid{grid-template-columns:1fr}}
</style></head><body><main class="shell"><header class="top"><div><div class="eyebrow">AUCTION / NAKED K / FORWARD TEST</div><h1>裸K策略</h1><p class="lead">集合竞价识别热点，只在最强板块选择一只结构最清晰的股票。单池、不抢占，所有决策保留解释和审计轨迹。</p></div><div class="stamp">前向观察 · 参数冻结</div></header>
<div class="grid"><section class="panel"><h2>今日状态</h2><div id="today"></div></section><section class="panel"><h2>自动任务</h2><div id="automation"></div></section><section class="panel"><h2>龙头评分</h2><div class="weights" id="weights"></div></section><section class="panel"><h2>入场纪律</h2><div id="entry"></div></section><section class="panel"><h2>退出纪律</h2><div id="exit"></div></section><section class="panel wide"><h2>人工成交回填</h2><form id="fillForm"><select id="side"><option value="buy">买入</option><option value="sell">卖出</option></select><input id="code" placeholder="股票代码" required><input id="price" type="number" step="0.001" min="0" placeholder="成交价" required><input id="lots" type="number" step="1" min="1" placeholder="手数" required><button>保存成交</button></form><div class="message" id="message">股票名称自动识别；1手等于100股，系统自动计算手续费。</div></section><section class="panel"><h2>当前账户</h2><div id="account"></div></section><section class="panel"><h2>成交历史</h2><div id="fills"></div></section></div><div class="foot"><span id="updated"></span><a href="/">返回交易平台</a></div></main>
<script>
const pct=v=>(Number(v)*100).toFixed(2)+'%';const money=v=>'¥'+Number(v).toLocaleString('zh-CN',{maximumFractionDigits:0});
const row=(a,b)=>'<div class="status-row"><span>'+a+'</span><b>'+b+'</b></div>';
async function refresh(){const r=await fetch('/api/naked-k/status'),s=await r.json();if(!r.ok)throw new Error(s.error||'加载失败');const c=s.strategy;
const sig=s.forward.signal,top=sig&&sig.top1;document.querySelector('#today').innerHTML=row('策略编号',c.strategy_id)+row('观察状态','<span class="tag">'+(s.forward.status==='observing'?'前向观察中':'等待开始')+'</span>')+row('严格前向起点',c.forward_observation_start)+row('今日决策',!sig?'等待09:25':sig.status==='failed'?'数据失败 · 禁止交易':sig.decision==='buy'?'买入 '+top.name+' '+top.ts_code:'空仓')+row('参考开盘价',top?Number(top.open).toFixed(2):'—')+row('持仓',s.forward.position?s.forward.position.name+' · '+s.forward.position.shares+'股':'空仓');
if(top&&!document.querySelector('#code').value)document.querySelector('#code').value=top.ts_code;
const phaseNames={preflight:'盘前预计算',open:'竞价决策',tail:'尾盘判断',settle:'收盘结算'};document.querySelector('#automation').innerHTML=s.automation.phases.map(p=>{const d=p.data,status=!d?'等待执行':d.status==='failed'?'失败':d.status==='awaiting_official_close_data'?'等待收盘数据':'已完成';return row(s.automation.schedule[p.phase]+' '+phaseNames[p.phase],status)}).join('');
const names={open_strength:'集合竞价强度',prior_5d_strength:'前 5 日强度',naked_k_structure:'裸K结构',limit_memory_60d:'60日涨停记忆'};document.querySelector('#weights').innerHTML=Object.entries(c.leader_weights).map(([k,v])=>'<div><div class="weight-head"><span>'+names[k]+'</span><b>'+pct(v)+'</b></div><div class="bar"><i style="width:'+pct(v)+'"></i></div></div>').join('');
document.querySelector('#entry').innerHTML=row('资金池','单池 · Top1')+row('热点板块','仅最强 1 个')+row('市场上涨家数','≥ '+pct(c.entry.min_market_open_breadth))+row('个股高开','≤ '+pct(c.entry.max_entry_gap))+row('前20日涨幅','≤ '+pct(c.entry.max_prior_return_20d))+row('抢占换仓',c.allow_preemption?'允许':'禁止');
document.querySelector('#exit').innerHTML=row('单日跌幅',pct(c.exit.daily_drop_threshold)+' 收盘退出')+row('MA5','收盘跌破前一日 MA5 退出')+row('最长持有',c.exit.max_holding_days+' 个交易日')+row('盘中固定止损',c.exit.fixed_stop_loss==null?'关闭':'开启')+row('交易费用','买万三 / 卖万八');
document.querySelector('#updated').textContent='更新于 '+new Date(s.now).toLocaleString('zh-CN');
document.querySelector('#account').innerHTML=row('持仓成本',money(s.account.openCostValue))+row('已实现盈亏',money(s.account.realizedPnl))+row('净交易现金流',money(s.account.netCashFlow))+row('持仓数量',s.account.positions.length+' 只')+s.account.positions.map(p=>row(p.name,p.shares+'股 @ '+p.cost.toFixed(3))).join('');const fs=s.account.fills;document.querySelector('#fills').innerHTML=fs.length?'<table><thead><tr><th>日期</th><th>方向</th><th>股票</th><th>价格</th><th>数量</th><th>费用</th></tr></thead><tbody>'+fs.slice(0,8).map(f=>'<tr><td>'+f.date+'</td><td>'+(f.side==='buy'?'买':'卖')+'</td><td>'+f.name+'</td><td>'+f.price.toFixed(3)+'</td><td>'+f.shares+'</td><td>'+f.fee.toFixed(2)+'</td></tr>').join('')+'</tbody></table>':'<div class="empty">尚无前向模拟成交。</div>';}
document.querySelector('#fillForm').addEventListener('submit',async e=>{e.preventDefault();const body={side:side.value,date:new Intl.DateTimeFormat('en-CA',{timeZone:'Asia/Shanghai'}).format(new Date()),code:code.value,price:Number(price.value),lots:Number(lots.value)};const r=await fetch('/api/naked-k/fills',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(body)}),d=await r.json();if(!r.ok){message.textContent=d.error||'保存失败';return}message.textContent=d.fill.name+' 成交已保存，手续费 '+d.fill.fee.toFixed(2)+' 元';e.target.reset();await refresh()});
refresh().catch(e=>document.querySelector('#today').innerHTML='<div class="empty">'+e.message+'</div>');
</script></body></html>`;
}
