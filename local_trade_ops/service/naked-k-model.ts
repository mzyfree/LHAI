import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { loadNakedKStatus } from "./naked-k.ts";

const OPS_HOME = resolve(new URL("..", import.meta.url).pathname);
const STATE_DIR = process.env.NAKED_K_MODEL_STATE_DIR
  ? resolve(process.env.NAKED_K_MODEL_STATE_DIR)
  : resolve(OPS_HOME, "state", "naked_k_model_forward");

type Phase = "market" | "hotspot" | "selection" | "entry" | "exit";
const PHASES: Phase[] = ["market", "hotspot", "selection", "entry", "exit"];

function decisionPath(date: string, phase: Phase) {
  return resolve(STATE_DIR, `${date}_${phase}.json`);
}

function promptFor(phase: Phase) {
  const status = loadNakedKStatus();
  const signal = status.forward.signal as Record<string, unknown> | null;
  const llm = (signal?.llmContext || {}) as Record<string, unknown>;
  const tail = status.automation.phases.find((item) => item.phase === "tail")?.data ?? null;
  const contexts: Record<Phase, unknown> = {
    market: { timestamp: status.now, market: llm.market || { openBreadth: signal?.marketOpenBreadth ?? null } },
    hotspot: { timestamp: status.now, market: llm.market || {}, sectorLeaderboard: llm.sectorLeaderboard || signal?.hotSectors || [] },
    selection: { timestamp: status.now, market: llm.market || {}, sectorLeaderboard: llm.sectorLeaderboard || signal?.hotSectors || [], stockPool: llm.stockPool || signal?.candidates || [] },
    entry: { timestamp: status.now, market: llm.market || {}, stockPool: llm.stockPool || signal?.candidates || [], currentPosition: status.forward.position },
    exit: { timestamp: status.now, currentPosition: status.forward.position, tailSnapshot: tail, originalOpenContext: signal },
  };
  const tasks: Record<Phase, string> = {
    market: "判断今天是否适合短线开新仓，以及建议风险档位。",
    hotspot: "识别最多3个具有真实联动而非少数个股脉冲的热点板块。",
    selection: "从股票池中选择最多1只结构、辨识度和板块联动最匹配的股票，也可以放弃。",
    entry: "判断目标股票此刻应买入、等待还是放弃，并给出可验证的触发条件。",
    exit: "对当前持仓判断继续持有或卖出，不得推荐新股票。",
  };
  const schemas: Record<Phase, string> = {
    market: '{"decision":"risk_on|cautious|risk_off","confidence":0到1,"reasons":["..."],"risks":["..."],"invalidation":"..."}',
    hotspot: '{"decision":"accept|none","sectors":["板块名"],"confidence":0到1,"reasons":["..."],"risks":["..."],"invalidation":"..."}',
    selection: '{"decision":"select|skip","selectedCode":"股票代码或null","confidence":0到1,"reasons":["..."],"risks":["..."],"invalidation":"..."}',
    entry: '{"decision":"buy|wait|skip","selectedCode":"股票代码或null","confidence":0到1,"reasons":["..."],"risks":["..."],"invalidation":"..."}',
    exit: '{"decision":"hold|sell","selectedCode":"持仓代码或null","confidence":0到1,"reasons":["..."],"risks":["..."],"invalidation":"..."}',
  };
  return `你是A股裸K短线交易系统中的独立决策节点，不是收益承诺者。\n\n本节点任务：${tasks[phase]}\n\n通用约束：\n1. 仅使用INPUT中的截至当前时点数据，不得补充新闻、传闻或未来数据。\n2. 不把涨幅大直接等同于强，不把模型信心解释为上涨概率。\n3. 数据缺失、样本过少、结论冲突时必须降低confidence并倾向保守。\n4. 股票只能从stockPool选择；退出节点只能处理currentPosition。\n5. reasons必须引用具体字段，risks必须写出最可能失败方式。\n6. 只返回一个符合OUTPUT_SCHEMA的JSON对象，不要Markdown或额外文字。\n\nOUTPUT_SCHEMA:\n${schemas[phase]}\n\nINPUT:\n${JSON.stringify(contexts[phase], null, 2)}`;
}

function parseDecision(value: unknown, phase: Phase) {
  const input = typeof value === "string" ? JSON.parse(value) : value;
  if (!input || typeof input !== "object" || Array.isArray(input)) throw new Error("模型结果必须是JSON对象");
  const data = input as Record<string, unknown>;
  const allowedByPhase: Record<Phase, string[]> = { market: ["risk_on", "cautious", "risk_off"], hotspot: ["accept", "none"], selection: ["select", "skip"], entry: ["buy", "wait", "skip"], exit: ["hold", "sell"] };
  const allowed = allowedByPhase[phase];
  if (!allowed.includes(String(data.decision))) throw new Error(`decision只能是 ${allowed.join(" / ")}`);
  const confidence = Number(data.confidence);
  if (!Number.isFinite(confidence) || confidence < 0 || confidence > 1) throw new Error("confidence必须在0到1之间");
  if (!Array.isArray(data.reasons) || !Array.isArray(data.risks)) throw new Error("reasons和risks必须是数组");
  return {
    decision: String(data.decision),
    selectedCode: data.selectedCode == null ? null : String(data.selectedCode),
    sectors: Array.isArray(data.sectors) ? data.sectors.map(String).slice(0, 3) : [],
    confidence,
    reasons: data.reasons.map(String).slice(0, 8),
    risks: data.risks.map(String).slice(0, 8),
    invalidation: String(data.invalidation || ""),
  };
}

export function loadNakedKModelStatus() {
  const base = loadNakedKStatus();
  const date = base.today;
  const read = (phase: Phase) => {
    const path = decisionPath(date, phase);
    return existsSync(path) ? JSON.parse(readFileSync(path, "utf8")) : null;
  };
  return { now: base.now, today: date, base, prompts: Object.fromEntries(PHASES.map((phase) => [phase, promptFor(phase)])), decisions: Object.fromEntries(PHASES.map((phase) => [phase, read(phase)])), mode: "shadow" };
}

export function saveNakedKModelDecision(body: Record<string, unknown>) {
  const phase = String(body.phase) as Phase;
  if (!PHASES.includes(phase)) throw new Error(`phase只能是${PHASES.join("、")}`);
  const base = loadNakedKStatus();
  const result = parseDecision(body.result, phase);
  const signal = base.forward.signal as { candidates?: Array<{ ts_code: string }>; llmContext?: { stockPool?: Array<{ ts_code: string }> } } | null;
  const candidates = (signal?.llmContext?.stockPool || signal?.candidates || []).map((item) => item.ts_code);
  if ((phase === "selection" && result.decision === "select") || (phase === "entry" && result.decision === "buy")) {
    if (!candidates.includes(result.selectedCode || "")) throw new Error("所选代码必须来自客观股票池");
  }
  const position = base.forward.position as { code?: string } | null;
  if (phase === "exit" && result.selectedCode && result.selectedCode !== position?.code) throw new Error("退出代码必须与当前持仓一致");
  const payload = { date: base.today, phase, mode: "shadow", model: String(body.model || "ChatGPT"), createdAt: new Date().toISOString(), prompt: promptFor(phase), result };
  mkdirSync(STATE_DIR, { recursive: true });
  writeFileSync(decisionPath(base.today, phase), `${JSON.stringify(payload, null, 2)}\n`, "utf8");
  return payload;
}

export function nakedKModelHtml() {
  return `<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>裸K模型版</title><style>
:root{--ink:#13221b;--paper:#f3efe4;--card:#fffdf7;--line:#c9c3b5;--orange:#c64b2d;--teal:#176c5a}*{box-sizing:border-box}body{margin:0;background:linear-gradient(135deg,#e8eee7,#f5ecdc);color:var(--ink);font-family:"Songti SC","STSong",serif}.shell{max-width:1180px;margin:auto;padding:30px 22px 60px}header{padding:24px 0;border-bottom:2px solid var(--ink);display:flex;justify-content:space-between;gap:20px}h1{font-size:58px;margin:4px 0 10px}.sub{color:#657067;line-height:1.7}.badge{height:max-content;border:2px solid var(--orange);color:var(--orange);padding:9px 12px;font:bold 13px monospace}.notice{margin:20px 0;padding:14px;background:#fff3ce;border-left:5px solid #c48a2c;font:14px sans-serif}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.card{background:var(--card);border:1px solid var(--line);box-shadow:5px 6px 0 #25352a16;padding:20px}.wide{grid-column:1/-1}h2{margin:0 0 14px}textarea{width:100%;min-height:310px;padding:13px;border:1px solid var(--line);background:#f8f6ef;font:12px/1.55 ui-monospace,monospace;resize:vertical}.response{min-height:180px}button,select{padding:10px 14px;border:1px solid var(--ink);background:var(--ink);color:white;font-weight:bold;margin:8px 7px 0 0;cursor:pointer}.secondary{background:transparent;color:var(--ink)}.result{font:14px/1.65 sans-serif}.muted{color:#69736b}.foot{margin-top:18px;display:flex;justify-content:space-between;font:13px sans-serif}.foot a{color:var(--ink)}@media(max-width:760px){.grid{grid-template-columns:1fr}.wide{grid-column:auto}h1{font-size:42px}header{display:block}.badge{display:inline-block;margin-top:12px}}
</style></head><body><main class="shell"><header><div><div class="muted">LLM FULL PIPELINE</div><h1>裸K模型版</h1><div class="sub">代码准备客观行情与硬风控，LLM依次参与市场判断、热点识别、选股、买点和卖点。当前仅记录影子决策。</div></div><div class="badge">全流程提示词 · 不下单</div></header><div class="notice">五个节点应按顺序执行并保留结果。当前页面不会修改规则版信号或自动成交，避免未经验证的模型判断进入实盘。</div><section class="grid"><div class="card"><h2>决策阶段</h2><select id="phase"><option value="market">1. 市场状态</option><option value="hotspot">2. 热点板块</option><option value="selection">3. 候选选股</option><option value="entry">4. 买入时点</option><option value="exit">5. 卖出时点</option></select><button id="copy">复制提示词</button><div id="message" class="muted"></div></div><div class="card"><h2>已保存决策</h2><div id="saved" class="result muted">尚未保存</div></div><div class="card wide"><h2>结构化提示词</h2><textarea id="prompt" readonly></textarea></div><div class="card wide"><h2>粘贴 ChatGPT JSON 结果</h2><textarea id="response" class="response" placeholder='粘贴该节点要求的JSON对象'></textarea><button id="save">保存影子决策</button></div></section><div class="foot"><span id="updated"></span><div><a href="/naked-k">规则版</a> · <a href="/">交易平台</a></div></div></main><script>
let state;const phase=document.querySelector('#phase'),prompt=document.querySelector('#prompt'),saved=document.querySelector('#saved'),message=document.querySelector('#message');const esc=s=>String(s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));function render(){const p=phase.value;prompt.value=state.prompts[p];const d=state.decisions[p];saved.innerHTML=d?'<b>'+esc(d.result.decision.toUpperCase())+'</b> · 置信度 '+Math.round(d.result.confidence*100)+'%<br>'+d.result.reasons.map(esc).join('；'):'尚未保存 '+p+' 决策';document.querySelector('#updated').textContent='更新于 '+new Date(state.now).toLocaleString('zh-CN')}async function refresh(){const r=await fetch('/api/naked-k-model/status');state=await r.json();if(!r.ok)throw Error(state.error||'加载失败');render()}phase.addEventListener('change',render);document.querySelector('#copy').onclick=async()=>{await navigator.clipboard.writeText(prompt.value);message.textContent='提示词已复制，请发送给 ChatGPT'};document.querySelector('#save').onclick=async()=>{try{const r=await fetch('/api/naked-k-model/decisions',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({phase:phase.value,model:'ChatGPT',result:document.querySelector('#response').value})}),d=await r.json();if(!r.ok)throw Error(d.error||'保存失败');message.textContent='影子决策已保存';document.querySelector('#response').value='';await refresh()}catch(e){message.textContent=e.message}};refresh().catch(e=>message.textContent=e.message);
</script></body></html>`;
}
