import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

const stateDir = mkdtempSync(join(tmpdir(), "naked-k-live-"));
process.env.NAKED_K_LIVE_STATE_DIR = stateDir;
process.env.NAKED_K_NATIVE_NOTIFICATIONS = "0";
delete process.env.INITIAL_CAPITAL;

const live = await import("../service/naked-k-live.ts");

test.after(() => rmSync(stateDir, { recursive: true, force: true }));

test("creates a 50k account and enforces the frozen single-position workflow", () => {
  const initial = live.loadNakedKLiveStatus();
  assert.equal(initial.account.initialCapital, 50_000);
  assert.equal(initial.account.cash, 50_000);
  assert.equal(initial.state, "PREMARKET");

  const open = live.recordNakedKLiveOpen({
    openBreadth: 0.42,
    top3: [{
      code: "600988", name: "赤峰黄金", industry: "黄金", price: 30,
      context: { prior_low: 28.4, prior_ma5_raw: 29.1 },
    }],
  });
  assert.equal(open.entryAllowed, true);
  assert.equal(live.loadNakedKLiveStatus().state, "WATCHING");

  const result = live.recordNakedKLiveDecision({
    id: "buy-1", action: "buy", code: "600988", name: "赤峰黄金",
    referencePrice: 30, confidence: 0.65, evidenceCount: 5,
  });
  assert.equal(result.decision.status, "pending");
  assert.equal(live.loadNakedKLiveStatus().state, "BUY_PENDING");

  const fill = live.recordNakedKLiveFill({
    requestId: "request-1", side: "buy", code: "600988", price: 30, shares: 100,
  });
  assert.equal(fill.fill.name, "赤峰黄金");
  assert.equal(fill.fill.linkedDecisionId, "buy-1");
  assert.deepEqual(fill.fill.entryContext, { prior_low: 28.4, prior_ma5_raw: 29.1 });
  assert.equal(fill.account.position?.shares, 100);
  assert.deepEqual(fill.account.position?.entryContext, { prior_low: 28.4, prior_ma5_raw: 29.1 });
  assert.equal(fill.account.cash, 46_995);
  assert.equal(live.loadNakedKLiveStatus().state, "T0_LOCKED");

  const duplicate = live.recordNakedKLiveFill({
    requestId: "request-1", side: "buy", code: "600988", price: 30, shares: 100,
  });
  assert.equal(duplicate.duplicate, true);
  assert.equal(duplicate.account.position?.shares, 100);

  assert.throws(() => live.recordNakedKLiveFill({
    requestId: "request-2", side: "buy", code: "601168", price: 20, shares: 100,
  }), /禁止新增或加仓/);
  assert.throws(() => live.recordNakedKLiveFill({
    requestId: "request-3", side: "sell", code: "600988", price: 31, shares: 100,
  }), /T\+1/);
});

test("blocks buy decisions when the 09:25 breadth gate is closed", () => {
  rmSync(stateDir, { recursive: true, force: true });
  mkdirSync(stateDir, { recursive: true });
  live.recordNakedKLiveOpen({ openBreadth: 0.36, top3: [] });
  assert.equal(live.loadNakedKLiveStatus().state, "ENTRY_BLOCKED");
  assert.throws(() => live.recordNakedKLiveDecision({
    id: "blocked-buy", action: "buy", code: "600111", referencePrice: 10,
  }), /没有开仓权限/);
});
