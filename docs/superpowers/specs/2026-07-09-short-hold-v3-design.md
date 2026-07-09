# Short Hold V3 Design

## 背景

当前短持策略的实盘口径是：

- T 日收盘后拿到最新 Qlib 数据并生成预测。
- T+1 开盘人工买入。
- T+2 尾盘或收盘卖出。

`short_hold_v2` 已经把主 label 调整为更贴近这个节奏的收益目标：

```text
Ref($close, -2) / Ref($open, -1) - 1
```

也就是用 T 日可见特征，预测从 T+1 开盘买入到 T+2 收盘卖出的收益。

目前的问题不在于流程不可用，而在于排序分数仍然偏单一：它主要表达预期收益，缺少“能不能买到”和“是否是 T+2 强势股/涨停股”的显式建模。因此模型经常把高收益但 T+1 开盘快速涨停、买不进去的股票排在前面。

## 目标

V3 的目标是把排序目标从“理论收益最高”升级为“可执行收益最高”。

需要满足：

- 继续沿用 T 信号、T+1 买入、T+2 卖出的短持节奏。
- 保留当前 `short_hold_v2` 收益模型作为核心收益分数。
- 增加可买入风险评分，降低 T+1 高开、涨停、一字板、盘口极难成交股票的排序。
- 增加强势评分，识别 T+2 更可能强收益或涨停的股票。
- 前端能解释每个候选的原始收益分、最终分、可买入风险、强势概率和过滤原因。
- 不影响老的 T/T+1 页面，也不替换现有 v2 模型包；V3 用独立 flavor 和配置。

## 非目标

本阶段不做以下事情：

- 不直接接入自动下单。
- 不把新闻/RAG 作为主排序模型，只把它作为后续风险解释或人工复核增强。
- 不立刻切换到 GNN、Transformer、LLM 端到端选股。
- 不承诺实盘收益，所有结果仍先以回测和观察盘验证。

## 推荐方案

采用“两层排序”设计：

1. 第一层继续使用现有 Qlib 三模型融合，得到 `return_score`。
2. 第二层用轻量 side model 或规则模型生成 `buyability_risk` 和 `strong_prob`。
3. 最终按 `final_score` 排序出 primary 和 backup。

推荐先做 sidecar，不急着改 Qlib 主训练结构：

```text
final_score = return_score
            + alpha * strong_prob
            - beta  * buyability_risk
            - gamma * liquidity_risk
```

这条路线比“直接重训复杂模型”更稳，因为可以快速回测每个因子的增益，也能清楚解释为什么某只股票从第一名被降权。

## Label 设计

### 收益 label

继续保留 `short_hold_v2` 的收益 label：

```text
return_label = Ref($close, -2) / Ref($open, -1) - 1
```

它对应 T 数据预测 T+1 开盘买、T+2 收盘卖的真实交易窗口。

### 可买入 label

训练时可以用未来数据生成标签，但推理时不能使用未来数据。

建议第一版把不可买入定义为以下情况之一：

- T+1 开盘涨幅超过 5%。
- T+1 开盘接近涨停。
- T+1 出现一字涨停或高度封板特征。
- T+1 日内可成交空间极低，例如 `high == low` 且接近涨停。

输出：

```text
buyability_risk in [0, 1]
```

值越大，越可能买不到或买入体验很差。

### 强势 label

强势目标用于补足“预测 T+2 会强势”的问题。

建议第一版定义为以下任一条件：

- T+1 开盘买入到 T+2 收盘收益进入当日候选池前分位。
- T+2 收盘接近涨停。
- T+2 收盘收益超过固定阈值，例如 6% 或 8%。

输出：

```text
strong_prob in [0, 1]
```

值越大，越像短持窗口内的强势股。

## Feature 设计

V3 第一阶段不新增复杂行业或新闻特征，先使用现有 Qlib 日线数据和当前模型输出。

候选级特征包括：

- `return_score`：三模型融合后的收益分。
- `model_rank`：收益模型原始排名。
- T 日收盘价、成交额、成交量、换手代理。
- 过去 5/10/20 日收益、波动、振幅。
- 过去若干日接近涨停次数、涨停后回落特征。
- T 日距离涨停价的距离。
- T 日成交额是否满足 `min_amount`。
- 是否 ST、是否停牌、是否科创/创业板权限受限。

这些特征都必须只使用 T 日及以前的信息。

## 订单生成逻辑

V3 的订单生成分为四步：

1. 生成或刷新最新 `short_hold_v2` 预测 pkl。
2. 对候选池计算 T 日可见特征。
3. 加载 V3 side model，得到 `buyability_risk`、`strong_prob`、`final_score`。
4. 用 `final_score` 排序生成 primary 和 backup。

前端表格建议展示：

- `model_rank`
- `return_score`
- `buyability_risk`
- `strong_prob`
- `final_score`
- `filter_reason`
- `order_role`

这样能明确区分“模型觉得收益高”和“系统觉得可执行价值高”。

## Backup 逻辑

Backup 不再只是“主单之后顺延的股票”，而是同一个 final_score 排序下的补位池。

规则：

- Primary 买不进、涨停、高开超过红线、未成交时，才考虑 backup。
- Backup 使用释放出来的主单预算重新 softmax 分配。
- Backup 也必须遵守 `price_5pct`、权限、ST、停牌、成交额过滤。
- 前端继续保留 backup，但明确显示它不是必须买。

## 训练流程

新增模型 flavor：

```text
short_hold_v3
```

第一阶段训练内容：

- Qlib 主收益模型仍使用 `short_hold_v2` label。
- Side model 使用历史候选样本训练 `buyability_risk` 和 `strong_prob`。
- Side model 可以先用 sklearn/lightgbm/xgboost 小模型，保存为独立 artifact。

本地配置建议：

```text
SHORT_HOLD_MODEL_FLAVOR=short_hold_v3
SHORT_HOLD_RETURN_MODEL_PACKAGE=latest_csi1000_short_hold_v2_model_package.tar.gz
SHORT_HOLD_SIDE_MODEL_PACKAGE=latest_csi1000_short_hold_v3_side_models.tar.gz
```

如果第一阶段验证有效，再考虑把 side labels 纳入完整 Qlib weekly training。

## 回测验证

必须和当前 v2 做对照，不看单日表现。

核心对照：

- v2：按 `return_score` 排序。
- v3-a：`return_score - beta * buyability_risk`。
- v3-b：`return_score + alpha * strong_prob`。
- v3-c：完整 `final_score`。

指标：

- 累计收益。
- 最大回撤。
- 信息比率。
- 平均仓位。
- 平均成交数量。
- 高开或涨停跳过数量。
- primary 买不到后 backup 的收益贡献。

至少跑 2026 YTD，并尽量扩展到更长历史区间。

## 风险

- 涨停和一字板只能通过日线近似，真实盘口排队成交无法完全还原。
- Buyability model 可能把强势股过度降权，导致错过大肉。
- 2026 YTD 样本较短，side model 必须尽量使用多年历史训练。
- 如果 final_score 权重调得太激进，容易从“收益模型”变成“规则模型”。
- 科创/创业权限、ST、涨跌停规则存在板块差异，需要前端继续提示人工确认。

## 推荐落地顺序

1. 先实现离线诊断脚本：从历史预测和 Qlib 日线生成 `buyability_risk`、`strong_prob` 标签和特征。
2. 用历史样本训练 side model，并输出每日候选诊断表。
3. 回测 v2 与 v3 final_score，确认不是只优化了少数日期。
4. 前端新增 V3 分数列和原因列，但默认仍可切回 v2。
5. 观察盘跑 1 到 2 周，再决定是否把 v3 设为默认。

## 当前结论

需要修正，但不建议马上大改主模型。

最优先的修正是：

- 保持 `short_hold_v2` 收益 label。
- 新增可买入风险和强势概率作为 side scoring。
- 用 final_score 重排 primary/backup。

这能直接回应两个核心问题：

- 模型目标和 T+1 买、T+2 卖的短持玩法是否一致。
- 第一名经常涨停买不进时，系统是否能提前降权或给出更可靠备选。
