# LHAI

一个用于实践 AI 在中国 A 股量化中应用的本地研究项目。

当前项目已经完成第一阶段工作：建立一个稳定、可复现、可对比的 A 股机器学习选股 baseline。

## 当前结论

第一阶段目前的主结论是：

- 主股票池选择 `csi500`
- 基线特征使用 `Alpha158`
- 基线模型使用 `LightGBM`
- 基线策略使用 `TopkDropout(topk=30, n_drop=3)`
- 该 baseline 在更靠后的测试窗口中仍然有效，但强度会下降

第一阶段最终输出见：

- [stage1_final_output.md](/Users/Dylan.Min/Documents/Code/learn/LHAI/reports/stage1_final_output.md)

## 项目目标

这个项目不是一上来就做复杂 AI 策略，而是按下面顺序推进：

1. 搭建稳定的量化研究闭环
2. 建立可复现的 baseline
3. 通过单变量实验找到更合适的市场和策略设定
4. 验证稳健性
5. 再逐步引入 AI 特征、文本信号和更复杂模型

## 目录说明

```text
LHAI/
├── README.md
├── requirements-qlib.txt
├── data/
│   ├── qlib_cn_data/
│   ├── raw/
│   └── processed/
├── mlruns/
├── notebooks/
├── qlib/
├── reports/
└── src/
    ├── backtest/
    ├── configs/
    ├── data/
    ├── features/
    ├── models/
    ├── strategies/
    ├── compare_stage1_experiments.py
    ├── generate_stage1_report.py
    └── run_qlib_workflow.py
```

### 根目录文件

- [README.md](/Users/Dylan.Min/Documents/Code/learn/LHAI/README.md)
  - 项目导航、目录说明、当前阶段结论
- [requirements-qlib.txt](/Users/Dylan.Min/Documents/Code/learn/LHAI/requirements-qlib.txt)
  - 本地环境需要的依赖列表

### `data/`

- [data/qlib_cn_data](/Users/Dylan.Min/Documents/Code/learn/LHAI/data/qlib_cn_data)
  - Qlib 使用的中国市场数据目录
- [data/raw](/Users/Dylan.Min/Documents/Code/learn/LHAI/data/raw)
  - 预留给未来原始数据下载和存档
  - 当前已补充文本事件原始数据规范：[data/raw/README.md](/Users/Dylan.Min/Documents/Code/learn/LHAI/data/raw/README.md)
- [data/processed](/Users/Dylan.Min/Documents/Code/learn/LHAI/data/processed)
  - 预留给未来特征处理中间结果

### `mlruns/`

- [mlruns](/Users/Dylan.Min/Documents/Code/learn/LHAI/mlruns)
  - MLflow 本地实验记录目录
  - 每次 Qlib 运行产生的参数、指标、artifact 都会落到这里

### `qlib/`

- [qlib](/Users/Dylan.Min/Documents/Code/learn/LHAI/qlib)
  - 本地引用的 Qlib 源码
  - 主要作为量化研究框架底座使用

### `src/`

- [run_qlib_workflow.py](/Users/Dylan.Min/Documents/Code/learn/LHAI/src/run_qlib_workflow.py)
  - 主运行入口
  - 读取 YAML 配置，初始化 Qlib，训练模型并执行回测

- [generate_stage1_report.py](/Users/Dylan.Min/Documents/Code/learn/LHAI/src/generate_stage1_report.py)
  - 根据最近一次 MLflow 运行生成单次实验报告

- [compare_stage1_experiments.py](/Users/Dylan.Min/Documents/Code/learn/LHAI/src/compare_stage1_experiments.py)
  - 生成 `CSI300 vs CSI500` 的对照报告

- [src/configs](/Users/Dylan.Min/Documents/Code/learn/LHAI/src/configs)
  - 放置所有实验配置文件

- [src/data/fetch_akshare_notices.py](/Users/Dylan.Min/Documents/Code/learn/LHAI/src/data/fetch_akshare_notices.py)
  - 从 AKShare 批量抓取 A 股公告
  - 直接输出项目可用的 `text_events.csv` 原始文本事件表

- [src/features/build_text_event_features.py](/Users/Dylan.Min/Documents/Code/learn/LHAI/src/features/build_text_event_features.py)
  - 第二阶段第一版文本事件特征聚合脚本
  - 将原始公告/新闻 csv 聚合成按 `datetime + instrument` 对齐的数值特征

- [src/run_text_feature_pilot.py](/Users/Dylan.Min/Documents/Code/learn/LHAI/src/run_text_feature_pilot.py)
  - 第二阶段文本特征 pilot 入口
  - 用于快速比较“基线特征”和“基线特征 + 文本事件特征”的预测层效果

### `reports/`

- [stage1_baseline.md](/Users/Dylan.Min/Documents/Code/learn/LHAI/reports/stage1_baseline.md)
  - 第一阶段 baseline 的单次报告

- [stage1_csi300_vs_csi500.md](/Users/Dylan.Min/Documents/Code/learn/LHAI/reports/stage1_csi300_vs_csi500.md)
  - `CSI300` 与 `CSI500` 的并排对照结果

- [stage1_experiment_board.md](/Users/Dylan.Min/Documents/Code/learn/LHAI/reports/stage1_experiment_board.md)
  - 第一阶段实验台账
  - 记录做过哪些实验、每次改了什么、结论是什么

- [stage1_mid_summary.md](/Users/Dylan.Min/Documents/Code/learn/LHAI/reports/stage1_mid_summary.md)
  - 第一阶段的中间总结

- [stage1_final_output.md](/Users/Dylan.Min/Documents/Code/learn/LHAI/reports/stage1_final_output.md)
  - 第一阶段最终输出
  - 是后续继续学习和迭代时最应该先读的一份文件

- [stage2_plan.md](/Users/Dylan.Min/Documents/Code/learn/LHAI/reports/stage2_plan.md)
  - 第二阶段总计划

- [stage2_label_comparison.md](/Users/Dylan.Min/Documents/Code/learn/LHAI/reports/stage2_label_comparison.md)
  - `短标签 / 5日 / 10日` 的时间尺度对比

- [stage2_ai_feature_entry.md](/Users/Dylan.Min/Documents/Code/learn/LHAI/reports/stage2_ai_feature_entry.md)
  - 第二阶段第一类 AI 特征入口说明

- [stage2_text_feature_pilot.md](/Users/Dylan.Min/Documents/Code/learn/LHAI/reports/stage2_text_feature_pilot.md)
  - 第一版文本特征试跑结果
  - 当前用于确认文本特征流程是否打通、覆盖率是否足够

## 关键配置文件说明

- [workflow_config_lightgbm_a_share.yaml](/Users/Dylan.Min/Documents/Code/learn/LHAI/src/configs/workflow_config_lightgbm_a_share.yaml)
  - 第一版 `CSI300` baseline

- [workflow_config_lightgbm_csi500.yaml](/Users/Dylan.Min/Documents/Code/learn/LHAI/src/configs/workflow_config_lightgbm_csi500.yaml)
  - 将股票池切换为 `CSI500` 的配置

- [workflow_config_lightgbm_csi500_top20.yaml](/Users/Dylan.Min/Documents/Code/learn/LHAI/src/configs/workflow_config_lightgbm_csi500_top20.yaml)
  - `CSI500` 下测试更集中持仓

- [workflow_config_lightgbm_csi500_drop5.yaml](/Users/Dylan.Min/Documents/Code/learn/LHAI/src/configs/workflow_config_lightgbm_csi500_drop5.yaml)
  - `CSI500` 下测试更高换手

- [workflow_config_lightgbm_csi500_robustness_2021.yaml](/Users/Dylan.Min/Documents/Code/learn/LHAI/src/configs/workflow_config_lightgbm_csi500_robustness_2021.yaml)
  - 时间窗口稳健性实验配置

- [workflow_config_lightgbm_csi500_h5.yaml](/Users/Dylan.Min/Documents/Code/learn/LHAI/src/configs/workflow_config_lightgbm_csi500_h5.yaml)
  - 第二阶段 `5日标签` 配置

- [workflow_config_lightgbm_csi500_h10.yaml](/Users/Dylan.Min/Documents/Code/learn/LHAI/src/configs/workflow_config_lightgbm_csi500_h10.yaml)
  - 第二阶段 `10日标签` 配置

## 第一阶段做了什么

第一阶段按下面顺序推进：

1. 配好 `Python 3.12 + Qlib`
2. 下载并接入 A 股 `cn_data`
3. 跑通 `CSI300` baseline
4. 改股票池，验证 `CSI500`
5. 改 `topk`
6. 改 `n_drop`
7. 做更后时间窗口的稳健性测试

## 第一阶段最重要的实验结论

### 1. `CSI500` 比 `CSI300` 更适合作为当前主研究池

- `CSI300` 扣成本超额年化：`7.54%`
- `CSI500` 扣成本超额年化：`14.24%`

### 2. 更集中持仓没有带来更好结果

- `topk=30` 优于 `topk=20`

### 3. 更高换手没有带来更好结果

- `n_drop=3` 优于 `n_drop=5`

### 4. 更后时间窗口中策略仍然有效

- 稳健性窗口中扣成本超额年化仍为 `9.39%`
- 说明这套 baseline 不是只在单一窗口内有效

## 运行方式

激活环境：

```bash
source .venv/bin/activate
```

运行某个实验：

```bash
python src/run_qlib_workflow.py --config src/configs/workflow_config_lightgbm_csi500.yaml --experiment-name lhai_stage1_csi500
```

生成最近一次实验报告：

```bash
python src/generate_stage1_report.py
```

生成阶段 1 对照报告：

```bash
python src/compare_stage1_experiments.py
```

## 推荐阅读顺序

如果是后续重新回来看这个项目，推荐按这个顺序读：

1. [stage1_final_output.md](/Users/Dylan.Min/Documents/Code/learn/LHAI/reports/stage1_final_output.md)
2. [stage1_experiment_board.md](/Users/Dylan.Min/Documents/Code/learn/LHAI/reports/stage1_experiment_board.md)
3. [stage1_csi300_vs_csi500.md](/Users/Dylan.Min/Documents/Code/learn/LHAI/reports/stage1_csi300_vs_csi500.md)
4. [run_qlib_workflow.py](/Users/Dylan.Min/Documents/Code/learn/LHAI/src/run_qlib_workflow.py)
5. [src/configs](/Users/Dylan.Min/Documents/Code/learn/LHAI/src/configs)

## 第二阶段建议

第二阶段建议优先做：

1. 调整预测周期或标签定义
2. 比较不同特征模板
3. 再引入第一类 AI 特征，例如公告、新闻或财报文本特征

第二阶段当前主结论更新为：

1. `10日标签` 是当前更值得继续的主标签
2. 第一类 AI 特征从“文本事件特征入口”开始建设

第二阶段当前入口文档：

- [stage2_plan.md](/Users/Dylan.Min/Documents/Code/learn/LHAI/reports/stage2_plan.md)
- [data/raw/README.md](/Users/Dylan.Min/Documents/Code/learn/LHAI/data/raw/README.md)
- [stage2_ai_feature_entry.md](/Users/Dylan.Min/Documents/Code/learn/LHAI/reports/stage2_ai_feature_entry.md)
- [stage2_text_feature_pilot.md](/Users/Dylan.Min/Documents/Code/learn/LHAI/reports/stage2_text_feature_pilot.md)

## 第二阶段文本数据入口

如果要继续做“公告/新闻文本特征”，建议按这个顺序使用：

1. 先阅读 [data/raw/README.md](/Users/Dylan.Min/Documents/Code/learn/LHAI/data/raw/README.md)
2. 把真实原始文本数据放到 [data/raw](/Users/Dylan.Min/Documents/Code/learn/LHAI/data/raw)
3. 运行 [build_text_event_features.py](/Users/Dylan.Min/Documents/Code/learn/LHAI/src/features/build_text_event_features.py)
4. 再运行 [run_text_feature_pilot.py](/Users/Dylan.Min/Documents/Code/learn/LHAI/src/run_text_feature_pilot.py)

当前最重要的实践原则是：

- 先保证文本数据能覆盖 `CSI500` 和测试区间
- 再判断文本特征是否带来增量
- 最后再把文本特征并入正式 Qlib workflow

如果走 AKShare 公告路线，推荐先这样跑：

```bash
source .venv/bin/activate
python src/data/fetch_akshare_notices.py --start-date 2017-01-01 --end-date 2017-01-31 --notice-types 全部 --market csi500 --output data/raw/text_events.csv
python src/features/build_text_event_features.py --input data/raw/text_events.csv --output data/processed/text_event_features.csv
python src/run_text_feature_pilot.py --raw-events data/raw/text_events.csv --text-features data/processed/text_event_features.csv
```
