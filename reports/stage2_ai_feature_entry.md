# Stage 2 AI Feature Entry

## Goal

在 `CSI500 + H10` 的当前主设定上，开始引入第一类 AI 特征。

## Why This Is The First AI Feature

公告、新闻、事件文本是最自然也最容易体现 AI 价值的一层：

- 它们本身不是纯价格数据
- 传统量价因子难以完整表达它们
- 后续可以自然升级为情绪模型、embedding 或 LLM 标签

## Current Entry Design

先搭建一个可运行的数据入口，不直接上复杂模型。

当前新增内容：

- `data/raw/text_events_template.csv`
  - 原始文本事件数据模板
- `src/features/build_text_event_features.py`
  - 将原始文本事件聚合成按日、按股票的事件特征

## Current Output Features

当前脚本会输出：

- `event_count`
- `text_length_mean`
- `positive_keyword_hits`
- `negative_keyword_hits`
- `sentiment_balance`

这些还不是最终的 AI 特征，更像是第一版“文本事件特征骨架”。

## How To Use

1. 准备原始事件数据  
   参考：`data/raw/text_events_template.csv`

2. 运行特征聚合脚本

```bash
python src/features/build_text_event_features.py
```

3. 输出文件位置

```bash
data/processed/text_event_features.csv
```

## Next Upgrade Path

在这个入口之上，后面可以继续升级：

1. 关键词计数 -> 情绪分类模型
2. 简单统计特征 -> embedding 特征
3. 单条事件情绪 -> 多事件聚合因子
4. 手工词典 -> LLM 事件标签与摘要特征

## Current Decision

第二阶段主线现在分成两部分：

1. 主标签使用 `H10`
2. 开始建设文本事件特征入口
