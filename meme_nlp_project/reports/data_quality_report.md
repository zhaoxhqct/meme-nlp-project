# 数据质量与划分报告

## 原始文件编码

- `hot_meme_dataset.csv`：gb18030
- `expansion/hot_meme_dataset.csv`：utf-8-sig
- `normal_negative_samples.csv`：gb18030
- `expansion/normal_samples.csv`：utf-8-sig
- `meme_dictionary.csv`：gb18030
- `expansion/meme_dictionary.csv`：utf-8-sig
- `dictionary_overrides.csv`：utf-8-sig

## 清洗规则

- 情绪标签 `中立` 统一为 `中性`。
- 原句包含热梗原词时标记为 `literal`，否则标记为 `variant`。
- 正常样本的 `meme` 统一标记为 `无`。
- 同一热梗的所有样本只进入一个数据划分，避免分组泄漏。

## 全量统计

- 总样本数：702
- 热梗词表数量：366
- 来源分布：{'hot_meme': 522, 'normal': 180}
- 是否热梗：{'yes': 522, 'no': 180}
- 热梗形式：{'variant': 8, 'literal': 514, 'none': 180}
- 情绪分布：{'中性': 297, '正向': 245, '负向': 160}

## 数据划分

### train

- 样本数：492
- 独立热梗/文本组：337
- 热梗样本：366
- 普通样本：126
- 情绪分布：{'正向': 173, '中性': 198, '负向': 121}

### dev

- 样本数：105
- 独立热梗/文本组：104
- 热梗样本：78
- 普通样本：27
- 情绪分布：{'中性': 50, '负向': 22, '正向': 33}

### test

- 样本数：105
- 独立热梗/文本组：102
- 热梗样本：78
- 普通样本：27
- 情绪分布：{'正向': 39, '中性': 49, '负向': 17}
