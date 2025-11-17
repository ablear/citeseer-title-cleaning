# CiteSeer Title Cleaning & Classification

自动识别 CiteSeer 数据集中**错误抽取的论文标题**，对标题进行二分类（正确 / 错误），并比较三种不同模型方案：

- Naive Bayes（朴素贝叶斯）
- Word2Vec + SVM
- BERT-based Classifier（含特征提取 + Fine-tune 两种方式）

## 1. 项目简介

CiteSeer 等学术搜索引擎在从 PDF 中抽取论文标题时，经常出现错误抽取的情况，例如：

- 正确标题：`The Social Life of Routers`
- 错误抽取：`Call for Papers......41 Fragments......42`

本项目的目标是：

1. 基于给定的正负样本标题，训练分类模型识别错误抽取的标题；  
2. 分别实现：
   - Task A：Naive Bayes 文本分类器  
   - Task B：Word2Vec + SVM 分类器  
   - Task C：BERT-based 分类器  
3. 对三种方法的性能进行对比（Accuracy、Macro-F1、Micro-F1 等）；  
4. 使用 t-SNE 等方法对向量空间与分类结果进行可视化与分析。

## 2. 数据集说明

> ⚠️ 出于版权/大小原因，原始数据不直接包含在仓库中，请按下述方式手动下载。

数据包含三部分：

- **正类训练集（Positive Training Set）**：正确的论文标题  
- **负类训练集（Negative Training Set）**：错误抽取的标题  
- **测试集（Test Set, 1000 条带标签）**：用于最终评估模型性能  

下载后，请按如下目录结构放置：

```text
data/
  raw/
    positive_train.txt
    negative_train.txt
    test_set.txt
  processed/
    # 预处理后的中间文件（由代码自动生成）
  external/
    # 参考资料/论文（可选）
