# src/demo_nb.py

import os
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix


DATA_DIR = "data/raw"


def read_title_lines(path: str, encoding: str = "utf-8") -> List[str]:
    """读取“每行一个标题”的纯文本文件。"""
    titles = []
    with open(path, "r", encoding=encoding) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            titles.append(line)
    return titles


def load_train_from_txt(data_dir: str = DATA_DIR) -> Tuple[list, list]:
    """
    只从 positive_trainingSet / negative_trainingSet 读取训练数据，
    返回 X（标题列表）和 y（标签列表）。
    """
    pos_path = os.path.join(data_dir, "positive_trainingSet")
    neg_path = os.path.join(data_dir, "negative_trainingSet")

    if not os.path.exists(pos_path):
        raise FileNotFoundError(f"未找到正类文件: {pos_path}")
    if not os.path.exists(neg_path):
        raise FileNotFoundError(f"未找到负类文件: {neg_path}")

    X_pos = read_title_lines(pos_path, encoding="gb18030")  # 或 "gbk"
    X_neg = read_title_lines(neg_path, encoding="gb18030")
    
    X = X_pos + X_neg
    y = [1] * len(X_pos) + [0] * len(X_neg)  # 1 = 正类, 0 = 负类

    print(f"[INFO] 正类样本数: {len(X_pos)}, 负类样本数: {len(X_neg)}, 总数: {len(X)}")
    return X, y


def run_nb_demo():
    # 1) 读数据
    X, y = load_train_from_txt()

    # 2) 划分一个简单的 train/valid，用来做 demo
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"[INFO] 训练集: {len(X_train)} 条, 验证集: {len(X_valid)} 条")

    # 3) 文本 -> TF-IDF 特征
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),    # uni + bi-gram
        min_df=2,              # 出现太少的词丢掉一点
        max_features=20000
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_valid_vec = vectorizer.transform(X_valid)

    # 4) 训练 Naive Bayes
    clf = MultinomialNB()
    clf.fit(X_train_vec, y_train)

    # 5) 验证集预测
    y_pred = clf.predict(X_valid_vec)

    # 6) 打印评估指标
    print("\n=== Classification Report (valid set) ===")
    print(classification_report(y_valid, y_pred, digits=4))

    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_valid, y_pred))


if __name__ == "__main__":
    run_nb_demo()
