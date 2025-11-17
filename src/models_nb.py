# src/models_nb.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

from src.data_loader import load_train_test_data


def train_and_evaluate_nb():
    # 1) 加载训练 + 测试数据
    X_train, y_train, X_test, y_test = load_train_test_data()

    print(f"[INFO] X_train: {len(X_train)}, X_test: {len(X_test)}")

    # 2) 文本 -> TF-IDF 特征
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),       # uni-gram + bi-gram
        min_df=2,                 # 过滤太稀有的词
        max_features=50000        # 给个上限，防止特征太多
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 3) 训练 Naive Bayes
    clf = MultinomialNB()
    clf.fit(X_train_vec, y_train)

    # 4) 在测试集上预测
    y_pred = clf.predict(X_test_vec)

    # 5) 打印评估指标
    print("\n=== [Naive Bayes] Classification Report (Test Set) ===")
    print(classification_report(y_test, y_pred, digits=4))

    print("=== [Naive Bayes] Confusion Matrix (Test Set) ===")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    train_and_evaluate_nb()
