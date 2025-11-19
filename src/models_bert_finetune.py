import os
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

from src.data_loader import load_train_test_data


# ======================
# 1. Dataset & 编码工具
# ======================

class TitleDataset(Dataset):
    """
    用预先编码好的 encodings + labels 构造 HuggingFace Trainer 可用的 Dataset。
    encodings: dict(input_ids / attention_mask / ...)
    labels: List[int]
    """
    def __init__(self, encodings: Dict[str, torch.Tensor], labels: List[int]):
        assert len(labels) == encodings["input_ids"].shape[0]
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def encode_texts(tokenizer, texts: List[str], max_length: int = 64) -> Dict[str, torch.Tensor]:
    """
    一次性将一批文本编码成张量，避免在 Dataset.__getitem__ 中重复 tokenizer 调用。
    动态 padding 交给 DataCollatorWithPadding 处理，因此这里 padding=False。
    """
    texts = [str(t) for t in texts]
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=False,          # 动态 padding 由 collator 负责
        max_length=max_length,
        return_tensors="pt",    # (N, L) 的张量
    )
    return encodings


# ======================
# 2. metrics 计算
# ======================

def compute_metrics(eval_pred):
    """
    给 Trainer 用的 metrics 回调。
    返回 accuracy / precision / recall / f1（binary, 以 1 为“正类”）。
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ======================
# 3. 主流程：加载数据、划分、微调 BERT、在 testSet 上评估
# ======================

def train_and_evaluate_bert(
    model_name: str = "bert-base-uncased",
    max_length: int = 64,
    num_epochs: int = 1,
    train_sample_cap: Optional[int] = None,
):
    """
    微调 BertForSequenceClassification 做二分类：
    - 训练集：来自 positive/negative txt
    - 验证集：从训练里切一部分
    - 测试集：来自 testSet.xlsx（官方数据）

    train_sample_cap:
        若为 None，则使用全部训练数据；
        若为正整数，则随机打乱后只取前 train_sample_cap 条作为训练（方便快速实验）。
    """

    # ----------------------
    # 3.1 加载数据
    # ----------------------
    X_train_all, y_train_all, X_test, y_test = load_train_test_data()
    print(f"[BERT] Total train samples: {len(X_train_all)}, test samples: {len(X_test)}")

    # 可选：裁剪训练样本数量（无论 CPU/GPU 都可用，方便快速调参）
    if train_sample_cap is not None and len(X_train_all) > train_sample_cap:
        from sklearn.utils import shuffle
        X_train_all, y_train_all = shuffle(X_train_all, y_train_all, random_state=42)
        X_train_all = X_train_all[:train_sample_cap]
        y_train_all = y_train_all[:train_sample_cap]
        print(f"[BERT] Using subset of train data: {len(X_train_all)} samples")

    # train / val 划分（stratify 保持类别比例）
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_all,
        y_train_all,
        test_size=0.1,
        random_state=42,
        stratify=y_train_all,
    )

    print(f"[BERT] Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # ----------------------
    # 3.2 tokenizer & 模型
    # ----------------------
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("[BERT] Using device:", device)
    print("[DEBUG] Model first param device:", next(model.parameters()).device)

    # ----------------------
    # 3.3 预先编码 & 构造 Dataset
    # ----------------------
    train_encodings = encode_texts(tokenizer, X_train, max_length=max_length)
    val_encodings = encode_texts(tokenizer, X_val, max_length=max_length)
    test_encodings = encode_texts(tokenizer, X_test, max_length=max_length)

    train_dataset = TitleDataset(train_encodings, y_train)
    val_dataset = TitleDataset(val_encodings, y_val)
    test_dataset = TitleDataset(test_encodings, y_test)

    # 动态 padding collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ----------------------
    # 3.4 TrainingArguments
    # ----------------------
    output_dir = os.path.join("experiments", "bert_finetune")
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        eval_strategy="epoch",          # 你当前 transformers 版本支持这个写法
        save_strategy="epoch",
        logging_steps=100,
        learning_rate=2e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to=[],                   # 不用 wandb / tensorboard
        fp16=True if device.type == "cuda" else False,  # GPU 上开启混合精度
        warmup_ratio=0.1,               # 前 10% step warmup
        lr_scheduler_type="linear",     # 线性学习率衰减
    )

    # ----------------------
    # 3.5 Trainer
    # ----------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # ----------------------
    # 3.6 训练
    # ----------------------
    print("[BERT] Start training...")
    trainer.train()

    # ----------------------
    # 3.7 在测试集上评估
    # ----------------------
    print("[BERT] Evaluating on test set...")
    preds_output = trainer.predict(test_dataset)
    logits = preds_output.predictions
    y_pred = np.argmax(logits, axis=-1)

    print("\n=== [BERT Fine-tune] Classification Report (Test Set) ===")
    print(classification_report(y_test, y_pred, digits=4))

    print("=== [BERT Fine-tune] Confusion Matrix (Test Set) ===")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    # 默认 1 epoch，跑全量，如果想快速调试，可以给 train_sample_cap 传一个值，比如 40000
    train_and_evaluate_bert(
        model_name="bert-base-uncased",
        max_length=64,
        num_epochs=1,
        train_sample_cap=None,   # 调参/快跑：改成 40000 之类
    )
