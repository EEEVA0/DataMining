import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.metrics import classification_report
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tqdm.keras import TqdmCallback
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error
from scipy.stats import pearsonr

# 设置随机种子保证可重复性
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# 1. 数据加载与预处理
def load_data(train_path, val_path):
    print("\nLoading data...")
    # 加载训练集和验证集
    train_df = pd.read_csv(train_path, names=['label', 'message'])
    val_df = pd.read_csv(val_path, names=['label', 'message'])

    # 打印原始标签分布
    print("Original label distribution (train):", train_df['label'].value_counts())
    print("Original label distribution (val):", val_df['label'].value_counts())

    # 确保标签是连续的0-3（BERT要求从0开始的连续标签）
    label_map = {0: 0, 1: 1, 2: 2}

    # 处理缺失值和无效标签
    train_df = train_df.dropna(subset=['label'])
    val_df = val_df.dropna(subset=['label'])

    # 只映射已知标签，其他转为NaN然后删除
    train_df['label'] = train_df['label'].map(label_map)
    val_df['label'] = val_df['label'].map(label_map)

    train_df = train_df.dropna(subset=['label'])
    val_df = val_df.dropna(subset=['label'])

    # 转换为整数类型
    train_df['label'] = train_df['label'].astype(int)
    val_df['label'] = val_df['label'].astype(int)

    # 检查是否有 NaN 标签
    assert train_df['label'].isna().sum() == 0, f"train_df contains {train_df['label'].isna().sum()} NaN labels!"
    assert val_df['label'].isna().sum() == 0, f"val_df contains {val_df['label'].isna().sum()} NaN labels!"

    print(f"Final train samples: {len(train_df)}, Validation samples: {len(val_df)}")
    return train_df, val_df


# 2. 文本编码器（使用Hugging Face分词器）
class TextEncoder:
    def __init__(self, model_path='./hf_cache/bert-base-uncased', max_length=128):
        self.tokenizer = BertTokenizer.from_pretrained('./hf_cache/bert-base-uncased')
        self.max_length = max_length

    def encode(self, texts):
        print("\nEncoding texts...")
        return self.tokenizer(
            texts.tolist(),
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='tf',
            verbose=True  # 显示分词进度
        )


# 3. 构建分类模型
def build_model(model_path='./hf_cache/bert-base-uncased', num_labels=3):
    model = TFBertForSequenceClassification.from_pretrained(
        model_path,
        from_pt=True,
        num_labels=num_labels,
        id2label={0: 'Anti', 1: 'Neutral', 2: 'Pro'},
        label2id={'Anti': 0, 'Neutral': 1, 'Pro': 2}
    )

    return model


# 4. 主流程
def create_dataset(encodings, labels, batch_size=32, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((dict(encodings), labels))
    if shuffle:
        dataset = dataset.shuffle(1000, seed=SEED)
    return dataset.batch(batch_size)

def main():
    # 加载数据
    train_df, val_df = load_data("train_clean.csv", "validation_clean.csv")

    # 初始化编码器
    encoder = TextEncoder(model_path='./hf_cache/bert-base-uncased', max_length=128)

    # 编码训练集
    print("\nEncoding training data...")
    train_encodings = encoder.encode(train_df['message'])
    y_train = train_df['label'].values

    # 编码验证集
    print("\nEncoding validation data...")
    val_encodings = encoder.encode(val_df['message'])
    y_val = val_df['label'].values

    # 构建模型
    print("\nBuilding model...")
    model = build_model(model_path='./hf_cache/bert-base-uncased', num_labels=3)

    # 构建 Dataset
    batch_size = 32
    train_dataset = create_dataset(train_encodings, y_train, batch_size=batch_size)
    val_dataset = create_dataset(val_encodings, y_val, batch_size=batch_size, shuffle=False)

    # 优化器、损失函数和指标
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    # 训练循环
    epochs = 2
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("Training...")
        for batch in tqdm(train_dataset, desc="Training"):
            x_batch, y_batch = batch
            with tf.GradientTape() as tape:
                logits = model(x_batch, training=True).logits
                loss_value = loss_fn(y_batch, logits)
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            train_acc_metric.update_state(y_batch, logits)
        train_acc = train_acc_metric.result().numpy()
        print(f"Training Accuracy: {train_acc:.4f}")
        train_acc_metric.reset_state()

        print("Validating...")
        for val_batch in val_dataset:
            x_val, y_val_batch = val_batch
            val_logits = model(x_val, training=False).logits
            val_acc_metric.update_state(y_val_batch, val_logits)
        val_acc = val_acc_metric.result().numpy()
        print(f"Validation Accuracy: {val_acc:.4f}")
        val_acc_metric.reset_state()

    # 最终评估
    print("\nFinal Evaluation:")
    all_logits = []
    all_labels = []

    for val_batch in val_dataset:
        x_val, y_val_batch = val_batch
        logits = model(x_val, training=False).logits
        all_logits.append(logits)
        all_labels.append(y_val_batch)

    predictions = tf.concat(all_logits, axis=0).numpy()
    y_pred = np.argmax(predictions, axis=1)
    y_true = tf.concat(all_labels, axis=0).numpy()

    print("\nClassification Report:")
    print(classification_report(
        y_true,
        y_pred,
        target_names=['Neutral(0)', 'Anti(1)', 'Pro(2)'],
        digits=4
    ))
    # ======= 额外指标 =======
    print("\n======= Classification Metrics =======")
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    mae = mean_absolute_error(y_true, y_pred)

    # 皮尔逊相关系数（需要确保非恒定标签）
    if np.std(y_true) > 0 and np.std(y_pred) > 0:
        correlation, _ = pearsonr(y_true, y_pred)
    else:
        correlation = float('nan')

    print(f"Accuracy:      {accuracy:.4f}")
    print(f"Precision:     {precision:.4f}")
    print(f"Recall:        {recall:.4f}")
    print(f"F1-score:      {f1:.4f}")
    print(f"MAE:           {mae:.4f}")
    print(f"Correlation:   {correlation:.4f}")
    return model



if __name__ == "__main__":
    # 安装必要依赖（首次运行时需要）
    # !pip install transformers tensorflow

    # 检查GPU可用性
    print("GPU Available:", tf.config.list_physical_devices('GPU'))

    # 运行主程序
    model = main()

    # 保存模型
    model.save_pretrained("saved_model")
    print("\nModel saved to 'saved_model' directory")