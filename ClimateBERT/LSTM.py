import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, Dataset
from sklearn.feature_extraction.text import CountVectorizer

# 配置参数
MAX_LEN = 100
BATCH_SIZE = 64
EPOCHS = 5
LR = 1e-3
HIDDEN_SIZE = 128
EMBEDDING_DIM = 100

# 自定义数据集类
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return torch.tensor(self.texts[idx], dtype=torch.long), self.labels[idx]

# 模型定义
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hn, _) = self.lstm(embedded)
        return self.fc(hn[-1])

# 加载数据
df = pd.read_csv('train_clean.csv', names=['label', 'message'])
texts = df['message'].astype(str).tolist()
df = df[df['label'].isin([0, 1, 2])]
labels = df['label'].tolist()

# 文本编码
vectorizer = CountVectorizer(max_features=5000, token_pattern=r'\b\w+\b')
vectorized_texts = vectorizer.fit_transform(texts).toarray()

# 填充或裁剪
def pad_sequences(arrs, max_len):
    padded = []
    for arr in arrs:
        if len(arr) > max_len:
            padded.append(arr[:max_len])
        else:
            padded.append(list(arr) + [0] * (max_len - len(arr)))
    return padded

# 生成词索引矩阵
word_to_index = {word: idx+1 for idx, word in enumerate(vectorizer.get_feature_names_out())}
encoded_texts = [
    [word_to_index.get(word, 0) for word in text.split()]
    for text in texts
]
padded_texts = pad_sequences(encoded_texts, MAX_LEN)

# 划分数据
X_train, X_test, y_train, y_test = train_test_split(padded_texts, labels, test_size=0.2, random_state=42)

train_dataset = TextDataset(X_train, y_train)
test_dataset = TextDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# 初始化模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMClassifier(vocab_size=5001, embedding_dim=EMBEDDING_DIM, hidden_size=HIDDEN_SIZE, num_classes=3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# 训练过程
for epoch in range(EPOCHS):
    model.train()
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{EPOCHS} 完成")

# 测试评估
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        outputs = model(batch_x)
        preds = torch.argmax(outputs, dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(batch_y.tolist())

# 评估指标
acc = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds, average='macro')
rec = recall_score(all_labels, all_preds, average='macro')
f1 = f1_score(all_labels, all_preds, average='macro')
mae = mean_absolute_error(all_labels, all_preds)
corr, _ = pearsonr(all_labels, all_preds)

print("\n======= 分类评估结果 =======")
print(f"Accuracy:      {acc:.4f}")
print(f"Precision:     {prec:.4f}")
print(f"Recall:        {rec:.4f}")
print(f"F1-score:      {f1:.4f}")
print(f"MAE:           {mae:.4f}")
print(f"Correlation:   {corr:.4f}")
