from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score,
    recall_score, f1_score, mean_absolute_error
)
from scipy.stats import pearsonr
import pandas as pd

# 读取数据
df = pd.read_csv('train_clean.csv', names=['label', 'message'])
val_df = pd.read_csv('validation_clean.csv',names=['label', 'message'])

# 文本向量化
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(df['message'])
X_val = vectorizer.transform(val_df['message'])

# 模型训练
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, df['label'])

# 模型评估
y_true = val_df['label']
y_pred = clf.predict(X_val)

# 打印详细分类报告
print(classification_report(y_true, y_pred, digits=4))

# 额外评价指标
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')
mae = mean_absolute_error(y_true, y_pred)
corr, _ = pearsonr(y_true, y_pred)

# 打印结果
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro Precision: {precision:.4f}")
print(f"Macro Recall: {recall:.4f}")
print(f"Macro F1-score: {f1:.4f}")
print(f"MAE: {mae:.4f}")
print(f"Pearson Correlation: {corr:.4f}")
