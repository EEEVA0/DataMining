import pandas as pd
from sklearn.model_selection import train_test_split

# 1. 加载数据并提取前20,000条
df = pd.read_csv('weka_compatible_twitter_data.csv').head(20000)

# 2. 首次拆分：60%训练集，40%临时集（将用于验证+测试）
train_df, temp_df = train_test_split(
    df,
    train_size=0.6,
    stratify=df['label'],  # 保持标签分布
    random_state=42
)

# 3. 二次拆分：将40%的temp_df均分为验证集和测试集（各20%）
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,  # 50% of temp_df = 20% of original
    stratify=temp_df['label'],
    random_state=42
)

# 4. 保存到CSV文件
train_df.to_csv('train.csv', index=False)
val_df.to_csv('validation.csv', index=False)
test_df.to_csv('test.csv', index=False)

# 5. 打印数据集信息
print(f"训练集: {len(train_df)}条 ({len(train_df)/20000:.0%})")
print(f"验证集: {len(val_df)}条 ({len(val_df)/20000:.0%})")
print(f"测试集: {len(test_df)}条 ({len(test_df)/20000:.0%})")
print("\n各类别分布:")
print("训练集:\n", train_df['label'].value_counts(normalize=True))
print("验证集:\n", val_df['label'].value_counts(normalize=True))
print("测试集:\n", test_df['label'].value_counts(normalize=True))