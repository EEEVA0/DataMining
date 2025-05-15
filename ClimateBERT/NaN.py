import pandas as pd

# 标签映射：将原始标签转换为从 0 开始的标签
LABEL_MAP = {0: 0, 1: 1, 2: 2}

def safe_int(x):
    try:
        return int(float(x))
    except:
        return None

def clean_csv(input_path, output_path):
    print(f"\nCleaning: {input_path}")

    # 读取原始数据（不设置header）
    df = pd.read_csv(input_path, names=['label', 'message'], dtype=str)

    # 删除全空行
    df = df.dropna(how='all')

    # 去除前后空格
    df['label'] = df['label'].str.strip()
    df['message'] = df['message'].str.strip()

    # 丢弃 label 或 message 缺失的行
    df = df.dropna(subset=['label', 'message'])

    # 删除空字符串的行
    df = df[(df['label'] != '') & (df['message'] != '')]

    # 转换 label 为整数
    df['label'] = df['label'].map(safe_int)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)

    # 映射 label 到 0-3（不在映射表内的行将变为 NaN）
    df['label'] = df['label'].map(LABEL_MAP)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)

    print(f"✔ Cleaned {len(df)} valid rows. Saving to: {output_path}")
    df.to_csv(output_path, index=False, header=False)

    # 检查是否仍有问题
    assert df['label'].isna().sum() == 0, "Still contains NaN labels!"
    assert df['message'].isna().sum() == 0, "Still contains NaN messages!"


if __name__ == "__main__":
    # 清洗训练集和验证集
    clean_csv("test_updated.csv", "test_cleaned.csv")

    print("\n✅ Data cleaning completed successfully.")
