import pandas as pd

# 读取 CSV 文件
df = pd.read_csv('test.csv', header=None)

# 假设第一列为 label，删除 label == 2 的行
df_filtered = df[df[0] != 2]

# 保存到新文件
df_filtered.to_csv('test_cleaned.csv', index=False, header=False)

print("已删除 label=2 的行，结果保存在 test_filtered.csv")
