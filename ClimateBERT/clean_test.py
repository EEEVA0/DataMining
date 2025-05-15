# import pandas as pd
#
# # 读取 CSV（有表头）
# df = pd.read_csv('test.csv')
#
# # 显示原始标签分布（可选）
# print("原始标签分布：")
# print(df['label'].value_counts())
#
# # 删除 label 为 2 的行
# df_filtered = df[df['label'] != 2]
#
# # 再次确认标签分布
# print("\n过滤后标签分布：")
# print(df_filtered['label'].value_counts())
#
# # 保存新文件
# df_filtered.to_csv('test_filtered.csv', index=False)
#
# print("\n已删除 label=2 的行，保存至 test_filtered.csv")
# import pandas as pd
#
# # 读取原始数据（包含表头）
# df = pd.read_csv('test_filtered.csv')
#
# # 替换 label=-1 为 2
# df['label'] = df['label'].replace(-1, 2)
#
# # 保存到新文件
# df.to_csv('test_updated.csv', index=False)
#
# print("已将 label=-1 替换为 2，保存至 test_updated.csv")
import pandas as pd


def map_specific_labels(input_file, output_file):
    """
    标签映射函数：
    -1 → 1
    1 → 2
    其他标签保持不变
    """
    # 定义映射规则
    label_mapping = {-1: 1, 1: 2}

    # 读取数据（假设CSV格式包含label列）
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"错误：文件 {input_file} 未找到")
        return

    # 验证标签列是否存在
    if 'label' not in df.columns:
        print("错误：CSV文件中必须包含'label'列")
        return

    # 执行标签映射（仅修改指定标签，其他保持不变）
    df['label'] = df['label'].apply(lambda x: label_mapping.get(x, x))

    # 保存处理后的数据
    df.to_csv(output_file, index=False)
    print(f"标签映射完成，结果已保存到 {output_file}")

    # 打印新旧标签对比
    print("\n标签分布变化：")
    print("原分布:", df['label'].value_counts().sort_index())

    # 验证映射结果
    print("\n映射验证：")
    sample = df.sample(2)[['label']] if len(df) >= 2 else df[['label']]
    print(sample)


# 使用示例
if __name__ == "__main__":
    input_csv = "test_filtered.csv"  # 替换为您的输入文件路径
    output_csv = "test_updated.csv"  # 替换为您想要的输出路径

    print("开始执行标签映射...")
    map_specific_labels(input_csv, output_csv)
    print("处理完成！")