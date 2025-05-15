import pandas as pd
import re


def clean_text(text):
    """彻底清理文本：删除所有单引号，转义双引号"""
    if pd.isna(text):
        return ""
    text = str(text).strip()
    # 删除所有单引号
    text = text.replace("'", "")
    # 转义双引号（WEKA要求字符串内的双引号用\"表示）
    text = text.replace('"', '\\"')
    # 移除控制字符
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
    return text


def csv_to_weka_arff(csv_path, arff_path):
    """生成严格兼容WEKA的ARFF文件（无单引号版本）"""
    df = pd.read_csv(csv_path, header=None, names=['label', 'text'])

    with open(arff_path, 'w', encoding='utf-8') as f:
        # 头部声明
        f.write("@RELATION text_classification\n\n")

        # 属性声明
        f.write("@ATTRIBUTE label {0,1,2,9}\n")
        f.write("@ATTRIBUTE text string\n")  # WEKA的string类型需用双引号包裹

        # 数据部分
        f.write("\n@DATA\n")
        for _, row in df.iterrows():
            label = int(float(row['label'])) if not pd.isna(row['label']) else '?'
            text = clean_text(row['text'])
            # 关键点：双引号包裹文本，且已删除所有单引号
            f.write(f'{label},"{text}"\n')  # 格式: label,"text"

    print(f"转换完成！WEKA兼容文件已保存到: {arff_path}")


# 使用示例
csv_to_weka_arff("validation_clean.csv", "validation3.arff")