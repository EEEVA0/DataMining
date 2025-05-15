import pandas as pd
import re
import csv


def clean_tweet(tweet):
    """
    清洗单条推文文本的函数，只保留英文和数字
    """
    if not isinstance(tweet, str):
        return ""

    # 只保留英文、数字和基本标点符号
    tweet = re.sub(r'[^a-zA-Z0-9\s\.,!?]', '', tweet)

    # 移除转推标记 (RT @username:)
    tweet = re.sub(r'RT\s*@\w+:\s*', '', tweet)

    # 移除URL链接
    tweet = re.sub(r'https?://\S+', '', tweet)

    # 移除用户提及 (@username)
    tweet = re.sub(r'@\w+', '', tweet)

    # 移除特殊字符序列如"k4h. -N"及其变体
    tweet = re.sub(r'\S*k4h\.\s*-N\S*', '', tweet)

    # 修正常见拼写错误
    tweet = tweet.replace('global warning', 'global warming')
    tweet = tweet.replace('helimate change', 'climate change')

    # 移除HTML实体
    tweet = tweet.replace('&amp;', '&')

    # 确保只保留英文和数字（更严格的过滤）
    tweet = re.sub(r'[^a-zA-Z0-9\s]', ' ', tweet)

    # 移除多余的空白字符
    tweet = ' '.join(tweet.split())

    return tweet.strip()


# 从本地CSV文件加载数据
try:
    # 读取CSV文件，指定正确的列名
    df = pd.read_csv('twitter_sentiment_data.csv')

    # 检查必要的列是否存在
    required_columns = ['sentiment', 'message', 'tweetid']
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        raise ValueError(f"CSV文件缺少必要的列: {missing}")

    # 删除不需要的tweetid列
    df = df.drop(columns=['tweetid'])

    # 应用清洗函数到message列
    df['cleaned_message'] = df['message'].apply(clean_tweet)

    # 删除原始message列，只保留清洗后的版本
    df = df.drop(columns=['message'])

    # 重命名列以保持一致性
    df = df.rename(columns={
        'sentiment': 'label',
        'cleaned_message': 'message'
    })

    # 重新排列列顺序
    df = df[['label', 'message']]

    # 去除重复的message（保留第一条出现的记录）
    df = df.drop_duplicates(subset=['message'], keep='first')

    # 保存为Weka兼容的CSV文件
    with open('weka_compatible_twitter_data.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 写入标题行
        writer.writerow(['label', 'message'])
        # 写入数据行
        for _, row in df.iterrows():
            # 确保最终内容只包含英文和数字
            final_message = re.sub(r'[^a-zA-Z0-9\s]', '', row['message'])
            writer.writerow([row['label'], final_message])

    print(f"数据清洗完成，共保留 {len(df)} 条唯一消息")
    print("已保存为 weka_compatible_twitter_data.csv")
    print("\n清洗后的数据前5行示例:")
    print(df.head())

except FileNotFoundError:
    print("错误：找不到 twitter_sentiment_data.csv 文件")
    print("请确保文件位于当前工作目录，或提供完整路径")
except Exception as e:
    print(f"发生错误: {str(e)}")