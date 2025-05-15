from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd

# 读取数据
df = pd.read_csv('train.csv')

# 标签映射
label_names = {0: 'Neutral', 1: 'Anti', 2: 'Pro'}

# 创建词云图
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for label, ax in zip([2, 0, 1], axes):  # 按 Pro, Neutral, Anti 顺序
    text = ' '.join(df[df['label'] == label]['message'].astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(f'{label_names[label]} Tweets', fontsize=16)
    ax.axis('off')

plt.tight_layout()
plt.show()
