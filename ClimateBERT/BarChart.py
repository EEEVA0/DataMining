import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("train_clean.csv", names=['label', 'text'])

# Clean and verify labels (ensure only 0,1,2 exist)
valid_labels = {0, 1, 2}
df = df[df['label'].isin(valid_labels)]
df['label'] = df['label'].astype(int)

# Create count plot
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")
ax = sns.countplot(x='label', data=df, palette=["#4C72B0", "#DD8452", "#55A868"])

# Customize plot
plt.title("Distribution of Climate Change Stance Labels", fontsize=14, pad=20)
plt.xlabel("Stance Label", fontsize=12)
plt.ylabel("Number of Tweets", fontsize=12)
ax.set_xticklabels(["Neutral (0)", "Anti (1)", "Pro (2)"])

# Add exact counts on bars
for p in ax.patches:
    ax.annotate(f'{p.get_height():,.0f}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center',
                xytext=(0, 5),
                textcoords='offset points')

# Save and display
plt.tight_layout()
plt.savefig("label_distribution.png", dpi=300)
plt.show()