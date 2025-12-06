import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("<<path-to-jacobi_bench.csv>>")
df.columns = df.columns.str.strip()

# Create cleaner names
df['algorithm'] = df['label'].str.replace('Jacobi2D', '')
plt.figure(figsize=(10, 6))

# Box plot by algorithm
ax2 = plt.subplot(2, 2, 1)
sns.boxplot(data=df, x='algorithm', y='clocks_inclusive', hue='label', ax=ax2)
ax2.set_title('Box plot')
ax2.set_xlabel('Algorithm')
ax2.set_ylabel('Clock Cycles')
ax2.tick_params(axis='x', rotation=45)
ax2.get_legend().remove()

# Violin plot
ax3 = plt.subplot(2, 2, 2)
sns.violinplot(data=df, x='algorithm', y='clocks_inclusive', hue='label',ax=ax3)
ax3.set_title('Violin plot')
ax3.set_xlabel('Algorithm')
ax3.set_ylabel('Clock Cycles')
ax3.tick_params(axis='x', rotation=45)
ax3.get_legend().remove()


# Histogram
ax1 = plt.subplot(2, 2, 3)
sns.histplot(data=df, x='clocks_inclusive', bins=50, hue='label', ax=ax1)
ax1.set_title('Histogram(bins=50)')
ax1.set_xlabel('Clock Cycles')
ax1.set_ylabel('Frequency')
ax1.get_legend().remove()

# Distribution for top 2 algorithms (individual)
top_algorithms = df.groupby('algorithm')['clocks_inclusive'].mean().nsmallest(2).index.tolist()
ax4 = plt.subplot(2, 2, 4)

color_map = {
    'Array': '#1f77b4',
    'SpreadBlockOnce': '#d62728'
}

for algo in top_algorithms:
    data = df[df['algorithm'] == algo]['clocks_inclusive']
    color = color_map.get(algo)
    sns.kdeplot(data=data, label=algo, ax=ax4, color=color)


ax4.set_title('Top 2 Algorithms (KDE)')
ax4.set_xlabel('Clock Cycles')
ax4.set_ylabel('Density')
ax4.legend()

plt.suptitle('Clock Cycle Distribution Analysis', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()