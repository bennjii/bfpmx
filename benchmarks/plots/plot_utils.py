import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Take the mean of runtime and group by chosen columns (='group_cols') 
def group_runtime(df: pd.DataFrame, 
                  group_cols = ['format', 'stress_function', 'steps', 'error (+/-)', 'label'],
                  value_col='runtime (ms)',agg='mean'):
    df.columns = df.columns.str.strip()
    grouped = df.groupby(group_cols)[value_col].agg(agg).reset_index()
    return grouped


def load_profiler_csv(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df



def lineplot(df, x, y, hue, figsize=(10,6), title="", subtitle="",label_display_names = None, label_offsets = None):
    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=figsize)
    
    ax = sns.lineplot(data=df, x=x, y=y, hue=hue, marker="o")

    if label_display_names:
        ax.legend().remove
    
    for spine in ax.spines.values():
        spine.set_color("white")
    ax.patch.set_facecolor("#f0f0f0")
    ax.grid(axis='y', linewidth = 2, color = "white")


    if label_display_names:
        for line, name in zip(ax.lines, df[hue].unique()):
            xf = line.get_xdata()[-1]
            yf = line.get_ydata()[-1]

            offsets = label_offsets.get(name, {"x_offset": 0, "y_offset": 0})

            ax.text(
                xf + offsets["x_offset"],
                yf + offsets["y_offset"],
                label_display_names[name.strip()],  
                color=line.get_color(),
                fontsize=12,
                fontfamily='Calibri',
                fontweight='bold',
                style='italic',
                va='center',
                ha='center'
            )

    if title:
        plt.suptitle(title, fontfamily='Calibri', fontsize = 16, fontweight = 'bold', x=0.165, y=0.95)
    if subtitle:
        plt.title(subtitle, fontfamily='Calibri', fontsize = 12,x=0.2)

    plt.xlabel(x.capitalize(), fontfamily='Calibri', fontsize = 14)
    plt.ylabel(y.capitalize(), fontfamily='Calibri', fontsize = 14)

    plt.tight_layout()
    plt.show()
            


def multi_distribution_plots(df, metric, hue = 'label'):
    df['algorithm'] = df['label'].str.replace(r'(Jacobi2D|Heat3D)', '', regex=True)
    plt.figure(figsize=(14,10))

    # Box plot by algorithm
    ax1 = plt.subplot(2, 2, 1)
    sns.boxplot(data=df, x='algorithm', y = metric, hue='label', ax=ax1)
    ax1.set_title('Box plot')
    ax1.set_xlabel('Algorithm')
    ax1.set_ylabel(metric.replace('_',' ').capitalize())
    if ax1.get_legend():
        ax1.get_legend().remove()

    # Violin plot
    ax2 = plt.subplot(2, 2, 2)
    sns.violinplot(data=df, x='algorithm', y = metric, hue='label',ax=ax2)
    ax2.set_title('Violin plot')
    ax2.set_xlabel('Algorithm')
    ax2.set_ylabel(metric.replace('_',' ').capitalize())
    if ax2.get_legend():
        ax2.get_legend().remove()


    # Histogram
    ax3 = plt.subplot(2, 2, 3)
    sns.histplot(data=df, x= metric, bins=50, hue='label', ax=ax3)
    ax3.set_title('Histogram(bins=50)')
    ax3.set_xlabel(metric.replace('_',' ').capitalize())
    ax3.set_ylabel('Frequency')

    # Distribution for top 2 algorithms (individual)
    top_algorithms = df.groupby('algorithm')[metric].mean().nsmallest(2).index.tolist()
    ax4 = plt.subplot(2, 2, 4)

    color_map = {
        'Array': '#1f77b4',
        'SpreadBlockOnce': '#d62728'
    }

    for algo in top_algorithms:
        data = df[df['algorithm'] == algo][metric]
        color = color_map.get(algo)
        sns.kdeplot(data=data, label=algo, ax=ax4, color=color)


    ax4.set_title('Top Algorithms (KDE)')
    ax4.set_xlabel(metric.replace('_',' ').capitalize())
    ax4.set_ylabel('Density')
    ax4.get_legend()


    plt.tight_layout(pad=3.0)

    plt.show()


