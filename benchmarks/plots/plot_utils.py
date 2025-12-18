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

def group_runtime_ALL(
    df: pd.DataFrame,
    steps=10000,
    size=None,
    i = [' Jacobi2DArray', ' Jacobi2DSpreadBlockOnce'],
    group_cols=['format', 'steps', 'input_size','iteration_id','label'],
    value_col='runtime (ms)',
    agg='mean'
):
    df['iteration_id'] = df['iteration_id'] % 100
    df = df.copy()
    df.columns = df.columns.str.strip()

    if i is not None:
        df = df[df['label'].isin(i)]
    if steps is not None:
        df = df[df['steps'] == steps]
    if size is not None:
        df = df[df['input_size'].isin(size)]

    grouped = (
        df
        .groupby(group_cols, as_index=False)[value_col]
        .agg(agg)
    )
    print(grouped)
    return grouped




def load_profiler_csv(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df



def lineplot(df, x, y, hue, figsize=(10,6), title="", subtitle="",label_display_names = None, label_offsets = None):
    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=figsize)
    
    ax = sns.lineplot(data=df, x=x, y=y, hue=hue, marker="o", errorbar='sd'
)

    if label_display_names:
        ax.legend().remove()
    
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
        plt.title(subtitle, fontfamily='Calibri', fontsize = 12,x=0.44)

    plt.xlabel(x.capitalize(), fontfamily='Calibri', fontsize = 14)
    plt.ylabel(y.capitalize(), fontfamily='Calibri', fontsize = 14)

    plt.tight_layout()
    plt.show()
            


def multi_distribution_plots(df, metric, hue = 'label'):
    df['algorithm'] = df['label'].str.replace(r'(Jacobi2D|Heat3D)', '', regex=True)

    plt.figure(figsize=(14,10))

    # Box plot by algorithm<a
    ax1 = plt.subplot(2, 2, 3)
    sns.boxplot(data=df, y='algorithm', x = metric, hue=hue, ax=ax1)
    ax1.set_title('Box plot')
    ax1.set_ylabel('Algorithm')
    ax1.set_xlabel(metric.replace('_',' ').capitalize())
    if ax1.get_legend():
        ax1.get_legend().remove()

    
    ax1.set_yticklabels([
    'Naive\nBlock',
    'Spread\nBlock\nEach',
    'Spread\nBlock\nOnce'
    ])

    # Violin plot
    ax2 = plt.subplot(2, 2, 2)
    sns.violinplot(data=df, x='algorithm', y = metric, hue=hue,ax=ax2)
    ax2.set_title('Violin plot')
    ax2.set_xlabel('Algorithm')
    ax2.set_ylabel(metric.replace('_',' ').capitalize())
    if ax2.get_legend():
        ax2.get_legend().remove()


    ax2.set_xticklabels([
    'NaiveBlock',
    'SpreadBlockEach',
    'SpreadBlockOnce'
    ])
    # Histogram
    ax3 = plt.subplot(2, 2, 1)
    sns.histplot(data=df, x= metric, bins=50, hue=hue, ax=ax3)
    ax3.set_title('Histogram(bins=50)')
    ax3.set_xlabel(metric.replace('_',' ').capitalize())
    ax3.set_ylabel('Frequency')

    plt.tight_layout(pad=3.0)

    plt.show()


