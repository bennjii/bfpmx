from plot_utils import *

path_to_csv = "C:\\Users\\Nico\\Desktop\\project\\bfpmx\\benchmarks\\jacobi2d.csv"
df = load_profiler_csv(path_to_csv)

lineplot(
    group_runtime(df), # (=filter dataframe)
    x = 'steps', 
    y = 'runtime (ms)',
    hue = 'label',
    title = 'Runtime [ms] vs Steps',
    subtitle = "Intel Core i7-13700H, 2.4GHz, N=128, Block=fp8::m4e3"
    # you can extend this for customization (see label_display_names and labels_offset in plot_utils.py)
)

multi_distribution_plots(df, metric='clocks_inclusive')
multi_distribution_plots(df, metric='runtime (ms)')
multi_distribution_plots(df, metric='error (+/-)')