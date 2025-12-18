from plot_utils import *

path_to_csv = "C:\\Users\\Nico\\Desktop\\project\\bfpmx\\benchmarks\\jacobi2dALLnew.csv"
df = load_profiler_csv(path_to_csv)

'''
#error vs iteration_id (fp4..fp32)
lineplot(
    group_runtime_ALL(df), # (=filter dataframe)
    x = 'iteration_id', 
    y = 'error (%)',
    hue = 'format',
    title = 'Errors(%) vs Iterations',
    subtitle = "Intel Core i7-13700H, 2.4GHz, QuantizationPolicy::SharedExponent, Alg::SpreadBlockOnce, input_size = 32, steps = 250",
    # you can extend this for customization (see label_display_names and labels_offset in plot_utils.py)
    label_display_names = {
    'primitive': 'Baseline',
    'FP32 E8M23': 'FP16/FP32',
    'FP16 E6M9': '',
    'FP4 E2M1': 'FP4',
    'FP6 E3M2': 'FP6',
    'FP8 E4M3': 'FP8'
    },
    label_offsets = {
    'primitive':   {"x_offset": 0, "y_offset": 5},
    'FP32 E8M23':  {"x_offset": -1, "y_offset": -1},
    'FP16 E6M9':   {"x_offset": -1, "y_offset": -1},
    'FP8 E4M3':    {"x_offset": -1, "y_offset": 1.5},
    'FP6 E3M2':    {"x_offset": 0.5, "y_offset": 2.5},
    'FP4 E2M1':    {"x_offset": 0.5, "y_offset": -1.5},
}
)

multi_distribution_plots(group_runtime_ALL(df), metric='error (%)', hue='format')

# error vs steps fp4..fp32
lineplot(
    group_runtime_ALL(df), # (=filter dataframe)
    x = 'steps', 
    y = 'runtime (ms)',
    hue = 'format',
    title = 'Runtime vs Steps',
    subtitle = "Intel Core i7-13700H, 2.4GHz, QuantizationPolicy::SharedExponent, Alg::SpreadBlockOnce, Input_size = 32               ",
    label_display_names = {
    'primitive': 'primitive',
    'FP32 E8M23': 'FP32',
    'FP16 E6M9': 'FP16',
    'FP4 E2M1': 'FP4',
    'FP6 E3M2': 'FP6',
    'FP8 E4M3': 'FP8'
    },
    label_offsets = {
    'primitive':   {"x_offset": -3, "y_offset": 0.2},
    'FP32 E8M23':  {"x_offset": 7, "y_offset": -0.15},
    'FP16 E6M9':   {"x_offset": -6, "y_offset": -0.5},
    'FP8 E4M3':    {"x_offset": 7, "y_offset": 0.1},
    'FP6 E3M2':    {"x_offset": 0.5, "y_offset": -0.3},
    'FP4 E2M1':    {"x_offset": -0.5, "y_offset": 0.2},
}
)


# (2) Here we have a fixed format (FP8) and compare different optimization methods (SpreadBlockOnce/Each, NaiveBlock, 2DArray=primitive)
lineplot(
    group_runtime_ALL(df), # (=filter dataframe)
    x = 'input_size', 
    y = 'runtime (ms)',
    hue = 'label',
    title = 'Runtime vs Input size',
    subtitle = "Intel Core i7-13700H, 2.4GHz, format=FP8::E4M3, QuantizationPolicy::SharedExponent, steps = 250               ",
    label_display_names = {
    'Jacobi2DArray': 'Jacobi2DArray',
    'Jacobi2DSpreadBlockEach': 'Jacobi2DSpreadBlockEach',
    'Jacobi2DSpreadBlockOnce': 'Jacobi2DSpreadBlockOnce',
    'Jacobi2DNaiveBlock': 'Jacobi2DNaiveBlock'
    },
    label_offsets = {
    ' Jacobi2DArray':   {"x_offset": -2, "y_offset": 5},
    ' Jacobi2DSpreadBlockEach':  {"x_offset": -2, "y_offset": 5},
    ' Jacobi2DSpreadBlockOnce':   {"x_offset": -2, "y_offset": -7},
    ' Jacobi2DNaiveBlock':    {"x_offset": -3, "y_offset": 0.1}
}
)


multi_distribution_plots(group_runtime_ALL(df), metric='error (%)', hue='format')
'''


lineplot(
    group_runtime_ALL(df), # (=filter dataframe)
    x = 'input_size', 
    y = 'runtime (ms)',
    hue = 'format',
    title = 'Runtime vs Steps',
    subtitle = "Intel Core i7-13700H, 2.4GHz, QuantizationPolicy::SharedExponent, Alg::SpreadBlockOnce, steps = 10000, Iter=100                "
)
