import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

CPU_FREQUENCY = 2.4 


df = pd.read_csv("<<path-to-jacobi_bench.csv>>")
f = CPU_FREQUENCY*1e9

# -- Group data and calculate runtime(s) for each label --
df.columns = df.columns.str.strip()
grouped = df.groupby(['format','stress_function', 'steps', 'label'])['clocks_inclusive'].mean().reset_index()
grouped['runtime(s)'] = grouped['clocks_inclusive'] / f 
grouped.to_csv("jacobi_runtimes.csv", index=False)

# -- PLOT settings --
# Remove border, set everything white&cool
sns.set_theme(style="white")
plt.figure(figsize=(10, 6))
ax = sns.lineplot(data=grouped, x="steps", y="runtime(s)", hue="label", marker="o")
for spine in ax.spines.values():
    spine.set_color("white")
    
# Customize legend

###############################
## UNCOMMENT ME to costumize ##
###############################
plt.legend().remove() 
label_offsets = {
    "Jacobi2DArray": {"x_offset": 0, "y_offset": 0.08},
    "Jacobi2DAlwaysFastMarshal": {"x_offset": -20, "y_offset": 0.1},
    "Jacobi2DSpreadBlockEach": {"x_offset": -20, "y_offset": -0.35}, 
    "Jacobi2DNaiveBlock":{"x_offset": -20, "y_offset": 0},  
    "Jacobi2DSpreadBlockOnce":{"x_offset": -20, "y_offset": -0.15},
}

label_display_names = {
    "Jacobi2DArray":"2DArray",
    "Jacobi2DAlwaysFastMarshal":"AlwaysFastMarshal",
    "Jacobi2DSpreadBlockEach":"SpreadBlockEach",
    "Jacobi2DNaiveBlock":"NaiveBlock",
    "Jacobi2DSpreadBlockOnce":"SpreadBlockOnce",
}

for line, name in zip(ax.lines, grouped['label'].unique()):
    x = line.get_xdata()[-1]  
    y = line.get_ydata()[-1]
    offsets = label_offsets.get(name.strip(), {"x_offset": 0, "y_offset": 0})
    plt.text(
        x + offsets["x_offset"],
        y + offsets["y_offset"],
        label_display_names[name.strip()],  
        color=line.get_color(),
        fontsize=12,
        fontfamily='Calibri',
        fontweight='bold',
        style='italic',
        va='center',
        ha='center'
    )


# Set titles
plt.suptitle("Runtime [s] vs. steps", fontfamily='Calibri', fontsize=16, fontweight='bold', x=0.16, y=0.95)
plt.title("Intel Core i7-13700H, 2.4GHz, N=128, Block=fp8::m4e3", fontfamily='Calibri', fontsize=12, x=0.20)



# Colors, Font
plt.gca().patch.set_facecolor("#f0f0f0")  
plt.grid(axis='y', linewidth=2, color="white")
plt.xlabel("Steps",fontfamily='Calibri', fontsize=14)
plt.ylabel("Runtime (s)",fontfamily='Calibri', fontsize=14)  
plt.tight_layout()

# Plot
plt.show()
