import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Set the style and context for better visualization
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# Read the data
df = pd.read_csv('data/reviews.csv')

# Filter out Mehta researcher
df_filtered = df[df['researcher'] != 'Mehta'].copy()

# Define metrics and their colors (same as plot.py)
metrics = {
    'technical_merit': '#2ecc71',    # emerald green
    'novelty': '#3498db',            # bright blue
    'feasibility': '#e74c3c',        # coral red
    'impact': '#9b59b6',             # amethyst purple
    'clarity': '#f1c40f',            # sunflower yellow
    'overall_score': '#1abc9c'       # turquoise
}

# Define LLM order
llm_order = ['openai', 'anthropic', 'deepseek']

# Create subplots with more vertical space
fig, axes = plt.subplots(2, 3, figsize=(20, 14))  # Made figure slightly taller
plt.subplots_adjust(hspace=0.1)  # Increase vertical spacing between subplots

fig.suptitle('Review Metrics Heatmap by LLM Combinations\n(Averaged Across All Researchers)', 
             fontsize=16, fontweight='bold', y=0.95)  # Adjusted title position

# Flatten axes for easier iteration
axes_flat = axes.flatten()

# Plot heatmap for each metric
for idx, (metric, color) in enumerate(metrics.items()):
    # Calculate mean scores for each LLM combination
    pivot_data = df_filtered.pivot_table(
        values=metric,
        index='generate_llm',
        columns='review_llm',
        aggfunc='mean'
    )
    
    # Reorder the index and columns
    pivot_data = pivot_data.reindex(index=llm_order, columns=llm_order)
    
    # Create custom colormap from white to the metric color
    cmap = sns.light_palette(color, as_cmap=True)
    
    # Plot heatmap
    hm = sns.heatmap(
        pivot_data,
        ax=axes_flat[idx],
        cmap=cmap,
        annot=True,
        fmt='.1f',
        annot_kws={'size': 12, 'weight': 'bold'},
        cbar_kws={'label': '', 'ticks': []},  # Remove label and ticks
        square=True,
        center=None  # Remove center to use local min/max
    )
    
    # Customize each subplot
    axes_flat[idx].set_title(metric.replace('_', ' ').title(), 
                            pad=10, 
                            fontsize=12, 
                            color=metrics[metric])
    axes_flat[idx].set_xlabel('Review LLM', fontsize=14, fontweight='bold')
    axes_flat[idx].set_ylabel('Generate LLM', fontsize=14, fontweight='bold')

# Save the plot
plt.savefig('plotters/imgs/review_heatmaps.png', 
            dpi=300, 
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none')

plt.show() 