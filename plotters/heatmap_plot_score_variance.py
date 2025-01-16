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

# Define LLM order
llm_order = ['openai', 'anthropic', 'deepseek']

# Create figure
plt.figure(figsize=(10, 8))

# Calculate variance for overall score
pivot_data = df_filtered.pivot_table(
    values='overall_score',
    index='generate_llm',
    columns='review_llm',
    aggfunc='var'
)

# Reorder the index and columns
pivot_data = pivot_data.reindex(index=llm_order, columns=llm_order)

# Create custom colormap from white to a neutral color (using a teal color)
cmap = sns.light_palette("#2980b9", as_cmap=True)

# Plot heatmap
hm = sns.heatmap(
    pivot_data,
    cmap=cmap,
    annot=True,
    fmt='.2f',
    annot_kws={'size': 14, 'weight': 'bold'},
    cbar_kws={
        'label': 'Variance',
        'shrink': 0.5,
        'aspect': 5
    },
    square=True
)

# Make LLM names title case and larger
llm_names = [name.title() for name in llm_order]
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
ax = plt.gca()
ax.set_xticklabels(llm_names, fontsize=20, fontweight='bold')
ax.set_yticklabels(llm_names, fontsize=20, fontweight='bold')

# Get the colorbar and modify its label properties
cbar = ax.collections[0].colorbar
cbar.ax.set_ylabel('Variance', fontsize=30, fontweight='bold')

# Customize plot
plt.title('Overall Score Variance\nby LLM Combinations', 
          fontsize=24, 
          fontweight='bold', 
          pad=20)
plt.xlabel('Review LLM', fontsize=30, fontweight='bold')
plt.ylabel('Generate LLM', fontsize=30, fontweight='bold')

# Save the plot
plt.savefig('plotters/imgs/overall_score_variance_heatmap.png', 
            dpi=300, 
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none')

plt.show() 