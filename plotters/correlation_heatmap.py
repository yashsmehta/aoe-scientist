import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Set style for publication-quality plot
sns.set_style("white")
sns.set_context("paper", font_scale=1.1)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'figure.dpi': 300,
    'savefig.dpi': 300
})

# Read the data
df = pd.read_csv('data/reviews.csv')

# Filter out Mehta researcher
df_filtered = df[df['researcher'] != 'Mehta'].copy()

# Define metrics
metrics = [
    'technical_merit',
    'novelty',
    'feasibility',
    'impact',
    'clarity',
    'overall_score'
]

# Create figure with Nature's single-column width (89mm)
width_mm = 89
height_mm = 89
width_inches = width_mm / 25.4
height_inches = height_mm / 25.4

fig, ax = plt.subplots(figsize=(width_inches, height_inches))

# Initialize correlation matrix accumulator
correlation_sum = None
num_metrics = len(metrics)

# For each metric
for metric in metrics:
    # Pivot the data to get scores by each review LLM
    pivot_data = df_filtered.pivot_table(
        values=metric,
        index=['researcher', 'generate_llm'],
        columns='review_llm',
        aggfunc='first'
    )
    
    # Get correlation between review LLMs
    corr = pivot_data.corr()
    
    # Add to accumulator
    if correlation_sum is None:
        correlation_sum = corr
    else:
        correlation_sum += corr

# Calculate average correlation
avg_correlation = correlation_sum / num_metrics

# Generate mask for upper triangle
mask = np.triu(np.ones_like(avg_correlation), k=1)

# Create custom diverging colormap (Nature-friendly colors)
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Plot heatmap
hm = sns.heatmap(
    avg_correlation,
    mask=mask,
    annot=True,
    fmt='.2f',
    cmap=cmap,
    vmin=-1,
    vmax=1,
    center=0,
    square=True,
    linewidths=0.5,
    annot_kws={'size': 10, 'weight': 'medium'},
    cbar_kws={
        'label': 'Average Correlation',
        'shrink': 0.5,  # Make colorbar shorter
        'aspect': 5,    # Make colorbar thicker
        'pad': 0.02    # Adjust spacing
    }
)

# Customize labels
plt.title('Inter-Rater Reliability of LLM Reviews', 
          fontsize=11, 
          fontweight='bold',
          pad=10)

# Make LLM names title case
llm_names = [name.title() for name in avg_correlation.index]
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
ax.set_xticklabels(llm_names, fontweight='medium')
ax.set_yticklabels(llm_names, fontweight='medium')

# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig('plotters/imgs/avg_review_llm_correlation.png', 
            dpi=300,
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none')

plt.show() 