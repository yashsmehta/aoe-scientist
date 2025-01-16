import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Set the style and context for better visualization
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# Read the data
df = pd.read_csv('data/reviews.csv')

# Filter for deepseek generate and review LLM
mask = (df['generate_llm'] == 'deepseek') & (df['review_llm'] == 'anthropic') & (df['researcher'] != 'Mehta')
df_filtered = df.loc[mask].copy()  # Create a copy to avoid SettingWithCopyWarning

# Fill empty researcher values with "Unspecified"
df_filtered['researcher'] = df_filtered['researcher'].fillna('No Researcher')

# Create subplots
metrics = ['technical_merit', 'novelty', 'feasibility', 'impact', 'clarity', 'overall_score']
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Idea Review Scores by Researcher Agent', 
             fontsize=16, fontweight='bold', y=1.02)

# Define colors for each metric
colors = {
    'technical_merit': '#2ecc71',    # emerald green
    'novelty': '#3498db',            # bright blue
    'feasibility': '#e74c3c',        # coral red
    'impact': '#9b59b6',             # amethyst purple
    'clarity': '#f1c40f',            # sunflower yellow
    'overall_score': '#1abc9c'       # turquoise
}

# Flatten axes for easier iteration
axes_flat = axes.flatten()

# Plot each metric in a subplot
for idx, (ax, metric) in enumerate(zip(axes_flat, metrics)):
    # Get min and max values for this metric
    min_val = df_filtered[metric].min()
    max_val = df_filtered[metric].max()
    
    sns.barplot(
        data=df_filtered,
        x='researcher',
        y=metric,
        ax=ax,
        color=colors[metric],
        saturation=0.75,
        errorbar='ci',
        width=0.7,
        alpha=0.85
    )
    
    # Customize each subplot
    ax.set_title(metric.replace('_', ' ').title(), pad=10, fontsize=12, color=colors[metric])
    ax.set_xlabel('')
    ax.set_ylabel('Score', fontsize=10)
    
    # Rotate x-axis labels properly
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    
    # Set y-axis range dynamically
    y_min = np.floor(min_val - 1)
    y_max = np.ceil(max_val + 1)
    ax.set_ylim(y_min, y_max)
    
    # Set integer ticks only
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    
    # Add light grid only on y-axis
    ax.grid(True, axis='y', alpha=0.2)
    
    # Remove spines
    sns.despine(ax=ax, left=True, bottom=True)

# Adjust layout
plt.tight_layout()

# Save the plot in high quality
plt.savefig('plotters/imgs/review_scores.png', 
            dpi=300, 
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none')

plt.show()
