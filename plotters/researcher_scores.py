import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Set style for publication-quality plot
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10
})

# Read the data
df = pd.read_csv('data/reviews.csv')

# Filter out Mehta researcher
df = df[df['researcher'] != 'Mehta']

# Define metrics with better labels
metrics = {
    'technical_merit': 'Technical Merit',
    'novelty': 'Novelty',
    'feasibility': 'Feasibility',
    'impact': 'Impact',
    'clarity': 'Clarity',
    'overall_score': 'Overall Score'
}

# Create a long-form dataframe for plotting
plot_data = df.melt(
    id_vars=['researcher', 'generate_llm', 'review_llm'],
    value_vars=list(metrics.keys()),
    var_name='metric',
    value_name='score'
)

# Calculate mean scores per researcher, LLM, and metric
plot_data = plot_data.groupby(['researcher', 'review_llm', 'metric'])['score'].mean().reset_index()

# Map metric codes to display names
plot_data['metric'] = plot_data['metric'].map(metrics)

# Create figure with extra space for legend
fig = plt.figure(figsize=(20, 12))

# Define markers and colors for each LLM
style_dict = {
    'openai': {'marker': 'o', 'color': '#2ecc71', 'label': 'OpenAI'},  # Bright green
    'anthropic': {'marker': '^', 'color': '#9b59b6', 'label': 'Anthropic'},  # Purple
    'deepseek': {'marker': 'D', 'color': '#3498db', 'label': 'DeepSeek'}  # Blue
}

# Get unique researchers and metrics
researchers = sorted(plot_data['researcher'].unique())
unique_metrics = list(metrics.values())

# Create subplot grid
ncols = 3
nrows = (len(unique_metrics) + ncols - 1) // ncols
gs = plt.GridSpec(nrows, ncols, figure=fig)

# Plot each metric in its own subplot
for i, metric in enumerate(unique_metrics):
    row = i // ncols
    col = i % ncols
    ax = fig.add_subplot(gs[row, col])
    
    for llm, style in style_dict.items():
        # Filter data for current metric and LLM
        metric_data = plot_data[
            (plot_data['metric'] == metric) & 
            (plot_data['review_llm'] == llm)
        ]
        
        # Plot points
        ax.scatter(
            x=metric_data['researcher'].map(lambda x: researchers.index(x)),
            y=metric_data['score'],
            marker=style['marker'],
            c=style['color'],
            s=150,
            alpha=0.8,
            label=style['label']
        )
    
    # Customize x-axis
    ax.set_xticks(range(len(researchers)))
    ax.set_xticklabels(researchers, rotation=45, ha='right')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Set title
    ax.set_title(metric, pad=20, fontsize=14, fontweight='bold')
    
    # Let y-axis adjust automatically to data range
    ax.margins(y=0.1)

# Add a single legend to the right of all subplots
handles = [plt.Line2D([0], [0], marker=style['marker'], color=style['color'],
                     label=style['label'], markersize=10, linestyle='None')
          for llm, style in style_dict.items()]

# Place legend to the right of the subplots
legend = fig.legend(handles=handles, 
                   title='Review LLM',
                   bbox_to_anchor=(0.98, 0.5),
                   loc='center left',
                   title_fontsize=12,
                   fontsize=10,
                   frameon=True,
                   borderaxespad=1)

# Adjust layout while preserving space for legend
plt.tight_layout()
# Adjust subplot spacing to make room for legend
plt.subplots_adjust(right=0.92)

# Save the plot
plt.savefig('plotters/imgs/researcher_scores.png', 
            dpi=300,
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none')

plt.close() 