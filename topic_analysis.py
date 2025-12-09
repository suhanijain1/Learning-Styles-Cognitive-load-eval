import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def topic_interaction_analysis():
    print("Loading data...")
    try:
        df = pd.read_excel('data.xlsx')
    except FileNotFoundError:
        print("Error: data.xlsx not found.")
        return

    # --- Feature Engineering ---
    df['z_NASA'] = stats.zscore(df['NASA_Total'])
    df['z_HR'] = stats.zscore(df['Delta_HR'])
    df['z_Pupil'] = stats.zscore(df['Delta_Pupil_mean'])
    df['z_Likert'] = stats.zscore(df['LIKERT_Total'])
    df['Strain_Index'] = (df['z_NASA'] + df['z_HR'] + df['z_Pupil'] + df['z_Likert']) / 4
    df['Learning_Ease'] = -1 * df['Strain_Index']

    # --- PLOT: Topic x Modality Interaction ---
    # Question: Does the effectiveness of a modality depend on the Topic Order (Fatigue)?
    
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    # We plot Ease on Y, Topic (1,2,3) on X, Colored by Modality
    # If lines are parallel -> No Interaction.
    # If lines cross -> Interaction (e.g., Reading gets harder over time, Kinesthetic stays easy).
    
    sns.lineplot(
        x='Topic', 
        y='Learning_Ease', 
        hue='Modality', 
        data=df, 
        marker='o', 
        linewidth=2.5,
        palette='Set2',
        errorbar=('se', 1) # Standard Error bars
    )
    
    plt.title('Interaction Effect: Learning Ease by Topic & Modality')
    plt.ylabel('Learning Ease (Z-Score)')
    plt.xlabel('Topic Order (1 -> 3)')
    plt.xticks([1, 2, 3], ['Topic 1 (Fresh)', 'Topic 2', 'Topic 3 (Fatigued)'])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('plot_topic_modality_interaction.png')
    
    print("Generated 'plot_topic_modality_interaction.png'")
    
    # --- Quick Stats for Interpretation ---
    # Compare Slope? 
    # Let's print mean ease per topic per modality
    print("\nMean Learning Ease by Topic & Modality:")
    summary = df.groupby(['Topic', 'Modality'])['Learning_Ease'].mean().unstack()
    print(summary)

if __name__ == "__main__":
    topic_interaction_analysis()

