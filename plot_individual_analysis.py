import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def plot_individual_analysis():
    print("Loading data...")
    try:
        df = pd.read_excel('data.xlsx')
    except FileNotFoundError:
        print("Error: data.xlsx not found.")
        return

    # --- Feature Engineering (Ease) ---
    df['z_NASA'] = stats.zscore(df['NASA_Total'])
    df['z_HR'] = stats.zscore(df['Delta_HR'])
    df['z_Pupil'] = stats.zscore(df['Delta_Pupil_mean'])
    df['z_Likert'] = stats.zscore(df['LIKERT_Total'])
    df['Strain_Index'] = (df['z_NASA'] + df['z_HR'] + df['z_Pupil'] + df['z_Likert']) / 4
    df['Learning_Ease'] = -1 * df['Strain_Index']

    # Set style
    sns.set_theme(style="whitegrid")

    # --- PLOT 1: SMALL MULTIPLES (3x3 Grid) ---
    # One subplot per Participant
    # Bar = Objective Score (Left Axis)
    # Line = Learning Ease (Right Axis)
    
    unique_ids = df['ID'].unique()
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, pid in enumerate(unique_ids):
        ax1 = axes[i]
        p_data = df[df['ID'] == pid]
        
        # Plot Score (Bar)
        sns.barplot(x='Modality', y='Objective_Percent', data=p_data, ax=ax1, palette='pastel', alpha=0.7)
        ax1.set_ylim(0, 100)
        ax1.set_ylabel('Score (%)')
        ax1.set_title(f'Participant {pid}')
        
        # Plot Ease (Line/Point) on secondary axis
        ax2 = ax1.twinx()
        sns.pointplot(x='Modality', y='Learning_Ease', data=p_data, ax=ax2, color='red', markers='o', scale=0.7)
        ax2.set_ylabel('Ease (Z-Score)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Add simple "Best" annotation
        best_mod = p_data.loc[p_data['Objective_Percent'].idxmax(), 'Modality']
        # ax1.text(0.5, 90, f"Best: {best_mod}", ha='center', transform=ax1.transData)

    plt.tight_layout()
    plt.savefig('plot_individual_profiles_grid.png')
    print("Generated 'plot_individual_profiles_grid.png'")
    plt.close()

    # --- PLOT 2: SCATTER (Colored by Participant) ---
    # Showing that Ease predicts Score for everyone
    
    plt.figure(figsize=(10, 7))
    # Using a categorical palette for 9 participants
    sns.scatterplot(x='Learning_Ease', y='Objective_Percent', hue='ID', data=df, palette='tab10', s=120, legend='full')
    
    # Add a single regression line for the whole group to show the trend
    sns.regplot(x='Learning_Ease', y='Objective_Percent', data=df, scatter=False, color='black', line_kws={'linestyle': '--', 'linewidth': 1.5})
    
    plt.title('Learning Ease vs. Performance (Color by Participant)')
    plt.xlabel('Learning Ease (Composite Z-Score)')
    plt.ylabel('Objective Score (%)')
    plt.legend(title='Participant ID', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('plot_ease_score_by_participant.png')
    print("Generated 'plot_ease_score_by_participant.png'")
    plt.close()

if __name__ == "__main__":
    plot_individual_analysis()

