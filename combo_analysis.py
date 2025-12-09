import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.anova import AnovaRM
import seaborn as sns
import matplotlib.pyplot as plt

def combo_analysis():
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
    
    # Create the Interaction Column
    df['Condition'] = "T" + df['Topic'].astype(str) + "_" + df['Modality']
    
    print("\n--- 9 Condition Groups ---")
    summary = df.groupby('Condition')[['Objective_Percent', 'Learning_Ease']].mean().sort_values(by='Objective_Percent', ascending=False)
    print(summary)
    
    # --- ANOVA on the 9 Groups ---
    # Testing if any specific combo is better
    # Note: With N=27, we have 3 samples per condition. Small, but calculating F-stat is possible.
    
    # One-Way ANOVA (Independent samples assumption for simplicity, or we can treat as different treatments)
    # Using scipy.stats.f_oneway
    
    groups_score = [df[df['Condition'] == c]['Objective_Percent'] for c in df['Condition'].unique()]
    f_score, p_score = stats.f_oneway(*groups_score)
    
    groups_ease = [df[df['Condition'] == c]['Learning_Ease'] for c in df['Condition'].unique()]
    f_ease, p_ease = stats.f_oneway(*groups_ease)
    
    print(f"\nANOVA (Score ~ Condition): F={f_score:.2f}, p={p_score:.4f}")
    print(f"ANOVA (Ease ~ Condition): F={f_ease:.2f}, p={p_ease:.4f}")
    
    if p_score < 0.05:
        print("RESULT: Significant difference between specific Topic-Modality combos (Score).")
    else:
        print("RESULT: No specific combo is significantly better (Score).")
        
    # --- Visualization ---
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))
    
    # Sort conditions by Topic then Modality for cleaner x-axis
    df = df.sort_values(by=['Topic', 'Modality'])
    
    sns.barplot(x='Condition', y='Learning_Ease', data=df, palette='viridis', errorbar=('se', 1))
    plt.xticks(rotation=45, ha='right')
    plt.title('Learning Ease by Condition (Topic + Modality)')
    plt.tight_layout()
    plt.savefig('plot_condition_ease.png')
    
    print("Generated 'plot_condition_ease.png'")

if __name__ == "__main__":
    combo_analysis()

