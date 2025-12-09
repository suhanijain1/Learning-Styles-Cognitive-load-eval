import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def quadrant_analysis():
    print("Loading data...")
    try:
        df = pd.read_excel('data.xlsx')
    except FileNotFoundError:
        print("Error: data.xlsx not found.")
        return

    # --- 1. Calculate Metrics ---
    # Ease (Assuming High Likert = Good)
    df['z_NASA'] = stats.zscore(df['NASA_Total'])
    df['z_HR'] = stats.zscore(df['Delta_HR'])
    df['z_Pupil'] = stats.zscore(df['Delta_Pupil_mean'])
    df['z_Likert'] = stats.zscore(df['LIKERT_Total'])
    df['Strain_Index'] = (df['z_NASA'] + df['z_HR'] + df['z_Pupil'] + df['z_Likert']) / 4
    df['Learning_Ease'] = -1 * df['Strain_Index']
    
    # --- 2. Classification Logic ---
    # Define Thresholds (Median Split)
    score_median = df['Objective_Percent'].median()
    ease_median = df['Learning_Ease'].median()
    
    print(f"Median Score: {score_median:.2f}")
    print(f"Median Ease (Z): {ease_median:.2f}")

    def categorize(row):
        high_score = row['Objective_Percent'] >= score_median
        high_ease = row['Learning_Ease'] >= ease_median
        
        if high_score and high_ease:
            return 'Flow (Easy & Effective)'
        elif high_score and not high_ease:
            return 'Brute Force (Hard but Effective)'
        elif not high_score and high_ease:
            return 'Illusion (Easy but Failed)'
        else:
            return 'Struggle (Hard & Failed)'

    df['Learning_Type'] = df.apply(categorize, axis=1)
    
    # --- 3. Analysis Table (Frequency) ---
    print("\n--- Quadrant Distribution by Modality ---")
    contingency_table = pd.crosstab(df['Modality'], df['Learning_Type'])
    print(contingency_table)
    
    # Chi-Square Test
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    print(f"\nChi-Square Test of Independence: Chi2={chi2:.2f}, p={p:.4f}")
    if p < 0.05:
        print("RESULT: Significant dependency between Modality and Learning Type.")
    else:
        print("RESULT: No significant dependency (Modality doesn't dictate Type).")

    # --- 4. Visualizations ---
    sns.set_theme(style="whitegrid")
    
    # Plot A: Quadrant Scatter
    plt.figure(figsize=(10, 8))
    
    # Define Quadrant Colors
    palette = {
        'Flow (Easy & Effective)': '#2ca02c',       # Green
        'Brute Force (Hard but Effective)': '#ff7f0e', # Orange
        'Illusion (Easy but Failed)': '#1f77b4',       # Blue
        'Struggle (Hard & Failed)': '#d62728'          # Red
    }
    
    sns.scatterplot(
        x='Learning_Ease', 
        y='Objective_Percent', 
        hue='Modality', 
        style='Modality', 
        data=df, 
        palette='Set2',
        s=150,
        alpha=0.9
    )
    
    # Add Quadrant Lines
    plt.axvline(x=ease_median, color='gray', linestyle='--')
    plt.axhline(y=score_median, color='gray', linestyle='--')
    
    # Annotate Regions
    plt.text(ease_median + 0.1, score_median + 2, "FLOW", color='green', fontweight='bold')
    plt.text(ease_median - 0.1, score_median + 2, "BRUTE FORCE", color='orange', fontweight='bold', ha='right')
    plt.text(ease_median + 0.1, score_median - 2, "ILLUSION", color='blue', fontweight='bold', va='top')
    plt.text(ease_median - 0.1, score_median - 2, "STRUGGLE", color='red', fontweight='bold', ha='right', va='top')
    
    plt.title('The 4 Quadrants of Learning State')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('plot_learning_quadrants.png')
    plt.close()
    
if __name__ == "__main__":
    quadrant_analysis()

