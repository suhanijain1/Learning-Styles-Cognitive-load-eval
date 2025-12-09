import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.anova import AnovaRM

def run_analysis_v4():
    print("Loading data...")
    try:
        df = pd.read_excel('data.xlsx')
    except FileNotFoundError:
        print("Error: data.xlsx not found.")
        return

    # --- 1. FEATURE ENGINEERING ---
    # Ease Calculation (High Likert = Good)
    df['z_NASA'] = stats.zscore(df['NASA_Total'])
    df['z_HR'] = stats.zscore(df['Delta_HR'])
    df['z_Pupil'] = stats.zscore(df['Delta_Pupil_mean'])
    df['z_Likert'] = stats.zscore(df['LIKERT_Total'])
    
    # Strain Index
    df['Strain_Index'] = (df['z_NASA'] + df['z_HR'] + df['z_Pupil'] + df['z_Likert']) / 4
    df['Learning_Ease'] = -1 * df['Strain_Index']

    # --- 2. STATISTICAL TESTS FUNCTION ---
    def test_variable(var_name, display_name):
        print(f"\n=== TESTING: {display_name} ===")
        
        # Normality Check
        print("--- Normality Check (Shapiro-Wilk) ---")
        modalities = ['Reading', 'Audio', 'Kinesthetic']
        all_normal = True
        
        for mod in modalities:
            data = df[df['Modality'] == mod][var_name]
            stat, p = stats.shapiro(data)
            print(f"{mod}: p={p:.4f} {'(Normal)' if p > 0.05 else '(NOT Normal)'}")
            if p < 0.05:
                all_normal = False
        
        # Hypothesis Testing
        print(f"\n--- Hypothesis Test for {display_name} ---")
        if all_normal:
            print(f">> Data is Normal. Using Repeated Measures ANOVA.")
            try:
                aov = AnovaRM(df, var_name, 'ID', within=['Modality']).fit()
                print(aov)
                # Check p-value extraction if possible, otherwise user reads table
            except ValueError as e:
                print(f"ANOVA Failed: {e}")
        else:
            print(f">> Data NOT Normal. Using Friedman Test.")
            pivot_df = df.pivot(index='ID', columns='Modality', values=var_name)
            stat, p = stats.friedmanchisquare(pivot_df['Reading'], pivot_df['Audio'], pivot_df['Kinesthetic'])
            print(f"Friedman Chi-Square: {stat:.3f}")
            print(f"p-value: {p:.4f}")
            if p < 0.05:
                print("RESULT: SIGNIFICANT difference found.")
            else:
                print("RESULT: No significant difference found.")

    # --- 3. RUN TESTS ---
    test_variable('Objective_Percent', 'Objective Score')
    test_variable('Learning_Ease', 'Learning Ease')
    test_variable('Strain_Index', 'Strain Index') # Same stats as Ease, just verifying

    # --- 4. SAVE RESULTS ---
    # Re-generating plots is optional if data changed significantly, but let's stick to stats output first
    print("\nAnalysis V4 Complete.")

if __name__ == "__main__":
    run_analysis_v4()

