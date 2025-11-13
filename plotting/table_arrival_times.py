import pandas as pd
import numpy as np

def build_residual_table(df, T_model, output_csv=None):
    """
    Construit un tableau avec :
        - Station
        - t_obs (s)
        - T_mod (s)
        - Résidu = t_obs - T_mod

    df : DataFrame contenant les colonnes d'origine, y compris t_obs_s
    T_model : array-like des temps modélisés dans le même ordre que df
    """

    df_tab = pd.DataFrame({
        "Station": df["Ville/Port"].str.replace("_", " "),
        "t_obs (s)": df["t_obs_s"].values,
        "T_mod (s)": T_model,
        "Résidu (s)": df["t_obs_s"].values - T_model
    })

    if output_csv:
        df_tab.to_csv(output_csv, index=False)
        print(f"[table_arrival_times] Tableau enregistré : {output_csv}")

    return df_tab


def print_latex_table(df_tab):
    """
    Imprime le tableau au format LaTeX.
    """
    latex = df_tab.to_latex(index=False, float_format="%.1f")
    print("\n===== TABLE LATEX =====\n")
    print(latex)
    print("========================\n")
