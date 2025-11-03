import pandas as pd

def load_arrival_times(csv_path):
    """
    Charge le fichier data_villes.csv et renvoie les heures d'arrivée en secondes.
    """

    df = pd.read_csv(csv_path, sep=r"\s+")

    df["datetime"] = pd.to_datetime(
        df["date_arrivee"] + " " + df["heure_arrivee"],
        format="%d-%b-%Y %H:%M:%S"
    )

    t_obs_s = (df["datetime"] - df["datetime"].min()).dt.total_seconds()

    return df, t_obs_s.values



"""
Example d'utilisation :
-----------------------
df, t_obs_s = load_arrival_times("data/data_villes.csv")

print(df[["Ville/Port", "datetime"]])
print("Heures d'arrivée en secondes :", t_obs_s)
"""
