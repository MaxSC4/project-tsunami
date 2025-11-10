import pandas as pd

def load_arrival_times(csv_path):
    """
    Charge le fichier data_villes.csv et renvoie les heures d'arrivée en secondes.
    """

    df = pd.read_csv(csv_path, sep=r"\s+")

    df["datetime"] = pd.to_datetime(
        df["date_arrivee"] + " " + df["heure_arrivee"],
        format="%d-%b-%Y %H:%M:%S",
        utc="UTC"
    )

    times_posix = df["datetime"].astype("int64") / 1e9   # ns -> s
    times_posix = times_posix.to_numpy(dtype=float)

    return df, times_posix
