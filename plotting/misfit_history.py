# plotting/misfit_history.py
import numpy as np
import matplotlib.pyplot as plt


def plot_misfit_history(
    history,
    metric="rmse_rel",
    title="RMSE vs iteration",
    savepath=None,
    show=True,
):
    """
    Plot l'évolution du misfit (RMSE) au cours des itérations d'inversion.

    Parameters
    ----------
    history : list of dict
        Liste de dictionnaires produits par l'inversion, avec au moins
        les clés "iter" et `metric` (par défaut "rmse_rel").
        Exemple d'entrée :
            {"iter": 1, "lat": ..., "lon": ..., "rmse_rel": 2800.0, "valid": 11}
    metric : str
        Clé du misfit à tracer ("rmse_rel" ou "rmse_abs").
    title : str
        Titre de la figure.
    savepath : str or None
        Si non None, chemin de sauvegarde du PNG.
    show : bool
        Si True, appelle plt.show().

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    if not history:
        raise ValueError("plot_misfit_history: empty history list.")

    # Extraction des itérations et du misfit
    iters = np.array([h.get("iter", i + 1) for i, h in enumerate(history)], dtype=int)
    values = np.array([h[metric] for h in history], dtype=float)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)

    ax.plot(iters, values, marker="o", linestyle="-")
    ax.set_xlabel("Iteration")
    ylabel = "RMS misfit (s)"
    if metric == "rmse_rel":
        ylabel = "Relative RMS misfit (s)"
    elif metric == "rmse_abs":
        ylabel = "Absolute RMS misfit (s)"
    ax.set_ylabel(ylabel)

    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.6)

    # Mettre en évidence le minimum
    idx_min = int(np.argmin(values))
    it_min = iters[idx_min]
    v_min = values[idx_min]
    ax.scatter([it_min], [v_min], color="red", zorder=5)
    ax.annotate(
        f"min = {v_min:.1f} s\n(iter {it_min})",
        xy=(it_min, v_min),
        xytext=(it_min + 0.2, v_min * 1.05),
        arrowprops=dict(arrowstyle="->", lw=0.8),
        fontsize=8,
    )

    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight", dpi=150)
    if show:
        plt.show()

    return fig, ax
