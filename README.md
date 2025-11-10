# 🌊 **Project Tsunami — Source Inversion & Travel-Time Modelling**

### M1 Geology — *Institut de Physique du Globe de Paris (IPGP)*
**Course:** *Analyse de données en Géosciences*
**Supervision:** *E. Gaier, C. Narteau*
**Authors:** [**Maxime Soares Correia**](https://maxsc4.github.io/) & **Matthieu Courcelles**

---

<p align="center">
  <img src="outputs/world_map_inversion.png" alt="Global bathymetry with stations and the inverted tsunami source" width="90%">
</p>

---

## 📘 Overview

**Project Tsunami** is an educational geophysics project developed as part of the *Analyse de données en Géosciences* course (M1 Geology, IPGP).
Its goal is to **model tsunami propagation** across the oceans and perform a **source inversion** based on observed arrival times at tide-gauge stations.

Starting from:
- a global bathymetric model (*ETOPO5*), and
- a dataset of tsunami arrival times at coastal stations,

the project estimates the **most likely tsunami source location** and **origin time** that best explain the observed data.

---

## ⚙️ Methodology

### 1. Bathymetry loading
The *ETOPO5* ASCII grid is loaded using `io_etopo.py`.
It produces a function:

```python
depth(lat, lon) → water depth (m)
```

which handles interpolation, missing data, and longitude wrapping.

---

### 2. Geometrical modelling
The tsunami is assumed to follow the **great-circle path** between the source and each station.
All paths and distances are computed on a spherical Earth using trigonometric formulas.

---

### 3. Velocity model
Wave phase speed is approximated by the shallow-water relation:

\[
v = \sqrt{g\,h}
\]

where:
- \( g = 9.81 \, \mathrm{m/s^2} \) is gravity,
- \( h \) is the local water depth (in meters).

---

### 4. Travel-time computation
The tsunami travel time between two points is obtained by integrating along the path:

\[
T = \int_{\text{path}} \frac{ds}{v(h(s))}
\]

The integration is **vectorized** for efficiency and ignores land or shallow coastal points automatically.

---

### 5. Source inversion
A **robust adaptive grid search** estimates:
- the **source latitude & longitude**, and
- the **origin time** \( t_0^* \)

by minimizing a physically meaningful **RMS misfit** between observed and modeled arrival times:

\[
\text{misfit} = \sqrt{\frac{1}{N}\sum_i \left(t_{\text{obs},i} - (t_0^* + T_{\text{model},i})\right)^2}
\]

Outliers are handled through *median-based estimation* and *clipping*.

---

### 6. Visualization
The module `world_map.py` generates a clear and customizable world map:

- Bathymetry with blue depth shading
- Source marker (gold star) + uncertainty circle
- Station positions with names
- Great-circle paths (red)
- Automatic legends and save options

---

## 🧠 Physical assumptions

The model relies on:
- Long-wavelength, linear shallow-water approximation
- Spherical Earth with constant gravity
- Negligible refraction, dispersion, and coastal reflections

Despite these simplifications, it captures the **first-order physics** of tsunami travel times, making it ideal for educational and exploratory purposes.

---

## 🚀 Running the pipeline

The entire workflow can be executed through:

```bash
python scripts/run_inversion.py
```

By default, it will:
1. Load the bathymetry and observation data
2. Perform the inversion (robust mode)
3. Display and save the resulting world map under `outputs/world_map_inversion.png`

You can also call the pipeline programmatically:

```python
from scripts.run_inversion import run_pipeline

results = run_pipeline(
    etopo_path="data/etopo5.grd",
    stations_csv="data/data_villes.csv",
    lon_mode="360",
    search_box=(-60, 60, 100, 290),  # entire Pacific Ocean
)
```

---

## 📂 Project structure

```
project-tsunami/
│
├── data/                     # Bathymetry & station data
│   ├── etopo5.grd
│   └── data_villes.csv
│
├── tsunami/                  # Core modules
│   ├── geo.py                # Great-circle geometry
│   ├── speed_model.py        # Tsunami velocity model
│   ├── speed_integrator.py   # Travel-time integration (vectorized)
│   ├── io_etopo.py           # ETOPO grid loading & interpolation
│   ├── inverse.py            # Source inversion (robust, adaptive)
│   └── observations.py       # Station data & arrival-time loader
│
├── plotting/
│   ├── world_map.py          # Global map & visualization tools
│
├── scripts/
│   └── run_inversion.py      # Main inversion pipeline
│
└── outputs/                  # Generated figures & results
```

---

## 🧮 Example result

<p align="center">
  <img src="outputs/world_map_inversion.png" alt="Inversion result map" width="85%">
</p>

---

## 👥 Authors

- [**Maxime Soares Correia**](https://maxsc4.github.io/)
- **Matthieu Courcelles**

Supervised by **Eric Gayer**, as part of the *U.E. Analyse de données en Géosciences* course,
M1 Geology — IPGP (2025).

---

## 🪶 License

This repository is intended for **academic and educational purposes only**.
Reuse and adaptation are permitted for research and teaching, with proper credit.

---

## 💡 Acknowledgments

Special thanks to the **Institut de Physique du Globe de Paris (IPGP)**
for providing the datasets and computational resources used in this project.
