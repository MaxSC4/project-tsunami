# 🌊 Project Tsunami — Source Inversion and Travel-Time Modelling

### M1 Geology — Institut de Physique du Globe de Paris (IPGP)
**Course:** *Analyse de données en Géosciences*  
**Supervision:** *E. Gaier*  
**Authors:** [**Maxime Soares Correia**](https://maxsc4.github.io/) & **Matthieu Courcelles**

---

## 📘 Overview

This project was developed as part of the *"Analyse de données en Géosciences"* course in the M1 Geology program at the **Institut de Physique du Globe de Paris (IPGP)**, under the supervision of **E. Gaier**.

The goal is to **model tsunami wave propagation** and perform a **source inversion** based on observed arrival times at several coastal tide stations.

Starting from a global bathymetric dataset (*ETOPO5*) and a set of observed arrival times, the project aims to:

1. Compute theoretical travel times along great-circle paths using a simplified physical model of tsunami propagation.  
2. Compare these modeled travel times with observed arrivals.  
3. Invert for the **most likely source location** and **origin time** that minimize the residuals between observed and modeled data.

---

## ⚙️ Methodology

1. **Bathymetry loading** — The *ETOPO5* grid is read and transformed into a function returning ocean depth at any geographic coordinate.  

2. **Geometrical modelling** — Great-circle distances and trajectories between points on the Earth’s surface are computed using spherical trigonometry.  

3. **Velocity model** — Tsunami phase velocity is approximated with the shallow-water relation $$v = \sqrt{gh}$$, where $$g = 9.81\,\mathrm{m/s^2}$$ and $$h$$ is the local water depth.

4. **Travel-time computation** — The propagation time between a source and each observation station is obtained by integrating the inverse velocity along the great-circle path:

$$
T = \int_{\text{path}} \frac{ds}{v\left(h(s)\right)}
$$

5. **Inversion** — A grid search minimizes the least-squares misfit between observed and modeled arrival times:

*WIP*
---

## 🧮 Physical assumptions

The model assumes:
- Long-wavelength tsunami propagation in the shallow-water regime.  
- Constant gravity and spherical Earth geometry.  
- Negligible effects of bathymetric refraction, non-linear dispersion, and coastline interactions.

While simplified, this approach captures first-order travel-time variations due to bathymetry and provides a framework for educational exploration of tsunami inversion techniques.

---

## 👥 Authors

- [**Maxime Soares Correia**](https://maxsc4.github.io/)  
- **Matthieu Courcelles**

Supervised by **Eric Gaier**, as part of the *U.E. Analyse de données en Géosciences* course,  
M1 Geology — IPGP, 2025.

---

## 🪶 License

This repository is intended for **academic and educational purposes only**.  
Reuse and adaptation are permitted for research and teaching with appropriate credit.
