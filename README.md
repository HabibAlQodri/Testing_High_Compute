# HPC — Benchmark Paralel Monte Carlo

Menghitung π dengan Monte Carlo, lalu membandingkan: **loop Python serial**, **multiprocessing (N worker)**, dan **vektorisasi NumPy**.
Menunjukkan konsep inti HPC: speedup, efisiensi paralel, overhead, dan pentingnya vektorisasi.

```bash
pip install numpy matplotlib
python hpc_benchmark.py
```
Keluaran: tabel di terminal + `hpc_hasil.png`.

**Catatan:** speedup baru terlihat di mesin multi-core. Pada 1 core, paralel justru lebih lambat (overhead proses).
**Lanjutan:** `mpi4py` untuk cluster, `numba`/`cupy` untuk akselerasi, Slurm array job.
