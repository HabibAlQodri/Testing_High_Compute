"""HPC mini: estimasi pi (Monte Carlo) -> serial vs paralel vs vektorisasi NumPy."""
import time, random
import numpy as np
from multiprocessing import Pool, cpu_count
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

N = 2_000_000  # total sampel

def loop_chunk(a):                       # Python murni (CPU-bound)
    n, seed = a; r = random.Random(seed); c = 0
    for _ in range(n):
        x, y = r.random(), r.random()
        c += x * x + y * y <= 1.0
    return c

def numpy_chunk(a):                      # vektorisasi
    n, seed = a; g = np.random.default_rng(seed)
    x, y = g.random(n), g.random(n)
    return int(np.count_nonzero(x * x + y * y <= 1.0))

def run(fn, workers):
    chunks = [(N // (workers * 4), s) for s in range(workers * 4)]
    t = time.perf_counter()
    if workers == 1: hits = sum(map(fn, chunks))
    else:
        with Pool(workers) as p: hits = sum(p.map(fn, chunks))
    return 4 * hits / sum(c[0] for c in chunks), time.perf_counter() - t

if __name__ == "__main__":
    cores = cpu_count() or 1
    ws = sorted({1, 2, 4, cores})
    print(f"CPU core terdeteksi: {cores}\n")
    res = {w: run(loop_chunk, w) for w in ws}
    pi_np, t_np = run(numpy_chunk, 1)
    t1 = res[1][1]
    print(f"{'worker':>6} {'waktu(s)':>9} {'speedup':>8} {'efisiensi':>9} {'pi':>8}")
    for w, (p, t) in res.items():
        print(f"{w:>6} {t:>9.2f} {t1 / t:>8.2f} {t1 / t / w:>9.0%} {p:>8.4f}")
    print(f"NumPy vektor (1 core): {t_np:.3f}s  -> {t1 / t_np:.0f}x lebih cepat dari loop Python\n")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ws, [res[w][1] for w in ws], "o-", color="#0072B2"); ax[0].set(title="Waktu eksekusi", xlabel="worker", ylabel="detik")
    ax[1].plot(ws, [t1 / res[w][1] for w in ws], "o-", color="#009E73", label="aktual")
    ax[1].plot(ws, ws, "--", color="gray", label="ideal"); ax[1].legend(); ax[1].set(title="Speedup", xlabel="worker")
    plt.tight_layout(); plt.savefig("hpc_hasil.png", dpi=120); print("Grafik: hpc_hasil.png")
