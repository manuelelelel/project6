import click
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import time
from numba import njit, prange

# -----------------------------
# NUMBA-KERNE
# -----------------------------

@njit(parallel=True, fastmath=True, cache=True)
def update_halo(field, num_halo):
    nz, ny, nx = field.shape
    h = num_halo

    # bottom/top (ohne Ecken)
    for k in prange(nz):
        # bottom rows j=0..h-1  <=  rows ny-2h..ny-h-1
        for j in range(h):
            jj_src = ny - 2*h + j
            for i in range(h, nx - h):
                field[k, j, i] = field[k, jj_src, i]
        # top rows j=ny-h..ny-1  <=  rows h..2h-1
        for j in range(h):
            jj_dst = ny - h + j
            jj_src = h + j
            for i in range(h, nx - h):
                field[k, jj_dst, i] = field[k, jj_src, i]

    # left/right (inkl. Ecken)
    for k in prange(nz):
        for j in range(ny):
            # left cols i=0..h-1  <=  cols nx-2h..nx-h-1
            for i in range(h):
                ii_src = nx - 2*h + i
                field[k, j, i] = field[k, j, ii_src]
            # right cols i=nx-h..nx-1  <=  cols h..2h-1
            for i in range(h):
                ii_dst = nx - h + i
                ii_src = h + i
                field[k, j, ii_dst] = field[k, j, ii_src]


@njit(parallel=True, fastmath=True, cache=True)
def laplacian(in_field, lap_field, num_halo, extend):
    """
    lap_field wird beschrieben (muss gleiche shape haben).
    extend=1: eine Zelle in den Halo hinein; extend=0: nur Innenbereich.
    """
    nz, ny, nx = in_field.shape
    h = num_halo
    ib = h - extend
    ie = nx - h + extend   # exklusiv
    jb = h - extend
    je = ny - h + extend   # exklusiv

    for k in range(nz):           # k seriell (oft kleiner)
        for j in prange(jb, je):  # prange über y (groß)
            for i in range(ib, ie):
                lap_field[k, j, i] = (
                    -4.0 * in_field[k, j, i]
                    + in_field[k, j, i - 1]
                    + in_field[k, j, i + 1]
                    + in_field[k, j - 1, i]
                    + in_field[k, j + 1, i]
                )


@njit(parallel=True, fastmath=True, cache=True)
def axpy_diffusion_step(in_field, lap2_field, out_field, alpha, num_halo):
    """
    out = in - alpha * lap2  (nur Innenbereich)
    """
    nz, ny, nx = in_field.shape
    h = num_halo
    for k in prange(nz):
        for j in range(h, ny - h):
            for i in range(h, nx - h):
                out_field[k, j, i] = in_field[k, j, i] - alpha * lap2_field[k, j, i]


@njit(fastmath=True, cache=True)
def copy3d(dst, src):
    nz, ny, nx = dst.shape
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                dst[k, j, i] = src[k, j, i]


@njit(fastmath=True, cache=True)
def apply_diffusion(in_field, out_field, alpha, num_halo, num_iter=1):
    """
    Führt num_iter Schritte aus. out_field enthält am Ende das Resultat.
    Keine Kopien im Kernel; Swap der Referenzen; gezielte Kopie nur wenn nötig.
    """
    a = in_field
    b = out_field
    tmp = np.empty_like(in_field)

    for n in range(num_iter):
        update_halo(a, num_halo)
        laplacian(a, tmp, num_halo=num_halo, extend=1)
        laplacian(tmp, b, num_halo=num_halo, extend=0)
        axpy_diffusion_step(a, b, b, alpha, num_halo)

        if n < num_iter - 1:
            a, b = b, a

    # Halos des finalen Feldes (liegen aktuell in b)
    update_halo(b, num_halo)

    # Ergebnis garantiert in out_field ablegen:
    # Ungerade Iterationszahl: Resultat liegt schon in out_field.
    # Gerade Iterationszahl: Resultat liegt in in_field -> kopieren.
    if (num_iter % 2) == 0:
        copy3d(out_field, in_field)
    # sonst: out_field enthält bereits das Ergebnis


# -----------------------------
# TREIBER
# -----------------------------
@click.command()
@click.option("--nx", type=int, required=True)
@click.option("--ny", type=int, required=True)
@click.option("--nz", type=int, required=True)
@click.option("--num_iter", type=int, required=True)
@click.option("--num_halo", type=int, default=2)
@click.option("--plot_result", is_flag=True, default=False)  # <— Flag ohne Argument
def main(nx, ny, nz, num_iter, num_halo=2, plot_result=False):
    assert 0 < nx <= 1024 * 1024
    assert 0 < ny <= 1024 * 1024
    assert 0 < nz <= 1024
    assert 0 < num_iter <= 1024 * 1024
    assert 2 <= num_halo <= 256

    alpha = np.float32(1.0 / 32.0)  # float32 wie Fortran wp=4

    in_field = np.zeros((nz, ny + 2 * num_halo, nx + 2 * num_halo), dtype=np.float32)
    in_field[
        nz // 4 : 3 * nz // 4,
        num_halo + ny // 4 : num_halo + 3 * ny // 4,
        num_halo + nx // 4 : num_halo + 3 * nx // 4,
    ] = 1.0
    out_field = in_field.copy()

    np.save("in_field_numba", in_field)

    if plot_result:
        plt.ioff()
        mid = in_field.shape[0] // 2
        plt.imshow(in_field[mid, :, :], origin="lower")
        plt.colorbar()
        plt.savefig("in_field_numba.png")
        plt.close()

    # Warmup (JIT & Caches); getrennte Arrays, damit das Timing sauber bleibt
    _in_w = in_field.copy()
    _out_w = out_field.copy()
    apply_diffusion(_in_w, _out_w, alpha, num_halo, num_iter=1)

    # Timing
    tic = time.time()
    apply_diffusion(in_field, out_field, alpha, num_halo, num_iter=num_iter)
    toc = time.time()
    print(f"Elapsed time for work = {toc - tic:.6f} s")

    np.save("out_field_numba", out_field)

    if plot_result:
        mid = out_field.shape[0] // 2
        plt.imshow(out_field[mid, :, :], origin="lower")
        plt.colorbar()
        plt.savefig("out_field_numba.png")
        plt.close()


if __name__ == "__main__":
    main()
