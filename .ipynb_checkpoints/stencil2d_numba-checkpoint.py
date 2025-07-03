import click
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import time
import numba
from numba import njit, prange


@njit(parallel=True)
def laplacian(in_field, lap_field, num_halo, extend=0):
    nz, ny, nx = in_field.shape
    ib = num_halo - extend
    ie = nx - num_halo + extend
    jb = num_halo - extend
    je = ny - num_halo + extend

    for k in prange(nz):
        for j in range(jb, je):
            for i in range(ib, ie):
                lap_field[k, j, i] = (
                    -4.0 * in_field[k, j, i]
                    + in_field[k, j, i - 1]
                    + in_field[k, j, i + 1]
                    + in_field[k, j - 1, i]
                    + in_field[k, j + 1, i]
                )


@njit(parallel=True)
def update_halo(field, num_halo):
    nz, ny, nx = field.shape

    for k in prange(nz):
        for j in range(num_halo):
            for i in range(num_halo, nx - num_halo):
                # Bottom edge
                field[k, j, i] = field[k, ny - 2 * num_halo + j, i]
                # Top edge
                field[k, ny - num_halo + j, i] = field[k, num_halo + j, i]

        for j in range(ny):
            for i in range(num_halo):
                # Left edge
                field[k, j, i] = field[k, j, nx - 2 * num_halo + i]
                # Right edge
                field[k, j, nx - num_halo + i] = field[k, j, num_halo + i]


@njit
def apply_diffusion(in_field, out_field, alpha, num_halo, num_iter=1):
    tmp_field = np.empty_like(in_field)

    for n in range(num_iter):
        update_halo(in_field, num_halo)

        laplacian(in_field, tmp_field, num_halo=num_halo, extend=1)
        laplacian(tmp_field, out_field, num_halo=num_halo, extend=0)

        nz, ny, nx = in_field.shape
        for k in range(nz):
            for j in range(num_halo, ny - num_halo):
                for i in range(num_halo, nx - num_halo):
                    out_field[k, j, i] = (
                        in_field[k, j, i] - alpha * out_field[k, j, i]
                    )

        if n < num_iter - 1:
            in_field[:], out_field[:] = out_field.copy(), in_field.copy()
        else:
            update_halo(out_field, num_halo)


@click.command()
@click.option("--nx", type=int, required=True)
@click.option("--ny", type=int, required=True)
@click.option("--nz", type=int, required=True)
@click.option("--num_iter", type=int, required=True)
@click.option("--num_halo", type=int, default=2)
@click.option("--plot_result", type=bool, default=False)
def main(nx, ny, nz, num_iter, num_halo=2, plot_result=False):
    assert 0 < nx <= 1024 * 1024
    assert 0 < ny <= 1024 * 1024
    assert 0 < nz <= 1024
    assert 0 < num_iter <= 1024 * 1024
    assert 2 <= num_halo <= 256
    alpha = 1.0 / 32.0

    in_field = np.zeros((nz, ny + 2 * num_halo, nx + 2 * num_halo))
    in_field[
        nz // 4 : 3 * nz // 4,
        num_halo + ny // 4 : num_halo + 3 * ny // 4,
        num_halo + nx // 4 : num_halo + 3 * nx // 4,
    ] = 1.0
    out_field = in_field.copy()

    np.save("in_field", in_field)

    if plot_result:
        plt.ioff()
        plt.imshow(in_field[in_field.shape[0] // 2, :, :], origin="lower")
        plt.colorbar()
        plt.savefig("in_field.png")
        plt.close()

    # Warm-up for JIT compilation
    apply_diffusion(in_field.copy(), out_field.copy(), alpha, num_halo, num_iter=1)

    tic = time.time()
    apply_diffusion(in_field, out_field, alpha, num_halo, num_iter=num_iter)
    toc = time.time()

    print(f"Elapsed time for work = {toc - tic} s")

    np.save("out_field", out_field)

    if plot_result:
        plt.imshow(out_field[out_field.shape[0] // 2, :, :], origin="lower")
        plt.colorbar()
        plt.savefig("out_field.png")
        plt.close()


if __name__ == "__main__":
    main()
