import click
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import time
from jax import lax
from functools import partial


# ---------------------------------------
# 1) Laplacian (with halo-aware slicing)
# ---------------------------------------
@partial(jax.jit, static_argnames=("num_halo", "extend"))
def laplacian(in_field, num_halo: int, extend: int = 0, out_field=None):
    """
    Compute the 2D (x,y) Laplacian on each z-slice using 2nd-order centered differences.
    Slices mirror your original NumPy code; 'extend' controls how far into the halo we compute.
    """
    ib = num_halo - extend
    ie = -num_halo + extend
    jb = num_halo - extend
    je = -num_halo + extend

    # Match original 'None' handling for ie/je == -1
    x_right_end = (ie + 1) if (ie != -1) else None
    y_up_end    = (je + 1) if (je != -1) else None

    lap = jnp.zeros_like(in_field) if out_field is None else out_field

    center = in_field[:, jb:je, ib:ie]
    left   = in_field[:, jb:je, ib - 1 : ie - 1]
    right  = in_field[:, jb:je, ib + 1 : x_right_end]
    down   = in_field[:, jb - 1 : je - 1, ib:ie]
    up     = in_field[:, jb + 1 : y_up_end,    ib:ie]

    interior_lap = -4.0 * center + left + right + down + up
    return lap.at[:, jb:je, ib:ie].set(interior_lap)


# ---------------------------------------
# 2) Halo update (up/down then left/right)
# ---------------------------------------
@partial(jax.jit, static_argnames=("num_halo",))
def update_halo(field, num_halo: int):
    """
    Update halo zones using:
      - bottom/top without corners in x,
      - left/right including corners (fills corners).
    Returns a new array (functional, no in-place mutation).
    """
    h = num_halo
    if h == 0:
        return field

    out = field
    # bottom edge (without corners): y[0:h, x[h:-h]] <= y[ny-2h:ny-h, x[h:-h]]
    out = out.at[:, :h, h:-h].set(out[:, -2*h:-h, h:-h])
    # top edge (without corners): y[ny-h:ny, x[h:-h]] <= y[h:2h, x[h:-h]]
    out = out.at[:, -h:, h:-h].set(out[:, h:2*h, h:-h])
    # left edge (including corners): x[0:h] <= x[nx-2h:nx-h]
    out = out.at[:, :, :h].set(out[:, :, -2*h:-h])
    # right edge (including corners): x[nx-h:nx] <= x[h:2h]
    out = out.at[:, :, -h:].set(out[:, :, h:2*h])
    return out


# ---------------------------------------
# 3) 4th-order diffusion apply (iterative)
# ---------------------------------------
@partial(jax.jit, static_argnames=("num_halo", "num_iter"))
def apply_diffusion(in_field, alpha: float, num_halo: int, num_iter: int = 1):
    """
    Perform 'num_iter' iterations of the 4th-order diffusion step:
      tmp = Lap(in, extend=1); lap2 = Lap(tmp, extend=0);
      out[interior] = in[interior] - alpha * lap2[interior].
    Swaps buffers except on the last iteration; finally updates halos.
    Returns the final field (same shape as input).
    """
    h = num_halo

    if h == 0:
        # No halo path — interior is the whole array
        def body(i, carry):
            inf, outf = carry
            tmp  = laplacian(inf, num_halo=h, extend=1)
            lap2 = laplacian(tmp, num_halo=h, extend=0)
            outf = outf.at[:].set(inf - alpha * lap2)
            # swap unless last iter
            inf, outf = lax.cond(i < num_iter - 1,
                                 lambda c: (c[1], c[0]),
                                 lambda c: c,
                                 (inf, outf))
            return (inf, outf)

        inf, outf = lax.fori_loop(0, num_iter, body, (in_field, jnp.zeros_like(in_field)))
        return outf

    # Regular halo path
    def body(i, carry):
        inf, outf = carry

        # 1) update halos of the input
        inf = update_halo(inf, h)

        # 2) two Laplacians (extend=1 then 0)
        tmp  = laplacian(inf, num_halo=h, extend=1)
        lap2 = laplacian(tmp, num_halo=h, extend=0)

        # 3) interior update only
        interior = inf[:, h:-h, h:-h] - alpha * lap2[:, h:-h, h:-h]
        outf = outf.at[:, h:-h, h:-h].set(interior)

        # 4) swap unless last iter
        inf, outf = lax.cond(i < num_iter - 1,
                             lambda c: (c[1], c[0]),
                             lambda c: c,
                             (inf, outf))
        return (inf, outf)

    inf, outf = lax.fori_loop(0, num_iter, body, (in_field, jnp.zeros_like(in_field)))
    # Final halo update on the result (matches your NumPy code path)
    return update_halo(outf, h)


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

    shape = (nz, ny + 2 * num_halo, nx + 2 * num_halo)
    in_field = jnp.zeros(shape)

    in_field = in_field.at[
        nz // 4 : 3 * nz // 4,
        num_halo + ny // 4 : num_halo + 3 * ny // 4,
        num_halo + nx // 4 : num_halo + 3 * nx // 4,
    ].set(1.0)

    if plot_result:
        plt.ioff()
        plt.imshow(jnp.array(in_field[in_field.shape[0] // 2, :, :]), origin="lower")
        plt.colorbar()
        plt.savefig("in_field_jax.png")
        plt.close()

    # JIT warm-up
    _ = apply_diffusion(in_field, alpha, num_halo, num_iter=1)

    tic = time.time()
    out_field = apply_diffusion(in_field, alpha, num_halo, num_iter=num_iter)
    toc = time.time()

    print(f"Elapsed time for work = {toc - tic} s")

    # Save output field
    with open("out_field_jax.npy", "wb") as f:
        jnp.save(f, out_field)

    if plot_result:
        plt.imshow(jnp.array(out_field[out_field.shape[0] // 2, :, :]), origin="lower")
        plt.colorbar()
        plt.savefig("out_field_jax.png")
        plt.close()


if __name__ == "__main__":
    main()
