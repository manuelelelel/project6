import click
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import time


@jax.jit
def laplacian(in_field, num_halo, extend=0):
    ib = num_halo - extend
    ie = -num_halo + extend
    jb = num_halo - extend
    je = -num_halo + extend

    # Pad edge values if required by slice behavior
    center = in_field[:, jb:je, ib:ie]
    left   = in_field[:, jb:je, ib - 1 : ie - 1]
    right  = in_field[:, jb:je, ib + 1 : ie + 1 if ie != -1 else None]
    top    = in_field[:, jb - 1 : je - 1, ib:ie]
    bottom = in_field[:, jb + 1 : je + 1 if je != -1 else None, ib:ie]

    return -4.0 * center + left + right + top + bottom


from jax import lax

from jax import lax

def update_halo(field, num_halo):
    nz, ny, nx = field.shape

    def slice_and_update(fld, src_start, dst_start, size, axis):
        # Move axis to front
        fld = jnp.moveaxis(fld, axis, 0)
        src = lax.dynamic_slice(fld, (src_start, 0, 0), (size, fld.shape[1], fld.shape[2]))
        fld = lax.dynamic_update_slice(fld, src, (dst_start, 0, 0))
        return jnp.moveaxis(fld, 0, axis)

    # Bottom edge
    field = slice_and_update(field, ny - 2 * num_halo, 0, num_halo, axis=1)
    # Top edge
    field = slice_and_update(field, num_halo, ny - num_halo, num_halo, axis=1)
    # Left edge
    field = slice_and_update(field, nx - 2 * num_halo, 0, num_halo, axis=2)
    # Right edge
    field = slice_and_update(field, num_halo, nx - num_halo, num_halo, axis=2)

    return field




def apply_diffusion(in_field, alpha, num_halo, num_iter=1):
    out_field = jnp.copy(in_field)
    tmp_field = jnp.copy(in_field)

    def body_fun(i, val):
        in_f, out_f = val

        in_f = update_halo(in_f, num_halo)
        tmp = laplacian(in_f, num_halo, extend=1)
        lap = laplacian(tmp, num_halo, extend=0)

        out_inner = in_f[:, num_halo:-num_halo, num_halo:-num_halo] - alpha * lap
        out_f = out_f.at[:, num_halo:-num_halo, num_halo:-num_halo].set(out_inner)

        return (out_f, in_f) if i < num_iter - 1 else (update_halo(out_f, num_halo), in_f)

    in_field, out_field = jax.lax.fori_loop(0, num_iter, body_fun, (in_field, out_field))
    return out_field


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
