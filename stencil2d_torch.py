import click
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import time


def laplacian(in_field, lap_field, num_halo, extend=0):
    ib = num_halo - extend
    ie = -num_halo + extend
    jb = num_halo - extend
    je = -num_halo + extend

    lap_field[:, jb:je, ib:ie] = (
        -4.0 * in_field[:, jb:je, ib:ie]
        + in_field[:, jb:je, ib - 1 : ie - 1]
        + in_field[:, jb:je, ib + 1 : ie + 1 if ie != -1 else None]
        + in_field[:, jb - 1 : je - 1, ib:ie]
        + in_field[:, jb + 1 : je + 1 if je != -1 else None, ib:ie]
    )


def update_halo(field, num_halo):
    # bottom edge (without corners)
    field[:, :num_halo, num_halo:-num_halo] = field[
        :, -2 * num_halo : -num_halo, num_halo:-num_halo
    ]

    # top edge (without corners)
    field[:, -num_halo:, num_halo:-num_halo] = field[
        :, num_halo : 2 * num_halo, num_halo:-num_halo
    ]

    # left edge (including corners)
    field[:, :, :num_halo] = field[:, :, -2 * num_halo : -num_halo]

    # right edge (including corners)
    field[:, :, -num_halo:] = field[:, :, num_halo : 2 * num_halo]


def apply_diffusion(in_field, out_field, alpha, num_halo, num_iter=1):
    tmp_field = torch.empty_like(in_field)

    for n in range(num_iter):
        update_halo(in_field, num_halo)

        laplacian(in_field, tmp_field, num_halo=num_halo, extend=1)
        laplacian(tmp_field, out_field, num_halo=num_halo, extend=0)

        out_field[:, num_halo:-num_halo, num_halo:-num_halo] = (
            in_field[:, num_halo:-num_halo, num_halo:-num_halo]
            - alpha * out_field[:, num_halo:-num_halo, num_halo:-num_halo]
        )

        if n < num_iter - 1:
            in_field, out_field = out_field.clone(), in_field
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

    device = torch.device("cpu")

    in_field = torch.zeros((nz, ny + 2 * num_halo, nx + 2 * num_halo), device=device)
    in_field[
        nz // 4 : 3 * nz // 4,
        num_halo + ny // 4 : num_halo + 3 * ny // 4,
        num_halo + nx // 4 : num_halo + 3 * nx // 4,
    ] = 1.0

    out_field = in_field.clone()

    torch.save(in_field, "in_field.pt")

    if plot_result:
        plt.ioff()
        plt.imshow(in_field[in_field.shape[0] // 2, :, :].cpu().numpy(), origin="lower")
        plt.colorbar()
        plt.savefig("in_field.png")
        plt.close()

    apply_diffusion(in_field.clone(), out_field.clone(), alpha, num_halo)

    tic = time.time()
    apply_diffusion(in_field, out_field, alpha, num_halo, num_iter=num_iter)
    toc = time.time()

    print(f"Elapsed time for work = {toc - tic} s")

    torch.save(out_field, "out_field.pt")

    if plot_result:
        plt.imshow(out_field[out_field.shape[0] // 2, :, :].cpu().numpy(), origin="lower")
        plt.colorbar()
        plt.savefig("out_field.png")
        plt.close()


if __name__ == "__main__":
    main()
