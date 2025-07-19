import numpy as np
import matplotlib.pyplot as plt
import scipy
import matplotlib

font = {'size'   : 14}
matplotlib.rc('font', **font)

matplotlib.rc('xtick', labelsize=12)
matplotlib.rc('ytick', labelsize=12)

def get_cdf(samples):
    values, counts = np.unique(samples, return_counts=True)
    cum_counts = np.cumsum(counts)
    normalized = cum_counts / cum_counts[-1]
    return values, normalized


def marcumq(a, b):
    return scipy.stats.ncx2.sf(b**2, 2, a**2)


def gen_rayleigh_va(rng, Omega_c, N):
    # 2 sigma^2 = Omega_c
    sigma = np.sqrt(Omega_c/2)
    x = rng.normal(0, sigma, N)
    y = rng.normal(0, sigma, N)

    return np.sqrt(x**2 + y**2)


def gen_rice_va(rng, Omega_c, K_r, N):
    # mu**2 = mu_x**2 + mu_y**2
    # K_r = mu**2 / (2 * sigma**2)
    #  => mu**2 = K_r * (2 * sigma**2)
    # Omega_c = mu**2 + 2 * sigma**2
    #  => Omega_c = K_r * (2 * sigma**2) + 2 * sigma**2
    #     Omega_c = (K_r + 1) * (2 * sigma**2)
    #     Omega_c / (2 * (K_r + 1)) =  sigma**2
    sigma = np.sqrt(
        Omega_c / (2 * (K_r + 1))
    )
    mu = np.sqrt(
        K_r * 2
    ) * sigma
    mu_y = mu_x = mu / np.sqrt(2)

    x = rng.normal(mu_x, sigma, N)
    y = rng.normal(mu_y, sigma, N)

    return np.sqrt(x*x + y*y)


def plot_rice_outage(
    snr_mean, va, K_r,
    fig=None,
    ax=None
):
    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot()

    x = np.linspace(-30, 30, 200)
    op = 1 - marcumq(
        np.sqrt(2 * K_r),
        np.sqrt(
            2 * (K_r + 1) / (10 ** (0.1 * snr_mean)) * 10 ** (0.1 * x)
        ),
    )

    clr = {
        -20: "red",
        0: "green",
        20: "blue"
    }[snr_mean]
    ax.semilogy(
        x, op,
        linestyle="-.",
        color=clr,
        label=r"Teórica $\overline{\gamma}_s = " + f"{snr_mean}$ dB"
    )
    ax.scatter(
        x[::10], op[::10],
        marker='s',
        s=50,
        facecolors='none',
        edgecolors=clr,
        linewidths=1.5,
    )

    va_x, va_cdf = get_cdf(va ** 2 * 10 **(0.1*snr_mean))
    va_x = 10 * np.log10(va_x)

    label = None
    if snr_mean == 20:
        label = "Simulação"
    ax.plot(
        va_x, va_cdf,
        linestyle="--",
        color="black",
        label=label
    )

    ax.grid(True, alpha=0.2, color="gray")

    ax.set_ylim((1e-6, 1.5))
    ax.set_xlim((-30, 30))
    ax.set_xlabel(r"Limiar de SNR - $\gamma_\text{th}$ (dB)")
    ax.set_ylabel(r"OP")
    # ax.set_title(r"Probabilidade de interrupção Rayleigh")

    ax.legend()

    return fig, ax


def plot_rayleigh_outage(
    snr_mean, va,
    fig=plt.figure(),
    ax=None
):
    if ax is None:
        ax = fig.add_subplot()

    x = np.linspace(-30, 30, 200)
    op = 1 - np.exp(- 10 ** (0.1 * (x - snr_mean)))

    clr = {
        -20: "red",
        0: "green",
        20: "blue"
    }[snr_mean]
    ax.semilogy(
        x, op,
        linestyle="-.",
        color=clr,
        label=r"Teórica $\overline{\gamma}_s = " + f"{snr_mean}$ dB"
    )
    ax.scatter(
        x[::10], op[::10],
        marker='s',
        s=50,
        facecolors='none',
        edgecolors=clr,
        linewidths=1.5,
    )

    va_x, va_cdf = get_cdf(va ** 2 * 10 **(0.1*snr_mean))
    va_x = 10 * np.log10(va_x)

    label = None
    if snr_mean == 20:
        label = "Simulação"
    ax.plot(
        va_x, va_cdf,
        linestyle="--",
        color="black",
        label=label
    )

    ax.grid(True, alpha=0.2, color="gray")

    ax.set_ylim((1e-6, 1.5))
    ax.set_xlim((-30, 30))
    ax.set_xlabel(r"Limiar de SNR - $\gamma_\text{th}$ (dB)")
    ax.set_ylabel(r"OP")
    # ax.set_title(r"Probabilidade de interrupção Rayleigh")

    ax.legend()

    return fig, ax


def get_greycode(k: int):
    return np.array([i ^ (i >> 1) for i in range(1 << k)])


def get_binary_repr(code: list):
    k = int(np.log2(len(code)))
    labels = [format(x, f"0{k}b") for x in code]
    return labels


def get_qam_constellation(M: int):
    """
    Return normalized QAM constellation in a tuple of
        ([IQ_of_symbols], [bit_repr_of_symbols])
    """
    k_bits = np.log2(M)
    assert k_bits == int(k_bits), "M needs to be a power of 2"
    k_bits = int(k_bits)
    assert k_bits % 2 == 0, "Constellation should be 2^(2*k) | k >= 1"

    i = np.arange(1, np.sqrt(M) + 1)
    # print("i", i)
    # normalizing term:
    d = np.sqrt(3 / 2 / (M - 1))
    # d = 1
    I = (2 * i - 1 - np.sqrt(M)) * d
    Q = I

    # permutation of all values, following column first repr
    I_grid, Q_grid = np.meshgrid(I, Q)
    # add axis at the end for I/Q values
    symbols_repr = np.stack([I_grid, Q_grid], axis=-1)
    # remove other dimensions
    symbols_repr = symbols_repr.reshape(-1, 2)

    # create greycode for both axis and join
    # them so that greycode is always followed
    I_greycode = get_greycode(k_bits // 2)
    Q_greycode = get_greycode(k_bits // 2) << k_bits // 2
    Igq, Qgq = np.meshgrid(I_greycode, Q_greycode)
    # print("Igq, Qgq", Igq, Qgq )
    greycode = (Igq + Qgq).reshape(-1)

    bits_repr = get_binary_repr(greycode)

    return (symbols_repr, bits_repr)

def plot_constellation(symbols, bits_repr):
    fig = plt.figure()
    ax = fig.add_subplot()

    ax.scatter(
        symbols[:, 0],
        symbols[:, 1],
        # facecolors='k',
        # edgecolors='k',
        color='k',
        linewidths=1.5,
    )

    mx = np.max(symbols)
    mn = np.min(symbols)

    pad = (mx - mn) / 10
    # place bit labels on the constellation
    for i, (x, y) in enumerate(symbols):
        ax.text(
            x,
            y+pad/2,
            bits_repr[i],
            fontsize=10,
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8, lw=0)
        )

    ax.grid(True, alpha=0.2, color="gray")

    ax.set_ylim((mn - pad, mx + pad))
    ax.set_xlim((mn - pad, mx + pad))
    # ax.set_ylim((-2, 2))
    # ax.set_xlim((-2, 2))
    # ax.set_xlabel(r"Envoltória do Canal - $\beta$")
    # ax.set_ylabel(r"FDA - $p_\beta(\beta)$")
    # ax.set_title(rf"$\Omega_c = {Omega_c}$, $K_R = {K_r}$, $N  = 10^{int(np.log10(len(va)))}$, $M = {M}$")

    ax.legend(loc="lower right")
    return fig

def plot_awgn_sep(
    M,
    fig=None,
    ax=None
):
    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot()

    mean_snr = np.linspace(-30, 30, 200)
    mean_snr_lin = 10 ** (0.1 * mean_snr)
    a = (1 - 1/np.sqrt(M))
    b = 1 - scipy.stats.norm.cdf(np.sqrt(3 * mean_snr_lin / (M-1)))

    Pe = 4 * a * b - 4 * a**2 * b**2

    clr = {
        4: "red",
        16: "green",
        64: "blue"
    }[M]
    ax.semilogy(mean_snr, Pe, color=clr, label=f"Teórica - $M={M}$")

    ax.set_xlabel(r"SNR média por símbolo - $\overline{\gamma}_s$")
    ax.set_ylabel(r"Probabilidade de erro")
    # ax.set_title(rf"$\Omega_c = {Omega_c}$, $N  = 10^{int(np.log10(len(va)))}$")
    ax.set_ylim((1e-1, 1.1))
    ax.set_xlim((-30, 30))
    ax.grid(True, alpha=0.2, color="gray")

    ax.legend()

    return fig, ax

def plot_rayleigh_sep(
    M,
    fig=None,
    ax=None
):
    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot()

    mean_snr = np.linspace(-30, 30, 200)
    mean_snr_lin = 10 ** (0.1 * mean_snr)
    a = (1 - 1/np.sqrt(M))
    Cm = np.sqrt(1.5 * mean_snr_lin / (M - 1 + 1.5*mean_snr_lin))
    b = 1 - Cm
    c = 1 - 4 * Cm * np.arctan2(1, Cm) / np.pi

    Pe = 2 * a * b - a**2 * c

    clr = {
        4: "red",
        16: "green",
        64: "blue"
    }[M]
    ax.semilogy(mean_snr, Pe, color=clr, label=f"Teórica - $M={M}$")

    ax.set_xlabel(r"SNR média por símbolo - $\overline{\gamma}_s$")
    ax.set_ylabel(r"Probabilidade de erro")
    # ax.set_title(rf"$\Omega_c = {Omega_c}$, $N  = 10^{int(np.log10(len(va)))}$")
    ax.set_ylim((1e-1, 1.1))
    ax.set_xlim((-30, 30))
    ax.grid(True, alpha=0.2, color="gray")

    ax.legend()

    return fig, ax


if __name__ == "__main__":
    n_tx_symbols = int(1e7)

    rng = np.random.default_rng(1)
    # P_tx = 1
    # Eb = 1
    # N0 = 1

    # mean_snr = P_tx * Omega_c * Eb / N0

    ################################################
    # OP Rayleigh
    # Omega_c = 1
    # va = gen_rayleigh_va(rng, Omega_c, n_tx_symbols)

    # fig, ax = plot_rayleigh_outage(
    #     -20, va
    # )
    # plot_rayleigh_outage(
    #     0, va, fig, ax
    # )
    # plot_rayleigh_outage(
    #     20, va, fig, ax
    # )
    # ax.legend(loc="lower right")
    # fig.legend(loc="lower right")
    # fig.show()
    # fig.savefig(f"figs/ray-1.png")



    ################################################
    # OP Rice
    # # Omega_c, K_r 
    # Omega_c = 1
    # for K_r in [0.1, 1, 10]:
    #     va = gen_rice_va(rng, Omega_c, K_r, n_tx_symbols)

    #     fig, ax = plot_rice_outage(
    #         -20, va, K_r
    #     )
    #     plot_rice_outage(
    #         0, va, K_r, fig, ax
    #     )
    #     plot_rice_outage(
    #         20, va, K_r, fig, ax
    #     )
    #     ax.legend(loc="lower right")
    #     fig.savefig(f"figs/rice-op-{K_r}.png")
    #     plt.close()


    # constellation order
    AWGN = False
    fig, ax = None, None
    n_tx_symbols = int(1e6)
    for M in [4, 16, 64]:
        print("starting for ", M)
        constellation_sym, constellation_bit_repr = get_qam_constellation(M)
        # mean_energy = np.sum(constellation_sym * constellation_sym) / M
        # print("mean_energy", mean_energy)
        # print("2/3 * (M -1)", 2/3 * (M -1))
        # plot_constellation(constellation_sym, constellation_bit_repr)
        mean_snrs = np.arange(-5, 30, 5)
        bers = []
        for mean_snr in mean_snrs:
            tx_syms_i = rng.choice(
                list(range(len(constellation_sym))), n_tx_symbols
            )
            tx_syms = constellation_sym[tx_syms_i]
            # print(tx_syms.shape)

            noise = np.sqrt(1/(2 * 10**(0.1*mean_snr))) * rng.normal(0, 1, tx_syms.shape)

            if AWGN:
                beta = np.ones(n_tx_symbols)
            else:
                Ohmega_c = 1
                beta = gen_rayleigh_va(rng, Ohmega_c, n_tx_symbols)
            beta = beta.reshape(-1, 1)

            rx_syms = beta * tx_syms + noise

            # (N, M, 2)
            scaled_const = beta[:, np.newaxis] * constellation_sym[np.newaxis, :, :]

            # better np.linalg.norm
            # (N, M, 2)
            diff = rx_syms[:, np.newaxis, :] - scaled_const
            # (N, M)
            dists = np.sum(diff**2, axis=-1)

            # index of nearest symbol
            rx_syms_i = np.argmin(dists, axis=1)

            n_errors = np.sum(rx_syms_i != tx_syms_i)

            bers.append(n_errors / n_tx_symbols)
        if AWGN:
            fig, ax = plot_awgn_sep(M, fig, ax)
        else:
            fig, ax = plot_rayleigh_sep(M, fig, ax)

        label = None
        if M == 64:
            label = "Simulação"
        ax.scatter(
            mean_snrs, bers,
            marker='x',
            s=75,
            facecolors='black',
            label=label
        )
    name = "awgn" if AWGN else "rayleigh"
    fig.savefig(f"figs/sep-{name}2.png")
    plt.close()
    # plt.show()
