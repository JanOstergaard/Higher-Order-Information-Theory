"""Example: Gaussian HCI, CI, ECI and the paper's synergy lower bound.

Requirements: numpy, scipy, matplotlib, idtxl (+ Java/JIDT), and hci_v3.py.
Run: python example_gaussian_hci.py

IDTxl/JIDT information estimates are in nats; this example converts to bits.
Set HCI_OUTPUT_UNIT to 'bits' if your hci_v3 implementation already returns bits.
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from idtxl.estimators_jidt import JidtKraskovMI, JidtKraskovCMI
from hci_v3 import compute_R2_R3

# Settings
N = 3000
RHO_VALUES = [-0.45, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9]
SEED = 42
KNN = 6
REPEATS = 40
MAX_ITR = 20
THRESHOLD = 0.1
AGGREGATION = "median"
HCI_OUTPUT_UNIT = "nats"  # Check hci_v3.py; set to "bits" if appropriate.


def gaussian_sources(n, rho, rng):
    """Generate n samples of three unit-variance equicorrelated Gaussians.

    The covariance has off-diagonal entries rho. It is positive definite
    for -1/2 < rho < 1; we exclude singular endpoints for MI estimation.
    """
    if not -0.5 < rho < 1.0:
        raise ValueError("For three nonsingular equicorrelated sources, -0.5 < rho < 1.")
    covariance = np.full((3, 3), rho, dtype=float)
    np.fill_diagonal(covariance, 1.0)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    transform = eigenvectors @ np.diag(np.sqrt(eigenvalues))
    independent = rng.standard_normal((n, 3))
    return independent @ transform.T


def zscore_columns(x):
    scale = x.std(axis=0, keepdims=True)
    if np.any(scale == 0):
        raise ValueError("Cannot standardize a constant source.")
    return (x - x.mean(axis=0, keepdims=True)) / scale


def gaussian_theory(rho):
    """Return theoretical (HCI, CI, ECI, synergy lower bound), in bits.

    HCI: paper Eq. (18); CI: paper Eq. (17); ECI: Eq. (6).
    Synergy lower bound: Section V, page 4 of the supplied paper.
    """
    hci = -0.5 * np.log2(1.0 - rho**4)
    mi = -0.5 * np.log2(1.0 - rho**2)
    partial_corr = rho / (1.0 + rho)
    cmi = -0.5 * np.log2(1.0 - partial_corr**2)
    ci = mi - cmi
    eci = max(ci - hci, 0.0)
    syn_lb = hci - ci if ci < 0.0 else 0.0
    return hci, ci, eci, syn_lb


def estimate_ci_bits(x, mi_estimator, cmi_estimator):
    """Average all six equivalent orientations to reduce estimator asymmetry."""
    values = []
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            k = 3 - i - j
            a = np.ascontiguousarray(x[:, i].reshape(-1, 1))
            b = np.ascontiguousarray(x[:, j].reshape(-1, 1))
            c = np.ascontiguousarray(x[:, k].reshape(-1, 1))
            mi_nats = mi_estimator.estimate(a, b)
            cmi_nats = cmi_estimator.estimate(a, b, c)
            values.append((mi_nats - cmi_nats) / np.log(2.0))
    return float(np.mean(values))


def main():
    rng = np.random.default_rng(SEED)
    mi_estimator = JidtKraskovMI({"kraskov_k": KNN})
    cmi_estimator = JidtKraskovCMI({"kraskov_k": KNN})
    estimated, theoretical = [], []

    print("All reported information quantities are in bits.")
    print("rho   HCI(est) HCI(th)   CI(est)  CI(th)   ECI(est) ECI(th)  SYN_LB(est) SYN_LB(th)")
    print("-" * 99)

    for rho in RHO_VALUES:
        x = zscore_columns(gaussian_sources(N, rho, rng))
        empirical_corr = np.corrcoef(x, rowvar=False)
        empirical_rho = np.mean(empirical_corr[np.triu_indices(3, 1)])
        print(f"  requested rho={rho:+.2f}; empirical mean pairwise rho={empirical_rho:+.3f}")
        r2, r3 = compute_R2_R3(
            x, target_idx=2, x1_idx=0, x2_idx=1,
            knn=KNN, repeats=REPEATS, aggregation=AGGREGATION,
            max_itr=MAX_ITR, threshold=THRESHOLD,
        )
        # r2 is returned for reference; r3 is the HCI estimate.
        hci = float(r3)
        if HCI_OUTPUT_UNIT == "nats":
            hci /= np.log(2.0)
        elif HCI_OUTPUT_UNIT != "bits":
            raise ValueError("HCI_OUTPUT_UNIT must be 'nats' or 'bits'.")

        ci = estimate_ci_bits(x, mi_estimator, cmi_estimator)
        eci = max(ci - hci, 0.0)
        # Section V synergy lower bound: zero for CI >= 0, HCI-CI otherwise.
        syn_lb = hci - ci if ci < 0.0 else 0.0
        estimated.append((hci, ci, eci, syn_lb))
        theoretical.append(gaussian_theory(rho))
        ht, ct, et, st = theoretical[-1]
        print(f"{rho:+.2f}  {hci:8.4f} {ht:7.4f}  {ci:8.4f} {ct:7.4f}  {eci:8.4f} {et:7.4f}  {syn_lb:11.4f} {st:10.4f}")

    estimated = np.asarray(estimated)
    theoretical = np.asarray(theoretical)
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["HCI", "CI", "ECI", "Synergy lower bound"]
    for j, label in enumerate(labels):
        line, = ax.plot(RHO_VALUES, theoretical[:, j], "-", label=f"{label} (theory)")
        ax.plot(RHO_VALUES, estimated[:, j], "o", color=line.get_color(),
                label=f"{label} (estimated)")
    ax.set(xlabel="Pairwise correlation coefficient rho", ylabel="Information (bits)",
           title="Equicorrelated Gaussian sources")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig("gaussian_hci_example.png", dpi=180)

    # Save values for further analysis or plotting.
    columns = ["rho", "hci_est", "ci_est", "eci_est", "syn_lb_est",
               "hci_theory", "ci_theory", "eci_theory", "syn_lb_theory"]
    with open("gaussian_hci_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for rho, est, th in zip(RHO_VALUES, estimated, theoretical):
            writer.writerow([rho, *est, *th])
    print("Saved gaussian_hci_results.csv")
    print("\nSaved gaussian_hci_example.png")
    plt.show()


if __name__ == "__main__":
    main()
