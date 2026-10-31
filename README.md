# Higher-Order-Information-Theory

**Version:** 1.0.0

Python implementations of Higher Order Common Information, Excess Co-information, Co-information, and Synergy.

This Python version is an AI conversion of the original Matlab implementation made by Jan Østergaard. 

It relies upon the (conditional) mutual information estimator IDXTL: https://pwollstadt.github.io/IDTxl/html/index.html

Use the example code for a demonstration of how to use it.

## Per-signal time delays

`compute_R2_R3`, `R3_approx_mi_fast`, and `R4_approx_mi_fast` accept
`delays=None` (default) or one integer sample delay per input column.
Existing calls keep their zero-lag behavior.

```python
import numpy as np
from hci_v3 import compute_R2_R3

# All signals must have the same sampling rate and original time origin.
X = np.column_stack([audio, eeg_left, eeg_right])
r2, r3 = compute_R2_R3(
    X, target_idx=0, x1_idx=1, x2_idx=2,
    delays=[14, 0, 0],  # audio[t] paired with both EEG signals at t+14
    knn=6, repeats=10, aggregation="median",
)
```

The convention is `aligned[t, c] = X[t - delays[c], c]`: positive delays
delay a signal; negative delays advance it. Thus `[14, 0, 0]` and
`[0, -14, -14]` are equivalent. At 64 Hz, 14 samples correspond to
218.75 ms. To pair EEG with earlier audio, delay the audio or advance the EEG.
For independently chosen EEG latencies, use e.g. `[0, -10, -14]`.
Convert milliseconds to samples explicitly, for example
`lag_samples = int(round(lag_ms * sfreq / 1000))`.

All columns are trimmed to one common overlap before any MI estimates or
auxiliaries are computed. No samples wrap around and no zeros are inserted.
The retained sample count is `N - (max(delays) - min(delays))`; unused columns
also participate in this trimming. Delays are indexed by the original columns,
not by the order of the `i, j, k, l` arguments. Delays leaving no overlap raise
`ValueError`; very short overlaps remain subject to the estimator's sample
requirements. Apply this separately within each trial to avoid crossing trial
boundaries. Different delays can give different sample counts in a lag sweep.

For consistent alignment of externally computed CI or synergy, use
`aligned = align_signals(X, delays)` and pass `aligned` to all measures with
their delay options omitted. Do not apply the same delays twice.

The estimator still computes MI between aligned observations. This option
changes their relative time alignment; it does not add temporal embeddings or
compute transfer entropy. Auxiliaries inherit the aligned source's time
reference, so their MI matching constraint is evaluated without another delay.

Run the alignment and integration tests with:
`python -m unittest discover -s tests -v`.
The integration tests use a deterministic Gaussian MI stand-in to check the
alignment plumbing without requiring IDTxl or a JVM.

For the mathematical details about the HCI and ECI please refer to the technical papers. 
If you use the toolbox in your research then please cite the following document: 

J. Ostergaard, "Higher-order Common Information". Available at https://arxiv.org/abs/2406.02001.

@misc{ostergaard2026higherordercommoninformation,
      title={Higher-order Common Information}, 
      author="Jan {\O}stergaard",
      year={2026},
      eprint={2406.02001},
      archivePrefix={arXiv},
      primaryClass={cs.IT},
      url={https://arxiv.org/abs/2406.02001}, 
}

J. Ostergaard, "Excess Co-information: A Non-Local Redundancy Measure". Submitted.



## HCI Estimation Parameters

The following parameters control the numerical estimation of Higher-Order Common Information (HCI):

```python
repeats = 10
aggregation = "median"
max_itr = 10
threshold = 0.1
```

### 1. `repeats = 10`

For each ordering of the sources, the algorithm constructs 10 independent realizations of the auxiliary variable:

```text
T = X_i + s * Z
```

where:

- `X_i` is the reference source.
- `Z` is standard Gaussian noise.
- `s` is the noise amplitude, determined numerically by the algorithm.

Each realization produces an estimate of the mutual information `I(T; X_k)`.

Increasing `repeats` provides more realizations over which to aggregate, but also increases computation time.

**Recommended value:** Between 10 and 20 for typical applications.

### 2. `aggregation = "median"`

This parameter specifies how the estimates obtained from the repeated realizations are combined.

The implementation supports three alternatives:

- `"mean"`: Average of all estimates.
- `"median"`: Median of all estimates, providing greater robustness to outliers.
- `"max"`: Maximum estimated value.

**Recommended value:** `"median"` for robust estimation.

### 3. `max_itr = 10`

This parameter controls the maximum number of iterations in the bisection search used to determine the Gaussian noise amplitude `s`.

The algorithm seeks an auxiliary variable:

```text
T = X_i + s * Z
```

that approximately satisfies the mutual information constraint:

```text
I(T; X_i) = I(X_i; X_j)
```

The search starts with the interval:

```text
s in [1e-8, 10]
```

The noise amplitude `s` is iteratively adjusted to match the two mutual information quantities.

With `max_itr = 10`, at most 10 bisection iterations are performed per realization.

Increasing `max_itr` allows more accurate matching of the mutual information constraint, at the cost of additional computation.

### 4. `threshold = 0.1`

This parameter specifies the relative tolerance for matching the mutual information constraint.

The bisection search stops when:

```text
abs(I(T; X_i) - I(X_i; X_j))
-------------------------------- < threshold
          abs(I(X_i; X_j))
```

Thus, `threshold = 0.1` permits a relative mismatch of 10%.

**Example:**

Suppose the mutual information between two sources is:

```text
I(X_i; X_j) = 0.2 bits
```

With `threshold = 0.1`, the algorithm stops when:

```text
abs(I(T; X_i) - 0.2) < 0.02 bits
```

Equivalently, the estimated mutual information must satisfy:

```text
0.18 < I(T; X_i) < 0.22 bits
```

The search also stops when `max_itr` is reached, even if the specified tolerance has not been satisfied.

**Recommended value:** `0.1` for initial experiments. Smaller values, such as `0.01`, can be used when more accurate matching is required, preferably together with a larger `max_itr`.

### Suggested Settings

| Parameter | Fast testing | More accurate estimation |
|---|---|---|
| `repeats` | 10 | 20–40 |
| `aggregation` | `"median"` | `"median"` |
| `max_itr` | 10 | 25 |
| `threshold` | 0.1 | 0.01 |

Increasing the number of repetitions or iterations does not necessarily guarantee more accurate HCI estimates. The results also depend on the accuracy of the underlying mutual information estimator and the available sample size.
