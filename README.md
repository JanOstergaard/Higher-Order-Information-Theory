# Higher-Order-Information-Theory

**Version:** 1.0.0

Python implementations of Higher Order Common Information, Excess Co-information, Co-information, and Synergy.

This Python version is an AI conversion of the original Matlab implementation made by Jan Østergaard. 

It relies upon the (conditional) mutual information estimator IDXTL: https://pwollstadt.github.io/IDTxl/html/index.html

Use the example code for a demonstration of how to use it.

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




