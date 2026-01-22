# risk_qae examples

These scripts are intended as **template reference entrypoints**. They demonstrate:
- low-qubit discretization of a 1D loss distribution
- end-to-end estimation of **mean / VaR / TVaR**
- switching AE modes:
  - `budgeted_fixed_schedule` (shots-first sampling baseline)
  - `grover_mle` (Grover powers + maximum-likelihood fit; QFT-free)

## Install

From the repo root:

```bash
pip install -e ".[quantum,aer]"
```

## Run

```bash
python examples/01_synthetic_gamma_grover_mle.py
python examples/02_synthetic_mixture_tailstress_compare_ae_methods.py
# later:
python examples/03_real_dataset_template.py --path /path/to/your/data.csv --column loss
```

## Notes

- VaR is reported as the **upper edge** of the selected discretized bin for conservatism.
  Because of this discretization convention, the relationship `TVaR >= VaR` is not guaranteed.
  A discretization-consistent sanity check is:

```python
assert tvar >= var_bracket_low
```

- In Grover-MLE mode you can optionally enable early stopping:
  - `epsilon_target`: stop when the half-width of the CI drops below this threshold
  - `alpha_confidence`: confidence level for the CI (e.g., 0.05 = 95% CI)
