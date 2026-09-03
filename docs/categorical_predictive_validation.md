# Categorical CVAE Phase A predictive validation

Protocol: `categorical-predictive-validation-v1`. Device: `cpu`.

Protocol/scenario manifest SHA-256: `722db9f185e3bacbcee6aec4d203d7f36d12915da0263003ab542c9d5ee4ba50`.

| environment field | value |
|---|---|
| package revision | `0.3.0` |
| imported package path | `python/multioutcome_cvae/__init__.py` |
| trainer source path | `python/multioutcome_cvae/model.py` |
| trainer source SHA-256 | `709304247799912f97b2415426c36d5ed31601534676b3f1334fb88a1777ccdd` |
| validation source path | `validation/categorical_predictive_validation.py` |
| validation source SHA-256 | `760534dee50b267a740bc67a87792af65a0aea418aa9a196543e493dc34628a0` |
| imported trainer matches checkout | `true` |
| git HEAD | `6adea6c74d12ec723f5f2a94495d1cff1408e32d` |
| git worktree dirty | `false` |
| Python | `3.11.15` |
| NumPy | `2.4.6` |
| PyTorch | `2.14.0` |
| platform | `macOS-26.6.2-arm64-arm-64bit` |

The scenario classes, primary score, and core acceptance criteria were posted before the first pilot fit. Exact coefficients, seeds, and model settings were fixed in the working source; they first become independently auditable in the committed manifest embedded below. A one-seed pilot was retained as an unfavorable result in the issue history. Before this canonical full run, code review corrected two design mechanics: model selection begins at the end of KL warm-up, and same-seed scenarios are clustered before seed-level t bounds. Neither correction changed a DGP strength or a predictive acceptance threshold. Gate failures remain visible and never abort report generation.

This is bounded engineering evidence for the four model-aligned core scenarios. It is not general proof of probabilistic validity, does not justify a production claim, and does not by itself close GitHub Issue #2.

## Predeclared protocol

Each scenario uses 4000 training rows, 1000 validation-only rows, and 5000 untouched test rows. Seeds: `1701, 9901, 31415`.

Each listed seed determines an independently generated train/validation/test replicate and separate derived initialization/training streams; no replicate or initialization is selected after fitting.

The fitted CVAE primary scores integrate its two-dimensional standard-normal prior with 31x31 tensor Gauss-Hermite nodes. Monte Carlo estimates are secondary deployment diagnostics only.

Fixed model settings:

| latent dim | hidden width | hidden layers | epochs | batch | learning rate | beta KL | warmup | selection starts | patience |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 48 | 2 | 30 | 128 | 0.001 | 0.2 | 8 | 8 | 6 |

Predeclared gates:

| gate | kind | applies to | criterion |
|---|---|---|---|
| `dependent_every_seed_positive` | `predictive` | each dependent DGP | minimum seed-level G > 0, where G = NLL(fitted marginal product) - NLL(shared) = log score(shared) - log score(product) |
| `dependent_aggregate_lcb` | `predictive` | all dependent DGP seeds | one-sided 95% LCB across top-level seeds, after equal-weighting dependent scenarios within seed, is G > 0.01 nats/vector |
| `conditional_independence_penalty_ucb` | `predictive` | conditional-independence DGP | one-sided 95% UCB across top-level seeds, after equal-weighting CI scenarios within seed, is P < 0.01 nats/vector, where P = -G = NLL(shared) - NLL(fitted marginal product) |
| `conditional_independence_each_seed` | `predictive` | conditional-independence DGP | maximum individual-seed conditional-independence penalty P < 0.02 |
| `dependent_oracle_headroom` | `prerequisite` | each dependent DGP | minimum seed-level oracle headroom H >= 0.05 nats/vector, where H = NLL(oracle marginal product) - NLL(oracle shared) |
| `marginal_log_score_lcb` | `predictive` | each DGP | one-sided 95% LCB for CVAE minus intercept-null marginal log score > 0 |
| `marginal_brier_lcb` | `predictive` | each DGP | one-sided 95% LCB for intercept-null minus CVAE marginal Brier score > 0 |
| `quadrature_gain_mean_stability` | `prerequisite` | every scenario seed on the fixed numerical subset | absolute change in mean G from GH-31 to GH-41 <= 0.002 |
| `quadrature_p99_stability` | `prerequisite` | every scenario seed on the fixed numerical subset | maximum p99 absolute per-row shared-score or G change from GH-31 to GH-41 <= 0.01 |
| `label_permutation_invariance` | `prerequisite` | CVAE, intercept-null, and oracle probabilities for every seed | maximum fixed-probability relabeling discrepancy <= 1e-10 |

Scenarios:

| scenario | cardinalities | conditional structure |
|---|---|---|
| `mixed_dependent` | 2/3/5 | shared latent dependence; Three 2/3/5-level outcomes with shared two-dimensional latent causes and observed covariate effects. |
| `mixed_conditional_independence` | 2/3/5 | independent given X; The same 2/3/5-level marginal regressions with strong shared X causes but zero latent effects; outcomes are independent given X. |
| `all_binary_dependent` | 2/2/2/2/2 | shared latent dependence; A vector of five two-level outcomes with shared two-dimensional latent causes, exercising the binary special case. |
| `all_binary_conditional_independence` | 2/2/2/2/2 | independent given X; Five two-level outcomes with the same strong shared X causes as the binary dependent DGP but zero latent effects. |

Exact machine-readable protocol manifest:

```json
{
  "config": {
    "batch_size": 128,
    "beta_kl": 0.2,
    "calibration_bins": 10,
    "early_stopping_min_delta": 0.0001,
    "early_stopping_patience": 6,
    "early_stopping_start_epoch": 8,
    "empirical_joint_alpha": 0.5,
    "fitted_quadrature_checks": [
      21,
      41
    ],
    "fitted_quadrature_order": 31,
    "hidden_dim": 48,
    "kl_warmup_epochs": 8,
    "latent_dim": 2,
    "learning_rate": 0.001,
    "mc_blocks": 8,
    "mc_sizes": [
      16,
      64,
      256,
      1024
    ],
    "mc_test_rows": 96,
    "n_hidden_layers": 2,
    "n_test": 5000,
    "n_train": 4000,
    "n_validation": 1000,
    "num_epochs": 30,
    "oracle_quadrature_order": 41,
    "oracle_x_batch_size": 256,
    "probability_epsilon": 1e-12,
    "quadrature_test_rows": 128,
    "quadrature_x_batch_size": 24,
    "seeds": [
      1701,
      9901,
      31415
    ]
  },
  "gates": {
    "conditional_independence_each_seed": {
      "applies_to": "conditional-independence DGP",
      "criterion": "maximum individual-seed conditional-independence penalty P < 0.02",
      "direction": "less",
      "kind": "predictive",
      "threshold": 0.02
    },
    "conditional_independence_penalty_ucb": {
      "applies_to": "conditional-independence DGP",
      "criterion": "one-sided 95% UCB across top-level seeds, after equal-weighting CI scenarios within seed, is P < 0.01 nats/vector, where P = -G = NLL(shared) - NLL(fitted marginal product)",
      "direction": "less",
      "kind": "predictive",
      "threshold": 0.01
    },
    "dependent_aggregate_lcb": {
      "applies_to": "all dependent DGP seeds",
      "criterion": "one-sided 95% LCB across top-level seeds, after equal-weighting dependent scenarios within seed, is G > 0.01 nats/vector",
      "direction": "greater",
      "kind": "predictive",
      "threshold": 0.01
    },
    "dependent_every_seed_positive": {
      "applies_to": "each dependent DGP",
      "criterion": "minimum seed-level G > 0, where G = NLL(fitted marginal product) - NLL(shared) = log score(shared) - log score(product)",
      "direction": "greater",
      "kind": "predictive",
      "threshold": 0.0
    },
    "dependent_oracle_headroom": {
      "applies_to": "each dependent DGP",
      "criterion": "minimum seed-level oracle headroom H >= 0.05 nats/vector, where H = NLL(oracle marginal product) - NLL(oracle shared)",
      "direction": "greater_or_equal",
      "kind": "prerequisite",
      "threshold": 0.05
    },
    "label_permutation_invariance": {
      "applies_to": "CVAE, intercept-null, and oracle probabilities for every seed",
      "criterion": "maximum fixed-probability relabeling discrepancy <= 1e-10",
      "direction": "less_or_equal",
      "kind": "prerequisite",
      "threshold": 1e-10
    },
    "marginal_brier_lcb": {
      "applies_to": "each DGP",
      "criterion": "one-sided 95% LCB for intercept-null minus CVAE marginal Brier score > 0",
      "direction": "greater",
      "kind": "predictive",
      "threshold": 0.0
    },
    "marginal_log_score_lcb": {
      "applies_to": "each DGP",
      "criterion": "one-sided 95% LCB for CVAE minus intercept-null marginal log score > 0",
      "direction": "greater",
      "kind": "predictive",
      "threshold": 0.0
    },
    "quadrature_gain_mean_stability": {
      "applies_to": "every scenario seed on the fixed numerical subset",
      "criterion": "absolute change in mean G from GH-31 to GH-41 <= 0.002",
      "direction": "less_or_equal",
      "kind": "prerequisite",
      "threshold": 0.002
    },
    "quadrature_p99_stability": {
      "applies_to": "every scenario seed on the fixed numerical subset",
      "criterion": "maximum p99 absolute per-row shared-score or G change from GH-31 to GH-41 <= 0.01",
      "direction": "less_or_equal",
      "kind": "prerequisite",
      "threshold": 0.01
    }
  },
  "primary_estimands": {
    "conditional_independence_penalty_P": "-joint_gain_G",
    "integration": "deterministic tensor Gauss-Hermite under the two-dimensional standard-normal prior",
    "joint_gain_G": "NLL(product of fitted CVAE marginals) - NLL(shared-latent CVAE joint)",
    "marginal_brier": "mean semantic-outcome multiclass Brier score",
    "marginal_log_score": "mean semantic-outcome log predictive probability"
  },
  "protocol": "categorical-predictive-validation-v1",
  "scenarios": {
    "all_binary_conditional_independence": {
      "conditionally_independent": true,
      "description": "Five two-level outcomes with the same strong shared X causes as the binary dependent DGP but zero latent effects.",
      "latent_dim": 2,
      "name": "all_binary_conditional_independence",
      "outcomes": [
        {
          "intercepts": [
            0.0,
            0.1
          ],
          "latent_weights": [
            [
              0.0,
              0.0
            ],
            [
              0.0,
              0.0
            ]
          ],
          "levels": [
            "0",
            "1"
          ],
          "name": "signal_a",
          "x_weights": [
            [
              0.0,
              0.7
            ],
            [
              0.0,
              -0.45
            ],
            [
              0.0,
              0.25
            ]
          ]
        },
        {
          "intercepts": [
            0.0,
            -0.15
          ],
          "latent_weights": [
            [
              0.0,
              0.0
            ],
            [
              0.0,
              0.0
            ]
          ],
          "levels": [
            "0",
            "1"
          ],
          "name": "signal_b",
          "x_weights": [
            [
              0.0,
              -0.55
            ],
            [
              0.0,
              0.65
            ],
            [
              0.0,
              0.35
            ]
          ]
        },
        {
          "intercepts": [
            0.0,
            0.05
          ],
          "latent_weights": [
            [
              0.0,
              0.0
            ],
            [
              0.0,
              0.0
            ]
          ],
          "levels": [
            "0",
            "1"
          ],
          "name": "signal_c",
          "x_weights": [
            [
              0.0,
              0.4
            ],
            [
              0.0,
              0.3
            ],
            [
              0.0,
              -0.75
            ]
          ]
        },
        {
          "intercepts": [
            0.0,
            -0.05
          ],
          "latent_weights": [
            [
              0.0,
              0.0
            ],
            [
              0.0,
              0.0
            ]
          ],
          "levels": [
            "0",
            "1"
          ],
          "name": "signal_d",
          "x_weights": [
            [
              0.0,
              -0.35
            ],
            [
              0.0,
              -0.6
            ],
            [
              0.0,
              0.55
            ]
          ]
        },
        {
          "intercepts": [
            0.0,
            0.2
          ],
          "latent_weights": [
            [
              0.0,
              0.0
            ],
            [
              0.0,
              0.0
            ]
          ],
          "levels": [
            "0",
            "1"
          ],
          "name": "signal_e",
          "x_weights": [
            [
              0.0,
              0.5
            ],
            [
              0.0,
              0.4
            ],
            [
              0.0,
              -0.3
            ]
          ]
        }
      ],
      "x_dim": 3
    },
    "all_binary_dependent": {
      "conditionally_independent": false,
      "description": "A vector of five two-level outcomes with shared two-dimensional latent causes, exercising the binary special case.",
      "latent_dim": 2,
      "name": "all_binary_dependent",
      "outcomes": [
        {
          "intercepts": [
            0.0,
            0.1
          ],
          "latent_weights": [
            [
              -1.0,
              1.0
            ],
            [
              -0.45,
              0.45
            ]
          ],
          "levels": [
            "0",
            "1"
          ],
          "name": "signal_a",
          "x_weights": [
            [
              0.0,
              0.7
            ],
            [
              0.0,
              -0.45
            ],
            [
              0.0,
              0.25
            ]
          ]
        },
        {
          "intercepts": [
            0.0,
            -0.15
          ],
          "latent_weights": [
            [
              -0.85,
              0.85
            ],
            [
              0.65,
              -0.65
            ]
          ],
          "levels": [
            "0",
            "1"
          ],
          "name": "signal_b",
          "x_weights": [
            [
              0.0,
              -0.55
            ],
            [
              0.0,
              0.65
            ],
            [
              0.0,
              0.35
            ]
          ]
        },
        {
          "intercepts": [
            0.0,
            0.05
          ],
          "latent_weights": [
            [
              0.9,
              -0.9
            ],
            [
              -0.7,
              0.7
            ]
          ],
          "levels": [
            "0",
            "1"
          ],
          "name": "signal_c",
          "x_weights": [
            [
              0.0,
              0.4
            ],
            [
              0.0,
              0.3
            ],
            [
              0.0,
              -0.75
            ]
          ]
        },
        {
          "intercepts": [
            0.0,
            -0.05
          ],
          "latent_weights": [
            [
              -0.7,
              0.7
            ],
            [
              -0.95,
              0.95
            ]
          ],
          "levels": [
            "0",
            "1"
          ],
          "name": "signal_d",
          "x_weights": [
            [
              0.0,
              -0.35
            ],
            [
              0.0,
              -0.6
            ],
            [
              0.0,
              0.55
            ]
          ]
        },
        {
          "intercepts": [
            0.0,
            0.2
          ],
          "latent_weights": [
            [
              0.75,
              -0.75
            ],
            [
              -0.55,
              0.55
            ]
          ],
          "levels": [
            "0",
            "1"
          ],
          "name": "signal_e",
          "x_weights": [
            [
              0.0,
              0.5
            ],
            [
              0.0,
              0.4
            ],
            [
              0.0,
              -0.3
            ]
          ]
        }
      ],
      "x_dim": 3
    },
    "mixed_conditional_independence": {
      "conditionally_independent": true,
      "description": "The same 2/3/5-level marginal regressions with strong shared X causes but zero latent effects; outcomes are independent given X.",
      "latent_dim": 2,
      "name": "mixed_conditional_independence",
      "outcomes": [
        {
          "intercepts": [
            0.0,
            0.15
          ],
          "latent_weights": [
            [
              0.0,
              0.0
            ],
            [
              0.0,
              0.0
            ]
          ],
          "levels": [
            "off",
            "on"
          ],
          "name": "switch",
          "x_weights": [
            [
              0.0,
              0.85
            ],
            [
              0.0,
              -0.5
            ],
            [
              0.0,
              0.35
            ]
          ]
        },
        {
          "intercepts": [
            0.2,
            -0.1,
            0.0
          ],
          "latent_weights": [
            [
              0.0,
              0.0,
              0.0
            ],
            [
              0.0,
              0.0,
              0.0
            ]
          ],
          "levels": [
            "circle",
            "square",
            "triangle"
          ],
          "name": "shape",
          "x_weights": [
            [
              0.55,
              -0.35,
              0.1
            ],
            [
              -0.45,
              0.25,
              0.6
            ],
            [
              0.2,
              0.45,
              -0.5
            ]
          ]
        },
        {
          "intercepts": [
            0.1,
            -0.1,
            0.15,
            -0.05,
            0.0
          ],
          "latent_weights": [
            [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ]
          ],
          "levels": [
            "amber",
            "blue",
            "coral",
            "green",
            "violet"
          ],
          "name": "color",
          "x_weights": [
            [
              0.55,
              -0.25,
              0.2,
              -0.4,
              0.1
            ],
            [
              -0.3,
              0.5,
              -0.2,
              0.35,
              -0.1
            ],
            [
              0.15,
              -0.45,
              0.55,
              -0.1,
              0.25
            ]
          ]
        }
      ],
      "x_dim": 3
    },
    "mixed_dependent": {
      "conditionally_independent": false,
      "description": "Three 2/3/5-level outcomes with shared two-dimensional latent causes and observed covariate effects.",
      "latent_dim": 2,
      "name": "mixed_dependent",
      "outcomes": [
        {
          "intercepts": [
            0.0,
            0.15
          ],
          "latent_weights": [
            [
              -0.9,
              0.9
            ],
            [
              -0.45,
              0.45
            ]
          ],
          "levels": [
            "off",
            "on"
          ],
          "name": "switch",
          "x_weights": [
            [
              0.0,
              0.85
            ],
            [
              0.0,
              -0.5
            ],
            [
              0.0,
              0.35
            ]
          ]
        },
        {
          "intercepts": [
            0.2,
            -0.1,
            0.0
          ],
          "latent_weights": [
            [
              -1.0,
              0.1,
              0.9
            ],
            [
              -0.5,
              0.8,
              -0.3
            ]
          ],
          "levels": [
            "circle",
            "square",
            "triangle"
          ],
          "name": "shape",
          "x_weights": [
            [
              0.55,
              -0.35,
              0.1
            ],
            [
              -0.45,
              0.25,
              0.6
            ],
            [
              0.2,
              0.45,
              -0.5
            ]
          ]
        },
        {
          "intercepts": [
            0.1,
            -0.1,
            0.15,
            -0.05,
            0.0
          ],
          "latent_weights": [
            [
              -1.3,
              -0.6,
              0.0,
              0.6,
              1.3
            ],
            [
              0.7,
              -0.8,
              0.3,
              0.8,
              -0.7
            ]
          ],
          "levels": [
            "amber",
            "blue",
            "coral",
            "green",
            "violet"
          ],
          "name": "color",
          "x_weights": [
            [
              0.55,
              -0.25,
              0.2,
              -0.4,
              0.1
            ],
            [
              -0.3,
              0.5,
              -0.2,
              0.35,
              -0.1
            ],
            [
              0.15,
              -0.45,
              0.55,
              -0.1,
              0.25
            ]
          ]
        }
      ],
      "x_dim": 3
    }
  }
}
```

## Gate results

Phase A disposition: **INCONCLUSIVE**.

At least one numerical or design prerequisite failed; predictive gate outcomes are shown but cannot establish a Phase A pass.

Predictive gates: **10 passed / 4 failed / 14 total**. Prerequisites: **20 passed / 18 failed / 38 total**.

| status | kind | gate | scope | value | rule | detail |
|---|---|---|---|---:|---:|---|
| FAIL | `predictive` | `dependent_every_seed_positive` | mixed_dependent | -0.01195 | > 0.00000 | minimum of the fixed seed-level paired mean gains |
| PASS | `prerequisite` | `dependent_oracle_headroom` | mixed_dependent | 0.13860 | >= 0.05000 | minimum realized-test oracle H across fixed seeds |
| PASS | `predictive` | `marginal_log_score_lcb` | mixed_dependent | 0.05009 | > 0.00000 | one-sided LCB computed across fixed seed-level paired means |
| PASS | `prerequisite` | `quadrature_gain_mean_stability` | mixed_dependent, seed=1701 | 0.00043 | <= 0.00200 | fixed-subset absolute mean G difference, GH-31 versus GH-41 |
| FAIL | `prerequisite` | `quadrature_p99_stability` | mixed_dependent, seed=1701 | 0.04521 | <= 0.01000 | maximum of p99 shared-score and G absolute changes |
| PASS | `prerequisite` | `label_permutation_invariance` | mixed_dependent, seed=1701 | 0.00000 | <= 0.00000 | maximum across CVAE, intercept-null, and oracle checks |
| FAIL | `prerequisite` | `quadrature_gain_mean_stability` | mixed_dependent, seed=9901 | 0.00614 | <= 0.00200 | fixed-subset absolute mean G difference, GH-31 versus GH-41 |
| FAIL | `prerequisite` | `quadrature_p99_stability` | mixed_dependent, seed=9901 | 0.19063 | <= 0.01000 | maximum of p99 shared-score and G absolute changes |
| PASS | `prerequisite` | `label_permutation_invariance` | mixed_dependent, seed=9901 | 0.00000 | <= 0.00000 | maximum across CVAE, intercept-null, and oracle checks |
| PASS | `prerequisite` | `quadrature_gain_mean_stability` | mixed_dependent, seed=31415 | 0.00052 | <= 0.00200 | fixed-subset absolute mean G difference, GH-31 versus GH-41 |
| FAIL | `prerequisite` | `quadrature_p99_stability` | mixed_dependent, seed=31415 | 0.16343 | <= 0.01000 | maximum of p99 shared-score and G absolute changes |
| PASS | `prerequisite` | `label_permutation_invariance` | mixed_dependent, seed=31415 | 0.00000 | <= 0.00000 | maximum across CVAE, intercept-null, and oracle checks |
| PASS | `predictive` | `marginal_brier_lcb` | mixed_dependent | 0.03241 | > 0.00000 | one-sided LCB computed across fixed seed-level paired means |
| FAIL | `predictive` | `conditional_independence_each_seed` | mixed_conditional_independence | 0.04992 | < 0.02000 | maximum individual fixed-seed penalty P |
| PASS | `predictive` | `marginal_log_score_lcb` | mixed_conditional_independence | 0.07317 | > 0.00000 | one-sided LCB computed across fixed seed-level paired means |
| FAIL | `prerequisite` | `quadrature_gain_mean_stability` | mixed_conditional_independence, seed=1701 | 0.01292 | <= 0.00200 | fixed-subset absolute mean G difference, GH-31 versus GH-41 |
| FAIL | `prerequisite` | `quadrature_p99_stability` | mixed_conditional_independence, seed=1701 | 0.28848 | <= 0.01000 | maximum of p99 shared-score and G absolute changes |
| PASS | `prerequisite` | `label_permutation_invariance` | mixed_conditional_independence, seed=1701 | 0.00000 | <= 0.00000 | maximum across CVAE, intercept-null, and oracle checks |
| PASS | `prerequisite` | `quadrature_gain_mean_stability` | mixed_conditional_independence, seed=9901 | 0.00167 | <= 0.00200 | fixed-subset absolute mean G difference, GH-31 versus GH-41 |
| FAIL | `prerequisite` | `quadrature_p99_stability` | mixed_conditional_independence, seed=9901 | 0.16446 | <= 0.01000 | maximum of p99 shared-score and G absolute changes |
| PASS | `prerequisite` | `label_permutation_invariance` | mixed_conditional_independence, seed=9901 | 0.00000 | <= 0.00000 | maximum across CVAE, intercept-null, and oracle checks |
| FAIL | `prerequisite` | `quadrature_gain_mean_stability` | mixed_conditional_independence, seed=31415 | 0.00241 | <= 0.00200 | fixed-subset absolute mean G difference, GH-31 versus GH-41 |
| FAIL | `prerequisite` | `quadrature_p99_stability` | mixed_conditional_independence, seed=31415 | 0.51651 | <= 0.01000 | maximum of p99 shared-score and G absolute changes |
| PASS | `prerequisite` | `label_permutation_invariance` | mixed_conditional_independence, seed=31415 | 0.00000 | <= 0.00000 | maximum across CVAE, intercept-null, and oracle checks |
| PASS | `predictive` | `marginal_brier_lcb` | mixed_conditional_independence | 0.05604 | > 0.00000 | one-sided LCB computed across fixed seed-level paired means |
| PASS | `predictive` | `dependent_every_seed_positive` | all_binary_dependent | 0.21158 | > 0.00000 | minimum of the fixed seed-level paired mean gains |
| PASS | `prerequisite` | `dependent_oracle_headroom` | all_binary_dependent | 0.27356 | >= 0.05000 | minimum realized-test oracle H across fixed seeds |
| PASS | `predictive` | `marginal_log_score_lcb` | all_binary_dependent | 0.02222 | > 0.00000 | one-sided LCB computed across fixed seed-level paired means |
| PASS | `prerequisite` | `quadrature_gain_mean_stability` | all_binary_dependent, seed=1701 | 0.00024 | <= 0.00200 | fixed-subset absolute mean G difference, GH-31 versus GH-41 |
| FAIL | `prerequisite` | `quadrature_p99_stability` | all_binary_dependent, seed=1701 | 0.10129 | <= 0.01000 | maximum of p99 shared-score and G absolute changes |
| PASS | `prerequisite` | `label_permutation_invariance` | all_binary_dependent, seed=1701 | 0.00000 | <= 0.00000 | maximum across CVAE, intercept-null, and oracle checks |
| PASS | `prerequisite` | `quadrature_gain_mean_stability` | all_binary_dependent, seed=9901 | 0.00027 | <= 0.00200 | fixed-subset absolute mean G difference, GH-31 versus GH-41 |
| FAIL | `prerequisite` | `quadrature_p99_stability` | all_binary_dependent, seed=9901 | 0.03908 | <= 0.01000 | maximum of p99 shared-score and G absolute changes |
| PASS | `prerequisite` | `label_permutation_invariance` | all_binary_dependent, seed=9901 | 0.00000 | <= 0.00000 | maximum across CVAE, intercept-null, and oracle checks |
| PASS | `prerequisite` | `quadrature_gain_mean_stability` | all_binary_dependent, seed=31415 | 0.00037 | <= 0.00200 | fixed-subset absolute mean G difference, GH-31 versus GH-41 |
| FAIL | `prerequisite` | `quadrature_p99_stability` | all_binary_dependent, seed=31415 | 0.06672 | <= 0.01000 | maximum of p99 shared-score and G absolute changes |
| PASS | `prerequisite` | `label_permutation_invariance` | all_binary_dependent, seed=31415 | 0.00000 | <= 0.00000 | maximum across CVAE, intercept-null, and oracle checks |
| PASS | `predictive` | `marginal_brier_lcb` | all_binary_dependent | 0.02174 | > 0.00000 | one-sided LCB computed across fixed seed-level paired means |
| FAIL | `predictive` | `conditional_independence_each_seed` | all_binary_conditional_independence | 0.05306 | < 0.02000 | maximum individual fixed-seed penalty P |
| PASS | `predictive` | `marginal_log_score_lcb` | all_binary_conditional_independence | 0.03693 | > 0.00000 | one-sided LCB computed across fixed seed-level paired means |
| FAIL | `prerequisite` | `quadrature_gain_mean_stability` | all_binary_conditional_independence, seed=1701 | 0.00216 | <= 0.00200 | fixed-subset absolute mean G difference, GH-31 versus GH-41 |
| FAIL | `prerequisite` | `quadrature_p99_stability` | all_binary_conditional_independence, seed=1701 | 0.11826 | <= 0.01000 | maximum of p99 shared-score and G absolute changes |
| PASS | `prerequisite` | `label_permutation_invariance` | all_binary_conditional_independence, seed=1701 | 0.00000 | <= 0.00000 | maximum across CVAE, intercept-null, and oracle checks |
| FAIL | `prerequisite` | `quadrature_gain_mean_stability` | all_binary_conditional_independence, seed=9901 | 0.00314 | <= 0.00200 | fixed-subset absolute mean G difference, GH-31 versus GH-41 |
| FAIL | `prerequisite` | `quadrature_p99_stability` | all_binary_conditional_independence, seed=9901 | 0.15365 | <= 0.01000 | maximum of p99 shared-score and G absolute changes |
| PASS | `prerequisite` | `label_permutation_invariance` | all_binary_conditional_independence, seed=9901 | 0.00000 | <= 0.00000 | maximum across CVAE, intercept-null, and oracle checks |
| FAIL | `prerequisite` | `quadrature_gain_mean_stability` | all_binary_conditional_independence, seed=31415 | 0.01070 | <= 0.00200 | fixed-subset absolute mean G difference, GH-31 versus GH-41 |
| FAIL | `prerequisite` | `quadrature_p99_stability` | all_binary_conditional_independence, seed=31415 | 0.20298 | <= 0.01000 | maximum of p99 shared-score and G absolute changes |
| PASS | `prerequisite` | `label_permutation_invariance` | all_binary_conditional_independence, seed=31415 | 0.00000 | <= 0.00000 | maximum across CVAE, intercept-null, and oracle checks |
| PASS | `predictive` | `marginal_brier_lcb` | all_binary_conditional_independence | 0.03583 | > 0.00000 | one-sided LCB computed across fixed seed-level paired means |
| PASS | `predictive` | `dependent_aggregate_lcb` | all dependent DGP seeds | 0.10800 | > 0.01000 | one-sided LCB across top-level seeds after averaging dependent scenarios within seed |
| FAIL | `predictive` | `conditional_independence_penalty_ucb` | all conditional-independence DGP seeds | 0.05766 | < 0.01000 | one-sided UCB across top-level seeds after averaging CI scenarios within seed |

## Scenario: `mixed_dependent`

### Seed-level primary results

| seed | epochs (best) | active z at best | KL by latent at best | CVAE marginal NLL | CVAE marginal Brier | joint NLL | fitted-product NLL | paired joint gain (SE) | marginal log gain (SE) | Brier improvement (SE) |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1701 | 30 (29) | 2 / 2 | 1.453 / 1.629 | 1.0568 | 0.6092 | 3.1823 | 3.1703 | -0.0120 (0.0157) | 0.0682 (0.0030) | 0.0420 (0.0019) |
| 9901 | 30 (30) | 2 / 2 | 1.542 / 1.911 | 1.0639 | 0.6100 | 3.0924 | 3.1917 | 0.0993 (0.0078) | 0.0614 (0.0025) | 0.0411 (0.0017) |
| 31415 | 30 (30) | 2 / 2 | 1.509 / 1.898 | 1.0697 | 0.6164 | 3.1228 | 3.2091 | 0.0863 (0.0069) | 0.0547 (0.0026) | 0.0346 (0.0018) |

### Seed 1701 detail

Restored checkpoint: epoch 29 of 30; validation beta-ELBO 1.5447; training KL/latent [1.45276, 1.62857]; active latent units 2 / 2; effective beta 0.2000.

Marginal predictive metrics:

| model | NLL | multiclass Brier | classwise ECE | max class-bin gap |
|---|---:|---:|---:|---:|
| `cvae` | 1.0568 | 0.6092 | 0.0223 | 0.5190 |
| `intercept_null` | 1.1250 | 0.6512 | 0.0103 | 0.0232 |
| `oracle` | 1.0450 | 0.6040 | 0.0138 | 0.6253 |

Classwise calibration-in-the-large and uncertainty:

| model | outcome | level | mean predicted | observed | observed Wilson 95% | CIL gap | CIL SE | CIL 95% | ECE | max gap |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cvae` | switch | off | 0.4776 | 0.4912 | [0.4774, 0.5051] | 0.0136 | 0.0068 | [0.0003, 0.0270] | 0.0183 | 0.1590 |
| `cvae` | switch | on | 0.5224 | 0.5088 | [0.4949, 0.5226] | -0.0136 | 0.0068 | [-0.0270, -0.0003] | 0.0183 | 0.1590 |
| `cvae` | shape | circle | 0.4203 | 0.3980 | [0.3845, 0.4116] | -0.0223 | 0.0065 | [-0.0351, -0.0095] | 0.0238 | 0.1305 |
| `cvae` | shape | square | 0.2601 | 0.2714 | [0.2593, 0.2839] | 0.0113 | 0.0060 | [-0.0005, 0.0232] | 0.0155 | 0.0427 |
| `cvae` | shape | triangle | 0.3196 | 0.3306 | [0.3177, 0.3438] | 0.0110 | 0.0063 | [-0.0013, 0.0233] | 0.0198 | 0.0677 |
| `cvae` | color | amber | 0.1889 | 0.2506 | [0.2388, 0.2628] | 0.0617 | 0.0060 | [0.0500, 0.0735] | 0.0617 | 0.2853 |
| `cvae` | color | blue | 0.2032 | 0.1916 | [0.1809, 0.2027] | -0.0116 | 0.0054 | [-0.0222, -0.0011] | 0.0119 | 0.0689 |
| `cvae` | color | coral | 0.1826 | 0.1488 | [0.1392, 0.1589] | -0.0338 | 0.0050 | [-0.0436, -0.0241] | 0.0338 | 0.5190 |
| `cvae` | color | green | 0.2066 | 0.1824 | [0.1719, 0.1933] | -0.0242 | 0.0053 | [-0.0346, -0.0137] | 0.0298 | 0.2500 |
| `cvae` | color | violet | 0.2187 | 0.2266 | [0.2152, 0.2384] | 0.0079 | 0.0059 | [-0.0037, 0.0195] | 0.0081 | 0.0909 |
| `intercept_null` | switch | off | 0.4680 | 0.4912 | [0.4774, 0.5051] | 0.0232 | 0.0071 | [0.0093, 0.0371] | 0.0232 | 0.0232 |
| `intercept_null` | switch | on | 0.5320 | 0.5088 | [0.4949, 0.5226] | -0.0232 | 0.0071 | [-0.0371, -0.0093] | 0.0232 | 0.0232 |
| `intercept_null` | shape | circle | 0.3965 | 0.3980 | [0.3845, 0.4116] | 0.0015 | 0.0069 | [-0.0120, 0.0151] | 0.0015 | 0.0015 |
| `intercept_null` | shape | square | 0.2710 | 0.2714 | [0.2593, 0.2839] | 0.0004 | 0.0063 | [-0.0120, 0.0127] | 0.0004 | 0.0004 |
| `intercept_null` | shape | triangle | 0.3325 | 0.3306 | [0.3177, 0.3438] | -0.0019 | 0.0067 | [-0.0149, 0.0111] | 0.0019 | 0.0019 |
| `intercept_null` | color | amber | 0.2455 | 0.2506 | [0.2388, 0.2628] | 0.0051 | 0.0061 | [-0.0069, 0.0171] | 0.0051 | 0.0051 |
| `intercept_null` | color | blue | 0.1855 | 0.1916 | [0.1809, 0.2027] | 0.0061 | 0.0056 | [-0.0048, 0.0170] | 0.0061 | 0.0061 |
| `intercept_null` | color | coral | 0.1473 | 0.1488 | [0.1392, 0.1589] | 0.0015 | 0.0050 | [-0.0083, 0.0114] | 0.0015 | 0.0015 |
| `intercept_null` | color | green | 0.1788 | 0.1824 | [0.1719, 0.1933] | 0.0036 | 0.0055 | [-0.0071, 0.0143] | 0.0036 | 0.0036 |
| `intercept_null` | color | violet | 0.2430 | 0.2266 | [0.2152, 0.2384] | -0.0164 | 0.0059 | [-0.0280, -0.0048] | 0.0164 | 0.0164 |
| `oracle` | switch | off | 0.4791 | 0.4912 | [0.4774, 0.5051] | 0.0121 | 0.0068 | [-0.0012, 0.0254] | 0.0157 | 0.1559 |
| `oracle` | switch | on | 0.5209 | 0.5088 | [0.4949, 0.5226] | -0.0121 | 0.0068 | [-0.0254, 0.0012] | 0.0157 | 0.1559 |
| `oracle` | shape | circle | 0.3922 | 0.3980 | [0.3845, 0.4116] | 0.0058 | 0.0065 | [-0.0070, 0.0186] | 0.0175 | 0.0506 |
| `oracle` | shape | square | 0.2784 | 0.2714 | [0.2593, 0.2839] | -0.0070 | 0.0060 | [-0.0188, 0.0049] | 0.0111 | 0.3071 |
| `oracle` | shape | triangle | 0.3294 | 0.3306 | [0.3177, 0.3438] | 0.0012 | 0.0063 | [-0.0111, 0.0135] | 0.0266 | 0.0618 |
| `oracle` | color | amber | 0.2522 | 0.2506 | [0.2388, 0.2628] | -0.0016 | 0.0059 | [-0.0133, 0.0101] | 0.0056 | 0.3954 |
| `oracle` | color | blue | 0.1888 | 0.1916 | [0.1809, 0.2027] | 0.0028 | 0.0054 | [-0.0077, 0.0133] | 0.0096 | 0.6253 |
| `oracle` | color | coral | 0.1508 | 0.1488 | [0.1392, 0.1589] | -0.0020 | 0.0049 | [-0.0117, 0.0077] | 0.0060 | 0.0526 |
| `oracle` | color | green | 0.1768 | 0.1824 | [0.1719, 0.1933] | 0.0056 | 0.0053 | [-0.0048, 0.0160] | 0.0065 | 0.1194 |
| `oracle` | color | violet | 0.2314 | 0.2266 | [0.2152, 0.2384] | -0.0048 | 0.0059 | [-0.0164, 0.0067] | 0.0081 | 0.3068 |

CVAE classwise reliability bins (signed gap = observed - predicted):

| outcome | level | bin | n | mean predicted | observed | observed Wilson 95% | signed gap | gap SE | gap 95% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| switch | off | 0 | 4 | 0.0910 | 0.2500 | [0.0456, 0.6994] | 0.1590 | 0.2515 | [-0.3338, 0.6519] |
| switch | off | 1 | 123 | 0.1663 | 0.1707 | [0.1145, 0.2469] | 0.0045 | 0.0343 | [-0.0628, 0.0717] |
| switch | off | 2 | 555 | 0.2577 | 0.3027 | [0.2659, 0.3422] | 0.0450 | 0.0195 | [0.0068, 0.0832] |
| switch | off | 3 | 947 | 0.3538 | 0.3865 | [0.3560, 0.4179] | 0.0327 | 0.0158 | [0.0017, 0.0637] |
| switch | off | 4 | 1122 | 0.4504 | 0.4519 | [0.4230, 0.4811] | 0.0015 | 0.0148 | [-0.0276, 0.0306] |
| switch | off | 5 | 1049 | 0.5500 | 0.5682 | [0.5380, 0.5978] | 0.0182 | 0.0152 | [-0.0117, 0.0481] |
| switch | off | 6 | 871 | 0.6462 | 0.6361 | [0.6036, 0.6673] | -0.0101 | 0.0163 | [-0.0421, 0.0219] |
| switch | off | 7 | 313 | 0.7374 | 0.7284 | [0.6766, 0.7747] | -0.0090 | 0.0253 | [-0.0586, 0.0406] |
| switch | off | 8 | 16 | 0.8261 | 0.9375 | [0.7167, 0.9889] | 0.1114 | 0.0611 | [-0.0084, 0.2312] |
| switch | on | 1 | 16 | 0.1739 | 0.0625 | [0.0111, 0.2833] | -0.1114 | 0.0611 | [-0.2312, 0.0084] |
| switch | on | 2 | 313 | 0.2626 | 0.2716 | [0.2253, 0.3234] | 0.0090 | 0.0253 | [-0.0406, 0.0586] |
| switch | on | 3 | 871 | 0.3538 | 0.3639 | [0.3327, 0.3964] | 0.0101 | 0.0163 | [-0.0219, 0.0421] |
| switch | on | 4 | 1049 | 0.4500 | 0.4318 | [0.4022, 0.4620] | -0.0182 | 0.0152 | [-0.0481, 0.0117] |
| switch | on | 5 | 1122 | 0.5496 | 0.5481 | [0.5189, 0.5770] | -0.0015 | 0.0148 | [-0.0306, 0.0276] |
| switch | on | 6 | 947 | 0.6462 | 0.6135 | [0.5821, 0.6440] | -0.0327 | 0.0158 | [-0.0637, -0.0017] |
| switch | on | 7 | 555 | 0.7423 | 0.6973 | [0.6578, 0.7341] | -0.0450 | 0.0195 | [-0.0832, -0.0068] |
| switch | on | 8 | 123 | 0.8337 | 0.8293 | [0.7531, 0.8855] | -0.0045 | 0.0343 | [-0.0717, 0.0628] |
| switch | on | 9 | 4 | 0.9090 | 0.7500 | [0.3006, 0.9544] | -0.1590 | 0.2515 | [-0.6519, 0.3338] |
| shape | circle | 0 | 60 | 0.0818 | 0.1333 | [0.0691, 0.2417] | 0.0515 | 0.0440 | [-0.0346, 0.1377] |
| shape | circle | 1 | 528 | 0.1585 | 0.1534 | [0.1252, 0.1866] | -0.0051 | 0.0157 | [-0.0358, 0.0256] |
| shape | circle | 2 | 830 | 0.2513 | 0.2289 | [0.2016, 0.2587] | -0.0224 | 0.0146 | [-0.0509, 0.0062] |
| shape | circle | 3 | 852 | 0.3501 | 0.3216 | [0.2911, 0.3537] | -0.0285 | 0.0159 | [-0.0598, 0.0027] |
| shape | circle | 4 | 951 | 0.4510 | 0.4416 | [0.4104, 0.4734] | -0.0094 | 0.0161 | [-0.0410, 0.0222] |
| shape | circle | 5 | 916 | 0.5490 | 0.5066 | [0.4742, 0.5388] | -0.0425 | 0.0165 | [-0.0748, -0.0101] |
| shape | circle | 6 | 648 | 0.6402 | 0.6096 | [0.5715, 0.6464] | -0.0306 | 0.0191 | [-0.0680, 0.0068] |
| shape | circle | 7 | 199 | 0.7356 | 0.7387 | [0.6735, 0.7948] | 0.0031 | 0.0311 | [-0.0579, 0.0640] |
| shape | circle | 8 | 16 | 0.8180 | 0.6875 | [0.4440, 0.8584] | -0.1305 | 0.1205 | [-0.3668, 0.1058] |
| shape | square | 0 | 361 | 0.0753 | 0.0748 | [0.0519, 0.1066] | -0.0005 | 0.0138 | [-0.0276, 0.0266] |
| shape | square | 1 | 1504 | 0.1534 | 0.1649 | [0.1470, 0.1845] | 0.0115 | 0.0095 | [-0.0072, 0.0302] |
| shape | square | 2 | 1460 | 0.2480 | 0.2589 | [0.2371, 0.2820] | 0.0109 | 0.0114 | [-0.0115, 0.0334] |
| shape | square | 3 | 929 | 0.3450 | 0.3811 | [0.3504, 0.4127] | 0.0360 | 0.0159 | [0.0048, 0.0672] |
| shape | square | 4 | 500 | 0.4450 | 0.4280 | [0.3853, 0.4718] | -0.0170 | 0.0222 | [-0.0605, 0.0265] |
| shape | square | 5 | 204 | 0.5429 | 0.5441 | [0.4756, 0.6110] | 0.0012 | 0.0352 | [-0.0678, 0.0703] |
| shape | square | 6 | 42 | 0.6379 | 0.5952 | [0.4449, 0.7296] | -0.0427 | 0.0776 | [-0.1948, 0.1094] |
| shape | triangle | 0 | 100 | 0.0871 | 0.0500 | [0.0215, 0.1118] | -0.0371 | 0.0219 | [-0.0799, 0.0057] |
| shape | triangle | 1 | 1139 | 0.1563 | 0.1589 | [0.1388, 0.1813] | 0.0027 | 0.0107 | [-0.0184, 0.0237] |
| shape | triangle | 2 | 1347 | 0.2487 | 0.2747 | [0.2515, 0.2991] | 0.0260 | 0.0121 | [0.0022, 0.0498] |
| shape | triangle | 3 | 1006 | 0.3476 | 0.3529 | [0.3240, 0.3829] | 0.0053 | 0.0150 | [-0.0242, 0.0347] |
| shape | triangle | 4 | 735 | 0.4464 | 0.4245 | [0.3892, 0.4605] | -0.0220 | 0.0182 | [-0.0577, 0.0138] |
| shape | triangle | 5 | 409 | 0.5431 | 0.5917 | [0.5434, 0.6383] | 0.0486 | 0.0243 | [0.0009, 0.0963] |
| shape | triangle | 6 | 198 | 0.6444 | 0.7121 | [0.6455, 0.7707] | 0.0677 | 0.0322 | [0.0047, 0.1308] |
| shape | triangle | 7 | 59 | 0.7295 | 0.6949 | [0.5685, 0.7975] | -0.0346 | 0.0602 | [-0.1526, 0.0833] |
| shape | triangle | 8 | 7 | 0.8214 | 0.8571 | [0.4869, 0.9743] | 0.0358 | 0.1484 | [-0.2551, 0.3266] |
| color | amber | 0 | 583 | 0.0681 | 0.0892 | [0.0687, 0.1151] | 0.0211 | 0.0117 | [-0.0019, 0.0440] |
| color | amber | 1 | 2005 | 0.1554 | 0.2050 | [0.1879, 0.2232] | 0.0495 | 0.0090 | [0.0319, 0.0672] |
| color | amber | 2 | 2356 | 0.2445 | 0.3213 | [0.3028, 0.3404] | 0.0768 | 0.0096 | [0.0581, 0.0956] |
| color | amber | 3 | 56 | 0.3040 | 0.5893 | [0.4588, 0.7083] | 0.2853 | 0.0664 | [0.1551, 0.4155] |
| color | blue | 0 | 931 | 0.0726 | 0.0548 | [0.0419, 0.0713] | -0.0178 | 0.0074 | [-0.0324, -0.0032] |
| color | blue | 1 | 1733 | 0.1477 | 0.1368 | [0.1214, 0.1537] | -0.0109 | 0.0082 | [-0.0270, 0.0052] |
| color | blue | 2 | 1361 | 0.2476 | 0.2439 | [0.2219, 0.2675] | -0.0036 | 0.0117 | [-0.0265, 0.0193] |
| color | blue | 3 | 761 | 0.3436 | 0.3377 | [0.3050, 0.3721] | -0.0059 | 0.0171 | [-0.0395, 0.0277] |
| color | blue | 4 | 200 | 0.4339 | 0.3650 | [0.3014, 0.4337] | -0.0689 | 0.0340 | [-0.1356, -0.0023] |
| color | blue | 5 | 14 | 0.5279 | 0.5714 | [0.3259, 0.7862] | 0.0436 | 0.1360 | [-0.2229, 0.3100] |
| color | coral | 0 | 619 | 0.0779 | 0.0436 | [0.0301, 0.0627] | -0.0342 | 0.0082 | [-0.0503, -0.0182] |
| color | coral | 1 | 2569 | 0.1485 | 0.1288 | [0.1164, 0.1424] | -0.0197 | 0.0066 | [-0.0326, -0.0068] |
| color | coral | 2 | 1398 | 0.2436 | 0.2031 | [0.1829, 0.2250] | -0.0404 | 0.0107 | [-0.0615, -0.0194] |
| color | coral | 3 | 372 | 0.3358 | 0.2312 | [0.1912, 0.2766] | -0.1046 | 0.0218 | [-0.1473, -0.0618] |
| color | coral | 4 | 41 | 0.4284 | 0.3902 | [0.2566, 0.5427] | -0.0382 | 0.0767 | [-0.1886, 0.1123] |
| color | coral | 5 | 1 | 0.5190 | 0.0000 | [0.0000, 0.7935] | -0.5190 | NA | [NA, NA] |
| color | green | 0 | 39 | 0.0934 | 0.0256 | [0.0045, 0.1318] | -0.0677 | 0.0256 | [-0.1179, -0.0176] |
| color | green | 1 | 2392 | 0.1616 | 0.1133 | [0.1012, 0.1266] | -0.0483 | 0.0064 | [-0.0609, -0.0357] |
| color | green | 2 | 2265 | 0.2401 | 0.2327 | [0.2157, 0.2505] | -0.0074 | 0.0088 | [-0.0247, 0.0099] |
| color | green | 3 | 301 | 0.3246 | 0.3688 | [0.3162, 0.4246] | 0.0442 | 0.0278 | [-0.0103, 0.0986] |
| color | green | 4 | 3 | 0.4167 | 0.6667 | [0.2077, 0.9385] | 0.2500 | 0.3330 | [-0.4028, 0.9028] |
| color | violet | 0 | 6 | 0.0909 | 0.0000 | [0.0000, 0.3903] | -0.0909 | 0.0025 | [-0.0957, -0.0860] |
| color | violet | 1 | 1247 | 0.1769 | 0.1812 | [0.1608, 0.2036] | 0.0043 | 0.0109 | [-0.0170, 0.0257] |
| color | violet | 2 | 3747 | 0.2328 | 0.2421 | [0.2286, 0.2560] | 0.0093 | 0.0070 | [-0.0044, 0.0230] |

Fixed label-permutation invariance checks:

| probability source | max metric discrepancy | max selected-log-mass discrepancy | overall maximum |
|---|---:|---:|---:|
| `cvae` | 0.000000000000 | 0.000000000000 | 0.000000000000 |
| `intercept_null` | 0.000000000000 | 0.000000000000 | 0.000000000000 |
| `oracle` | 0.000000000000 | 0.000000000000 | 0.000000000000 |

Joint predictive scores and paired CVAE comparisons:

| model | joint NLL | NLL SE | CVAE-minus-model log gain | paired SE | one-sided LCB |
|---|---:|---:|---:|---:|---:|
| `cvae` | 3.1823 | 0.0183 | NA | NA | NA |
| `independent_fitted_marginal` | 3.1703 | 0.0090 | -0.0120 | 0.0157 | -0.0377 |
| `intercept_null` | 3.3750 | 0.0036 | 0.1928 | 0.0180 | 0.1631 |
| `empirical_joint` | 3.2584 | 0.0082 | 0.0761 | 0.0156 | 0.0504 |
| `oracle` | 2.9963 | 0.0117 | -0.1860 | 0.0115 | -0.2049 |

Fitted-CVAE quadrature sensitivity (fixed subset, relative to primary order):

| check order | joint mean delta | joint RMSE | product RMSE | mean G change | p99 shared change | p99 G change |
|---:|---:|---:|---:|---:|---:|---:|
| 21 | 0.000132 | 0.027984 | 0.011337 | 0.001082 | 0.079318 | 0.072879 |
| 41 | 0.000787 | 0.012081 | 0.005427 | 0.000427 | 0.042616 | 0.045209 |

Secondary nested common-panel Monte Carlo diagnostic:

| M | joint NLL | product NLL | mean gain | joint RMSE vs GH | product RMSE vs GH |
|---:|---:|---:|---:|---:|---:|
| 16 | 3.5034 | 3.2603 | -0.2431 | 1.1346 | 0.6973 |
| 64 | 3.2214 | 3.1677 | -0.0536 | 0.3663 | 0.2782 |
| 256 | 3.2106 | 3.1503 | -0.0603 | 0.1646 | 0.1473 |
| 1024 | 3.2136 | 3.1655 | -0.0481 | 0.0935 | 0.0667 |

MC block gain SD: 0.0271; jackknife-corrected mean gain: -0.0463; median joint-integrand ESS: 162.4 (0.159 of M).

### Seed 9901 detail

Restored checkpoint: epoch 30 of 30; validation beta-ELBO 1.1694; training KL/latent [1.54171, 1.91145]; active latent units 2 / 2; effective beta 0.2000.

Marginal predictive metrics:

| model | NLL | multiclass Brier | classwise ECE | max class-bin gap |
|---|---:|---:|---:|---:|
| `cvae` | 1.0639 | 0.6100 | 0.0309 | 0.2684 |
| `intercept_null` | 1.1253 | 0.6511 | 0.0050 | 0.0140 |
| `oracle` | 1.0428 | 0.6004 | 0.0108 | 0.6273 |

Classwise calibration-in-the-large and uncertainty:

| model | outcome | level | mean predicted | observed | observed Wilson 95% | CIL gap | CIL SE | CIL 95% | ECE | max gap |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cvae` | switch | off | 0.4740 | 0.4800 | [0.4662, 0.4939] | 0.0060 | 0.0068 | [-0.0072, 0.0193] | 0.0276 | 0.1748 |
| `cvae` | switch | on | 0.5260 | 0.5200 | [0.5061, 0.5338] | -0.0060 | 0.0068 | [-0.0193, 0.0072] | 0.0276 | 0.1748 |
| `cvae` | shape | circle | 0.3919 | 0.3968 | [0.3833, 0.4104] | 0.0049 | 0.0065 | [-0.0079, 0.0176] | 0.0282 | 0.0512 |
| `cvae` | shape | square | 0.2439 | 0.2762 | [0.2640, 0.2888] | 0.0323 | 0.0061 | [0.0204, 0.0442] | 0.0323 | 0.2684 |
| `cvae` | shape | triangle | 0.3642 | 0.3270 | [0.3141, 0.3401] | -0.0372 | 0.0063 | [-0.0494, -0.0249] | 0.0385 | 0.1782 |
| `cvae` | color | amber | 0.2518 | 0.2440 | [0.2323, 0.2561] | -0.0078 | 0.0060 | [-0.0196, 0.0040] | 0.0159 | 0.1967 |
| `cvae` | color | blue | 0.2119 | 0.1814 | [0.1710, 0.1923] | -0.0305 | 0.0053 | [-0.0409, -0.0201] | 0.0411 | 0.0895 |
| `cvae` | color | coral | 0.1441 | 0.1586 | [0.1487, 0.1690] | 0.0145 | 0.0051 | [0.0045, 0.0244] | 0.0190 | 0.0528 |
| `cvae` | color | green | 0.1942 | 0.1790 | [0.1686, 0.1899] | -0.0152 | 0.0054 | [-0.0258, -0.0047] | 0.0446 | 0.0707 |
| `cvae` | color | violet | 0.1979 | 0.2370 | [0.2254, 0.2490] | 0.0391 | 0.0060 | [0.0274, 0.0508] | 0.0391 | 0.0481 |
| `intercept_null` | switch | off | 0.4828 | 0.4800 | [0.4662, 0.4939] | -0.0028 | 0.0071 | [-0.0166, 0.0111] | 0.0028 | 0.0028 |
| `intercept_null` | switch | on | 0.5172 | 0.5200 | [0.5061, 0.5338] | 0.0028 | 0.0071 | [-0.0111, 0.0166] | 0.0028 | 0.0028 |
| `intercept_null` | shape | circle | 0.3942 | 0.3968 | [0.3833, 0.4104] | 0.0026 | 0.0069 | [-0.0110, 0.0161] | 0.0026 | 0.0026 |
| `intercept_null` | shape | square | 0.2858 | 0.2762 | [0.2640, 0.2888] | -0.0096 | 0.0063 | [-0.0220, 0.0028] | 0.0096 | 0.0096 |
| `intercept_null` | shape | triangle | 0.3200 | 0.3270 | [0.3141, 0.3401] | 0.0070 | 0.0066 | [-0.0060, 0.0200] | 0.0070 | 0.0070 |
| `intercept_null` | color | amber | 0.2512 | 0.2440 | [0.2323, 0.2561] | -0.0072 | 0.0061 | [-0.0191, 0.0047] | 0.0072 | 0.0072 |
| `intercept_null` | color | blue | 0.1875 | 0.1814 | [0.1710, 0.1923] | -0.0061 | 0.0055 | [-0.0168, 0.0046] | 0.0061 | 0.0061 |
| `intercept_null` | color | coral | 0.1583 | 0.1586 | [0.1487, 0.1690] | 0.0003 | 0.0052 | [-0.0098, 0.0105] | 0.0003 | 0.0003 |
| `intercept_null` | color | green | 0.1800 | 0.1790 | [0.1686, 0.1899] | -0.0010 | 0.0054 | [-0.0116, 0.0096] | 0.0010 | 0.0010 |
| `intercept_null` | color | violet | 0.2230 | 0.2370 | [0.2254, 0.2490] | 0.0140 | 0.0060 | [0.0022, 0.0258] | 0.0140 | 0.0140 |
| `oracle` | switch | off | 0.4785 | 0.4800 | [0.4662, 0.4939] | 0.0015 | 0.0067 | [-0.0117, 0.0147] | 0.0153 | 0.0859 |
| `oracle` | switch | on | 0.5215 | 0.5200 | [0.5061, 0.5338] | -0.0015 | 0.0067 | [-0.0147, 0.0117] | 0.0153 | 0.0859 |
| `oracle` | shape | circle | 0.3940 | 0.3968 | [0.3833, 0.4104] | 0.0028 | 0.0065 | [-0.0100, 0.0155] | 0.0085 | 0.0272 |
| `oracle` | shape | square | 0.2779 | 0.2762 | [0.2640, 0.2888] | -0.0017 | 0.0060 | [-0.0136, 0.0101] | 0.0089 | 0.0542 |
| `oracle` | shape | triangle | 0.3281 | 0.3270 | [0.3141, 0.3401] | -0.0011 | 0.0062 | [-0.0133, 0.0111] | 0.0087 | 0.0857 |
| `oracle` | color | amber | 0.2527 | 0.2440 | [0.2323, 0.2561] | -0.0087 | 0.0059 | [-0.0203, 0.0029] | 0.0104 | 0.0317 |
| `oracle` | color | blue | 0.1880 | 0.1814 | [0.1710, 0.1923] | -0.0066 | 0.0053 | [-0.0169, 0.0037] | 0.0091 | 0.6273 |
| `oracle` | color | coral | 0.1513 | 0.1586 | [0.1487, 0.1690] | 0.0073 | 0.0051 | [-0.0026, 0.0172] | 0.0105 | 0.0708 |
| `oracle` | color | green | 0.1761 | 0.1790 | [0.1686, 0.1899] | 0.0029 | 0.0053 | [-0.0074, 0.0133] | 0.0060 | 0.0539 |
| `oracle` | color | violet | 0.2319 | 0.2370 | [0.2254, 0.2490] | 0.0051 | 0.0060 | [-0.0066, 0.0168] | 0.0062 | 0.1993 |

CVAE classwise reliability bins (signed gap = observed - predicted):

| outcome | level | bin | n | mean predicted | observed | observed Wilson 95% | signed gap | gap SE | gap 95% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| switch | off | 1 | 13 | 0.1748 | 0.0000 | [0.0000, 0.2281] | -0.1748 | 0.0057 | [-0.1860, -0.1636] |
| switch | off | 2 | 305 | 0.2631 | 0.1705 | [0.1324, 0.2167] | -0.0927 | 0.0214 | [-0.1347, -0.0506] |
| switch | off | 3 | 965 | 0.3569 | 0.3326 | [0.3036, 0.3630] | -0.0242 | 0.0151 | [-0.0539, 0.0054] |
| switch | off | 4 | 1628 | 0.4488 | 0.4552 | [0.4311, 0.4794] | 0.0063 | 0.0123 | [-0.0178, 0.0305] |
| switch | off | 5 | 1438 | 0.5474 | 0.5640 | [0.5382, 0.5894] | 0.0165 | 0.0131 | [-0.0091, 0.0421] |
| switch | off | 6 | 558 | 0.6389 | 0.7204 | [0.6818, 0.7561] | 0.0815 | 0.0189 | [0.0445, 0.1185] |
| switch | off | 7 | 88 | 0.7326 | 0.7841 | [0.6872, 0.8572] | 0.0515 | 0.0438 | [-0.0344, 0.1375] |
| switch | off | 8 | 5 | 0.8152 | 0.8000 | [0.3755, 0.9638] | -0.0152 | 0.2006 | [-0.4083, 0.3779] |
| switch | on | 1 | 5 | 0.1848 | 0.2000 | [0.0362, 0.6245] | 0.0152 | 0.2006 | [-0.3779, 0.4083] |
| switch | on | 2 | 88 | 0.2674 | 0.2159 | [0.1428, 0.3128] | -0.0515 | 0.0438 | [-0.1375, 0.0344] |
| switch | on | 3 | 558 | 0.3611 | 0.2796 | [0.2439, 0.3182] | -0.0815 | 0.0189 | [-0.1185, -0.0445] |
| switch | on | 4 | 1438 | 0.4526 | 0.4360 | [0.4106, 0.4618] | -0.0165 | 0.0131 | [-0.0421, 0.0091] |
| switch | on | 5 | 1628 | 0.5512 | 0.5448 | [0.5206, 0.5689] | -0.0063 | 0.0123 | [-0.0305, 0.0178] |
| switch | on | 6 | 965 | 0.6431 | 0.6674 | [0.6370, 0.6964] | 0.0242 | 0.0151 | [-0.0054, 0.0539] |
| switch | on | 7 | 305 | 0.7369 | 0.8295 | [0.7833, 0.8676] | 0.0927 | 0.0214 | [0.0506, 0.1347] |
| switch | on | 8 | 13 | 0.8252 | 1.0000 | [0.7719, 1.0000] | 0.1748 | 0.0057 | [0.1636, 0.1860] |
| shape | circle | 0 | 71 | 0.0850 | 0.0986 | [0.0486, 0.1898] | 0.0136 | 0.0355 | [-0.0559, 0.0831] |
| shape | circle | 1 | 782 | 0.1580 | 0.1714 | [0.1466, 0.1994] | 0.0134 | 0.0134 | [-0.0130, 0.0397] |
| shape | circle | 2 | 1014 | 0.2480 | 0.2682 | [0.2419, 0.2964] | 0.0202 | 0.0139 | [-0.0070, 0.0475] |
| shape | circle | 3 | 899 | 0.3482 | 0.3760 | [0.3449, 0.4081] | 0.0278 | 0.0162 | [-0.0039, 0.0595] |
| shape | circle | 4 | 760 | 0.4492 | 0.4829 | [0.4475, 0.5184] | 0.0337 | 0.0181 | [-0.0018, 0.0692] |
| shape | circle | 5 | 654 | 0.5482 | 0.5107 | [0.4724, 0.5488] | -0.0375 | 0.0195 | [-0.0758, 0.0008] |
| shape | circle | 6 | 492 | 0.6447 | 0.5935 | [0.5495, 0.6360] | -0.0512 | 0.0221 | [-0.0944, -0.0079] |
| shape | circle | 7 | 265 | 0.7412 | 0.7132 | [0.6560, 0.7643] | -0.0280 | 0.0279 | [-0.0827, 0.0266] |
| shape | circle | 8 | 63 | 0.8278 | 0.8095 | [0.6959, 0.8875] | -0.0183 | 0.0499 | [-0.1161, 0.0796] |
| shape | square | 0 | 315 | 0.0769 | 0.0952 | [0.0675, 0.1327] | 0.0184 | 0.0165 | [-0.0140, 0.0508] |
| shape | square | 1 | 1689 | 0.1526 | 0.1687 | [0.1516, 0.1873] | 0.0161 | 0.0091 | [-0.0017, 0.0340] |
| shape | square | 2 | 1641 | 0.2467 | 0.2852 | [0.2639, 0.3075] | 0.0385 | 0.0111 | [0.0166, 0.0603] |
| shape | square | 3 | 857 | 0.3431 | 0.3699 | [0.3382, 0.4027] | 0.0268 | 0.0165 | [-0.0055, 0.0590] |
| shape | square | 4 | 350 | 0.4423 | 0.5286 | [0.4762, 0.5803] | 0.0863 | 0.0267 | [0.0340, 0.1387] |
| shape | square | 5 | 112 | 0.5388 | 0.6250 | [0.5326, 0.7091] | 0.0862 | 0.0457 | [-0.0034, 0.1758] |
| shape | square | 6 | 32 | 0.6383 | 0.6875 | [0.5143, 0.8205] | 0.0492 | 0.0843 | [-0.1160, 0.2144] |
| shape | square | 7 | 4 | 0.7316 | 1.0000 | [0.5101, 1.0000] | 0.2684 | 0.0112 | [0.2464, 0.2904] |
| shape | triangle | 0 | 138 | 0.0764 | 0.0797 | [0.0451, 0.1371] | 0.0033 | 0.0229 | [-0.0417, 0.0482] |
| shape | triangle | 1 | 684 | 0.1550 | 0.1462 | [0.1217, 0.1747] | -0.0088 | 0.0135 | [-0.0353, 0.0178] |
| shape | triangle | 2 | 1033 | 0.2506 | 0.2062 | [0.1826, 0.2319] | -0.0444 | 0.0126 | [-0.0690, -0.0198] |
| shape | triangle | 3 | 1104 | 0.3488 | 0.2835 | [0.2577, 0.3108] | -0.0653 | 0.0135 | [-0.0919, -0.0388] |
| shape | triangle | 4 | 980 | 0.4488 | 0.4122 | [0.3818, 0.4433] | -0.0365 | 0.0157 | [-0.0673, -0.0057] |
| shape | triangle | 5 | 697 | 0.5465 | 0.5165 | [0.4794, 0.5534] | -0.0300 | 0.0189 | [-0.0669, 0.0070] |
| shape | triangle | 6 | 303 | 0.6418 | 0.6139 | [0.5579, 0.6669] | -0.0279 | 0.0280 | [-0.0828, 0.0270] |
| shape | triangle | 7 | 58 | 0.7364 | 0.7759 | [0.6534, 0.8641] | 0.0394 | 0.0548 | [-0.0679, 0.1468] |
| shape | triangle | 8 | 3 | 0.8218 | 1.0000 | [0.4385, 1.0000] | 0.1782 | 0.0078 | [0.1630, 0.1935] |
| color | amber | 1 | 2 | 0.1967 | 0.0000 | [0.0000, 0.6576] | -0.1967 | 0.0015 | [-0.1997, -0.1937] |
| color | amber | 2 | 4641 | 0.2454 | 0.2327 | [0.2208, 0.2451] | -0.0127 | 0.0062 | [-0.0247, -0.0006] |
| color | amber | 3 | 337 | 0.3314 | 0.3858 | [0.3354, 0.4387] | 0.0543 | 0.0264 | [0.0026, 0.1060] |
| color | amber | 4 | 20 | 0.4076 | 0.5000 | [0.2993, 0.7007] | 0.0924 | 0.1148 | [-0.1327, 0.3175] |
| color | blue | 0 | 84 | 0.0812 | 0.0357 | [0.0122, 0.0998] | -0.0455 | 0.0202 | [-0.0851, -0.0060] |
| color | blue | 1 | 2031 | 0.1678 | 0.0827 | [0.0715, 0.0955] | -0.0851 | 0.0061 | [-0.0971, -0.0731] |
| color | blue | 2 | 2591 | 0.2383 | 0.2374 | [0.2214, 0.2541] | -0.0009 | 0.0083 | [-0.0172, 0.0154] |
| color | blue | 3 | 294 | 0.3221 | 0.4116 | [0.3568, 0.4686] | 0.0895 | 0.0286 | [0.0335, 0.1455] |
| color | coral | 0 | 837 | 0.0774 | 0.0645 | [0.0498, 0.0832] | -0.0128 | 0.0084 | [-0.0294, 0.0037] |
| color | coral | 1 | 3667 | 0.1479 | 0.1639 | [0.1523, 0.1762] | 0.0160 | 0.0061 | [0.0041, 0.0279] |
| color | coral | 2 | 475 | 0.2251 | 0.2779 | [0.2395, 0.3198] | 0.0528 | 0.0205 | [0.0126, 0.0930] |
| color | coral | 3 | 21 | 0.3172 | 0.2857 | [0.1381, 0.4996] | -0.0315 | 0.1010 | [-0.2294, 0.1664] |
| color | green | 1 | 3278 | 0.1802 | 0.1345 | [0.1233, 0.1466] | -0.0457 | 0.0059 | [-0.0573, -0.0340] |
| color | green | 2 | 1698 | 0.2198 | 0.2621 | [0.2417, 0.2835] | 0.0423 | 0.0107 | [0.0214, 0.0632] |
| color | green | 3 | 24 | 0.3043 | 0.3750 | [0.2116, 0.5729] | 0.0707 | 0.1007 | [-0.1266, 0.2680] |
| color | violet | 0 | 110 | 0.0850 | 0.0909 | [0.0501, 0.1593] | 0.0059 | 0.0275 | [-0.0479, 0.0598] |
| color | violet | 1 | 2242 | 0.1822 | 0.2123 | [0.1959, 0.2297] | 0.0301 | 0.0086 | [0.0132, 0.0470] |
| color | violet | 2 | 2648 | 0.2158 | 0.2640 | [0.2475, 0.2811] | 0.0481 | 0.0086 | [0.0313, 0.0649] |

Fixed label-permutation invariance checks:

| probability source | max metric discrepancy | max selected-log-mass discrepancy | overall maximum |
|---|---:|---:|---:|
| `cvae` | 0.000000000000 | 0.000000000000 | 0.000000000000 |
| `intercept_null` | 0.000000000000 | 0.000000000000 | 0.000000000000 |
| `oracle` | 0.000000000000 | 0.000000000000 | 0.000000000000 |

Joint predictive scores and paired CVAE comparisons:

| model | joint NLL | NLL SE | CVAE-minus-model log gain | paired SE | one-sided LCB |
|---|---:|---:|---:|---:|---:|
| `cvae` | 3.0924 | 0.0107 | NA | NA | NA |
| `independent_fitted_marginal` | 3.1917 | 0.0081 | 0.0993 | 0.0078 | 0.0865 |
| `intercept_null` | 3.3759 | 0.0032 | 0.2836 | 0.0102 | 0.2667 |
| `empirical_joint` | 3.2564 | 0.0079 | 0.1641 | 0.0092 | 0.1490 |
| `oracle` | 2.9868 | 0.0118 | -0.1055 | 0.0059 | -0.1153 |

Fitted-CVAE quadrature sensitivity (fixed subset, relative to primary order):

| check order | joint mean delta | joint RMSE | product RMSE | mean G change | p99 shared change | p99 G change |
|---:|---:|---:|---:|---:|---:|---:|
| 21 | 0.003571 | 0.141193 | 0.097989 | 0.009250 | 0.505383 | 0.310817 |
| 41 | 0.016800 | 0.075194 | 0.045678 | 0.006143 | 0.190628 | 0.142539 |

Secondary nested common-panel Monte Carlo diagnostic:

| M | joint NLL | product NLL | mean gain | joint RMSE vs GH | product RMSE vs GH |
|---:|---:|---:|---:|---:|---:|
| 16 | 3.7270 | 3.3708 | -0.3562 | 1.7907 | 0.7298 |
| 64 | 3.2715 | 3.2529 | -0.0186 | 0.4805 | 0.3474 |
| 256 | 3.1415 | 3.2000 | 0.0585 | 0.2327 | 0.1559 |
| 1024 | 3.1349 | 3.1875 | 0.0526 | 0.1334 | 0.0918 |

MC block gain SD: 0.0226; jackknife-corrected mean gain: 0.0554; median joint-integrand ESS: 99.4 (0.097 of M).

### Seed 31415 detail

Restored checkpoint: epoch 30 of 30; validation beta-ELBO 1.2321; training KL/latent [1.50874, 1.89815]; active latent units 2 / 2; effective beta 0.2000.

Marginal predictive metrics:

| model | NLL | multiclass Brier | classwise ECE | max class-bin gap |
|---|---:|---:|---:|---:|
| `cvae` | 1.0697 | 0.6164 | 0.0505 | 0.4055 |
| `intercept_null` | 1.1244 | 0.6510 | 0.0081 | 0.0127 |
| `oracle` | 1.0398 | 0.6016 | 0.0119 | 0.4173 |

Classwise calibration-in-the-large and uncertainty:

| model | outcome | level | mean predicted | observed | observed Wilson 95% | CIL gap | CIL SE | CIL 95% | ECE | max gap |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cvae` | switch | off | 0.4478 | 0.4846 | [0.4708, 0.4985] | 0.0368 | 0.0068 | [0.0234, 0.0502] | 0.0463 | 0.1770 |
| `cvae` | switch | on | 0.5522 | 0.5154 | [0.5015, 0.5292] | -0.0368 | 0.0068 | [-0.0502, -0.0234] | 0.0463 | 0.1770 |
| `cvae` | shape | circle | 0.3383 | 0.3852 | [0.3718, 0.3988] | 0.0469 | 0.0065 | [0.0341, 0.0597] | 0.0696 | 0.1994 |
| `cvae` | shape | square | 0.2674 | 0.2772 | [0.2650, 0.2898] | 0.0098 | 0.0061 | [-0.0021, 0.0218] | 0.0318 | 0.3977 |
| `cvae` | shape | triangle | 0.3943 | 0.3376 | [0.3246, 0.3508] | -0.0567 | 0.0064 | [-0.0692, -0.0442] | 0.0758 | 0.2740 |
| `cvae` | color | amber | 0.2237 | 0.2564 | [0.2445, 0.2687] | 0.0327 | 0.0060 | [0.0208, 0.0445] | 0.0423 | 0.0920 |
| `cvae` | color | blue | 0.2099 | 0.1762 | [0.1659, 0.1870] | -0.0337 | 0.0052 | [-0.0439, -0.0235] | 0.0453 | 0.4055 |
| `cvae` | color | coral | 0.1913 | 0.1482 | [0.1386, 0.1583] | -0.0431 | 0.0049 | [-0.0528, -0.0334] | 0.0431 | 0.0550 |
| `cvae` | color | green | 0.2018 | 0.1830 | [0.1725, 0.1940] | -0.0188 | 0.0054 | [-0.0293, -0.0083] | 0.0370 | 0.0721 |
| `cvae` | color | violet | 0.1733 | 0.2362 | [0.2246, 0.2482] | 0.0629 | 0.0060 | [0.0512, 0.0746] | 0.0634 | 0.2283 |
| `intercept_null` | switch | off | 0.4738 | 0.4846 | [0.4708, 0.4985] | 0.0108 | 0.0071 | [-0.0030, 0.0247] | 0.0108 | 0.0108 |
| `intercept_null` | switch | on | 0.5262 | 0.5154 | [0.5015, 0.5292] | -0.0108 | 0.0071 | [-0.0247, 0.0030] | 0.0108 | 0.0108 |
| `intercept_null` | shape | circle | 0.3870 | 0.3852 | [0.3718, 0.3988] | -0.0018 | 0.0069 | [-0.0153, 0.0117] | 0.0018 | 0.0018 |
| `intercept_null` | shape | square | 0.2828 | 0.2772 | [0.2650, 0.2898] | -0.0056 | 0.0063 | [-0.0180, 0.0068] | 0.0056 | 0.0056 |
| `intercept_null` | shape | triangle | 0.3303 | 0.3376 | [0.3246, 0.3508] | 0.0073 | 0.0067 | [-0.0058, 0.0205] | 0.0073 | 0.0073 |
| `intercept_null` | color | amber | 0.2622 | 0.2564 | [0.2445, 0.2687] | -0.0058 | 0.0062 | [-0.0179, 0.0063] | 0.0058 | 0.0058 |
| `intercept_null` | color | blue | 0.1823 | 0.1762 | [0.1659, 0.1870] | -0.0061 | 0.0054 | [-0.0166, 0.0045] | 0.0061 | 0.0061 |
| `intercept_null` | color | coral | 0.1580 | 0.1482 | [0.1386, 0.1583] | -0.0098 | 0.0050 | [-0.0197, 0.0000] | 0.0098 | 0.0098 |
| `intercept_null` | color | green | 0.1740 | 0.1830 | [0.1725, 0.1940] | 0.0090 | 0.0055 | [-0.0017, 0.0197] | 0.0090 | 0.0090 |
| `intercept_null` | color | violet | 0.2235 | 0.2362 | [0.2246, 0.2482] | 0.0127 | 0.0060 | [0.0009, 0.0245] | 0.0127 | 0.0127 |
| `oracle` | switch | off | 0.4807 | 0.4846 | [0.4708, 0.4985] | 0.0039 | 0.0068 | [-0.0094, 0.0173] | 0.0153 | 0.4173 |
| `oracle` | switch | on | 0.5193 | 0.5154 | [0.5015, 0.5292] | -0.0039 | 0.0068 | [-0.0173, 0.0094] | 0.0153 | 0.4173 |
| `oracle` | shape | circle | 0.3900 | 0.3852 | [0.3718, 0.3988] | -0.0048 | 0.0064 | [-0.0174, 0.0078] | 0.0154 | 0.0757 |
| `oracle` | shape | square | 0.2831 | 0.2772 | [0.2650, 0.2898] | -0.0059 | 0.0060 | [-0.0178, 0.0059] | 0.0082 | 0.1715 |
| `oracle` | shape | triangle | 0.3268 | 0.3376 | [0.3246, 0.3508] | 0.0108 | 0.0063 | [-0.0016, 0.0231] | 0.0141 | 0.1954 |
| `oracle` | color | amber | 0.2506 | 0.2564 | [0.2445, 0.2687] | 0.0058 | 0.0060 | [-0.0059, 0.0176] | 0.0076 | 0.3724 |
| `oracle` | color | blue | 0.1880 | 0.1762 | [0.1659, 0.1870] | -0.0118 | 0.0052 | [-0.0219, -0.0017] | 0.0121 | 0.3972 |
| `oracle` | color | coral | 0.1523 | 0.1482 | [0.1386, 0.1583] | -0.0041 | 0.0049 | [-0.0137, 0.0056] | 0.0074 | 0.0718 |
| `oracle` | color | green | 0.1774 | 0.1830 | [0.1725, 0.1940] | 0.0056 | 0.0053 | [-0.0048, 0.0160] | 0.0056 | 0.0472 |
| `oracle` | color | violet | 0.2317 | 0.2362 | [0.2246, 0.2482] | 0.0045 | 0.0060 | [-0.0072, 0.0162] | 0.0065 | 0.3016 |

CVAE classwise reliability bins (signed gap = observed - predicted):

| outcome | level | bin | n | mean predicted | observed | observed Wilson 95% | signed gap | gap SE | gap 95% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| switch | off | 0 | 4 | 0.0730 | 0.2500 | [0.0456, 0.6994] | 0.1770 | 0.2542 | [-0.3212, 0.6752] |
| switch | off | 1 | 255 | 0.1669 | 0.2510 | [0.2017, 0.3076] | 0.0841 | 0.0271 | [0.0310, 0.1372] |
| switch | off | 2 | 795 | 0.2567 | 0.3384 | [0.3063, 0.3720] | 0.0816 | 0.0167 | [0.0488, 0.1144] |
| switch | off | 3 | 1095 | 0.3498 | 0.4201 | [0.3912, 0.4496] | 0.0703 | 0.0149 | [0.0411, 0.0995] |
| switch | off | 4 | 1010 | 0.4489 | 0.4743 | [0.4436, 0.5051] | 0.0254 | 0.0157 | [-0.0053, 0.0561] |
| switch | off | 5 | 802 | 0.5484 | 0.5686 | [0.5341, 0.6024] | 0.0202 | 0.0175 | [-0.0141, 0.0545] |
| switch | off | 6 | 632 | 0.6467 | 0.6092 | [0.5706, 0.6464] | -0.0375 | 0.0195 | [-0.0757, 0.0007] |
| switch | off | 7 | 336 | 0.7386 | 0.7411 | [0.6917, 0.7850] | 0.0025 | 0.0238 | [-0.0441, 0.0492] |
| switch | off | 8 | 70 | 0.8314 | 0.8429 | [0.7401, 0.9099] | 0.0115 | 0.0438 | [-0.0743, 0.0972] |
| switch | off | 9 | 1 | 0.9027 | 1.0000 | [0.2065, 1.0000] | 0.0973 | NA | [NA, NA] |
| switch | on | 0 | 1 | 0.0973 | 0.0000 | [0.0000, 0.7935] | -0.0973 | NA | [NA, NA] |
| switch | on | 1 | 70 | 0.1686 | 0.1571 | [0.0901, 0.2599] | -0.0115 | 0.0438 | [-0.0972, 0.0743] |
| switch | on | 2 | 336 | 0.2614 | 0.2589 | [0.2150, 0.3083] | -0.0025 | 0.0238 | [-0.0492, 0.0441] |
| switch | on | 3 | 632 | 0.3533 | 0.3908 | [0.3536, 0.4294] | 0.0375 | 0.0195 | [-0.0007, 0.0757] |
| switch | on | 4 | 802 | 0.4516 | 0.4314 | [0.3976, 0.4659] | -0.0202 | 0.0175 | [-0.0545, 0.0141] |
| switch | on | 5 | 1010 | 0.5511 | 0.5257 | [0.4949, 0.5564] | -0.0254 | 0.0157 | [-0.0561, 0.0053] |
| switch | on | 6 | 1095 | 0.6502 | 0.5799 | [0.5504, 0.6088] | -0.0703 | 0.0149 | [-0.0995, -0.0411] |
| switch | on | 7 | 795 | 0.7433 | 0.6616 | [0.6280, 0.6937] | -0.0816 | 0.0167 | [-0.1144, -0.0488] |
| switch | on | 8 | 255 | 0.8331 | 0.7490 | [0.6924, 0.7983] | -0.0841 | 0.0271 | [-0.1372, -0.0310] |
| switch | on | 9 | 4 | 0.9270 | 0.7500 | [0.3006, 0.9544] | -0.1770 | 0.2542 | [-0.6752, 0.3212] |
| shape | circle | 0 | 11 | 0.0859 | 0.0000 | [0.0000, 0.2588] | -0.0859 | 0.0020 | [-0.0898, -0.0820] |
| shape | circle | 1 | 357 | 0.1741 | 0.1036 | [0.0761, 0.1396] | -0.0704 | 0.0161 | [-0.1020, -0.0389] |
| shape | circle | 2 | 1344 | 0.2526 | 0.2299 | [0.2082, 0.2532] | -0.0227 | 0.0114 | [-0.0451, -0.0003] |
| shape | circle | 3 | 1967 | 0.3515 | 0.4001 | [0.3787, 0.4219] | 0.0486 | 0.0110 | [0.0271, 0.0701] |
| shape | circle | 4 | 1116 | 0.4374 | 0.5762 | [0.5470, 0.6048] | 0.1388 | 0.0147 | [0.1099, 0.1677] |
| shape | circle | 5 | 201 | 0.5319 | 0.7313 | [0.6661, 0.7879] | 0.1994 | 0.0314 | [0.1378, 0.2610] |
| shape | circle | 6 | 4 | 0.6186 | 0.7500 | [0.3006, 0.9544] | 0.1314 | 0.2463 | [-0.3514, 0.6143] |
| shape | square | 0 | 96 | 0.0823 | 0.0417 | [0.0163, 0.1023] | -0.0407 | 0.0204 | [-0.0807, -0.0007] |
| shape | square | 1 | 1160 | 0.1631 | 0.1397 | [0.1209, 0.1608] | -0.0235 | 0.0101 | [-0.0433, -0.0036] |
| shape | square | 2 | 1943 | 0.2495 | 0.2373 | [0.2189, 0.2567] | -0.0122 | 0.0096 | [-0.0311, 0.0066] |
| shape | square | 3 | 1443 | 0.3436 | 0.3839 | [0.3592, 0.4093] | 0.0403 | 0.0128 | [0.0153, 0.0654] |
| shape | square | 4 | 321 | 0.4342 | 0.5576 | [0.5029, 0.6110] | 0.1234 | 0.0277 | [0.0691, 0.1777] |
| shape | square | 5 | 35 | 0.5327 | 0.6857 | [0.5202, 0.8145] | 0.1530 | 0.0801 | [-0.0040, 0.3100] |
| shape | square | 6 | 2 | 0.6023 | 1.0000 | [0.3424, 1.0000] | 0.3977 | 0.0012 | [0.3953, 0.4000] |
| shape | triangle | 1 | 12 | 0.1799 | 0.0000 | [0.0000, 0.2425] | -0.1799 | 0.0054 | [-0.1904, -0.1694] |
| shape | triangle | 2 | 744 | 0.2697 | 0.1142 | [0.0933, 0.1391] | -0.1555 | 0.0116 | [-0.1782, -0.1328] |
| shape | triangle | 3 | 2042 | 0.3513 | 0.2620 | [0.2434, 0.2815] | -0.0893 | 0.0097 | [-0.1083, -0.0703] |
| shape | triangle | 4 | 1530 | 0.4457 | 0.4255 | [0.4009, 0.4504] | -0.0202 | 0.0126 | [-0.0449, 0.0044] |
| shape | triangle | 5 | 582 | 0.5360 | 0.5962 | [0.5559, 0.6353] | 0.0602 | 0.0203 | [0.0205, 0.1000] |
| shape | triangle | 6 | 85 | 0.6335 | 0.7647 | [0.6643, 0.8422] | 0.1312 | 0.0465 | [0.0400, 0.2224] |
| shape | triangle | 7 | 5 | 0.7260 | 1.0000 | [0.5655, 1.0000] | 0.2740 | 0.0065 | [0.2613, 0.2867] |
| color | amber | 0 | 107 | 0.0821 | 0.0374 | [0.0146, 0.0922] | -0.0447 | 0.0186 | [-0.0811, -0.0084] |
| color | amber | 1 | 1798 | 0.1698 | 0.1591 | [0.1429, 0.1767] | -0.0107 | 0.0086 | [-0.0275, 0.0061] |
| color | amber | 2 | 2402 | 0.2404 | 0.2918 | [0.2740, 0.3103] | 0.0515 | 0.0092 | [0.0333, 0.0696] |
| color | amber | 3 | 693 | 0.3279 | 0.4199 | [0.3837, 0.4570] | 0.0920 | 0.0187 | [0.0553, 0.1287] |
| color | blue | 0 | 346 | 0.0849 | 0.0318 | [0.0178, 0.0560] | -0.0531 | 0.0094 | [-0.0716, -0.0347] |
| color | blue | 1 | 1536 | 0.1571 | 0.0827 | [0.0699, 0.0975] | -0.0744 | 0.0070 | [-0.0881, -0.0606] |
| color | blue | 2 | 2627 | 0.2325 | 0.2078 | [0.1928, 0.2238] | -0.0247 | 0.0079 | [-0.0401, -0.0092] |
| color | blue | 3 | 480 | 0.3406 | 0.3917 | [0.3490, 0.4360] | 0.0510 | 0.0223 | [0.0074, 0.0946] |
| color | blue | 4 | 11 | 0.4126 | 0.8182 | [0.5230, 0.9486] | 0.4055 | 0.1220 | [0.1665, 0.6446] |
| color | coral | 0 | 205 | 0.0854 | 0.0488 | [0.0267, 0.0875] | -0.0367 | 0.0152 | [-0.0664, -0.0069] |
| color | coral | 1 | 2973 | 0.1666 | 0.1117 | [0.1008, 0.1235] | -0.0550 | 0.0058 | [-0.0662, -0.0437] |
| color | coral | 2 | 1658 | 0.2358 | 0.2117 | [0.1927, 0.2320] | -0.0241 | 0.0100 | [-0.0437, -0.0045] |
| color | coral | 3 | 164 | 0.3202 | 0.2927 | [0.2284, 0.3664] | -0.0275 | 0.0356 | [-0.0973, 0.0423] |
| color | green | 0 | 59 | 0.0835 | 0.0508 | [0.0174, 0.1392] | -0.0326 | 0.0290 | [-0.0894, 0.0242] |
| color | green | 1 | 2456 | 0.1668 | 0.1107 | [0.0989, 0.1238] | -0.0560 | 0.0063 | [-0.0684, -0.0437] |
| color | green | 2 | 2319 | 0.2330 | 0.2475 | [0.2304, 0.2655] | 0.0145 | 0.0089 | [-0.0030, 0.0320] |
| color | green | 3 | 166 | 0.3255 | 0.3976 | [0.3263, 0.4735] | 0.0721 | 0.0378 | [-0.0021, 0.1463] |
| color | violet | 0 | 71 | 0.0891 | 0.0704 | [0.0305, 0.1545] | -0.0187 | 0.0305 | [-0.0785, 0.0412] |
| color | violet | 1 | 4922 | 0.1745 | 0.2383 | [0.2266, 0.2504] | 0.0639 | 0.0061 | [0.0520, 0.0757] |
| color | violet | 2 | 7 | 0.2002 | 0.4286 | [0.1582, 0.7495] | 0.2283 | 0.2021 | [-0.1677, 0.6243] |

Fixed label-permutation invariance checks:

| probability source | max metric discrepancy | max selected-log-mass discrepancy | overall maximum |
|---|---:|---:|---:|
| `cvae` | 0.000000000000 | 0.000000000000 | 0.000000000000 |
| `intercept_null` | 0.000000000000 | 0.000000000000 | 0.000000000000 |
| `oracle` | 0.000000000000 | 0.000000000000 | 0.000000000000 |

Joint predictive scores and paired CVAE comparisons:

| model | joint NLL | NLL SE | CVAE-minus-model log gain | paired SE | one-sided LCB |
|---|---:|---:|---:|---:|---:|
| `cvae` | 3.1228 | 0.0095 | NA | NA | NA |
| `independent_fitted_marginal` | 3.2091 | 0.0072 | 0.0863 | 0.0069 | 0.0749 |
| `intercept_null` | 3.3733 | 0.0034 | 0.2505 | 0.0096 | 0.2347 |
| `empirical_joint` | 3.2479 | 0.0082 | 0.1251 | 0.0092 | 0.1099 |
| `oracle` | 2.9665 | 0.0115 | -0.1563 | 0.0069 | -0.1676 |

Fitted-CVAE quadrature sensitivity (fixed subset, relative to primary order):

| check order | joint mean delta | joint RMSE | product RMSE | mean G change | p99 shared change | p99 G change |
|---:|---:|---:|---:|---:|---:|---:|
| 21 | -0.023342 | 0.123411 | 0.112439 | 0.000985 | 0.257011 | 0.270133 |
| 41 | -0.001439 | 0.055654 | 0.049023 | 0.000521 | 0.163428 | 0.128566 |

Secondary nested common-panel Monte Carlo diagnostic:

| M | joint NLL | product NLL | mean gain | joint RMSE vs GH | product RMSE vs GH |
|---:|---:|---:|---:|---:|---:|
| 16 | 3.6156 | 3.4742 | -0.1414 | 1.4587 | 0.8204 |
| 64 | 3.1908 | 3.3168 | 0.1260 | 0.4632 | 0.2961 |
| 256 | 3.1666 | 3.3023 | 0.1357 | 0.1856 | 0.1480 |
| 1024 | 3.1733 | 3.3008 | 0.1275 | 0.1333 | 0.0971 |

MC block gain SD: 0.0195; jackknife-corrected mean gain: 0.1307; median joint-integrand ESS: 104.0 (0.102 of M).

## Scenario: `mixed_conditional_independence`

### Seed-level primary results

| seed | epochs (best) | active z at best | KL by latent at best | CVAE marginal NLL | CVAE marginal Brier | joint NLL | fitted-product NLL | paired joint gain (SE) | marginal log gain (SE) | Brier improvement (SE) |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1701 | 30 (29) | 2 / 2 | 1.421 / 2.350 | 1.0499 | 0.5903 | 3.1702 | 3.1498 | -0.0204 (0.0034) | 0.0796 (0.0036) | 0.0626 (0.0020) |
| 9901 | 30 (30) | 2 / 2 | 1.954 / 1.722 | 1.0410 | 0.5894 | 3.1570 | 3.1229 | -0.0341 (0.0040) | 0.0890 (0.0036) | 0.0639 (0.0022) |
| 31415 | 30 (29) | 2 / 2 | 2.042 / 1.662 | 1.0526 | 0.5964 | 3.2078 | 3.1579 | -0.0499 (0.0055) | 0.0791 (0.0034) | 0.0578 (0.0021) |

### Seed 1701 detail

Restored checkpoint: epoch 29 of 30; validation beta-ELBO 0.9811; training KL/latent [1.42070, 2.35035]; active latent units 2 / 2; effective beta 0.2000.

Marginal predictive metrics:

| model | NLL | multiclass Brier | classwise ECE | max class-bin gap |
|---|---:|---:|---:|---:|
| `cvae` | 1.0499 | 0.5903 | 0.0616 | 0.5142 |
| `intercept_null` | 1.1295 | 0.6528 | 0.0102 | 0.0183 |
| `oracle` | 0.9945 | 0.5671 | 0.0116 | 0.7213 |

Classwise calibration-in-the-large and uncertainty:

| model | outcome | level | mean predicted | observed | observed Wilson 95% | CIL gap | CIL SE | CIL 95% | ECE | max gap |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cvae` | switch | off | 0.4946 | 0.4730 | [0.4592, 0.4869] | -0.0216 | 0.0065 | [-0.0343, -0.0090] | 0.0576 | 0.1055 |
| `cvae` | switch | on | 0.5054 | 0.5270 | [0.5131, 0.5408] | 0.0216 | 0.0065 | [0.0090, 0.0343] | 0.0576 | 0.1055 |
| `cvae` | shape | circle | 0.3650 | 0.3918 | [0.3784, 0.4054] | 0.0268 | 0.0063 | [0.0145, 0.0391] | 0.0583 | 0.1770 |
| `cvae` | shape | square | 0.2552 | 0.2802 | [0.2679, 0.2928] | 0.0250 | 0.0060 | [0.0132, 0.0368] | 0.0366 | 0.0985 |
| `cvae` | shape | triangle | 0.3798 | 0.3280 | [0.3151, 0.3411] | -0.0518 | 0.0062 | [-0.0639, -0.0397] | 0.1008 | 0.2753 |
| `cvae` | color | amber | 0.2847 | 0.2180 | [0.2068, 0.2297] | -0.0667 | 0.0057 | [-0.0778, -0.0555] | 0.0667 | 0.0901 |
| `cvae` | color | blue | 0.2726 | 0.1986 | [0.1878, 0.2099] | -0.0740 | 0.0054 | [-0.0846, -0.0634] | 0.0746 | 0.4907 |
| `cvae` | color | coral | 0.2043 | 0.2186 | [0.2074, 0.2303] | 0.0143 | 0.0057 | [0.0031, 0.0254] | 0.0361 | 0.1706 |
| `cvae` | color | green | 0.1689 | 0.1948 | [0.1841, 0.2060] | 0.0259 | 0.0055 | [0.0152, 0.0366] | 0.0314 | 0.5142 |
| `cvae` | color | violet | 0.0695 | 0.1700 | [0.1598, 0.1807] | 0.1005 | 0.0053 | [0.0901, 0.1109] | 0.1005 | 0.1030 |
| `intercept_null` | switch | off | 0.4588 | 0.4730 | [0.4592, 0.4869] | 0.0142 | 0.0071 | [0.0004, 0.0281] | 0.0142 | 0.0142 |
| `intercept_null` | switch | on | 0.5412 | 0.5270 | [0.5131, 0.5408] | -0.0142 | 0.0071 | [-0.0281, -0.0004] | 0.0142 | 0.0142 |
| `intercept_null` | shape | circle | 0.3967 | 0.3918 | [0.3784, 0.4054] | -0.0049 | 0.0069 | [-0.0185, 0.0086] | 0.0049 | 0.0049 |
| `intercept_null` | shape | square | 0.2870 | 0.2802 | [0.2679, 0.2928] | -0.0068 | 0.0064 | [-0.0193, 0.0056] | 0.0068 | 0.0068 |
| `intercept_null` | shape | triangle | 0.3163 | 0.3280 | [0.3151, 0.3411] | 0.0117 | 0.0066 | [-0.0013, 0.0248] | 0.0117 | 0.0117 |
| `intercept_null` | color | amber | 0.2207 | 0.2180 | [0.2068, 0.2297] | -0.0027 | 0.0058 | [-0.0142, 0.0087] | 0.0027 | 0.0027 |
| `intercept_null` | color | blue | 0.1943 | 0.1986 | [0.1878, 0.2099] | 0.0043 | 0.0056 | [-0.0067, 0.0154] | 0.0043 | 0.0043 |
| `intercept_null` | color | coral | 0.2157 | 0.2186 | [0.2074, 0.2303] | 0.0029 | 0.0058 | [-0.0086, 0.0143] | 0.0029 | 0.0029 |
| `intercept_null` | color | green | 0.1810 | 0.1948 | [0.1841, 0.2060] | 0.0138 | 0.0056 | [0.0028, 0.0248] | 0.0138 | 0.0138 |
| `intercept_null` | color | violet | 0.1883 | 0.1700 | [0.1598, 0.1807] | -0.0183 | 0.0053 | [-0.0287, -0.0078] | 0.0183 | 0.0183 |
| `oracle` | switch | off | 0.4698 | 0.4730 | [0.4592, 0.4869] | 0.0032 | 0.0064 | [-0.0093, 0.0157] | 0.0114 | 0.0740 |
| `oracle` | switch | on | 0.5302 | 0.5270 | [0.5131, 0.5408] | -0.0032 | 0.0064 | [-0.0157, 0.0093] | 0.0114 | 0.0740 |
| `oracle` | shape | circle | 0.3887 | 0.3918 | [0.3784, 0.4054] | 0.0031 | 0.0062 | [-0.0090, 0.0152] | 0.0175 | 0.0474 |
| `oracle` | shape | square | 0.2827 | 0.2802 | [0.2679, 0.2928] | -0.0025 | 0.0059 | [-0.0141, 0.0091] | 0.0107 | 0.2544 |
| `oracle` | shape | triangle | 0.3286 | 0.3280 | [0.3151, 0.3411] | -0.0006 | 0.0060 | [-0.0122, 0.0111] | 0.0129 | 0.0772 |
| `oracle` | color | amber | 0.2197 | 0.2180 | [0.2068, 0.2297] | -0.0017 | 0.0056 | [-0.0128, 0.0093] | 0.0094 | 0.2841 |
| `oracle` | color | blue | 0.1985 | 0.1986 | [0.1878, 0.2099] | 0.0001 | 0.0054 | [-0.0104, 0.0106] | 0.0126 | 0.7213 |
| `oracle` | color | coral | 0.2217 | 0.2186 | [0.2074, 0.2303] | -0.0031 | 0.0057 | [-0.0142, 0.0080] | 0.0082 | 0.1239 |
| `oracle` | color | green | 0.1871 | 0.1948 | [0.1841, 0.2060] | 0.0077 | 0.0054 | [-0.0029, 0.0183] | 0.0101 | 0.0458 |
| `oracle` | color | violet | 0.1730 | 0.1700 | [0.1598, 0.1807] | -0.0030 | 0.0053 | [-0.0134, 0.0074] | 0.0076 | 0.0115 |

CVAE classwise reliability bins (signed gap = observed - predicted):

| outcome | level | bin | n | mean predicted | observed | observed Wilson 95% | signed gap | gap SE | gap 95% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| switch | off | 0 | 6 | 0.0915 | 0.0000 | [0.0000, 0.3903] | -0.0915 | 0.0017 | [-0.0948, -0.0883] |
| switch | off | 1 | 79 | 0.1662 | 0.0886 | [0.0436, 0.1718] | -0.0775 | 0.0325 | [-0.1412, -0.0139] |
| switch | off | 2 | 471 | 0.2469 | 0.1507 | [0.1213, 0.1859] | -0.0961 | 0.0164 | [-0.1282, -0.0640] |
| switch | off | 3 | 855 | 0.3640 | 0.2585 | [0.2303, 0.2889] | -0.1055 | 0.0149 | [-0.1348, -0.0763] |
| switch | off | 4 | 1203 | 0.4401 | 0.4156 | [0.3881, 0.4437] | -0.0245 | 0.0142 | [-0.0522, 0.0033] |
| switch | off | 5 | 927 | 0.5562 | 0.5275 | [0.4953, 0.5595] | -0.0287 | 0.0164 | [-0.0608, 0.0034] |
| switch | off | 6 | 978 | 0.6325 | 0.6892 | [0.6595, 0.7174] | 0.0566 | 0.0148 | [0.0277, 0.0856] |
| switch | off | 7 | 389 | 0.7515 | 0.8380 | [0.7982, 0.8713] | 0.0865 | 0.0187 | [0.0498, 0.1232] |
| switch | off | 8 | 85 | 0.8195 | 0.8235 | [0.7290, 0.8900] | 0.0040 | 0.0414 | [-0.0770, 0.0851] |
| switch | off | 9 | 7 | 0.9187 | 1.0000 | [0.6457, 1.0000] | 0.0813 | 0.0025 | [0.0764, 0.0861] |
| switch | on | 0 | 7 | 0.0813 | 0.0000 | [0.0000, 0.3543] | -0.0813 | 0.0025 | [-0.0861, -0.0764] |
| switch | on | 1 | 85 | 0.1805 | 0.1765 | [0.1100, 0.2710] | -0.0040 | 0.0414 | [-0.0851, 0.0770] |
| switch | on | 2 | 389 | 0.2485 | 0.1620 | [0.1287, 0.2018] | -0.0865 | 0.0187 | [-0.1232, -0.0498] |
| switch | on | 3 | 978 | 0.3675 | 0.3108 | [0.2826, 0.3405] | -0.0566 | 0.0148 | [-0.0856, -0.0277] |
| switch | on | 4 | 927 | 0.4438 | 0.4725 | [0.4405, 0.5047] | 0.0287 | 0.0164 | [-0.0034, 0.0608] |
| switch | on | 5 | 1203 | 0.5599 | 0.5844 | [0.5563, 0.6119] | 0.0245 | 0.0142 | [-0.0033, 0.0522] |
| switch | on | 6 | 855 | 0.6360 | 0.7415 | [0.7111, 0.7697] | 0.1055 | 0.0149 | [0.0763, 0.1348] |
| switch | on | 7 | 471 | 0.7531 | 0.8493 | [0.8141, 0.8787] | 0.0961 | 0.0164 | [0.0640, 0.1282] |
| switch | on | 8 | 79 | 0.8338 | 0.9114 | [0.8282, 0.9564] | 0.0775 | 0.0325 | [0.0139, 0.1412] |
| switch | on | 9 | 6 | 0.9085 | 1.0000 | [0.6097, 1.0000] | 0.0915 | 0.0017 | [0.0883, 0.0948] |
| shape | circle | 0 | 114 | 0.0749 | 0.0965 | [0.0547, 0.1646] | 0.0216 | 0.0277 | [-0.0327, 0.0759] |
| shape | circle | 1 | 599 | 0.1529 | 0.1035 | [0.0816, 0.1305] | -0.0494 | 0.0123 | [-0.0735, -0.0252] |
| shape | circle | 2 | 1194 | 0.2530 | 0.2119 | [0.1897, 0.2360] | -0.0411 | 0.0118 | [-0.0643, -0.0180] |
| shape | circle | 3 | 1055 | 0.3495 | 0.3782 | [0.3494, 0.4079] | 0.0287 | 0.0148 | [-0.0003, 0.0578] |
| shape | circle | 4 | 1029 | 0.4479 | 0.5151 | [0.4845, 0.5455] | 0.0672 | 0.0155 | [0.0367, 0.0976] |
| shape | circle | 5 | 659 | 0.5439 | 0.6388 | [0.6015, 0.6746] | 0.0949 | 0.0186 | [0.0584, 0.1315] |
| shape | circle | 6 | 261 | 0.6459 | 0.7778 | [0.7235, 0.8240] | 0.1319 | 0.0257 | [0.0815, 0.1822] |
| shape | circle | 7 | 82 | 0.7350 | 0.8902 | [0.8044, 0.9412] | 0.1552 | 0.0349 | [0.0868, 0.2237] |
| shape | circle | 8 | 7 | 0.8230 | 1.0000 | [0.6457, 1.0000] | 0.1770 | 0.0053 | [0.1665, 0.1875] |
| shape | square | 0 | 406 | 0.0761 | 0.0567 | [0.0380, 0.0836] | -0.0194 | 0.0115 | [-0.0419, 0.0031] |
| shape | square | 1 | 1272 | 0.1545 | 0.1384 | [0.1205, 0.1584] | -0.0162 | 0.0097 | [-0.0351, 0.0028] |
| shape | square | 2 | 1741 | 0.2468 | 0.2740 | [0.2535, 0.2954] | 0.0272 | 0.0106 | [0.0063, 0.0481] |
| shape | square | 3 | 929 | 0.3465 | 0.3994 | [0.3683, 0.4312] | 0.0529 | 0.0160 | [0.0215, 0.0843] |
| shape | square | 4 | 545 | 0.4392 | 0.5376 | [0.4956, 0.5791] | 0.0985 | 0.0214 | [0.0566, 0.1403] |
| shape | square | 5 | 98 | 0.5324 | 0.5714 | [0.4726, 0.6649] | 0.0390 | 0.0503 | [-0.0596, 0.1376] |
| shape | square | 6 | 9 | 0.6207 | 0.5556 | [0.2667, 0.8112] | -0.0651 | 0.1763 | [-0.4106, 0.2804] |
| shape | triangle | 1 | 125 | 0.1712 | 0.0320 | [0.0125, 0.0794] | -0.1392 | 0.0158 | [-0.1701, -0.1083] |
| shape | triangle | 2 | 887 | 0.2635 | 0.1251 | [0.1050, 0.1485] | -0.1383 | 0.0110 | [-0.1600, -0.1167] |
| shape | triangle | 3 | 2046 | 0.3511 | 0.2331 | [0.2153, 0.2519] | -0.1180 | 0.0093 | [-0.1362, -0.0998] |
| shape | triangle | 4 | 1371 | 0.4443 | 0.4573 | [0.4311, 0.4838] | 0.0131 | 0.0134 | [-0.0131, 0.0393] |
| shape | triangle | 5 | 478 | 0.5392 | 0.7155 | [0.6734, 0.7541] | 0.1763 | 0.0205 | [0.1362, 0.2164] |
| shape | triangle | 6 | 88 | 0.6265 | 0.8409 | [0.7505, 0.9028] | 0.2144 | 0.0388 | [0.1384, 0.2904] |
| shape | triangle | 7 | 5 | 0.7247 | 1.0000 | [0.5655, 1.0000] | 0.2753 | 0.0107 | [0.2543, 0.2963] |
| color | amber | 0 | 30 | 0.0765 | 0.0333 | [0.0059, 0.1667] | -0.0432 | 0.0330 | [-0.1080, 0.0215] |
| color | amber | 1 | 930 | 0.1653 | 0.0849 | [0.0687, 0.1046] | -0.0804 | 0.0091 | [-0.0982, -0.0626] |
| color | amber | 2 | 1669 | 0.2567 | 0.1666 | [0.1495, 0.1852] | -0.0901 | 0.0091 | [-0.1079, -0.0723] |
| color | amber | 3 | 2118 | 0.3455 | 0.2998 | [0.2807, 0.3197] | -0.0457 | 0.0099 | [-0.0651, -0.0262] |
| color | amber | 4 | 253 | 0.4235 | 0.3834 | [0.3257, 0.4446] | -0.0401 | 0.0307 | [-0.1002, 0.0200] |
| color | blue | 0 | 274 | 0.0616 | 0.0109 | [0.0037, 0.0317] | -0.0506 | 0.0065 | [-0.0634, -0.0378] |
| color | blue | 1 | 904 | 0.1568 | 0.0586 | [0.0451, 0.0759] | -0.0981 | 0.0079 | [-0.1135, -0.0827] |
| color | blue | 2 | 1604 | 0.2564 | 0.1540 | [0.1372, 0.1725] | -0.1024 | 0.0090 | [-0.1200, -0.0849] |
| color | blue | 3 | 1864 | 0.3454 | 0.2892 | [0.2690, 0.3102] | -0.0562 | 0.0105 | [-0.0767, -0.0357] |
| color | blue | 4 | 353 | 0.4220 | 0.4249 | [0.3744, 0.4770] | 0.0030 | 0.0263 | [-0.0486, 0.0545] |
| color | blue | 5 | 1 | 0.5093 | 1.0000 | [0.2065, 1.0000] | 0.4907 | NA | [NA, NA] |
| color | coral | 0 | 122 | 0.0868 | 0.0328 | [0.0128, 0.0813] | -0.0540 | 0.0162 | [-0.0857, -0.0223] |
| color | coral | 1 | 2371 | 0.1568 | 0.1371 | [0.1238, 0.1515] | -0.0197 | 0.0070 | [-0.0334, -0.0059] |
| color | coral | 2 | 2104 | 0.2416 | 0.2928 | [0.2737, 0.3126] | 0.0512 | 0.0099 | [0.0318, 0.0706] |
| color | coral | 3 | 395 | 0.3237 | 0.3696 | [0.3235, 0.4183] | 0.0459 | 0.0243 | [-0.0016, 0.0935] |
| color | coral | 4 | 8 | 0.4206 | 0.2500 | [0.0715, 0.5907] | -0.1706 | 0.1599 | [-0.4840, 0.1428] |
| color | green | 0 | 390 | 0.0826 | 0.0564 | [0.0375, 0.0839] | -0.0262 | 0.0116 | [-0.0490, -0.0034] |
| color | green | 1 | 3276 | 0.1447 | 0.1554 | [0.1434, 0.1682] | 0.0107 | 0.0063 | [-0.0016, 0.0230] |
| color | green | 2 | 1130 | 0.2387 | 0.3204 | [0.2938, 0.3481] | 0.0816 | 0.0138 | [0.0545, 0.1088] |
| color | green | 3 | 187 | 0.3263 | 0.4118 | [0.3437, 0.4834] | 0.0855 | 0.0359 | [0.0150, 0.1559] |
| color | green | 4 | 16 | 0.4314 | 0.2500 | [0.1018, 0.4950] | -0.1814 | 0.1149 | [-0.4065, 0.0437] |
| color | green | 5 | 1 | 0.5142 | 0.0000 | [0.0000, 0.7935] | -0.5142 | NA | [NA, NA] |
| color | violet | 0 | 4709 | 0.0671 | 0.1701 | [0.1596, 0.1811] | 0.1030 | 0.0055 | [0.0923, 0.1137] |
| color | violet | 1 | 291 | 0.1087 | 0.1684 | [0.1298, 0.2156] | 0.0597 | 0.0220 | [0.0165, 0.1028] |

Fixed label-permutation invariance checks:

| probability source | max metric discrepancy | max selected-log-mass discrepancy | overall maximum |
|---|---:|---:|---:|
| `cvae` | 0.000000000000 | 0.000000000000 | 0.000000000000 |
| `intercept_null` | 0.000000000000 | 0.000000000000 | 0.000000000000 |
| `oracle` | 0.000000000000 | 0.000000000000 | 0.000000000000 |

Joint predictive scores and paired CVAE comparisons:

| model | joint NLL | NLL SE | CVAE-minus-model log gain | paired SE | one-sided LCB |
|---|---:|---:|---:|---:|---:|
| `cvae` | 3.1702 | 0.0120 | NA | NA | NA |
| `independent_fitted_marginal` | 3.1498 | 0.0113 | -0.0204 | 0.0034 | -0.0260 |
| `intercept_null` | 3.3885 | 0.0029 | 0.2183 | 0.0116 | 0.1993 |
| `empirical_joint` | 3.3430 | 0.0049 | 0.1728 | 0.0103 | 0.1559 |
| `oracle` | 2.9834 | 0.0124 | -0.1868 | 0.0086 | -0.2009 |

Fitted-CVAE quadrature sensitivity (fixed subset, relative to primary order):

| check order | joint mean delta | joint RMSE | product RMSE | mean G change | p99 shared change | p99 G change |
|---:|---:|---:|---:|---:|---:|---:|
| 21 | 0.020061 | 0.171550 | 0.063621 | 0.018126 | 0.536469 | 0.505119 |
| 41 | -0.015296 | 0.101218 | 0.033457 | -0.012918 | 0.254336 | 0.288480 |

Secondary nested common-panel Monte Carlo diagnostic:

| M | joint NLL | product NLL | mean gain | joint RMSE vs GH | product RMSE vs GH |
|---:|---:|---:|---:|---:|---:|
| 16 | 4.7442 | 3.3648 | -1.3794 | 3.8823 | 0.8157 |
| 64 | 3.4354 | 3.1897 | -0.2456 | 1.4351 | 0.3419 |
| 256 | 3.2120 | 3.1368 | -0.0752 | 0.4719 | 0.1673 |
| 1024 | 3.1586 | 3.1193 | -0.0392 | 0.2115 | 0.0800 |

MC block gain SD: 0.0245; jackknife-corrected mean gain: -0.0320; median joint-integrand ESS: 76.6 (0.075 of M).

Conditional-independence oracle common-panel shared-vs-product maximum absolute differences: M=16: 8.882e-16, M=64: 8.882e-16, M=256: 8.882e-16, M=1024: 8.882e-16.

### Seed 9901 detail

Restored checkpoint: epoch 30 of 30; validation beta-ELBO 1.1812; training KL/latent [1.95435, 1.72235]; active latent units 2 / 2; effective beta 0.2000.

Marginal predictive metrics:

| model | NLL | multiclass Brier | classwise ECE | max class-bin gap |
|---|---:|---:|---:|---:|
| `cvae` | 1.0410 | 0.5894 | 0.0590 | 0.6951 |
| `intercept_null` | 1.1300 | 0.6533 | 0.0110 | 0.0196 |
| `oracle` | 0.9917 | 0.5658 | 0.0145 | 0.5179 |

Classwise calibration-in-the-large and uncertainty:

| model | outcome | level | mean predicted | observed | observed Wilson 95% | CIL gap | CIL SE | CIL 95% | ECE | max gap |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cvae` | switch | off | 0.4886 | 0.4816 | [0.4678, 0.4955] | -0.0070 | 0.0064 | [-0.0196, 0.0056] | 0.0361 | 0.0842 |
| `cvae` | switch | on | 0.5114 | 0.5184 | [0.5045, 0.5322] | 0.0070 | 0.0064 | [-0.0056, 0.0196] | 0.0361 | 0.0842 |
| `cvae` | shape | circle | 0.4447 | 0.3920 | [0.3786, 0.4056] | -0.0527 | 0.0063 | [-0.0651, -0.0402] | 0.0820 | 0.2292 |
| `cvae` | shape | square | 0.2129 | 0.2848 | [0.2725, 0.2975] | 0.0719 | 0.0060 | [0.0601, 0.0838] | 0.0771 | 0.3342 |
| `cvae` | shape | triangle | 0.3424 | 0.3232 | [0.3104, 0.3363] | -0.0192 | 0.0060 | [-0.0310, -0.0075] | 0.0585 | 0.1715 |
| `cvae` | color | amber | 0.1224 | 0.2116 | [0.2005, 0.2231] | 0.0892 | 0.0056 | [0.0782, 0.1003] | 0.0892 | 0.6951 |
| `cvae` | color | blue | 0.2156 | 0.2032 | [0.1923, 0.2146] | -0.0124 | 0.0053 | [-0.0229, -0.0019] | 0.0194 | 0.3893 |
| `cvae` | color | coral | 0.3017 | 0.2224 | [0.2111, 0.2341] | -0.0793 | 0.0058 | [-0.0906, -0.0681] | 0.0795 | 0.3972 |
| `cvae` | color | green | 0.1082 | 0.1868 | [0.1762, 0.1978] | 0.0786 | 0.0054 | [0.0680, 0.0893] | 0.0786 | 0.1776 |
| `cvae` | color | violet | 0.2521 | 0.1760 | [0.1657, 0.1868] | -0.0761 | 0.0054 | [-0.0866, -0.0656] | 0.0761 | 0.4100 |
| `intercept_null` | switch | off | 0.4620 | 0.4816 | [0.4678, 0.4955] | 0.0196 | 0.0071 | [0.0057, 0.0334] | 0.0196 | 0.0196 |
| `intercept_null` | switch | on | 0.5380 | 0.5184 | [0.5045, 0.5322] | -0.0196 | 0.0071 | [-0.0334, -0.0057] | 0.0196 | 0.0196 |
| `intercept_null` | shape | circle | 0.3912 | 0.3920 | [0.3786, 0.4056] | 0.0008 | 0.0069 | [-0.0128, 0.0143] | 0.0008 | 0.0008 |
| `intercept_null` | shape | square | 0.2728 | 0.2848 | [0.2725, 0.2975] | 0.0120 | 0.0064 | [-0.0005, 0.0245] | 0.0120 | 0.0120 |
| `intercept_null` | shape | triangle | 0.3360 | 0.3232 | [0.3104, 0.3363] | -0.0128 | 0.0066 | [-0.0258, 0.0002] | 0.0128 | 0.0128 |
| `intercept_null` | color | amber | 0.2205 | 0.2116 | [0.2005, 0.2231] | -0.0089 | 0.0058 | [-0.0202, 0.0024] | 0.0089 | 0.0089 |
| `intercept_null` | color | blue | 0.1955 | 0.2032 | [0.1923, 0.2146] | 0.0077 | 0.0057 | [-0.0035, 0.0189] | 0.0077 | 0.0077 |
| `intercept_null` | color | coral | 0.2252 | 0.2224 | [0.2111, 0.2341] | -0.0028 | 0.0059 | [-0.0144, 0.0087] | 0.0028 | 0.0028 |
| `intercept_null` | color | green | 0.1823 | 0.1868 | [0.1762, 0.1978] | 0.0045 | 0.0055 | [-0.0063, 0.0153] | 0.0045 | 0.0045 |
| `intercept_null` | color | violet | 0.1765 | 0.1760 | [0.1657, 0.1868] | -0.0005 | 0.0054 | [-0.0111, 0.0100] | 0.0005 | 0.0005 |
| `oracle` | switch | off | 0.4686 | 0.4816 | [0.4678, 0.4955] | 0.0130 | 0.0064 | [0.0005, 0.0255] | 0.0202 | 0.0422 |
| `oracle` | switch | on | 0.5314 | 0.5184 | [0.5045, 0.5322] | -0.0130 | 0.0064 | [-0.0255, -0.0005] | 0.0202 | 0.0422 |
| `oracle` | shape | circle | 0.3912 | 0.3920 | [0.3786, 0.4056] | 0.0008 | 0.0062 | [-0.0114, 0.0129] | 0.0184 | 0.0594 |
| `oracle` | shape | square | 0.2820 | 0.2848 | [0.2725, 0.2975] | 0.0028 | 0.0059 | [-0.0088, 0.0145] | 0.0138 | 0.1069 |
| `oracle` | shape | triangle | 0.3268 | 0.3232 | [0.3104, 0.3363] | -0.0036 | 0.0059 | [-0.0152, 0.0080] | 0.0091 | 0.0770 |
| `oracle` | color | amber | 0.2201 | 0.2116 | [0.2005, 0.2231] | -0.0085 | 0.0055 | [-0.0193, 0.0023] | 0.0128 | 0.0658 |
| `oracle` | color | blue | 0.1977 | 0.2032 | [0.1923, 0.2146] | 0.0055 | 0.0053 | [-0.0049, 0.0160] | 0.0113 | 0.2726 |
| `oracle` | color | coral | 0.2226 | 0.2224 | [0.2111, 0.2341] | -0.0002 | 0.0057 | [-0.0114, 0.0111] | 0.0125 | 0.1343 |
| `oracle` | color | green | 0.1862 | 0.1868 | [0.1762, 0.1978] | 0.0006 | 0.0054 | [-0.0099, 0.0111] | 0.0085 | 0.5179 |
| `oracle` | color | violet | 0.1735 | 0.1760 | [0.1657, 0.1868] | 0.0025 | 0.0053 | [-0.0080, 0.0130] | 0.0028 | 0.0086 |

CVAE classwise reliability bins (signed gap = observed - predicted):

| outcome | level | bin | n | mean predicted | observed | observed Wilson 95% | signed gap | gap SE | gap 95% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| switch | off | 0 | 10 | 0.0809 | 0.0000 | [0.0000, 0.2775] | -0.0809 | 0.0042 | [-0.0891, -0.0727] |
| switch | off | 1 | 193 | 0.1682 | 0.0984 | [0.0639, 0.1486] | -0.0698 | 0.0212 | [-0.1114, -0.0281] |
| switch | off | 2 | 546 | 0.2566 | 0.1960 | [0.1648, 0.2313] | -0.0606 | 0.0169 | [-0.0938, -0.0275] |
| switch | off | 3 | 932 | 0.3545 | 0.3305 | [0.3010, 0.3613] | -0.0240 | 0.0153 | [-0.0541, 0.0060] |
| switch | off | 4 | 1008 | 0.4493 | 0.4117 | [0.3817, 0.4424] | -0.0376 | 0.0155 | [-0.0680, -0.0071] |
| switch | off | 5 | 946 | 0.5476 | 0.5655 | [0.5337, 0.5968] | 0.0179 | 0.0161 | [-0.0136, 0.0494] |
| switch | off | 6 | 700 | 0.6470 | 0.6671 | [0.6314, 0.7011] | 0.0202 | 0.0177 | [-0.0145, 0.0548] |
| switch | off | 7 | 463 | 0.7437 | 0.7991 | [0.7602, 0.8331] | 0.0554 | 0.0186 | [0.0189, 0.0920] |
| switch | off | 8 | 174 | 0.8353 | 0.9195 | [0.8695, 0.9515] | 0.0842 | 0.0206 | [0.0438, 0.1246] |
| switch | off | 9 | 28 | 0.9192 | 0.9643 | [0.8229, 0.9937] | 0.0451 | 0.0353 | [-0.0240, 0.1142] |
| switch | on | 0 | 28 | 0.0808 | 0.0357 | [0.0063, 0.1771] | -0.0451 | 0.0353 | [-0.1142, 0.0240] |
| switch | on | 1 | 174 | 0.1647 | 0.0805 | [0.0485, 0.1305] | -0.0842 | 0.0206 | [-0.1246, -0.0438] |
| switch | on | 2 | 463 | 0.2563 | 0.2009 | [0.1669, 0.2398] | -0.0554 | 0.0186 | [-0.0920, -0.0189] |
| switch | on | 3 | 700 | 0.3530 | 0.3329 | [0.2989, 0.3686] | -0.0202 | 0.0177 | [-0.0548, 0.0145] |
| switch | on | 4 | 946 | 0.4524 | 0.4345 | [0.4032, 0.4663] | -0.0179 | 0.0161 | [-0.0494, 0.0136] |
| switch | on | 5 | 1008 | 0.5507 | 0.5883 | [0.5576, 0.6183] | 0.0376 | 0.0155 | [0.0071, 0.0680] |
| switch | on | 6 | 932 | 0.6455 | 0.6695 | [0.6387, 0.6990] | 0.0240 | 0.0153 | [-0.0060, 0.0541] |
| switch | on | 7 | 546 | 0.7434 | 0.8040 | [0.7687, 0.8352] | 0.0606 | 0.0169 | [0.0275, 0.0938] |
| switch | on | 8 | 193 | 0.8318 | 0.9016 | [0.8514, 0.9361] | 0.0698 | 0.0212 | [0.0281, 0.1114] |
| switch | on | 9 | 10 | 0.9191 | 1.0000 | [0.7225, 1.0000] | 0.0809 | 0.0042 | [0.0727, 0.0891] |
| shape | circle | 0 | 13 | 0.0807 | 0.0000 | [0.0000, 0.2281] | -0.0807 | 0.0044 | [-0.0892, -0.0721] |
| shape | circle | 1 | 149 | 0.1642 | 0.0738 | [0.0417, 0.1274] | -0.0904 | 0.0215 | [-0.1326, -0.0482] |
| shape | circle | 2 | 522 | 0.2602 | 0.1015 | [0.0785, 0.1304] | -0.1586 | 0.0132 | [-0.1845, -0.1328] |
| shape | circle | 3 | 1170 | 0.3552 | 0.2308 | [0.2075, 0.2558] | -0.1244 | 0.0122 | [-0.1484, -0.1004] |
| shape | circle | 4 | 1417 | 0.4511 | 0.3867 | [0.3617, 0.4124] | -0.0644 | 0.0129 | [-0.0896, -0.0392] |
| shape | circle | 5 | 1139 | 0.5457 | 0.5435 | [0.5144, 0.5722] | -0.0022 | 0.0147 | [-0.0310, 0.0265] |
| shape | circle | 6 | 494 | 0.6392 | 0.7429 | [0.7026, 0.7795] | 0.1037 | 0.0196 | [0.0652, 0.1422] |
| shape | circle | 7 | 95 | 0.7287 | 0.9579 | [0.8967, 0.9835] | 0.2292 | 0.0205 | [0.1890, 0.2694] |
| shape | circle | 8 | 1 | 0.8031 | 1.0000 | [0.2065, 1.0000] | 0.1969 | NA | [NA, NA] |
| shape | square | 0 | 388 | 0.0774 | 0.0438 | [0.0275, 0.0690] | -0.0336 | 0.0104 | [-0.0540, -0.0131] |
| shape | square | 1 | 2056 | 0.1516 | 0.1639 | [0.1485, 0.1805] | 0.0123 | 0.0081 | [-0.0036, 0.0282] |
| shape | square | 2 | 1727 | 0.2441 | 0.3526 | [0.3305, 0.3755] | 0.1086 | 0.0115 | [0.0861, 0.1310] |
| shape | square | 3 | 650 | 0.3415 | 0.5262 | [0.4877, 0.5643] | 0.1846 | 0.0195 | [0.1464, 0.2229] |
| shape | square | 4 | 165 | 0.4356 | 0.6485 | [0.5730, 0.7172] | 0.2129 | 0.0372 | [0.1400, 0.2857] |
| shape | square | 5 | 14 | 0.5230 | 0.8571 | [0.6006, 0.9599] | 0.3342 | 0.0955 | [0.1469, 0.5214] |
| shape | triangle | 0 | 132 | 0.0750 | 0.0455 | [0.0210, 0.0956] | -0.0296 | 0.0180 | [-0.0648, 0.0056] |
| shape | triangle | 1 | 749 | 0.1580 | 0.0774 | [0.0604, 0.0988] | -0.0805 | 0.0098 | [-0.0998, -0.0613] |
| shape | triangle | 2 | 1345 | 0.2515 | 0.1844 | [0.1646, 0.2060] | -0.0671 | 0.0105 | [-0.0877, -0.0465] |
| shape | triangle | 3 | 1165 | 0.3517 | 0.3176 | [0.2915, 0.3449] | -0.0341 | 0.0136 | [-0.0608, -0.0074] |
| shape | triangle | 4 | 767 | 0.4454 | 0.4811 | [0.4459, 0.5165] | 0.0357 | 0.0180 | [0.0004, 0.0710] |
| shape | triangle | 5 | 559 | 0.5432 | 0.6154 | [0.5744, 0.6548] | 0.0722 | 0.0204 | [0.0322, 0.1122] |
| shape | triangle | 6 | 204 | 0.6449 | 0.7451 | [0.6811, 0.8000] | 0.1002 | 0.0303 | [0.0408, 0.1597] |
| shape | triangle | 7 | 67 | 0.7338 | 0.8507 | [0.7466, 0.9169] | 0.1170 | 0.0441 | [0.0306, 0.2034] |
| shape | triangle | 8 | 12 | 0.8285 | 1.0000 | [0.7575, 1.0000] | 0.1715 | 0.0065 | [0.1588, 0.1843] |
| color | amber | 0 | 1553 | 0.0747 | 0.0953 | [0.0817, 0.1109] | 0.0206 | 0.0074 | [0.0060, 0.0351] |
| color | amber | 1 | 3248 | 0.1392 | 0.2478 | [0.2333, 0.2630] | 0.1086 | 0.0075 | [0.0939, 0.1233] |
| color | amber | 2 | 198 | 0.2187 | 0.5253 | [0.4559, 0.5937] | 0.3066 | 0.0355 | [0.2370, 0.3762] |
| color | amber | 3 | 1 | 0.3049 | 1.0000 | [0.2065, 1.0000] | 0.6951 | NA | [NA, NA] |
| color | blue | 0 | 1131 | 0.0632 | 0.0424 | [0.0322, 0.0558] | -0.0207 | 0.0060 | [-0.0325, -0.0090] |
| color | blue | 1 | 1405 | 0.1468 | 0.1295 | [0.1130, 0.1481] | -0.0172 | 0.0090 | [-0.0348, 0.0003] |
| color | blue | 2 | 1088 | 0.2476 | 0.2261 | [0.2022, 0.2519] | -0.0215 | 0.0127 | [-0.0464, 0.0033] |
| color | blue | 3 | 865 | 0.3484 | 0.3387 | [0.3080, 0.3709] | -0.0097 | 0.0161 | [-0.0413, 0.0219] |
| color | blue | 4 | 449 | 0.4386 | 0.4566 | [0.4111, 0.5028] | 0.0179 | 0.0234 | [-0.0278, 0.0637] |
| color | blue | 5 | 61 | 0.5248 | 0.6721 | [0.5472, 0.7766] | 0.1474 | 0.0603 | [0.0292, 0.2655] |
| color | blue | 6 | 1 | 0.6107 | 1.0000 | [0.2065, 1.0000] | 0.3893 | NA | [NA, NA] |
| color | coral | 0 | 144 | 0.0802 | 0.0486 | [0.0237, 0.0969] | -0.0316 | 0.0180 | [-0.0668, 0.0036] |
| color | coral | 1 | 822 | 0.1557 | 0.0973 | [0.0789, 0.1195] | -0.0584 | 0.0104 | [-0.0787, -0.0381] |
| color | coral | 2 | 1298 | 0.2567 | 0.1764 | [0.1567, 0.1981] | -0.0803 | 0.0106 | [-0.1011, -0.0596] |
| color | coral | 3 | 1878 | 0.3508 | 0.2630 | [0.2436, 0.2834] | -0.0878 | 0.0101 | [-0.1076, -0.0679] |
| color | coral | 4 | 803 | 0.4336 | 0.3499 | [0.3177, 0.3836] | -0.0836 | 0.0168 | [-0.1166, -0.0506] |
| color | coral | 5 | 54 | 0.5237 | 0.3704 | [0.2542, 0.5037] | -0.1533 | 0.0661 | [-0.2828, -0.0239] |
| color | coral | 6 | 1 | 0.6028 | 1.0000 | [0.2065, 1.0000] | 0.3972 | NA | [NA, NA] |
| color | green | 0 | 2585 | 0.0804 | 0.1137 | [0.1021, 0.1265] | 0.0333 | 0.0062 | [0.0211, 0.0455] |
| color | green | 1 | 2276 | 0.1321 | 0.2562 | [0.2386, 0.2745] | 0.1241 | 0.0091 | [0.1061, 0.1420] |
| color | green | 2 | 133 | 0.2284 | 0.4060 | [0.3263, 0.4910] | 0.1776 | 0.0430 | [0.0934, 0.2619] |
| color | green | 3 | 6 | 0.3303 | 0.5000 | [0.1876, 0.8124] | 0.1697 | 0.2246 | [-0.2706, 0.6100] |
| color | violet | 1 | 545 | 0.1822 | 0.0991 | [0.0767, 0.1270] | -0.0832 | 0.0128 | [-0.1082, -0.0581] |
| color | violet | 2 | 3919 | 0.2525 | 0.1812 | [0.1694, 0.1935] | -0.0713 | 0.0061 | [-0.0833, -0.0593] |
| color | violet | 3 | 535 | 0.3203 | 0.2168 | [0.1840, 0.2537] | -0.1035 | 0.0179 | [-0.1385, -0.0685] |
| color | violet | 4 | 1 | 0.4100 | 0.0000 | [0.0000, 0.7935] | -0.4100 | NA | [NA, NA] |

Fixed label-permutation invariance checks:

| probability source | max metric discrepancy | max selected-log-mass discrepancy | overall maximum |
|---|---:|---:|---:|
| `cvae` | 0.000000000000 | 0.000000000000 | 0.000000000000 |
| `intercept_null` | 0.000000000000 | 0.000000000000 | 0.000000000000 |
| `oracle` | 0.000000000000 | 0.000000000000 | 0.000000000000 |

Joint predictive scores and paired CVAE comparisons:

| model | joint NLL | NLL SE | CVAE-minus-model log gain | paired SE | one-sided LCB |
|---|---:|---:|---:|---:|---:|
| `cvae` | 3.1570 | 0.0125 | NA | NA | NA |
| `independent_fitted_marginal` | 3.1229 | 0.0113 | -0.0341 | 0.0040 | -0.0406 |
| `intercept_null` | 3.3899 | 0.0029 | 0.2329 | 0.0119 | 0.2133 |
| `empirical_joint` | 3.3386 | 0.0055 | 0.1816 | 0.0105 | 0.1643 |
| `oracle` | 2.9751 | 0.0127 | -0.1820 | 0.0087 | -0.1963 |

Fitted-CVAE quadrature sensitivity (fixed subset, relative to primary order):

| check order | joint mean delta | joint RMSE | product RMSE | mean G change | p99 shared change | p99 G change |
|---:|---:|---:|---:|---:|---:|---:|
| 21 | 0.002261 | 0.110157 | 0.031221 | 0.000595 | 0.369995 | 0.340112 |
| 41 | -0.003449 | 0.047006 | 0.016908 | -0.001668 | 0.124907 | 0.164458 |

Secondary nested common-panel Monte Carlo diagnostic:

| M | joint NLL | product NLL | mean gain | joint RMSE vs GH | product RMSE vs GH |
|---:|---:|---:|---:|---:|---:|
| 16 | 3.9494 | 3.2642 | -0.6852 | 2.4547 | 0.6669 |
| 64 | 3.2420 | 3.1520 | -0.0900 | 0.6177 | 0.3062 |
| 256 | 3.1726 | 3.0933 | -0.0793 | 0.2758 | 0.1378 |
| 1024 | 3.1468 | 3.0979 | -0.0489 | 0.1400 | 0.0691 |

MC block gain SD: 0.0378; jackknife-corrected mean gain: -0.0432; median joint-integrand ESS: 95.9 (0.094 of M).

Conditional-independence oracle common-panel shared-vs-product maximum absolute differences: M=16: 4.441e-16, M=64: 8.882e-16, M=256: 8.882e-16, M=1024: 8.882e-16.

### Seed 31415 detail

Restored checkpoint: epoch 29 of 30; validation beta-ELBO 1.0617; training KL/latent [2.04210, 1.66159]; active latent units 2 / 2; effective beta 0.2000.

Marginal predictive metrics:

| model | NLL | multiclass Brier | classwise ECE | max class-bin gap |
|---|---:|---:|---:|---:|
| `cvae` | 1.0526 | 0.5964 | 0.0640 | 0.3659 |
| `intercept_null` | 1.1318 | 0.6542 | 0.0132 | 0.0225 |
| `oracle` | 0.9968 | 0.5680 | 0.0162 | 0.9152 |

Classwise calibration-in-the-large and uncertainty:

| model | outcome | level | mean predicted | observed | observed Wilson 95% | CIL gap | CIL SE | CIL 95% | ECE | max gap |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cvae` | switch | off | 0.5124 | 0.4806 | [0.4668, 0.4945] | -0.0318 | 0.0064 | [-0.0445, -0.0192] | 0.0510 | 0.1153 |
| `cvae` | switch | on | 0.4876 | 0.5194 | [0.5055, 0.5332] | 0.0318 | 0.0064 | [0.0192, 0.0445] | 0.0510 | 0.1153 |
| `cvae` | shape | circle | 0.3743 | 0.3782 | [0.3649, 0.3917] | 0.0039 | 0.0063 | [-0.0085, 0.0163] | 0.0691 | 0.1731 |
| `cvae` | shape | square | 0.2083 | 0.2846 | [0.2723, 0.2973] | 0.0763 | 0.0061 | [0.0644, 0.0883] | 0.0779 | 0.3659 |
| `cvae` | shape | triangle | 0.4174 | 0.3372 | [0.3242, 0.3504] | -0.0802 | 0.0062 | [-0.0924, -0.0681] | 0.0972 | 0.1771 |
| `cvae` | color | amber | 0.1818 | 0.2222 | [0.2109, 0.2339] | 0.0404 | 0.0057 | [0.0293, 0.0515] | 0.0450 | 0.1968 |
| `cvae` | color | blue | 0.2070 | 0.1982 | [0.1874, 0.2095] | -0.0088 | 0.0054 | [-0.0194, 0.0019] | 0.0357 | 0.0896 |
| `cvae` | color | coral | 0.1135 | 0.2062 | [0.1952, 0.2176] | 0.0927 | 0.0056 | [0.0817, 0.1038] | 0.0927 | 0.2514 |
| `cvae` | color | green | 0.2681 | 0.1880 | [0.1774, 0.1991] | -0.0801 | 0.0054 | [-0.0908, -0.0694] | 0.0801 | 0.1175 |
| `cvae` | color | violet | 0.2297 | 0.1854 | [0.1749, 0.1964] | -0.0443 | 0.0055 | [-0.0551, -0.0336] | 0.0443 | 0.1329 |
| `intercept_null` | switch | off | 0.4605 | 0.4806 | [0.4668, 0.4945] | 0.0201 | 0.0071 | [0.0062, 0.0339] | 0.0201 | 0.0201 |
| `intercept_null` | switch | on | 0.5395 | 0.5194 | [0.5055, 0.5332] | -0.0201 | 0.0071 | [-0.0339, -0.0062] | 0.0201 | 0.0201 |
| `intercept_null` | shape | circle | 0.3842 | 0.3782 | [0.3649, 0.3917] | -0.0060 | 0.0069 | [-0.0195, 0.0074] | 0.0060 | 0.0060 |
| `intercept_null` | shape | square | 0.2888 | 0.2846 | [0.2723, 0.2973] | -0.0042 | 0.0064 | [-0.0167, 0.0083] | 0.0042 | 0.0042 |
| `intercept_null` | shape | triangle | 0.3270 | 0.3372 | [0.3242, 0.3504] | 0.0102 | 0.0067 | [-0.0029, 0.0233] | 0.0102 | 0.0102 |
| `intercept_null` | color | amber | 0.2142 | 0.2222 | [0.2109, 0.2339] | 0.0080 | 0.0059 | [-0.0036, 0.0195] | 0.0080 | 0.0080 |
| `intercept_null` | color | blue | 0.1943 | 0.1982 | [0.1874, 0.2095] | 0.0039 | 0.0056 | [-0.0071, 0.0150] | 0.0039 | 0.0039 |
| `intercept_null` | color | coral | 0.2287 | 0.2062 | [0.1952, 0.2176] | -0.0225 | 0.0057 | [-0.0337, -0.0113] | 0.0225 | 0.0225 |
| `intercept_null` | color | green | 0.1970 | 0.1880 | [0.1774, 0.1991] | -0.0090 | 0.0055 | [-0.0198, 0.0018] | 0.0090 | 0.0090 |
| `intercept_null` | color | violet | 0.1658 | 0.1854 | [0.1749, 0.1964] | 0.0196 | 0.0055 | [0.0089, 0.0304] | 0.0196 | 0.0196 |
| `oracle` | switch | off | 0.4715 | 0.4806 | [0.4668, 0.4945] | 0.0091 | 0.0064 | [-0.0034, 0.0216] | 0.0221 | 0.0478 |
| `oracle` | switch | on | 0.5285 | 0.5194 | [0.5055, 0.5332] | -0.0091 | 0.0064 | [-0.0216, 0.0034] | 0.0221 | 0.0478 |
| `oracle` | shape | circle | 0.3862 | 0.3782 | [0.3649, 0.3917] | -0.0080 | 0.0062 | [-0.0201, 0.0041] | 0.0201 | 0.0636 |
| `oracle` | shape | square | 0.2883 | 0.2846 | [0.2723, 0.2973] | -0.0037 | 0.0059 | [-0.0153, 0.0079] | 0.0095 | 0.9152 |
| `oracle` | shape | triangle | 0.3254 | 0.3372 | [0.3242, 0.3504] | 0.0118 | 0.0060 | [-0.0001, 0.0236] | 0.0148 | 0.0930 |
| `oracle` | color | amber | 0.2176 | 0.2222 | [0.2109, 0.2339] | 0.0046 | 0.0056 | [-0.0065, 0.0156] | 0.0058 | 0.2474 |
| `oracle` | color | blue | 0.1973 | 0.1982 | [0.1874, 0.2095] | 0.0009 | 0.0053 | [-0.0095, 0.0114] | 0.0078 | 0.2863 |
| `oracle` | color | coral | 0.2238 | 0.2062 | [0.1952, 0.2176] | -0.0176 | 0.0056 | [-0.0285, -0.0067] | 0.0190 | 0.1087 |
| `oracle` | color | green | 0.1878 | 0.1880 | [0.1774, 0.1991] | 0.0002 | 0.0054 | [-0.0104, 0.0107] | 0.0123 | 0.0775 |
| `oracle` | color | violet | 0.1735 | 0.1854 | [0.1749, 0.1964] | 0.0119 | 0.0055 | [0.0012, 0.0226] | 0.0134 | 0.0239 |

CVAE classwise reliability bins (signed gap = observed - predicted):

| outcome | level | bin | n | mean predicted | observed | observed Wilson 95% | signed gap | gap SE | gap 95% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| switch | off | 0 | 11 | 0.0710 | 0.0000 | [0.0000, 0.2588] | -0.0710 | 0.0050 | [-0.0809, -0.0612] |
| switch | off | 1 | 175 | 0.1693 | 0.1143 | [0.0752, 0.1699] | -0.0550 | 0.0239 | [-0.1019, -0.0082] |
| switch | off | 2 | 259 | 0.2620 | 0.1467 | [0.1088, 0.1950] | -0.1153 | 0.0220 | [-0.1584, -0.0722] |
| switch | off | 3 | 850 | 0.3492 | 0.2376 | [0.2103, 0.2674] | -0.1116 | 0.0146 | [-0.1401, -0.0830] |
| switch | off | 4 | 821 | 0.4567 | 0.3837 | [0.3510, 0.4174] | -0.0730 | 0.0169 | [-0.1062, -0.0399] |
| switch | off | 5 | 1313 | 0.5401 | 0.5308 | [0.5038, 0.5577] | -0.0092 | 0.0137 | [-0.0361, 0.0176] |
| switch | off | 6 | 909 | 0.6478 | 0.6546 | [0.6231, 0.6848] | 0.0068 | 0.0157 | [-0.0241, 0.0376] |
| switch | off | 7 | 568 | 0.7324 | 0.7870 | [0.7514, 0.8187] | 0.0546 | 0.0170 | [0.0212, 0.0880] |
| switch | off | 8 | 91 | 0.8299 | 0.9451 | [0.8778, 0.9763] | 0.1151 | 0.0243 | [0.0676, 0.1627] |
| switch | off | 9 | 3 | 0.9054 | 1.0000 | [0.4385, 1.0000] | 0.0946 | 0.0021 | [0.0905, 0.0986] |
| switch | on | 0 | 3 | 0.0946 | 0.0000 | [0.0000, 0.5615] | -0.0946 | 0.0021 | [-0.0986, -0.0905] |
| switch | on | 1 | 91 | 0.1701 | 0.0549 | [0.0237, 0.1222] | -0.1151 | 0.0243 | [-0.1627, -0.0676] |
| switch | on | 2 | 568 | 0.2676 | 0.2130 | [0.1813, 0.2486] | -0.0546 | 0.0170 | [-0.0880, -0.0212] |
| switch | on | 3 | 909 | 0.3522 | 0.3454 | [0.3152, 0.3769] | -0.0068 | 0.0157 | [-0.0376, 0.0241] |
| switch | on | 4 | 1313 | 0.4599 | 0.4692 | [0.4423, 0.4962] | 0.0092 | 0.0137 | [-0.0176, 0.0361] |
| switch | on | 5 | 821 | 0.5433 | 0.6163 | [0.5826, 0.6490] | 0.0730 | 0.0169 | [0.0399, 0.1062] |
| switch | on | 6 | 850 | 0.6508 | 0.7624 | [0.7326, 0.7897] | 0.1116 | 0.0146 | [0.0830, 0.1401] |
| switch | on | 7 | 259 | 0.7380 | 0.8533 | [0.8050, 0.8912] | 0.1153 | 0.0220 | [0.0722, 0.1584] |
| switch | on | 8 | 175 | 0.8307 | 0.8857 | [0.8301, 0.9248] | 0.0550 | 0.0239 | [0.0082, 0.1019] |
| switch | on | 9 | 11 | 0.9290 | 1.0000 | [0.7412, 1.0000] | 0.0710 | 0.0050 | [0.0612, 0.0809] |
| shape | circle | 0 | 43 | 0.0791 | 0.0465 | [0.0128, 0.1546] | -0.0326 | 0.0324 | [-0.0962, 0.0310] |
| shape | circle | 1 | 440 | 0.1749 | 0.0773 | [0.0558, 0.1060] | -0.0977 | 0.0127 | [-0.1225, -0.0728] |
| shape | circle | 2 | 1040 | 0.2396 | 0.1750 | [0.1531, 0.1993] | -0.0646 | 0.0118 | [-0.0877, -0.0415] |
| shape | circle | 3 | 1246 | 0.3606 | 0.3194 | [0.2941, 0.3458] | -0.0412 | 0.0132 | [-0.0670, -0.0154] |
| shape | circle | 4 | 1546 | 0.4412 | 0.5084 | [0.4835, 0.5333] | 0.0672 | 0.0126 | [0.0424, 0.0920] |
| shape | circle | 5 | 373 | 0.5429 | 0.6327 | [0.5827, 0.6800] | 0.0898 | 0.0249 | [0.0410, 0.1386] |
| shape | circle | 6 | 256 | 0.6507 | 0.7930 | [0.7392, 0.8381] | 0.1423 | 0.0253 | [0.0926, 0.1919] |
| shape | circle | 7 | 50 | 0.7269 | 0.9000 | [0.7864, 0.9565] | 0.1731 | 0.0423 | [0.0902, 0.2560] |
| shape | circle | 8 | 6 | 0.8488 | 0.8333 | [0.4365, 0.9699] | -0.0155 | 0.1657 | [-0.3402, 0.3092] |
| shape | square | 0 | 281 | 0.0815 | 0.0676 | [0.0437, 0.1032] | -0.0139 | 0.0150 | [-0.0432, 0.0154] |
| shape | square | 1 | 2137 | 0.1565 | 0.1638 | [0.1487, 0.1801] | 0.0073 | 0.0079 | [-0.0083, 0.0228] |
| shape | square | 2 | 1983 | 0.2392 | 0.3530 | [0.3323, 0.3743] | 0.1138 | 0.0107 | [0.0928, 0.1347] |
| shape | square | 3 | 525 | 0.3371 | 0.5638 | [0.5211, 0.6056] | 0.2267 | 0.0215 | [0.1845, 0.2689] |
| shape | square | 4 | 69 | 0.4312 | 0.7971 | [0.6878, 0.8751] | 0.3659 | 0.0488 | [0.2704, 0.4615] |
| shape | square | 5 | 5 | 0.5564 | 0.6000 | [0.2307, 0.8824] | 0.0436 | 0.2384 | [-0.4237, 0.5108] |
| shape | triangle | 0 | 11 | 0.0668 | 0.0909 | [0.0162, 0.3774] | 0.0241 | 0.0906 | [-0.1535, 0.2017] |
| shape | triangle | 1 | 229 | 0.1679 | 0.0349 | [0.0178, 0.0674] | -0.1329 | 0.0120 | [-0.1565, -0.1094] |
| shape | triangle | 2 | 459 | 0.2575 | 0.0828 | [0.0609, 0.1116] | -0.1747 | 0.0129 | [-0.2000, -0.1493] |
| shape | triangle | 3 | 1588 | 0.3574 | 0.2003 | [0.1813, 0.2206] | -0.1572 | 0.0100 | [-0.1767, -0.1376] |
| shape | triangle | 4 | 1535 | 0.4453 | 0.3909 | [0.3668, 0.4155] | -0.0544 | 0.0124 | [-0.0787, -0.0301] |
| shape | triangle | 5 | 820 | 0.5464 | 0.5537 | [0.5195, 0.5874] | 0.0073 | 0.0173 | [-0.0266, 0.0412] |
| shape | triangle | 6 | 321 | 0.6351 | 0.7383 | [0.6876, 0.7834] | 0.1032 | 0.0244 | [0.0554, 0.1510] |
| shape | triangle | 7 | 36 | 0.7239 | 0.8056 | [0.6497, 0.9025] | 0.0816 | 0.0667 | [-0.0490, 0.2123] |
| shape | triangle | 8 | 1 | 0.8229 | 1.0000 | [0.2065, 1.0000] | 0.1771 | NA | [NA, NA] |
| color | amber | 0 | 644 | 0.0825 | 0.0668 | [0.0499, 0.0887] | -0.0157 | 0.0098 | [-0.0349, 0.0035] |
| color | amber | 1 | 2611 | 0.1467 | 0.1723 | [0.1583, 0.1873] | 0.0257 | 0.0074 | [0.0112, 0.0401] |
| color | amber | 2 | 1300 | 0.2419 | 0.3146 | [0.2900, 0.3404] | 0.0728 | 0.0128 | [0.0476, 0.0979] |
| color | amber | 3 | 369 | 0.3385 | 0.4444 | [0.3946, 0.4955] | 0.1059 | 0.0258 | [0.0553, 0.1565] |
| color | amber | 4 | 70 | 0.4309 | 0.6143 | [0.4972, 0.7195] | 0.1834 | 0.0582 | [0.0693, 0.2974] |
| color | amber | 5 | 6 | 0.5301 | 0.3333 | [0.0968, 0.7000] | -0.1968 | 0.2020 | [-0.5926, 0.1990] |
| color | blue | 0 | 404 | 0.0736 | 0.0272 | [0.0153, 0.0481] | -0.0464 | 0.0081 | [-0.0623, -0.0305] |
| color | blue | 1 | 1852 | 0.1568 | 0.1069 | [0.0936, 0.1218] | -0.0499 | 0.0071 | [-0.0639, -0.0360] |
| color | blue | 2 | 2315 | 0.2457 | 0.2592 | [0.2417, 0.2774] | 0.0135 | 0.0090 | [-0.0042, 0.0312] |
| color | blue | 3 | 392 | 0.3314 | 0.4209 | [0.3730, 0.4703] | 0.0896 | 0.0249 | [0.0408, 0.1383] |
| color | blue | 4 | 37 | 0.4284 | 0.4595 | [0.3104, 0.6162] | 0.0310 | 0.0819 | [-0.1295, 0.1916] |
| color | coral | 0 | 1835 | 0.0770 | 0.1243 | [0.1099, 0.1401] | 0.0472 | 0.0077 | [0.0322, 0.0623] |
| color | coral | 1 | 3141 | 0.1341 | 0.2521 | [0.2373, 0.2676] | 0.1181 | 0.0077 | [0.1030, 0.1332] |
| color | coral | 2 | 24 | 0.2069 | 0.4583 | [0.2789, 0.6493] | 0.2514 | 0.1040 | [0.0475, 0.4553] |
| color | green | 0 | 79 | 0.0778 | 0.0633 | [0.0273, 0.1397] | -0.0145 | 0.0274 | [-0.0683, 0.0392] |
| color | green | 1 | 609 | 0.1618 | 0.0443 | [0.0306, 0.0637] | -0.1175 | 0.0083 | [-0.1338, -0.1012] |
| color | green | 2 | 2575 | 0.2648 | 0.1891 | [0.1745, 0.2047] | -0.0757 | 0.0077 | [-0.0907, -0.0607] |
| color | green | 3 | 1737 | 0.3189 | 0.2424 | [0.2228, 0.2631] | -0.0766 | 0.0103 | [-0.0967, -0.0564] |
| color | violet | 1 | 1158 | 0.1896 | 0.1408 | [0.1219, 0.1620] | -0.0489 | 0.0102 | [-0.0689, -0.0289] |
| color | violet | 2 | 3552 | 0.2352 | 0.1996 | [0.1868, 0.2131] | -0.0356 | 0.0067 | [-0.0487, -0.0225] |
| color | violet | 3 | 290 | 0.3226 | 0.1897 | [0.1487, 0.2387] | -0.1329 | 0.0231 | [-0.1781, -0.0877] |

Fixed label-permutation invariance checks:

| probability source | max metric discrepancy | max selected-log-mass discrepancy | overall maximum |
|---|---:|---:|---:|
| `cvae` | 0.000000000000 | 0.000000000000 | 0.000000000000 |
| `intercept_null` | 0.000000000000 | 0.000000000000 | 0.000000000000 |
| `oracle` | 0.000000000000 | 0.000000000000 | 0.000000000000 |

Joint predictive scores and paired CVAE comparisons:

| model | joint NLL | NLL SE | CVAE-minus-model log gain | paired SE | one-sided LCB |
|---|---:|---:|---:|---:|---:|
| `cvae` | 3.2078 | 0.0114 | NA | NA | NA |
| `independent_fitted_marginal` | 3.1579 | 0.0098 | -0.0499 | 0.0055 | -0.0590 |
| `intercept_null` | 3.3953 | 0.0027 | 0.1875 | 0.0116 | 0.1683 |
| `empirical_joint` | 3.3537 | 0.0051 | 0.1459 | 0.0106 | 0.1285 |
| `oracle` | 2.9905 | 0.0125 | -0.2173 | 0.0096 | -0.2330 |

Fitted-CVAE quadrature sensitivity (fixed subset, relative to primary order):

| check order | joint mean delta | joint RMSE | product RMSE | mean G change | p99 shared change | p99 G change |
|---:|---:|---:|---:|---:|---:|---:|
| 21 | -0.019942 | 0.189679 | 0.053403 | -0.014898 | 0.713166 | 0.603328 |
| 41 | 0.005575 | 0.113272 | 0.041073 | 0.002407 | 0.516510 | 0.439770 |

Secondary nested common-panel Monte Carlo diagnostic:

| M | joint NLL | product NLL | mean gain | joint RMSE vs GH | product RMSE vs GH |
|---:|---:|---:|---:|---:|---:|
| 16 | 4.0736 | 3.2890 | -0.7846 | 2.8708 | 0.5769 |
| 64 | 3.2895 | 3.2130 | -0.0766 | 0.7628 | 0.3058 |
| 256 | 3.1918 | 3.1268 | -0.0650 | 0.3496 | 0.1272 |
| 1024 | 3.1690 | 3.1363 | -0.0326 | 0.1943 | 0.0791 |

MC block gain SD: 0.0276; jackknife-corrected mean gain: -0.0268; median joint-integrand ESS: 78.4 (0.077 of M).

Conditional-independence oracle common-panel shared-vs-product maximum absolute differences: M=16: 8.882e-16, M=64: 8.882e-16, M=256: 8.882e-16, M=1024: 1.332e-15.

## Scenario: `all_binary_dependent`

### Seed-level primary results

| seed | epochs (best) | active z at best | KL by latent at best | CVAE marginal NLL | CVAE marginal Brier | joint NLL | fitted-product NLL | paired joint gain (SE) | marginal log gain (SE) | Brier improvement (SE) |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1701 | 30 (30) | 2 / 2 | 1.678 / 1.306 | 0.6683 | 0.4756 | 3.0901 | 3.3416 | 0.2516 (0.0121) | 0.0245 (0.0016) | 0.0241 (0.0015) |
| 9901 | 30 (30) | 2 / 2 | 1.511 / 1.341 | 0.6687 | 0.4761 | 3.1319 | 3.3434 | 0.2116 (0.0149) | 0.0240 (0.0015) | 0.0234 (0.0014) |
| 31415 | 30 (30) | 2 / 2 | 1.609 / 1.408 | 0.6703 | 0.4776 | 3.1177 | 3.3513 | 0.2336 (0.0146) | 0.0227 (0.0014) | 0.0223 (0.0014) |

### Seed 1701 detail

Restored checkpoint: epoch 30 of 30; validation beta-ELBO 1.5964; training KL/latent [1.67837, 1.30609]; active latent units 2 / 2; effective beta 0.2000.

Marginal predictive metrics:

| model | NLL | multiclass Brier | classwise ECE | max class-bin gap |
|---|---:|---:|---:|---:|
| `cvae` | 0.6683 | 0.4756 | 0.0374 | 0.2079 |
| `intercept_null` | 0.6928 | 0.4997 | 0.0062 | 0.0199 |
| `oracle` | 0.6631 | 0.4707 | 0.0122 | 0.4861 |

Classwise calibration-in-the-large and uncertainty:

| model | outcome | level | mean predicted | observed | observed Wilson 95% | CIL gap | CIL SE | CIL 95% | ECE | max gap |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cvae` | signal_a | 0 | 0.4720 | 0.4962 | [0.4823, 0.5101] | 0.0242 | 0.0069 | [0.0106, 0.0378] | 0.0352 | 0.0967 |
| `cvae` | signal_a | 1 | 0.5280 | 0.5038 | [0.4899, 0.5177] | -0.0242 | 0.0069 | [-0.0378, -0.0106] | 0.0352 | 0.0967 |
| `cvae` | signal_b | 0 | 0.5551 | 0.5166 | [0.5027, 0.5304] | -0.0385 | 0.0069 | [-0.0520, -0.0251] | 0.0385 | 0.0786 |
| `cvae` | signal_b | 1 | 0.4449 | 0.4834 | [0.4696, 0.4973] | 0.0385 | 0.0069 | [0.0251, 0.0520] | 0.0385 | 0.0786 |
| `cvae` | signal_c | 0 | 0.4817 | 0.4994 | [0.4855, 0.5133] | 0.0177 | 0.0069 | [0.0042, 0.0311] | 0.0273 | 0.1704 |
| `cvae` | signal_c | 1 | 0.5183 | 0.5006 | [0.4867, 0.5145] | -0.0177 | 0.0069 | [-0.0311, -0.0042] | 0.0273 | 0.1704 |
| `cvae` | signal_d | 0 | 0.5157 | 0.5086 | [0.4947, 0.5224] | -0.0071 | 0.0068 | [-0.0205, 0.0063] | 0.0288 | 0.1768 |
| `cvae` | signal_d | 1 | 0.4843 | 0.4914 | [0.4776, 0.5053] | 0.0071 | 0.0068 | [-0.0063, 0.0205] | 0.0288 | 0.1768 |
| `cvae` | signal_e | 0 | 0.4133 | 0.4704 | [0.4566, 0.4843] | 0.0571 | 0.0069 | [0.0436, 0.0707] | 0.0573 | 0.2079 |
| `cvae` | signal_e | 1 | 0.5867 | 0.5296 | [0.5157, 0.5434] | -0.0571 | 0.0069 | [-0.0707, -0.0436] | 0.0573 | 0.2079 |
| `intercept_null` | signal_a | 0 | 0.4763 | 0.4962 | [0.4823, 0.5101] | 0.0199 | 0.0071 | [0.0061, 0.0338] | 0.0199 | 0.0199 |
| `intercept_null` | signal_a | 1 | 0.5237 | 0.5038 | [0.4899, 0.5177] | -0.0199 | 0.0071 | [-0.0338, -0.0061] | 0.0199 | 0.0199 |
| `intercept_null` | signal_b | 0 | 0.5142 | 0.5166 | [0.5027, 0.5304] | 0.0024 | 0.0071 | [-0.0115, 0.0162] | 0.0024 | 0.0024 |
| `intercept_null` | signal_b | 1 | 0.4858 | 0.4834 | [0.4696, 0.4973] | -0.0024 | 0.0071 | [-0.0162, 0.0115] | 0.0024 | 0.0024 |
| `intercept_null` | signal_c | 0 | 0.5067 | 0.4994 | [0.4855, 0.5133] | -0.0073 | 0.0071 | [-0.0212, 0.0065] | 0.0073 | 0.0073 |
| `intercept_null` | signal_c | 1 | 0.4933 | 0.5006 | [0.4867, 0.5145] | 0.0073 | 0.0071 | [-0.0065, 0.0212] | 0.0073 | 0.0073 |
| `intercept_null` | signal_d | 0 | 0.5077 | 0.5086 | [0.4947, 0.5224] | 0.0009 | 0.0071 | [-0.0130, 0.0147] | 0.0009 | 0.0009 |
| `intercept_null` | signal_d | 1 | 0.4923 | 0.4914 | [0.4776, 0.5053] | -0.0009 | 0.0071 | [-0.0147, 0.0130] | 0.0009 | 0.0009 |
| `intercept_null` | signal_e | 0 | 0.4700 | 0.4704 | [0.4566, 0.4843] | 0.0004 | 0.0071 | [-0.0134, 0.0142] | 0.0004 | 0.0004 |
| `intercept_null` | signal_e | 1 | 0.5300 | 0.5296 | [0.5157, 0.5434] | -0.0004 | 0.0071 | [-0.0142, 0.0134] | 0.0004 | 0.0004 |
| `oracle` | signal_a | 0 | 0.4863 | 0.4962 | [0.4823, 0.5101] | 0.0099 | 0.0069 | [-0.0036, 0.0234] | 0.0169 | 0.1557 |
| `oracle` | signal_a | 1 | 0.5137 | 0.5038 | [0.4899, 0.5177] | -0.0099 | 0.0069 | [-0.0234, 0.0036] | 0.0169 | 0.1557 |
| `oracle` | signal_b | 0 | 0.5222 | 0.5166 | [0.5027, 0.5304] | -0.0056 | 0.0068 | [-0.0190, 0.0078] | 0.0108 | 0.1089 |
| `oracle` | signal_b | 1 | 0.4778 | 0.4834 | [0.4696, 0.4973] | 0.0056 | 0.0068 | [-0.0078, 0.0190] | 0.0108 | 0.1089 |
| `oracle` | signal_c | 0 | 0.4921 | 0.4994 | [0.4855, 0.5133] | 0.0073 | 0.0068 | [-0.0061, 0.0207] | 0.0142 | 0.1826 |
| `oracle` | signal_c | 1 | 0.5079 | 0.5006 | [0.4867, 0.5145] | -0.0073 | 0.0068 | [-0.0207, 0.0061] | 0.0142 | 0.1826 |
| `oracle` | signal_d | 0 | 0.5070 | 0.5086 | [0.4947, 0.5224] | 0.0016 | 0.0068 | [-0.0118, 0.0150] | 0.0134 | 0.0503 |
| `oracle` | signal_d | 1 | 0.4930 | 0.4914 | [0.4776, 0.5053] | -0.0016 | 0.0068 | [-0.0150, 0.0118] | 0.0134 | 0.0503 |
| `oracle` | signal_e | 0 | 0.4694 | 0.4704 | [0.4566, 0.4843] | 0.0010 | 0.0069 | [-0.0126, 0.0145] | 0.0057 | 0.4861 |
| `oracle` | signal_e | 1 | 0.5306 | 0.5296 | [0.5157, 0.5434] | -0.0010 | 0.0069 | [-0.0145, 0.0126] | 0.0057 | 0.4861 |

CVAE classwise reliability bins (signed gap = observed - predicted):

| outcome | level | bin | n | mean predicted | observed | observed Wilson 95% | signed gap | gap SE | gap 95% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| signal_a | 0 | 1 | 32 | 0.1778 | 0.2500 | [0.1325, 0.4211] | 0.0722 | 0.0779 | [-0.0805, 0.2249] |
| signal_a | 0 | 2 | 370 | 0.2660 | 0.3054 | [0.2607, 0.3541] | 0.0394 | 0.0239 | [-0.0074, 0.0863] |
| signal_a | 0 | 3 | 1145 | 0.3559 | 0.3956 | [0.3677, 0.4243] | 0.0397 | 0.0144 | [0.0114, 0.0680] |
| signal_a | 0 | 4 | 1479 | 0.4489 | 0.4733 | [0.4479, 0.4988] | 0.0244 | 0.0130 | [-0.0011, 0.0499] |
| signal_a | 0 | 5 | 1119 | 0.5456 | 0.5889 | [0.5598, 0.6174] | 0.0434 | 0.0147 | [0.0146, 0.0721] |
| signal_a | 0 | 6 | 627 | 0.6431 | 0.6204 | [0.5818, 0.6576] | -0.0227 | 0.0194 | [-0.0607, 0.0154] |
| signal_a | 0 | 7 | 201 | 0.7373 | 0.6716 | [0.6040, 0.7328] | -0.0657 | 0.0335 | [-0.1314, -0.0000] |
| signal_a | 0 | 8 | 25 | 0.8258 | 0.8800 | [0.7004, 0.9583] | 0.0542 | 0.0657 | [-0.0747, 0.1831] |
| signal_a | 0 | 9 | 2 | 0.9033 | 1.0000 | [0.3424, 1.0000] | 0.0967 | 0.0032 | [0.0905, 0.1029] |
| signal_a | 1 | 0 | 2 | 0.0967 | 0.0000 | [0.0000, 0.6576] | -0.0967 | 0.0032 | [-0.1029, -0.0905] |
| signal_a | 1 | 1 | 25 | 0.1742 | 0.1200 | [0.0417, 0.2996] | -0.0542 | 0.0657 | [-0.1831, 0.0747] |
| signal_a | 1 | 2 | 201 | 0.2627 | 0.3284 | [0.2672, 0.3960] | 0.0657 | 0.0335 | [0.0000, 0.1314] |
| signal_a | 1 | 3 | 627 | 0.3569 | 0.3796 | [0.3424, 0.4182] | 0.0227 | 0.0194 | [-0.0154, 0.0607] |
| signal_a | 1 | 4 | 1119 | 0.4544 | 0.4111 | [0.3826, 0.4402] | -0.0434 | 0.0147 | [-0.0721, -0.0146] |
| signal_a | 1 | 5 | 1479 | 0.5511 | 0.5267 | [0.5012, 0.5521] | -0.0244 | 0.0130 | [-0.0499, 0.0011] |
| signal_a | 1 | 6 | 1145 | 0.6441 | 0.6044 | [0.5757, 0.6323] | -0.0397 | 0.0144 | [-0.0680, -0.0114] |
| signal_a | 1 | 7 | 370 | 0.7340 | 0.6946 | [0.6459, 0.7393] | -0.0394 | 0.0239 | [-0.0863, 0.0074] |
| signal_a | 1 | 8 | 32 | 0.8222 | 0.7500 | [0.5789, 0.8675] | -0.0722 | 0.0779 | [-0.2249, 0.0805] |
| signal_b | 0 | 2 | 10 | 0.2786 | 0.2000 | [0.0567, 0.5098] | -0.0786 | 0.1306 | [-0.3346, 0.1775] |
| signal_b | 0 | 3 | 595 | 0.3671 | 0.2891 | [0.2541, 0.3268] | -0.0780 | 0.0185 | [-0.1143, -0.0417] |
| signal_b | 0 | 4 | 1244 | 0.4507 | 0.4365 | [0.4092, 0.4642] | -0.0142 | 0.0141 | [-0.0418, 0.0134] |
| signal_b | 0 | 5 | 1276 | 0.5497 | 0.5321 | [0.5047, 0.5594] | -0.0176 | 0.0140 | [-0.0449, 0.0098] |
| signal_b | 0 | 6 | 1163 | 0.6477 | 0.5899 | [0.5613, 0.6178] | -0.0579 | 0.0144 | [-0.0861, -0.0297] |
| signal_b | 0 | 7 | 600 | 0.7424 | 0.6917 | [0.6536, 0.7273] | -0.0507 | 0.0189 | [-0.0877, -0.0137] |
| signal_b | 0 | 8 | 112 | 0.8358 | 0.7679 | [0.6816, 0.8364] | -0.0679 | 0.0402 | [-0.1467, 0.0108] |
| signal_b | 1 | 1 | 112 | 0.1642 | 0.2321 | [0.1636, 0.3184] | 0.0679 | 0.0402 | [-0.0108, 0.1467] |
| signal_b | 1 | 2 | 600 | 0.2576 | 0.3083 | [0.2727, 0.3464] | 0.0507 | 0.0189 | [0.0137, 0.0877] |
| signal_b | 1 | 3 | 1163 | 0.3523 | 0.4101 | [0.3822, 0.4387] | 0.0579 | 0.0144 | [0.0297, 0.0861] |
| signal_b | 1 | 4 | 1276 | 0.4503 | 0.4679 | [0.4406, 0.4953] | 0.0176 | 0.0140 | [-0.0098, 0.0449] |
| signal_b | 1 | 5 | 1244 | 0.5493 | 0.5635 | [0.5358, 0.5908] | 0.0142 | 0.0141 | [-0.0134, 0.0418] |
| signal_b | 1 | 6 | 595 | 0.6329 | 0.7109 | [0.6732, 0.7459] | 0.0780 | 0.0185 | [0.0417, 0.1143] |
| signal_b | 1 | 7 | 10 | 0.7214 | 0.8000 | [0.4902, 0.9433] | 0.0786 | 0.1306 | [-0.1775, 0.3346] |
| signal_c | 0 | 1 | 3 | 0.1672 | 0.0000 | [0.0000, 0.5615] | -0.1672 | 0.0169 | [-0.2002, -0.1341] |
| signal_c | 0 | 2 | 140 | 0.2700 | 0.2214 | [0.1606, 0.2971] | -0.0486 | 0.0349 | [-0.1171, 0.0199] |
| signal_c | 0 | 3 | 984 | 0.3624 | 0.3455 | [0.3165, 0.3758] | -0.0169 | 0.0151 | [-0.0465, 0.0127] |
| signal_c | 0 | 4 | 1734 | 0.4507 | 0.4666 | [0.4432, 0.4901] | 0.0159 | 0.0119 | [-0.0075, 0.0393] |
| signal_c | 0 | 5 | 1533 | 0.5474 | 0.5851 | [0.5603, 0.6095] | 0.0377 | 0.0126 | [0.0131, 0.0624] |
| signal_c | 0 | 6 | 525 | 0.6365 | 0.6876 | [0.6467, 0.7258] | 0.0511 | 0.0202 | [0.0115, 0.0906] |
| signal_c | 0 | 7 | 80 | 0.7260 | 0.7250 | [0.6186, 0.8108] | -0.0010 | 0.0505 | [-0.0999, 0.0979] |
| signal_c | 0 | 8 | 1 | 0.8296 | 1.0000 | [0.2065, 1.0000] | 0.1704 | NA | [NA, NA] |
| signal_c | 1 | 1 | 1 | 0.1704 | 0.0000 | [0.0000, 0.7935] | -0.1704 | NA | [NA, NA] |
| signal_c | 1 | 2 | 80 | 0.2740 | 0.2750 | [0.1892, 0.3814] | 0.0010 | 0.0505 | [-0.0979, 0.0999] |
| signal_c | 1 | 3 | 525 | 0.3635 | 0.3124 | [0.2742, 0.3533] | -0.0511 | 0.0202 | [-0.0906, -0.0115] |
| signal_c | 1 | 4 | 1533 | 0.4526 | 0.4149 | [0.3905, 0.4397] | -0.0377 | 0.0126 | [-0.0624, -0.0131] |
| signal_c | 1 | 5 | 1734 | 0.5493 | 0.5334 | [0.5099, 0.5568] | -0.0159 | 0.0119 | [-0.0393, 0.0075] |
| signal_c | 1 | 6 | 984 | 0.6376 | 0.6545 | [0.6242, 0.6835] | 0.0169 | 0.0151 | [-0.0127, 0.0465] |
| signal_c | 1 | 7 | 140 | 0.7300 | 0.7786 | [0.7029, 0.8394] | 0.0486 | 0.0349 | [-0.0199, 0.1171] |
| signal_c | 1 | 8 | 3 | 0.8328 | 1.0000 | [0.4385, 1.0000] | 0.1672 | 0.0169 | [0.1341, 0.2002] |
| signal_d | 0 | 1 | 2 | 0.1768 | 0.0000 | [0.0000, 0.6576] | -0.1768 | 0.0214 | [-0.2186, -0.1349] |
| signal_d | 0 | 2 | 85 | 0.2745 | 0.2235 | [0.1480, 0.3229] | -0.0510 | 0.0454 | [-0.1399, 0.0379] |
| signal_d | 0 | 3 | 600 | 0.3653 | 0.3317 | [0.2952, 0.3703] | -0.0336 | 0.0191 | [-0.0711, 0.0039] |
| signal_d | 0 | 4 | 1477 | 0.4541 | 0.4103 | [0.3855, 0.4356] | -0.0438 | 0.0127 | [-0.0688, -0.0188] |
| signal_d | 0 | 5 | 1805 | 0.5464 | 0.5568 | [0.5338, 0.5796] | 0.0104 | 0.0117 | [-0.0125, 0.0332] |
| signal_d | 0 | 6 | 836 | 0.6398 | 0.6758 | [0.6434, 0.7067] | 0.0360 | 0.0162 | [0.0044, 0.0677] |
| signal_d | 0 | 7 | 184 | 0.7314 | 0.7609 | [0.6943, 0.8168] | 0.0295 | 0.0316 | [-0.0325, 0.0914] |
| signal_d | 0 | 8 | 11 | 0.8162 | 0.8182 | [0.5230, 0.9486] | 0.0020 | 0.1198 | [-0.2329, 0.2369] |
| signal_d | 1 | 1 | 11 | 0.1838 | 0.1818 | [0.0514, 0.4770] | -0.0020 | 0.1198 | [-0.2369, 0.2329] |
| signal_d | 1 | 2 | 184 | 0.2686 | 0.2391 | [0.1832, 0.3057] | -0.0295 | 0.0316 | [-0.0914, 0.0325] |
| signal_d | 1 | 3 | 836 | 0.3602 | 0.3242 | [0.2933, 0.3566] | -0.0360 | 0.0162 | [-0.0677, -0.0044] |
| signal_d | 1 | 4 | 1805 | 0.4536 | 0.4432 | [0.4204, 0.4662] | -0.0104 | 0.0117 | [-0.0332, 0.0125] |
| signal_d | 1 | 5 | 1477 | 0.5459 | 0.5897 | [0.5644, 0.6145] | 0.0438 | 0.0127 | [0.0188, 0.0688] |
| signal_d | 1 | 6 | 600 | 0.6347 | 0.6683 | [0.6297, 0.7048] | 0.0336 | 0.0191 | [-0.0039, 0.0711] |
| signal_d | 1 | 7 | 85 | 0.7255 | 0.7765 | [0.6771, 0.8520] | 0.0510 | 0.0454 | [-0.0379, 0.1399] |
| signal_d | 1 | 8 | 2 | 0.8232 | 1.0000 | [0.3424, 1.0000] | 0.1768 | 0.0214 | [0.1349, 0.2186] |
| signal_e | 0 | 0 | 1 | 0.0859 | 0.0000 | [0.0000, 0.7935] | -0.0859 | NA | [NA, NA] |
| signal_e | 0 | 1 | 66 | 0.1788 | 0.2273 | [0.1429, 0.3417] | 0.0485 | 0.0520 | [-0.0535, 0.1505] |
| signal_e | 0 | 2 | 619 | 0.2626 | 0.3215 | [0.2859, 0.3593] | 0.0588 | 0.0188 | [0.0221, 0.0956] |
| signal_e | 0 | 3 | 1557 | 0.3551 | 0.4297 | [0.4053, 0.4544] | 0.0746 | 0.0125 | [0.0500, 0.0991] |
| signal_e | 0 | 4 | 1746 | 0.4486 | 0.4885 | [0.4651, 0.5120] | 0.0400 | 0.0119 | [0.0166, 0.0633] |
| signal_e | 0 | 5 | 898 | 0.5403 | 0.6024 | [0.5701, 0.6340] | 0.0622 | 0.0163 | [0.0303, 0.0941] |
| signal_e | 0 | 6 | 111 | 0.6226 | 0.6667 | [0.5747, 0.7475] | 0.0441 | 0.0449 | [-0.0440, 0.1321] |
| signal_e | 0 | 7 | 2 | 0.7079 | 0.5000 | [0.0945, 0.9055] | -0.2079 | 0.4979 | [-1.1837, 0.7679] |
| signal_e | 1 | 2 | 2 | 0.2921 | 0.5000 | [0.0945, 0.9055] | 0.2079 | 0.4979 | [-0.7679, 1.1837] |
| signal_e | 1 | 3 | 111 | 0.3774 | 0.3333 | [0.2525, 0.4253] | -0.0441 | 0.0449 | [-0.1321, 0.0440] |
| signal_e | 1 | 4 | 898 | 0.4597 | 0.3976 | [0.3660, 0.4299] | -0.0622 | 0.0163 | [-0.0941, -0.0303] |
| signal_e | 1 | 5 | 1746 | 0.5514 | 0.5115 | [0.4880, 0.5349] | -0.0400 | 0.0119 | [-0.0633, -0.0166] |
| signal_e | 1 | 6 | 1557 | 0.6449 | 0.5703 | [0.5456, 0.5947] | -0.0746 | 0.0125 | [-0.0991, -0.0500] |
| signal_e | 1 | 7 | 619 | 0.7374 | 0.6785 | [0.6407, 0.7141] | -0.0588 | 0.0188 | [-0.0956, -0.0221] |
| signal_e | 1 | 8 | 66 | 0.8212 | 0.7727 | [0.6583, 0.8571] | -0.0485 | 0.0520 | [-0.1505, 0.0535] |
| signal_e | 1 | 9 | 1 | 0.9141 | 1.0000 | [0.2065, 1.0000] | 0.0859 | NA | [NA, NA] |

Fixed label-permutation invariance checks:

| probability source | max metric discrepancy | max selected-log-mass discrepancy | overall maximum |
|---|---:|---:|---:|
| `cvae` | 0.000000000000 | 0.000000000000 | 0.000000000000 |
| `intercept_null` | 0.000000000000 | 0.000000000000 | 0.000000000000 |
| `oracle` | 0.000000000000 | 0.000000000000 | 0.000000000000 |

Joint predictive scores and paired CVAE comparisons:

| model | joint NLL | NLL SE | CVAE-minus-model log gain | paired SE | one-sided LCB |
|---|---:|---:|---:|---:|---:|
| `cvae` | 3.0901 | 0.0137 | NA | NA | NA |
| `independent_fitted_marginal` | 3.3416 | 0.0081 | 0.2516 | 0.0121 | 0.2317 |
| `intercept_null` | 3.4642 | 0.0011 | 0.3741 | 0.0135 | 0.3518 |
| `empirical_joint` | 3.1885 | 0.0108 | 0.0984 | 0.0093 | 0.0831 |
| `oracle` | 3.0302 | 0.0124 | -0.0598 | 0.0053 | -0.0685 |

Fitted-CVAE quadrature sensitivity (fixed subset, relative to primary order):

| check order | joint mean delta | joint RMSE | product RMSE | mean G change | p99 shared change | p99 G change |
|---:|---:|---:|---:|---:|---:|---:|
| 21 | -0.001236 | 0.071826 | 0.030737 | 0.002383 | 0.268801 | 0.269267 |
| 41 | 0.004058 | 0.028829 | 0.014355 | 0.000239 | 0.101295 | 0.098143 |

Secondary nested common-panel Monte Carlo diagnostic:

| M | joint NLL | product NLL | mean gain | joint RMSE vs GH | product RMSE vs GH |
|---:|---:|---:|---:|---:|---:|
| 16 | 3.4346 | 3.4242 | -0.0105 | 1.0002 | 0.5877 |
| 64 | 3.0834 | 3.2778 | 0.1944 | 0.3217 | 0.2830 |
| 256 | 3.0420 | 3.2709 | 0.2289 | 0.1617 | 0.1213 |
| 1024 | 3.0373 | 3.2622 | 0.2250 | 0.0813 | 0.0708 |

MC block gain SD: 0.0200; jackknife-corrected mean gain: 0.2264; median joint-integrand ESS: 179.9 (0.176 of M).

### Seed 9901 detail

Restored checkpoint: epoch 30 of 30; validation beta-ELBO 1.6445; training KL/latent [1.51134, 1.34107]; active latent units 2 / 2; effective beta 0.2000.

Marginal predictive metrics:

| model | NLL | multiclass Brier | classwise ECE | max class-bin gap |
|---|---:|---:|---:|---:|
| `cvae` | 0.6687 | 0.4761 | 0.0240 | 0.2862 |
| `intercept_null` | 0.6927 | 0.4995 | 0.0110 | 0.0194 |
| `oracle` | 0.6643 | 0.4720 | 0.0119 | 0.1597 |

Classwise calibration-in-the-large and uncertainty:

| model | outcome | level | mean predicted | observed | observed Wilson 95% | CIL gap | CIL SE | CIL 95% | ECE | max gap |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cvae` | signal_a | 0 | 0.4913 | 0.4838 | [0.4700, 0.4977] | -0.0075 | 0.0069 | [-0.0210, 0.0060] | 0.0247 | 0.1010 |
| `cvae` | signal_a | 1 | 0.5087 | 0.5162 | [0.5023, 0.5300] | 0.0075 | 0.0069 | [-0.0060, 0.0210] | 0.0247 | 0.1010 |
| `cvae` | signal_b | 0 | 0.5425 | 0.5214 | [0.5075, 0.5352] | -0.0211 | 0.0069 | [-0.0346, -0.0077] | 0.0257 | 0.2862 |
| `cvae` | signal_b | 1 | 0.4575 | 0.4786 | [0.4648, 0.4925] | 0.0211 | 0.0069 | [0.0077, 0.0346] | 0.0257 | 0.2862 |
| `cvae` | signal_c | 0 | 0.4921 | 0.4974 | [0.4835, 0.5113] | 0.0053 | 0.0069 | [-0.0081, 0.0188] | 0.0144 | 0.1827 |
| `cvae` | signal_c | 1 | 0.5079 | 0.5026 | [0.4887, 0.5165] | -0.0053 | 0.0069 | [-0.0188, 0.0081] | 0.0144 | 0.1827 |
| `cvae` | signal_d | 0 | 0.5467 | 0.5148 | [0.5009, 0.5286] | -0.0319 | 0.0069 | [-0.0455, -0.0183] | 0.0319 | 0.1314 |
| `cvae` | signal_d | 1 | 0.4533 | 0.4852 | [0.4714, 0.4991] | 0.0319 | 0.0069 | [0.0183, 0.0455] | 0.0319 | 0.1314 |
| `cvae` | signal_e | 0 | 0.4589 | 0.4680 | [0.4542, 0.4818] | 0.0091 | 0.0069 | [-0.0044, 0.0227] | 0.0235 | 0.1955 |
| `cvae` | signal_e | 1 | 0.5411 | 0.5320 | [0.5182, 0.5458] | -0.0091 | 0.0069 | [-0.0227, 0.0044] | 0.0235 | 0.1955 |
| `intercept_null` | signal_a | 0 | 0.4888 | 0.4838 | [0.4700, 0.4977] | -0.0050 | 0.0071 | [-0.0188, 0.0089] | 0.0050 | 0.0050 |
| `intercept_null` | signal_a | 1 | 0.5112 | 0.5162 | [0.5023, 0.5300] | 0.0050 | 0.0071 | [-0.0089, 0.0188] | 0.0050 | 0.0050 |
| `intercept_null` | signal_b | 0 | 0.5322 | 0.5214 | [0.5075, 0.5352] | -0.0108 | 0.0071 | [-0.0247, 0.0030] | 0.0108 | 0.0108 |
| `intercept_null` | signal_b | 1 | 0.4678 | 0.4786 | [0.4648, 0.4925] | 0.0108 | 0.0071 | [-0.0030, 0.0247] | 0.0108 | 0.0108 |
| `intercept_null` | signal_c | 0 | 0.4780 | 0.4974 | [0.4835, 0.5113] | 0.0194 | 0.0071 | [0.0055, 0.0333] | 0.0194 | 0.0194 |
| `intercept_null` | signal_c | 1 | 0.5220 | 0.5026 | [0.4887, 0.5165] | -0.0194 | 0.0071 | [-0.0333, -0.0055] | 0.0194 | 0.0194 |
| `intercept_null` | signal_d | 0 | 0.5077 | 0.5148 | [0.5009, 0.5286] | 0.0071 | 0.0071 | [-0.0068, 0.0209] | 0.0071 | 0.0071 |
| `intercept_null` | signal_d | 1 | 0.4923 | 0.4852 | [0.4714, 0.4991] | -0.0071 | 0.0071 | [-0.0209, 0.0068] | 0.0071 | 0.0071 |
| `intercept_null` | signal_e | 0 | 0.4550 | 0.4680 | [0.4542, 0.4818] | 0.0130 | 0.0071 | [-0.0008, 0.0268] | 0.0130 | 0.0130 |
| `intercept_null` | signal_e | 1 | 0.5450 | 0.5320 | [0.5182, 0.5458] | -0.0130 | 0.0071 | [-0.0268, 0.0008] | 0.0130 | 0.0130 |
| `oracle` | signal_a | 0 | 0.4857 | 0.4838 | [0.4700, 0.4977] | -0.0019 | 0.0069 | [-0.0154, 0.0115] | 0.0097 | 0.0761 |
| `oracle` | signal_a | 1 | 0.5143 | 0.5162 | [0.5023, 0.5300] | 0.0019 | 0.0069 | [-0.0115, 0.0154] | 0.0097 | 0.0761 |
| `oracle` | signal_b | 0 | 0.5233 | 0.5214 | [0.5075, 0.5352] | -0.0019 | 0.0068 | [-0.0153, 0.0115] | 0.0103 | 0.1105 |
| `oracle` | signal_b | 1 | 0.4767 | 0.4786 | [0.4648, 0.4925] | 0.0019 | 0.0068 | [-0.0115, 0.0153] | 0.0103 | 0.1105 |
| `oracle` | signal_c | 0 | 0.4930 | 0.4974 | [0.4835, 0.5113] | 0.0044 | 0.0069 | [-0.0090, 0.0179] | 0.0114 | 0.0552 |
| `oracle` | signal_c | 1 | 0.5070 | 0.5026 | [0.4887, 0.5165] | -0.0044 | 0.0069 | [-0.0179, 0.0090] | 0.0114 | 0.0552 |
| `oracle` | signal_d | 0 | 0.5054 | 0.5148 | [0.5009, 0.5286] | 0.0094 | 0.0069 | [-0.0042, 0.0229] | 0.0160 | 0.1131 |
| `oracle` | signal_d | 1 | 0.4946 | 0.4852 | [0.4714, 0.4991] | -0.0094 | 0.0069 | [-0.0229, 0.0042] | 0.0160 | 0.1131 |
| `oracle` | signal_e | 0 | 0.4708 | 0.4680 | [0.4542, 0.4818] | -0.0028 | 0.0069 | [-0.0163, 0.0107] | 0.0119 | 0.1597 |
| `oracle` | signal_e | 1 | 0.5292 | 0.5320 | [0.5182, 0.5458] | 0.0028 | 0.0069 | [-0.0107, 0.0163] | 0.0119 | 0.1597 |

CVAE classwise reliability bins (signed gap = observed - predicted):

| outcome | level | bin | n | mean predicted | observed | observed Wilson 95% | signed gap | gap SE | gap 95% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| signal_a | 0 | 1 | 27 | 0.1751 | 0.0741 | [0.0206, 0.2337] | -0.1010 | 0.0514 | [-0.2017, -0.0003] |
| signal_a | 0 | 2 | 308 | 0.2620 | 0.2695 | [0.2230, 0.3216] | 0.0074 | 0.0251 | [-0.0417, 0.0566] |
| signal_a | 0 | 3 | 952 | 0.3552 | 0.3676 | [0.3376, 0.3988] | 0.0124 | 0.0156 | [-0.0182, 0.0431] |
| signal_a | 0 | 4 | 1330 | 0.4530 | 0.4729 | [0.4462, 0.4998] | 0.0199 | 0.0137 | [-0.0069, 0.0467] |
| signal_a | 0 | 5 | 1364 | 0.5472 | 0.5242 | [0.4977, 0.5506] | -0.0230 | 0.0135 | [-0.0495, 0.0034] |
| signal_a | 0 | 6 | 749 | 0.6438 | 0.5848 | [0.5491, 0.6195] | -0.0590 | 0.0180 | [-0.0942, -0.0237] |
| signal_a | 0 | 7 | 235 | 0.7349 | 0.7447 | [0.6853, 0.7962] | 0.0098 | 0.0285 | [-0.0460, 0.0656] |
| signal_a | 0 | 8 | 35 | 0.8302 | 0.7714 | [0.6098, 0.8793] | -0.0588 | 0.0716 | [-0.1991, 0.0816] |
| signal_a | 1 | 1 | 35 | 0.1698 | 0.2286 | [0.1207, 0.3902] | 0.0588 | 0.0716 | [-0.0816, 0.1991] |
| signal_a | 1 | 2 | 235 | 0.2651 | 0.2553 | [0.2038, 0.3147] | -0.0098 | 0.0285 | [-0.0656, 0.0460] |
| signal_a | 1 | 3 | 749 | 0.3562 | 0.4152 | [0.3805, 0.4509] | 0.0590 | 0.0180 | [0.0237, 0.0942] |
| signal_a | 1 | 4 | 1364 | 0.4528 | 0.4758 | [0.4494, 0.5023] | 0.0230 | 0.0135 | [-0.0034, 0.0495] |
| signal_a | 1 | 5 | 1330 | 0.5470 | 0.5271 | [0.5002, 0.5538] | -0.0199 | 0.0137 | [-0.0467, 0.0069] |
| signal_a | 1 | 6 | 952 | 0.6448 | 0.6324 | [0.6012, 0.6624] | -0.0124 | 0.0156 | [-0.0431, 0.0182] |
| signal_a | 1 | 7 | 308 | 0.7380 | 0.7305 | [0.6784, 0.7770] | -0.0074 | 0.0251 | [-0.0566, 0.0417] |
| signal_a | 1 | 8 | 27 | 0.8249 | 0.9259 | [0.7663, 0.9794] | 0.1010 | 0.0514 | [0.0003, 0.2017] |
| signal_b | 0 | 2 | 3 | 0.2862 | 0.0000 | [0.0000, 0.5615] | -0.2862 | 0.0026 | [-0.2914, -0.2811] |
| signal_b | 0 | 3 | 293 | 0.3747 | 0.2867 | [0.2379, 0.3410] | -0.0880 | 0.0265 | [-0.1399, -0.0362] |
| signal_b | 0 | 4 | 1450 | 0.4554 | 0.4103 | [0.3853, 0.4359] | -0.0451 | 0.0129 | [-0.0703, -0.0199] |
| signal_b | 0 | 5 | 1884 | 0.5490 | 0.5377 | [0.5151, 0.5601] | -0.0113 | 0.0114 | [-0.0337, 0.0111] |
| signal_b | 0 | 6 | 1089 | 0.6425 | 0.6529 | [0.6241, 0.6806] | 0.0104 | 0.0144 | [-0.0178, 0.0386] |
| signal_b | 0 | 7 | 265 | 0.7333 | 0.7245 | [0.6678, 0.7748] | -0.0088 | 0.0274 | [-0.0624, 0.0449] |
| signal_b | 0 | 8 | 16 | 0.8292 | 0.7500 | [0.5050, 0.8982] | -0.0792 | 0.1119 | [-0.2984, 0.1401] |
| signal_b | 1 | 1 | 16 | 0.1708 | 0.2500 | [0.1018, 0.4950] | 0.0792 | 0.1119 | [-0.1401, 0.2984] |
| signal_b | 1 | 2 | 265 | 0.2667 | 0.2755 | [0.2252, 0.3322] | 0.0088 | 0.0274 | [-0.0449, 0.0624] |
| signal_b | 1 | 3 | 1089 | 0.3575 | 0.3471 | [0.3194, 0.3759] | -0.0104 | 0.0144 | [-0.0386, 0.0178] |
| signal_b | 1 | 4 | 1884 | 0.4510 | 0.4623 | [0.4399, 0.4849] | 0.0113 | 0.0114 | [-0.0111, 0.0337] |
| signal_b | 1 | 5 | 1450 | 0.5446 | 0.5897 | [0.5641, 0.6147] | 0.0451 | 0.0129 | [0.0199, 0.0703] |
| signal_b | 1 | 6 | 293 | 0.6253 | 0.7133 | [0.6590, 0.7621] | 0.0880 | 0.0265 | [0.0362, 0.1399] |
| signal_b | 1 | 7 | 3 | 0.7138 | 1.0000 | [0.4385, 1.0000] | 0.2862 | 0.0026 | [0.2811, 0.2914] |
| signal_c | 0 | 1 | 47 | 0.1678 | 0.2553 | [0.1525, 0.3951] | 0.0876 | 0.0633 | [-0.0365, 0.2116] |
| signal_c | 0 | 2 | 273 | 0.2602 | 0.2747 | [0.2252, 0.3305] | 0.0146 | 0.0269 | [-0.0382, 0.0673] |
| signal_c | 0 | 3 | 857 | 0.3541 | 0.3722 | [0.3405, 0.4051] | 0.0182 | 0.0166 | [-0.0143, 0.0506] |
| signal_c | 0 | 4 | 1233 | 0.4533 | 0.4509 | [0.4234, 0.4788] | -0.0023 | 0.0142 | [-0.0301, 0.0254] |
| signal_c | 0 | 5 | 1574 | 0.5506 | 0.5381 | [0.5134, 0.5626] | -0.0125 | 0.0125 | [-0.0370, 0.0121] |
| signal_c | 0 | 6 | 952 | 0.6369 | 0.6597 | [0.6290, 0.6891] | 0.0227 | 0.0153 | [-0.0073, 0.0528] |
| signal_c | 0 | 7 | 63 | 0.7180 | 0.7778 | [0.6609, 0.8627] | 0.0597 | 0.0529 | [-0.0439, 0.1634] |
| signal_c | 0 | 8 | 1 | 0.8173 | 1.0000 | [0.2065, 1.0000] | 0.1827 | NA | [NA, NA] |
| signal_c | 1 | 1 | 1 | 0.1827 | 0.0000 | [0.0000, 0.7935] | -0.1827 | NA | [NA, NA] |
| signal_c | 1 | 2 | 63 | 0.2820 | 0.2222 | [0.1373, 0.3391] | -0.0597 | 0.0529 | [-0.1634, 0.0439] |
| signal_c | 1 | 3 | 952 | 0.3631 | 0.3403 | [0.3109, 0.3710] | -0.0227 | 0.0153 | [-0.0528, 0.0073] |
| signal_c | 1 | 4 | 1574 | 0.4494 | 0.4619 | [0.4374, 0.4866] | 0.0125 | 0.0125 | [-0.0121, 0.0370] |
| signal_c | 1 | 5 | 1233 | 0.5467 | 0.5491 | [0.5212, 0.5766] | 0.0023 | 0.0142 | [-0.0254, 0.0301] |
| signal_c | 1 | 6 | 857 | 0.6459 | 0.6278 | [0.5949, 0.6595] | -0.0182 | 0.0166 | [-0.0506, 0.0143] |
| signal_c | 1 | 7 | 273 | 0.7398 | 0.7253 | [0.6695, 0.7748] | -0.0146 | 0.0269 | [-0.0673, 0.0382] |
| signal_c | 1 | 8 | 47 | 0.8322 | 0.7447 | [0.6049, 0.8475] | -0.0876 | 0.0633 | [-0.2116, 0.0365] |
| signal_d | 0 | 2 | 14 | 0.2743 | 0.1429 | [0.0401, 0.3994] | -0.1314 | 0.0953 | [-0.3181, 0.0553] |
| signal_d | 0 | 3 | 410 | 0.3680 | 0.3244 | [0.2809, 0.3712] | -0.0436 | 0.0231 | [-0.0888, 0.0017] |
| signal_d | 0 | 4 | 1178 | 0.4534 | 0.4338 | [0.4057, 0.4623] | -0.0196 | 0.0144 | [-0.0478, 0.0087] |
| signal_d | 0 | 5 | 1822 | 0.5526 | 0.5241 | [0.5012, 0.5470] | -0.0285 | 0.0117 | [-0.0514, -0.0055] |
| signal_d | 0 | 6 | 1297 | 0.6398 | 0.5998 | [0.5729, 0.6262] | -0.0400 | 0.0136 | [-0.0666, -0.0134] |
| signal_d | 0 | 7 | 251 | 0.7363 | 0.6892 | [0.6295, 0.7433] | -0.0471 | 0.0291 | [-0.1041, 0.0099] |
| signal_d | 0 | 8 | 28 | 0.8251 | 0.7857 | [0.6046, 0.8979] | -0.0394 | 0.0795 | [-0.1952, 0.1165] |
| signal_d | 1 | 1 | 28 | 0.1749 | 0.2143 | [0.1021, 0.3954] | 0.0394 | 0.0795 | [-0.1165, 0.1952] |
| signal_d | 1 | 2 | 251 | 0.2637 | 0.3108 | [0.2567, 0.3705] | 0.0471 | 0.0291 | [-0.0099, 0.1041] |
| signal_d | 1 | 3 | 1297 | 0.3602 | 0.4002 | [0.3738, 0.4271] | 0.0400 | 0.0136 | [0.0134, 0.0666] |
| signal_d | 1 | 4 | 1822 | 0.4474 | 0.4759 | [0.4530, 0.4988] | 0.0285 | 0.0117 | [0.0055, 0.0514] |
| signal_d | 1 | 5 | 1178 | 0.5466 | 0.5662 | [0.5377, 0.5943] | 0.0196 | 0.0144 | [-0.0087, 0.0478] |
| signal_d | 1 | 6 | 410 | 0.6320 | 0.6756 | [0.6288, 0.7191] | 0.0436 | 0.0231 | [-0.0017, 0.0888] |
| signal_d | 1 | 7 | 14 | 0.7257 | 0.8571 | [0.6006, 0.9599] | 0.1314 | 0.0953 | [-0.0553, 0.3181] |
| signal_e | 0 | 1 | 37 | 0.1757 | 0.3243 | [0.1963, 0.4854] | 0.1486 | 0.0779 | [-0.0041, 0.3013] |
| signal_e | 0 | 2 | 365 | 0.2646 | 0.3068 | [0.2618, 0.3560] | 0.0423 | 0.0241 | [-0.0050, 0.0895] |
| signal_e | 0 | 3 | 1204 | 0.3559 | 0.3812 | [0.3542, 0.4090] | 0.0253 | 0.0140 | [-0.0021, 0.0527] |
| signal_e | 0 | 4 | 1566 | 0.4503 | 0.4432 | [0.4187, 0.4679] | -0.0071 | 0.0126 | [-0.0317, 0.0175] |
| signal_e | 0 | 5 | 1248 | 0.5462 | 0.5657 | [0.5380, 0.5930] | 0.0195 | 0.0140 | [-0.0078, 0.0469] |
| signal_e | 0 | 6 | 514 | 0.6380 | 0.5895 | [0.5465, 0.6312] | -0.0485 | 0.0216 | [-0.0909, -0.0062] |
| signal_e | 0 | 7 | 65 | 0.7290 | 0.8154 | [0.7045, 0.8911] | 0.0863 | 0.0484 | [-0.0085, 0.1812] |
| signal_e | 0 | 8 | 1 | 0.8045 | 1.0000 | [0.2065, 1.0000] | 0.1955 | NA | [NA, NA] |
| signal_e | 1 | 1 | 1 | 0.1955 | 0.0000 | [0.0000, 0.7935] | -0.1955 | NA | [NA, NA] |
| signal_e | 1 | 2 | 65 | 0.2710 | 0.1846 | [0.1089, 0.2955] | -0.0863 | 0.0484 | [-0.1812, 0.0085] |
| signal_e | 1 | 3 | 514 | 0.3620 | 0.4105 | [0.3688, 0.4535] | 0.0485 | 0.0216 | [0.0062, 0.0909] |
| signal_e | 1 | 4 | 1248 | 0.4538 | 0.4343 | [0.4070, 0.4620] | -0.0195 | 0.0140 | [-0.0469, 0.0078] |
| signal_e | 1 | 5 | 1566 | 0.5497 | 0.5568 | [0.5321, 0.5813] | 0.0071 | 0.0126 | [-0.0175, 0.0317] |
| signal_e | 1 | 6 | 1204 | 0.6441 | 0.6188 | [0.5910, 0.6458] | -0.0253 | 0.0140 | [-0.0527, 0.0021] |
| signal_e | 1 | 7 | 365 | 0.7354 | 0.6932 | [0.6440, 0.7382] | -0.0423 | 0.0241 | [-0.0895, 0.0050] |
| signal_e | 1 | 8 | 37 | 0.8243 | 0.6757 | [0.5146, 0.8037] | -0.1486 | 0.0779 | [-0.3013, 0.0041] |

Fixed label-permutation invariance checks:

| probability source | max metric discrepancy | max selected-log-mass discrepancy | overall maximum |
|---|---:|---:|---:|
| `cvae` | 0.000000000000 | 0.000000000000 | 0.000000000000 |
| `intercept_null` | 0.000000000000 | 0.000000000000 | 0.000000000000 |
| `oracle` | 0.000000000000 | 0.000000000000 | 0.000000000000 |

Joint predictive scores and paired CVAE comparisons:

| model | joint NLL | NLL SE | CVAE-minus-model log gain | paired SE | one-sided LCB |
|---|---:|---:|---:|---:|---:|
| `cvae` | 3.1319 | 0.0165 | NA | NA | NA |
| `independent_fitted_marginal` | 3.3434 | 0.0077 | 0.2116 | 0.0149 | 0.1871 |
| `intercept_null` | 3.4633 | 0.0021 | 0.3314 | 0.0164 | 0.3044 |
| `empirical_joint` | 3.2003 | 0.0108 | 0.0685 | 0.0109 | 0.0506 |
| `oracle` | 3.0480 | 0.0126 | -0.0838 | 0.0065 | -0.0944 |

Fitted-CVAE quadrature sensitivity (fixed subset, relative to primary order):

| check order | joint mean delta | joint RMSE | product RMSE | mean G change | p99 shared change | p99 G change |
|---:|---:|---:|---:|---:|---:|---:|
| 21 | -0.000721 | 0.032202 | 0.013888 | -0.000182 | 0.102124 | 0.092312 |
| 41 | 0.000549 | 0.014163 | 0.005544 | 0.000267 | 0.039081 | 0.037933 |

Secondary nested common-panel Monte Carlo diagnostic:

| M | joint NLL | product NLL | mean gain | joint RMSE vs GH | product RMSE vs GH |
|---:|---:|---:|---:|---:|---:|
| 16 | 3.4409 | 3.4034 | -0.0375 | 0.9949 | 0.5258 |
| 64 | 3.1485 | 3.3223 | 0.1738 | 0.3083 | 0.2586 |
| 256 | 3.1104 | 3.2979 | 0.1875 | 0.1513 | 0.1206 |
| 1024 | 3.0745 | 3.2746 | 0.2001 | 0.0691 | 0.0610 |

MC block gain SD: 0.0205; jackknife-corrected mean gain: 0.2011; median joint-integrand ESS: 209.9 (0.205 of M).

### Seed 31415 detail

Restored checkpoint: epoch 30 of 30; validation beta-ELBO 1.4979; training KL/latent [1.60891, 1.40772]; active latent units 2 / 2; effective beta 0.2000.

Marginal predictive metrics:

| model | NLL | multiclass Brier | classwise ECE | max class-bin gap |
|---|---:|---:|---:|---:|
| `cvae` | 0.6703 | 0.4776 | 0.0290 | 0.1930 |
| `intercept_null` | 0.6930 | 0.4999 | 0.0085 | 0.0132 |
| `oracle` | 0.6660 | 0.4735 | 0.0139 | 0.3147 |

Classwise calibration-in-the-large and uncertainty:

| model | outcome | level | mean predicted | observed | observed Wilson 95% | CIL gap | CIL SE | CIL 95% | ECE | max gap |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cvae` | signal_a | 0 | 0.4557 | 0.4906 | [0.4768, 0.5045] | 0.0349 | 0.0069 | [0.0213, 0.0485] | 0.0362 | 0.1001 |
| `cvae` | signal_a | 1 | 0.5443 | 0.5094 | [0.4955, 0.5232] | -0.0349 | 0.0069 | [-0.0485, -0.0213] | 0.0362 | 0.1001 |
| `cvae` | signal_b | 0 | 0.5466 | 0.5078 | [0.4939, 0.5216] | -0.0388 | 0.0069 | [-0.0522, -0.0253] | 0.0396 | 0.0830 |
| `cvae` | signal_b | 1 | 0.4534 | 0.4922 | [0.4784, 0.5061] | 0.0388 | 0.0069 | [0.0253, 0.0522] | 0.0396 | 0.0830 |
| `cvae` | signal_c | 0 | 0.5098 | 0.4948 | [0.4810, 0.5087] | -0.0150 | 0.0069 | [-0.0284, -0.0015] | 0.0232 | 0.1930 |
| `cvae` | signal_c | 1 | 0.4902 | 0.5052 | [0.4913, 0.5190] | 0.0150 | 0.0069 | [0.0015, 0.0284] | 0.0232 | 0.1930 |
| `cvae` | signal_d | 0 | 0.5032 | 0.5022 | [0.4883, 0.5161] | -0.0010 | 0.0069 | [-0.0147, 0.0126] | 0.0122 | 0.1389 |
| `cvae` | signal_d | 1 | 0.4968 | 0.4978 | [0.4839, 0.5117] | 0.0010 | 0.0069 | [-0.0126, 0.0147] | 0.0122 | 0.1389 |
| `cvae` | signal_e | 0 | 0.4479 | 0.4764 | [0.4626, 0.4903] | 0.0285 | 0.0069 | [0.0149, 0.0420] | 0.0338 | 0.1896 |
| `cvae` | signal_e | 1 | 0.5521 | 0.5236 | [0.5097, 0.5374] | -0.0285 | 0.0069 | [-0.0420, -0.0149] | 0.0338 | 0.1896 |
| `intercept_null` | signal_a | 0 | 0.4835 | 0.4906 | [0.4768, 0.5045] | 0.0071 | 0.0071 | [-0.0068, 0.0210] | 0.0071 | 0.0071 |
| `intercept_null` | signal_a | 1 | 0.5165 | 0.5094 | [0.4955, 0.5232] | -0.0071 | 0.0071 | [-0.0210, 0.0068] | 0.0071 | 0.0071 |
| `intercept_null` | signal_b | 0 | 0.5210 | 0.5078 | [0.4939, 0.5216] | -0.0132 | 0.0071 | [-0.0271, 0.0007] | 0.0132 | 0.0132 |
| `intercept_null` | signal_b | 1 | 0.4790 | 0.4922 | [0.4784, 0.5061] | 0.0132 | 0.0071 | [-0.0007, 0.0271] | 0.0132 | 0.0132 |
| `intercept_null` | signal_c | 0 | 0.4913 | 0.4948 | [0.4810, 0.5087] | 0.0035 | 0.0071 | [-0.0103, 0.0174] | 0.0035 | 0.0035 |
| `intercept_null` | signal_c | 1 | 0.5087 | 0.5052 | [0.4913, 0.5190] | -0.0035 | 0.0071 | [-0.0174, 0.0103] | 0.0035 | 0.0035 |
| `intercept_null` | signal_d | 0 | 0.5100 | 0.5022 | [0.4883, 0.5161] | -0.0078 | 0.0071 | [-0.0217, 0.0061] | 0.0078 | 0.0078 |
| `intercept_null` | signal_d | 1 | 0.4900 | 0.4978 | [0.4839, 0.5117] | 0.0078 | 0.0071 | [-0.0061, 0.0217] | 0.0078 | 0.0078 |
| `intercept_null` | signal_e | 0 | 0.4658 | 0.4764 | [0.4626, 0.4903] | 0.0106 | 0.0071 | [-0.0032, 0.0245] | 0.0106 | 0.0106 |
| `intercept_null` | signal_e | 1 | 0.5342 | 0.5236 | [0.5097, 0.5374] | -0.0106 | 0.0071 | [-0.0245, 0.0032] | 0.0106 | 0.0106 |
| `oracle` | signal_a | 0 | 0.4878 | 0.4906 | [0.4768, 0.5045] | 0.0028 | 0.0069 | [-0.0108, 0.0163] | 0.0129 | 0.0302 |
| `oracle` | signal_a | 1 | 0.5122 | 0.5094 | [0.4955, 0.5232] | -0.0028 | 0.0069 | [-0.0163, 0.0108] | 0.0129 | 0.0302 |
| `oracle` | signal_b | 0 | 0.5185 | 0.5078 | [0.4939, 0.5216] | -0.0107 | 0.0068 | [-0.0241, 0.0026] | 0.0162 | 0.0995 |
| `oracle` | signal_b | 1 | 0.4815 | 0.4922 | [0.4784, 0.5061] | 0.0107 | 0.0068 | [-0.0026, 0.0241] | 0.0162 | 0.0995 |
| `oracle` | signal_c | 0 | 0.4957 | 0.4948 | [0.4810, 0.5087] | -0.0009 | 0.0068 | [-0.0144, 0.0125] | 0.0061 | 0.0560 |
| `oracle` | signal_c | 1 | 0.5043 | 0.5052 | [0.4913, 0.5190] | 0.0009 | 0.0068 | [-0.0125, 0.0144] | 0.0061 | 0.0560 |
| `oracle` | signal_d | 0 | 0.5046 | 0.5022 | [0.4883, 0.5161] | -0.0024 | 0.0069 | [-0.0160, 0.0112] | 0.0208 | 0.0968 |
| `oracle` | signal_d | 1 | 0.4954 | 0.4978 | [0.4839, 0.5117] | 0.0024 | 0.0069 | [-0.0112, 0.0160] | 0.0208 | 0.0968 |
| `oracle` | signal_e | 0 | 0.4719 | 0.4764 | [0.4626, 0.4903] | 0.0045 | 0.0069 | [-0.0090, 0.0180] | 0.0135 | 0.3147 |
| `oracle` | signal_e | 1 | 0.5281 | 0.5236 | [0.5097, 0.5374] | -0.0045 | 0.0069 | [-0.0180, 0.0090] | 0.0135 | 0.3147 |

CVAE classwise reliability bins (signed gap = observed - predicted):

| outcome | level | bin | n | mean predicted | observed | observed Wilson 95% | signed gap | gap SE | gap 95% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| signal_a | 0 | 1 | 32 | 0.1744 | 0.0938 | [0.0324, 0.2422] | -0.0807 | 0.0528 | [-0.1841, 0.0228] |
| signal_a | 0 | 2 | 436 | 0.2599 | 0.3257 | [0.2834, 0.3710] | 0.0658 | 0.0225 | [0.0217, 0.1099] |
| signal_a | 0 | 3 | 1240 | 0.3551 | 0.4097 | [0.3826, 0.4373] | 0.0545 | 0.0140 | [0.0272, 0.0819] |
| signal_a | 0 | 4 | 1484 | 0.4490 | 0.4805 | [0.4551, 0.5059] | 0.0315 | 0.0130 | [0.0061, 0.0569] |
| signal_a | 0 | 5 | 1213 | 0.5468 | 0.5664 | [0.5383, 0.5940] | 0.0196 | 0.0142 | [-0.0083, 0.0475] |
| signal_a | 0 | 6 | 505 | 0.6401 | 0.6475 | [0.6049, 0.6879] | 0.0074 | 0.0213 | [-0.0342, 0.0491] |
| signal_a | 0 | 7 | 83 | 0.7328 | 0.8193 | [0.7230, 0.8873] | 0.0865 | 0.0419 | [0.0043, 0.1687] |
| signal_a | 0 | 8 | 7 | 0.8144 | 0.7143 | [0.3589, 0.9178] | -0.1001 | 0.1846 | [-0.4619, 0.2617] |
| signal_a | 1 | 1 | 7 | 0.1856 | 0.2857 | [0.0822, 0.6411] | 0.1001 | 0.1846 | [-0.2617, 0.4619] |
| signal_a | 1 | 2 | 83 | 0.2672 | 0.1807 | [0.1127, 0.2770] | -0.0865 | 0.0419 | [-0.1687, -0.0043] |
| signal_a | 1 | 3 | 505 | 0.3599 | 0.3525 | [0.3121, 0.3951] | -0.0074 | 0.0213 | [-0.0491, 0.0342] |
| signal_a | 1 | 4 | 1213 | 0.4532 | 0.4336 | [0.4060, 0.4617] | -0.0196 | 0.0142 | [-0.0475, 0.0083] |
| signal_a | 1 | 5 | 1484 | 0.5510 | 0.5195 | [0.4941, 0.5449] | -0.0315 | 0.0130 | [-0.0569, -0.0061] |
| signal_a | 1 | 6 | 1240 | 0.6449 | 0.5903 | [0.5627, 0.6174] | -0.0545 | 0.0140 | [-0.0819, -0.0272] |
| signal_a | 1 | 7 | 436 | 0.7401 | 0.6743 | [0.6290, 0.7166] | -0.0658 | 0.0225 | [-0.1099, -0.0217] |
| signal_a | 1 | 8 | 32 | 0.8256 | 0.9062 | [0.7578, 0.9676] | 0.0807 | 0.0528 | [-0.0228, 0.1841] |
| signal_b | 0 | 0 | 1 | 0.0830 | 0.0000 | [0.0000, 0.7935] | -0.0830 | NA | [NA, NA] |
| signal_b | 0 | 1 | 33 | 0.1667 | 0.1818 | [0.0861, 0.3439] | 0.0151 | 0.0677 | [-0.1177, 0.1478] |
| signal_b | 0 | 2 | 175 | 0.2613 | 0.2171 | [0.1625, 0.2839] | -0.0441 | 0.0313 | [-0.1054, 0.0172] |
| signal_b | 0 | 3 | 538 | 0.3582 | 0.3420 | [0.3032, 0.3831] | -0.0162 | 0.0204 | [-0.0563, 0.0238] |
| signal_b | 0 | 4 | 948 | 0.4535 | 0.4230 | [0.3919, 0.4547] | -0.0305 | 0.0160 | [-0.0619, 0.0009] |
| signal_b | 0 | 5 | 1419 | 0.5514 | 0.5159 | [0.4898, 0.5418] | -0.0356 | 0.0132 | [-0.0615, -0.0096] |
| signal_b | 0 | 6 | 1296 | 0.6459 | 0.5926 | [0.5656, 0.6190] | -0.0533 | 0.0137 | [-0.0801, -0.0265] |
| signal_b | 0 | 7 | 536 | 0.7368 | 0.6791 | [0.6384, 0.7172] | -0.0577 | 0.0200 | [-0.0969, -0.0184] |
| signal_b | 0 | 8 | 54 | 0.8219 | 0.8519 | [0.7340, 0.9230] | 0.0299 | 0.0487 | [-0.0656, 0.1254] |
| signal_b | 1 | 1 | 54 | 0.1781 | 0.1481 | [0.0770, 0.2660] | -0.0299 | 0.0487 | [-0.1254, 0.0656] |
| signal_b | 1 | 2 | 536 | 0.2632 | 0.3209 | [0.2828, 0.3616] | 0.0577 | 0.0200 | [0.0184, 0.0969] |
| signal_b | 1 | 3 | 1296 | 0.3541 | 0.4074 | [0.3810, 0.4344] | 0.0533 | 0.0137 | [0.0265, 0.0801] |
| signal_b | 1 | 4 | 1419 | 0.4486 | 0.4841 | [0.4582, 0.5102] | 0.0356 | 0.0132 | [0.0096, 0.0615] |
| signal_b | 1 | 5 | 948 | 0.5465 | 0.5770 | [0.5453, 0.6081] | 0.0305 | 0.0160 | [-0.0009, 0.0619] |
| signal_b | 1 | 6 | 538 | 0.6418 | 0.6580 | [0.6169, 0.6968] | 0.0162 | 0.0204 | [-0.0238, 0.0563] |
| signal_b | 1 | 7 | 175 | 0.7387 | 0.7829 | [0.7161, 0.8375] | 0.0441 | 0.0313 | [-0.0172, 0.1054] |
| signal_b | 1 | 8 | 33 | 0.8333 | 0.8182 | [0.6561, 0.9139] | -0.0151 | 0.0677 | [-0.1478, 0.1177] |
| signal_b | 1 | 9 | 1 | 0.9170 | 1.0000 | [0.2065, 1.0000] | 0.0830 | NA | [NA, NA] |
| signal_c | 0 | 2 | 38 | 0.2733 | 0.1053 | [0.0417, 0.2413] | -0.1680 | 0.0513 | [-0.2686, -0.0674] |
| signal_c | 0 | 3 | 475 | 0.3666 | 0.2842 | [0.2455, 0.3264] | -0.0824 | 0.0206 | [-0.1229, -0.0420] |
| signal_c | 0 | 4 | 1856 | 0.4543 | 0.4289 | [0.4065, 0.4515] | -0.0255 | 0.0114 | [-0.0479, -0.0030] |
| signal_c | 0 | 5 | 1779 | 0.5459 | 0.5481 | [0.5249, 0.5711] | 0.0022 | 0.0117 | [-0.0208, 0.0252] |
| signal_c | 0 | 6 | 771 | 0.6364 | 0.6576 | [0.6234, 0.6902] | 0.0212 | 0.0171 | [-0.0122, 0.0546] |
| signal_c | 0 | 7 | 79 | 0.7299 | 0.6962 | [0.5877, 0.7866] | -0.0337 | 0.0527 | [-0.1369, 0.0695] |
| signal_c | 0 | 8 | 2 | 0.8070 | 1.0000 | [0.3424, 1.0000] | 0.1930 | 0.0029 | [0.1874, 0.1986] |
| signal_c | 1 | 1 | 2 | 0.1930 | 0.0000 | [0.0000, 0.6576] | -0.1930 | 0.0029 | [-0.1986, -0.1874] |
| signal_c | 1 | 2 | 79 | 0.2701 | 0.3038 | [0.2134, 0.4123] | 0.0337 | 0.0527 | [-0.0695, 0.1369] |
| signal_c | 1 | 3 | 771 | 0.3636 | 0.3424 | [0.3098, 0.3766] | -0.0212 | 0.0171 | [-0.0546, 0.0122] |
| signal_c | 1 | 4 | 1779 | 0.4541 | 0.4519 | [0.4289, 0.4751] | -0.0022 | 0.0117 | [-0.0252, 0.0208] |
| signal_c | 1 | 5 | 1856 | 0.5457 | 0.5711 | [0.5485, 0.5935] | 0.0255 | 0.0114 | [0.0030, 0.0479] |
| signal_c | 1 | 6 | 475 | 0.6334 | 0.7158 | [0.6736, 0.7545] | 0.0824 | 0.0206 | [0.0420, 0.1229] |
| signal_c | 1 | 7 | 38 | 0.7267 | 0.8947 | [0.7587, 0.9583] | 0.1680 | 0.0513 | [0.0674, 0.2686] |
| signal_d | 0 | 1 | 3 | 0.1944 | 0.3333 | [0.0615, 0.7923] | 0.1389 | 0.3311 | [-0.5101, 0.7879] |
| signal_d | 0 | 2 | 92 | 0.2685 | 0.3370 | [0.2486, 0.4383] | 0.0684 | 0.0495 | [-0.0285, 0.1654] |
| signal_d | 0 | 3 | 620 | 0.3620 | 0.3581 | [0.3213, 0.3966] | -0.0039 | 0.0193 | [-0.0417, 0.0338] |
| signal_d | 0 | 4 | 1526 | 0.4552 | 0.4509 | [0.4260, 0.4759] | -0.0044 | 0.0127 | [-0.0292, 0.0205] |
| signal_d | 0 | 5 | 2084 | 0.5470 | 0.5355 | [0.5141, 0.5568] | -0.0115 | 0.0109 | [-0.0329, 0.0099] |
| signal_d | 0 | 6 | 627 | 0.6332 | 0.6603 | [0.6223, 0.6963] | 0.0270 | 0.0189 | [-0.0101, 0.0642] |
| signal_d | 0 | 7 | 48 | 0.7257 | 0.8125 | [0.6806, 0.8981] | 0.0868 | 0.0566 | [-0.0241, 0.1976] |
| signal_d | 1 | 2 | 48 | 0.2743 | 0.1875 | [0.1019, 0.3194] | -0.0868 | 0.0566 | [-0.1976, 0.0241] |
| signal_d | 1 | 3 | 627 | 0.3668 | 0.3397 | [0.3037, 0.3777] | -0.0270 | 0.0189 | [-0.0642, 0.0101] |
| signal_d | 1 | 4 | 2084 | 0.4530 | 0.4645 | [0.4432, 0.4859] | 0.0115 | 0.0109 | [-0.0099, 0.0329] |
| signal_d | 1 | 5 | 1526 | 0.5448 | 0.5491 | [0.5241, 0.5740] | 0.0044 | 0.0127 | [-0.0205, 0.0292] |
| signal_d | 1 | 6 | 620 | 0.6380 | 0.6419 | [0.6034, 0.6787] | 0.0039 | 0.0193 | [-0.0338, 0.0417] |
| signal_d | 1 | 7 | 92 | 0.7315 | 0.6630 | [0.5617, 0.7514] | -0.0684 | 0.0495 | [-0.1654, 0.0285] |
| signal_d | 1 | 8 | 3 | 0.8056 | 0.6667 | [0.2077, 0.9385] | -0.1389 | 0.3311 | [-0.7879, 0.5101] |
| signal_e | 0 | 1 | 5 | 0.1751 | 0.0000 | [0.0000, 0.4345] | -0.1751 | 0.0067 | [-0.1882, -0.1621] |
| signal_e | 0 | 2 | 329 | 0.2690 | 0.2553 | [0.2112, 0.3051] | -0.0137 | 0.0240 | [-0.0608, 0.0334] |
| signal_e | 0 | 3 | 1460 | 0.3562 | 0.3897 | [0.3650, 0.4150] | 0.0336 | 0.0128 | [0.0085, 0.0586] |
| signal_e | 0 | 4 | 1739 | 0.4465 | 0.4784 | [0.4550, 0.5019] | 0.0319 | 0.0120 | [0.0085, 0.0554] |
| signal_e | 0 | 5 | 994 | 0.5446 | 0.5815 | [0.5506, 0.6118] | 0.0368 | 0.0156 | [0.0062, 0.0675] |
| signal_e | 0 | 6 | 382 | 0.6398 | 0.6780 | [0.6296, 0.7229] | 0.0383 | 0.0241 | [-0.0089, 0.0854] |
| signal_e | 0 | 7 | 80 | 0.7362 | 0.6625 | [0.5536, 0.7565] | -0.0737 | 0.0541 | [-0.1797, 0.0323] |
| signal_e | 0 | 8 | 11 | 0.8260 | 0.6364 | [0.3538, 0.8483] | -0.1896 | 0.1528 | [-0.4891, 0.1099] |
| signal_e | 1 | 1 | 11 | 0.1740 | 0.3636 | [0.1517, 0.6462] | 0.1896 | 0.1528 | [-0.1099, 0.4891] |
| signal_e | 1 | 2 | 80 | 0.2638 | 0.3375 | [0.2435, 0.4464] | 0.0737 | 0.0541 | [-0.0323, 0.1797] |
| signal_e | 1 | 3 | 382 | 0.3602 | 0.3220 | [0.2771, 0.3704] | -0.0383 | 0.0241 | [-0.0854, 0.0089] |
| signal_e | 1 | 4 | 994 | 0.4554 | 0.4185 | [0.3882, 0.4494] | -0.0368 | 0.0156 | [-0.0675, -0.0062] |
| signal_e | 1 | 5 | 1739 | 0.5535 | 0.5216 | [0.4981, 0.5450] | -0.0319 | 0.0120 | [-0.0554, -0.0085] |
| signal_e | 1 | 6 | 1460 | 0.6438 | 0.6103 | [0.5850, 0.6350] | -0.0336 | 0.0128 | [-0.0586, -0.0085] |
| signal_e | 1 | 7 | 329 | 0.7310 | 0.7447 | [0.6949, 0.7888] | 0.0137 | 0.0240 | [-0.0334, 0.0608] |
| signal_e | 1 | 8 | 5 | 0.8249 | 1.0000 | [0.5655, 1.0000] | 0.1751 | 0.0067 | [0.1621, 0.1882] |

Fixed label-permutation invariance checks:

| probability source | max metric discrepancy | max selected-log-mass discrepancy | overall maximum |
|---|---:|---:|---:|
| `cvae` | 0.000000000000 | 0.000000000000 | 0.000000000000 |
| `intercept_null` | 0.000000000000 | 0.000000000000 | 0.000000000000 |
| `oracle` | 0.000000000000 | 0.000000000000 | 0.000000000000 |

Joint predictive scores and paired CVAE comparisons:

| model | joint NLL | NLL SE | CVAE-minus-model log gain | paired SE | one-sided LCB |
|---|---:|---:|---:|---:|---:|
| `cvae` | 3.1177 | 0.0159 | NA | NA | NA |
| `independent_fitted_marginal` | 3.3513 | 0.0074 | 0.2336 | 0.0146 | 0.2096 |
| `intercept_null` | 3.4651 | 0.0014 | 0.3473 | 0.0158 | 0.3214 |
| `empirical_joint` | 3.1883 | 0.0113 | 0.0705 | 0.0101 | 0.0539 |
| `oracle` | 3.0373 | 0.0125 | -0.0805 | 0.0066 | -0.0914 |

Fitted-CVAE quadrature sensitivity (fixed subset, relative to primary order):

| check order | joint mean delta | joint RMSE | product RMSE | mean G change | p99 shared change | p99 G change |
|---:|---:|---:|---:|---:|---:|---:|
| 21 | -0.001729 | 0.048861 | 0.015304 | -0.003045 | 0.156857 | 0.158542 |
| 41 | -0.000756 | 0.021555 | 0.007826 | -0.000369 | 0.066718 | 0.062671 |

Secondary nested common-panel Monte Carlo diagnostic:

| M | joint NLL | product NLL | mean gain | joint RMSE vs GH | product RMSE vs GH |
|---:|---:|---:|---:|---:|---:|
| 16 | 3.6583 | 3.5573 | -0.1010 | 1.0366 | 0.5578 |
| 64 | 3.4072 | 3.5145 | 0.1073 | 0.3538 | 0.2789 |
| 256 | 3.3756 | 3.5045 | 0.1289 | 0.1725 | 0.1506 |
| 1024 | 3.3675 | 3.4860 | 0.1185 | 0.0782 | 0.0705 |

MC block gain SD: 0.0176; jackknife-corrected mean gain: 0.1200; median joint-integrand ESS: 155.6 (0.152 of M).

## Scenario: `all_binary_conditional_independence`

### Seed-level primary results

| seed | epochs (best) | active z at best | KL by latent at best | CVAE marginal NLL | CVAE marginal Brier | joint NLL | fitted-product NLL | paired joint gain (SE) | marginal log gain (SE) | Brier improvement (SE) |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1701 | 30 (30) | 2 / 2 | 1.667 / 1.695 | 0.6362 | 0.4456 | 3.2196 | 3.1810 | -0.0386 (0.0043) | 0.0551 (0.0018) | 0.0525 (0.0017) |
| 9901 | 30 (30) | 2 / 2 | 1.660 / 1.770 | 0.6317 | 0.4416 | 3.2117 | 3.1587 | -0.0531 (0.0051) | 0.0598 (0.0017) | 0.0568 (0.0016) |
| 31415 | 30 (30) | 2 / 2 | 1.862 / 1.846 | 0.6503 | 0.4587 | 3.2995 | 3.2514 | -0.0481 (0.0051) | 0.0421 (0.0017) | 0.0406 (0.0016) |

### Seed 1701 detail

Restored checkpoint: epoch 30 of 30; validation beta-ELBO 1.3529; training KL/latent [1.66686, 1.69485]; active latent units 2 / 2; effective beta 0.2000.

Marginal predictive metrics:

| model | NLL | multiclass Brier | classwise ECE | max class-bin gap |
|---|---:|---:|---:|---:|
| `cvae` | 0.6362 | 0.4456 | 0.0586 | 0.2087 |
| `intercept_null` | 0.6913 | 0.4981 | 0.0087 | 0.0151 |
| `oracle` | 0.6200 | 0.4310 | 0.0142 | 0.2438 |

Classwise calibration-in-the-large and uncertainty:

| model | outcome | level | mean predicted | observed | observed Wilson 95% | CIL gap | CIL SE | CIL 95% | ECE | max gap |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cvae` | signal_a | 0 | 0.5250 | 0.4826 | [0.4688, 0.4965] | -0.0424 | 0.0066 | [-0.0553, -0.0294] | 0.0425 | 0.0758 |
| `cvae` | signal_a | 1 | 0.4750 | 0.5174 | [0.5035, 0.5312] | 0.0424 | 0.0066 | [0.0294, 0.0553] | 0.0425 | 0.0758 |
| `cvae` | signal_b | 0 | 0.4319 | 0.5338 | [0.5200, 0.5476] | 0.1019 | 0.0067 | [0.0888, 0.1150] | 0.1085 | 0.2087 |
| `cvae` | signal_b | 1 | 0.5681 | 0.4662 | [0.4524, 0.4800] | -0.1019 | 0.0067 | [-0.1150, -0.0888] | 0.1085 | 0.2087 |
| `cvae` | signal_c | 0 | 0.4956 | 0.4888 | [0.4750, 0.5027] | -0.0068 | 0.0066 | [-0.0197, 0.0061] | 0.0230 | 0.0806 |
| `cvae` | signal_c | 1 | 0.5044 | 0.5112 | [0.4973, 0.5250] | 0.0068 | 0.0066 | [-0.0061, 0.0197] | 0.0230 | 0.0806 |
| `cvae` | signal_d | 0 | 0.4891 | 0.5088 | [0.4949, 0.5226] | 0.0197 | 0.0065 | [0.0069, 0.0326] | 0.0581 | 0.1126 |
| `cvae` | signal_d | 1 | 0.5109 | 0.4912 | [0.4774, 0.5051] | -0.0197 | 0.0065 | [-0.0326, -0.0069] | 0.0581 | 0.1126 |
| `cvae` | signal_e | 0 | 0.4556 | 0.4402 | [0.4265, 0.4540] | -0.0154 | 0.0068 | [-0.0287, -0.0021] | 0.0611 | 0.1678 |
| `cvae` | signal_e | 1 | 0.5444 | 0.5598 | [0.5460, 0.5735] | 0.0154 | 0.0068 | [0.0021, 0.0287] | 0.0611 | 0.1678 |
| `intercept_null` | signal_a | 0 | 0.4675 | 0.4826 | [0.4688, 0.4965] | 0.0151 | 0.0071 | [0.0012, 0.0289] | 0.0151 | 0.0151 |
| `intercept_null` | signal_a | 1 | 0.5325 | 0.5174 | [0.5035, 0.5312] | -0.0151 | 0.0071 | [-0.0289, -0.0012] | 0.0151 | 0.0151 |
| `intercept_null` | signal_b | 0 | 0.5290 | 0.5338 | [0.5200, 0.5476] | 0.0048 | 0.0071 | [-0.0090, 0.0186] | 0.0048 | 0.0048 |
| `intercept_null` | signal_b | 1 | 0.4710 | 0.4662 | [0.4524, 0.4800] | -0.0048 | 0.0071 | [-0.0186, 0.0090] | 0.0048 | 0.0048 |
| `intercept_null` | signal_c | 0 | 0.4878 | 0.4888 | [0.4750, 0.5027] | 0.0010 | 0.0071 | [-0.0128, 0.0149] | 0.0010 | 0.0010 |
| `intercept_null` | signal_c | 1 | 0.5122 | 0.5112 | [0.4973, 0.5250] | -0.0010 | 0.0071 | [-0.0149, 0.0128] | 0.0010 | 0.0010 |
| `intercept_null` | signal_d | 0 | 0.5167 | 0.5088 | [0.4949, 0.5226] | -0.0079 | 0.0071 | [-0.0218, 0.0059] | 0.0079 | 0.0079 |
| `intercept_null` | signal_d | 1 | 0.4833 | 0.4912 | [0.4774, 0.5051] | 0.0079 | 0.0071 | [-0.0059, 0.0218] | 0.0079 | 0.0079 |
| `intercept_null` | signal_e | 0 | 0.4548 | 0.4402 | [0.4265, 0.4540] | -0.0146 | 0.0070 | [-0.0283, -0.0008] | 0.0146 | 0.0146 |
| `intercept_null` | signal_e | 1 | 0.5452 | 0.5598 | [0.5460, 0.5735] | 0.0146 | 0.0070 | [0.0008, 0.0283] | 0.0146 | 0.0146 |
| `oracle` | signal_a | 0 | 0.4785 | 0.4826 | [0.4688, 0.4965] | 0.0041 | 0.0066 | [-0.0088, 0.0170] | 0.0108 | 0.1110 |
| `oracle` | signal_a | 1 | 0.5215 | 0.5174 | [0.5035, 0.5312] | -0.0041 | 0.0066 | [-0.0170, 0.0088] | 0.0108 | 0.1110 |
| `oracle` | signal_b | 0 | 0.5340 | 0.5338 | [0.5200, 0.5476] | -0.0002 | 0.0065 | [-0.0130, 0.0126] | 0.0134 | 0.0577 |
| `oracle` | signal_b | 1 | 0.4660 | 0.4662 | [0.4524, 0.4800] | 0.0002 | 0.0065 | [-0.0126, 0.0130] | 0.0134 | 0.0577 |
| `oracle` | signal_c | 0 | 0.4873 | 0.4888 | [0.4750, 0.5027] | 0.0015 | 0.0065 | [-0.0113, 0.0143] | 0.0132 | 0.0310 |
| `oracle` | signal_c | 1 | 0.5127 | 0.5112 | [0.4973, 0.5250] | -0.0015 | 0.0065 | [-0.0143, 0.0113] | 0.0132 | 0.0310 |
| `oracle` | signal_d | 0 | 0.5115 | 0.5088 | [0.4949, 0.5226] | -0.0027 | 0.0065 | [-0.0154, 0.0100] | 0.0184 | 0.0366 |
| `oracle` | signal_d | 1 | 0.4885 | 0.4912 | [0.4774, 0.5051] | 0.0027 | 0.0065 | [-0.0100, 0.0154] | 0.0184 | 0.0366 |
| `oracle` | signal_e | 0 | 0.4549 | 0.4402 | [0.4265, 0.4540] | -0.0147 | 0.0067 | [-0.0278, -0.0016] | 0.0152 | 0.2438 |
| `oracle` | signal_e | 1 | 0.5451 | 0.5598 | [0.5460, 0.5735] | 0.0147 | 0.0067 | [0.0016, 0.0278] | 0.0152 | 0.2438 |

CVAE classwise reliability bins (signed gap = observed - predicted):

| outcome | level | bin | n | mean predicted | observed | observed Wilson 95% | signed gap | gap SE | gap 95% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| signal_a | 0 | 0 | 8 | 0.0902 | 0.1250 | [0.0224, 0.4709] | 0.0348 | 0.1238 | [-0.2078, 0.2773] |
| signal_a | 0 | 1 | 130 | 0.1645 | 0.1077 | [0.0652, 0.1727] | -0.0569 | 0.0274 | [-0.1106, -0.0032] |
| signal_a | 0 | 2 | 459 | 0.2582 | 0.2244 | [0.1886, 0.2648] | -0.0338 | 0.0193 | [-0.0716, 0.0040] |
| signal_a | 0 | 3 | 754 | 0.3532 | 0.3050 | [0.2732, 0.3388] | -0.0482 | 0.0168 | [-0.0810, -0.0153] |
| signal_a | 0 | 4 | 928 | 0.4513 | 0.4159 | [0.3846, 0.4479] | -0.0354 | 0.0162 | [-0.0671, -0.0037] |
| signal_a | 0 | 5 | 924 | 0.5495 | 0.4924 | [0.4603, 0.5246] | -0.0571 | 0.0164 | [-0.0893, -0.0248] |
| signal_a | 0 | 6 | 852 | 0.6488 | 0.6056 | [0.5724, 0.6379] | -0.0432 | 0.0167 | [-0.0759, -0.0104] |
| signal_a | 0 | 7 | 618 | 0.7455 | 0.7265 | [0.6901, 0.7602] | -0.0190 | 0.0180 | [-0.0542, 0.0162] |
| signal_a | 0 | 8 | 288 | 0.8398 | 0.7847 | [0.7337, 0.8283] | -0.0550 | 0.0242 | [-0.1025, -0.0076] |
| signal_a | 0 | 9 | 39 | 0.9219 | 0.8462 | [0.7027, 0.9275] | -0.0758 | 0.0584 | [-0.1901, 0.0386] |
| signal_a | 1 | 0 | 39 | 0.0781 | 0.1538 | [0.0725, 0.2973] | 0.0758 | 0.0584 | [-0.0386, 0.1901] |
| signal_a | 1 | 1 | 288 | 0.1602 | 0.2153 | [0.1717, 0.2663] | 0.0550 | 0.0242 | [0.0076, 0.1025] |
| signal_a | 1 | 2 | 618 | 0.2545 | 0.2735 | [0.2398, 0.3099] | 0.0190 | 0.0180 | [-0.0162, 0.0542] |
| signal_a | 1 | 3 | 852 | 0.3512 | 0.3944 | [0.3621, 0.4276] | 0.0432 | 0.0167 | [0.0104, 0.0759] |
| signal_a | 1 | 4 | 924 | 0.4505 | 0.5076 | [0.4754, 0.5397] | 0.0571 | 0.0164 | [0.0248, 0.0893] |
| signal_a | 1 | 5 | 928 | 0.5487 | 0.5841 | [0.5521, 0.6154] | 0.0354 | 0.0162 | [0.0037, 0.0671] |
| signal_a | 1 | 6 | 754 | 0.6468 | 0.6950 | [0.6612, 0.7268] | 0.0482 | 0.0168 | [0.0153, 0.0810] |
| signal_a | 1 | 7 | 459 | 0.7418 | 0.7756 | [0.7352, 0.8114] | 0.0338 | 0.0193 | [-0.0040, 0.0716] |
| signal_a | 1 | 8 | 130 | 0.8355 | 0.8923 | [0.8273, 0.9348] | 0.0569 | 0.0274 | [0.0032, 0.1106] |
| signal_a | 1 | 9 | 8 | 0.9098 | 0.8750 | [0.5291, 0.9776] | -0.0348 | 0.1238 | [-0.2773, 0.2078] |
| signal_b | 0 | 1 | 53 | 0.1757 | 0.1698 | [0.0920, 0.2923] | -0.0059 | 0.0525 | [-0.1087, 0.0970] |
| signal_b | 0 | 2 | 473 | 0.2625 | 0.2283 | [0.1928, 0.2683] | -0.0342 | 0.0193 | [-0.0720, 0.0036] |
| signal_b | 0 | 3 | 1327 | 0.3556 | 0.3911 | [0.3652, 0.4176] | 0.0355 | 0.0133 | [0.0094, 0.0616] |
| signal_b | 0 | 4 | 1899 | 0.4509 | 0.5671 | [0.5447, 0.5893] | 0.1163 | 0.0113 | [0.0941, 0.1384] |
| signal_b | 0 | 5 | 1039 | 0.5401 | 0.7488 | [0.7215, 0.7742] | 0.2087 | 0.0135 | [0.1823, 0.2351] |
| signal_b | 0 | 6 | 170 | 0.6342 | 0.8353 | [0.7723, 0.8835] | 0.2011 | 0.0286 | [0.1451, 0.2572] |
| signal_b | 0 | 7 | 35 | 0.7338 | 0.9143 | [0.7762, 0.9704] | 0.1805 | 0.0497 | [0.0830, 0.2779] |
| signal_b | 0 | 8 | 4 | 0.8194 | 1.0000 | [0.5101, 1.0000] | 0.1806 | 0.0095 | [0.1620, 0.1991] |
| signal_b | 1 | 1 | 4 | 0.1806 | 0.0000 | [0.0000, 0.4899] | -0.1806 | 0.0095 | [-0.1991, -0.1620] |
| signal_b | 1 | 2 | 35 | 0.2662 | 0.0857 | [0.0296, 0.2238] | -0.1805 | 0.0497 | [-0.2779, -0.0830] |
| signal_b | 1 | 3 | 170 | 0.3658 | 0.1647 | [0.1165, 0.2277] | -0.2011 | 0.0286 | [-0.2572, -0.1451] |
| signal_b | 1 | 4 | 1039 | 0.4599 | 0.2512 | [0.2258, 0.2785] | -0.2087 | 0.0135 | [-0.2351, -0.1823] |
| signal_b | 1 | 5 | 1899 | 0.5491 | 0.4329 | [0.4107, 0.4553] | -0.1163 | 0.0113 | [-0.1384, -0.0941] |
| signal_b | 1 | 6 | 1327 | 0.6444 | 0.6089 | [0.5824, 0.6348] | -0.0355 | 0.0133 | [-0.0616, -0.0094] |
| signal_b | 1 | 7 | 473 | 0.7375 | 0.7717 | [0.7317, 0.8072] | 0.0342 | 0.0193 | [-0.0036, 0.0720] |
| signal_b | 1 | 8 | 53 | 0.8243 | 0.8302 | [0.7077, 0.9080] | 0.0059 | 0.0525 | [-0.0970, 0.1087] |
| signal_c | 0 | 0 | 11 | 0.0806 | 0.0000 | [0.0000, 0.2588] | -0.0806 | 0.0056 | [-0.0915, -0.0696] |
| signal_c | 0 | 1 | 156 | 0.1644 | 0.1410 | [0.0950, 0.2043] | -0.0234 | 0.0279 | [-0.0780, 0.0312] |
| signal_c | 0 | 2 | 495 | 0.2540 | 0.2081 | [0.1746, 0.2460] | -0.0460 | 0.0182 | [-0.0817, -0.0102] |
| signal_c | 0 | 3 | 901 | 0.3527 | 0.3274 | [0.2976, 0.3587] | -0.0253 | 0.0156 | [-0.0559, 0.0053] |
| signal_c | 0 | 4 | 1054 | 0.4529 | 0.4393 | [0.4096, 0.4694] | -0.0136 | 0.0153 | [-0.0436, 0.0163] |
| signal_c | 0 | 5 | 949 | 0.5506 | 0.5806 | [0.5490, 0.6116] | 0.0300 | 0.0160 | [-0.0014, 0.0614] |
| signal_c | 0 | 6 | 783 | 0.6448 | 0.6386 | [0.6043, 0.6715] | -0.0062 | 0.0171 | [-0.0398, 0.0273] |
| signal_c | 0 | 7 | 470 | 0.7446 | 0.7702 | [0.7301, 0.8060] | 0.0257 | 0.0194 | [-0.0124, 0.0637] |
| signal_c | 0 | 8 | 164 | 0.8385 | 0.8110 | [0.7442, 0.8635] | -0.0275 | 0.0305 | [-0.0873, 0.0323] |
| signal_c | 0 | 9 | 17 | 0.9157 | 0.8824 | [0.6566, 0.9671] | -0.0334 | 0.0803 | [-0.1908, 0.1240] |
| signal_c | 1 | 0 | 17 | 0.0843 | 0.1176 | [0.0329, 0.3434] | 0.0334 | 0.0803 | [-0.1240, 0.1908] |
| signal_c | 1 | 1 | 164 | 0.1615 | 0.1890 | [0.1365, 0.2558] | 0.0275 | 0.0305 | [-0.0323, 0.0873] |
| signal_c | 1 | 2 | 470 | 0.2554 | 0.2298 | [0.1940, 0.2699] | -0.0257 | 0.0194 | [-0.0637, 0.0124] |
| signal_c | 1 | 3 | 783 | 0.3552 | 0.3614 | [0.3285, 0.3957] | 0.0062 | 0.0171 | [-0.0273, 0.0398] |
| signal_c | 1 | 4 | 949 | 0.4494 | 0.4194 | [0.3884, 0.4510] | -0.0300 | 0.0160 | [-0.0614, 0.0014] |
| signal_c | 1 | 5 | 1054 | 0.5471 | 0.5607 | [0.5306, 0.5904] | 0.0136 | 0.0153 | [-0.0163, 0.0436] |
| signal_c | 1 | 6 | 901 | 0.6473 | 0.6726 | [0.6413, 0.7024] | 0.0253 | 0.0156 | [-0.0053, 0.0559] |
| signal_c | 1 | 7 | 495 | 0.7460 | 0.7919 | [0.7540, 0.8254] | 0.0460 | 0.0182 | [0.0102, 0.0817] |
| signal_c | 1 | 8 | 156 | 0.8356 | 0.8590 | [0.7957, 0.9050] | 0.0234 | 0.0279 | [-0.0312, 0.0780] |
| signal_c | 1 | 9 | 11 | 0.9194 | 1.0000 | [0.7412, 1.0000] | 0.0806 | 0.0056 | [0.0696, 0.0915] |
| signal_d | 0 | 0 | 1 | 0.0662 | 0.0000 | [0.0000, 0.7935] | -0.0662 | NA | [NA, NA] |
| signal_d | 0 | 1 | 62 | 0.1703 | 0.0645 | [0.0254, 0.1545] | -0.1058 | 0.0318 | [-0.1680, -0.0435] |
| signal_d | 0 | 2 | 421 | 0.2591 | 0.1900 | [0.1554, 0.2302] | -0.0691 | 0.0192 | [-0.1067, -0.0315] |
| signal_d | 0 | 3 | 867 | 0.3577 | 0.2884 | [0.2592, 0.3194] | -0.0693 | 0.0153 | [-0.0994, -0.0393] |
| signal_d | 0 | 4 | 1270 | 0.4504 | 0.4654 | [0.4381, 0.4929] | 0.0149 | 0.0140 | [-0.0124, 0.0423] |
| signal_d | 0 | 5 | 1279 | 0.5506 | 0.5942 | [0.5671, 0.6208] | 0.0436 | 0.0137 | [0.0167, 0.0704] |
| signal_d | 0 | 6 | 791 | 0.6421 | 0.7547 | [0.7236, 0.7834] | 0.1126 | 0.0153 | [0.0827, 0.1426] |
| signal_d | 0 | 7 | 276 | 0.7398 | 0.8406 | [0.7928, 0.8790] | 0.1008 | 0.0222 | [0.0572, 0.1444] |
| signal_d | 0 | 8 | 33 | 0.8248 | 0.9091 | [0.7643, 0.9686] | 0.0843 | 0.0510 | [-0.0156, 0.1841] |
| signal_d | 1 | 1 | 33 | 0.1752 | 0.0909 | [0.0314, 0.2357] | -0.0843 | 0.0510 | [-0.1841, 0.0156] |
| signal_d | 1 | 2 | 276 | 0.2602 | 0.1594 | [0.1210, 0.2072] | -0.1008 | 0.0222 | [-0.1444, -0.0572] |
| signal_d | 1 | 3 | 791 | 0.3579 | 0.2453 | [0.2166, 0.2764] | -0.1126 | 0.0153 | [-0.1426, -0.0827] |
| signal_d | 1 | 4 | 1279 | 0.4494 | 0.4058 | [0.3792, 0.4329] | -0.0436 | 0.0137 | [-0.0704, -0.0167] |
| signal_d | 1 | 5 | 1270 | 0.5496 | 0.5346 | [0.5071, 0.5619] | -0.0149 | 0.0140 | [-0.0423, 0.0124] |
| signal_d | 1 | 6 | 867 | 0.6423 | 0.7116 | [0.6806, 0.7408] | 0.0693 | 0.0153 | [0.0393, 0.0994] |
| signal_d | 1 | 7 | 421 | 0.7409 | 0.8100 | [0.7698, 0.8446] | 0.0691 | 0.0192 | [0.0315, 0.1067] |
| signal_d | 1 | 8 | 62 | 0.8297 | 0.9355 | [0.8455, 0.9746] | 0.1058 | 0.0318 | [0.0435, 0.1680] |
| signal_d | 1 | 9 | 1 | 0.9338 | 1.0000 | [0.2065, 1.0000] | 0.0662 | NA | [NA, NA] |
| signal_e | 0 | 0 | 1 | 0.0829 | 0.0000 | [0.0000, 0.7935] | -0.0829 | NA | [NA, NA] |
| signal_e | 0 | 1 | 4 | 0.1678 | 0.0000 | [0.0000, 0.4899] | -0.1678 | 0.0188 | [-0.2047, -0.1309] |
| signal_e | 0 | 2 | 117 | 0.2713 | 0.1624 | [0.1065, 0.2398] | -0.1089 | 0.0342 | [-0.1760, -0.0419] |
| signal_e | 0 | 3 | 925 | 0.3631 | 0.2541 | [0.2271, 0.2831] | -0.1091 | 0.0142 | [-0.1370, -0.0811] |
| signal_e | 0 | 4 | 2684 | 0.4540 | 0.4255 | [0.4069, 0.4443] | -0.0285 | 0.0095 | [-0.0471, -0.0099] |
| signal_e | 0 | 5 | 1145 | 0.5352 | 0.6201 | [0.5916, 0.6478] | 0.0849 | 0.0143 | [0.0569, 0.1129] |
| signal_e | 0 | 6 | 119 | 0.6296 | 0.7731 | [0.6900, 0.8391] | 0.1435 | 0.0383 | [0.0684, 0.2186] |
| signal_e | 0 | 7 | 5 | 0.7225 | 0.6000 | [0.2307, 0.8824] | -0.1225 | 0.2403 | [-0.5935, 0.3484] |
| signal_e | 1 | 2 | 5 | 0.2775 | 0.4000 | [0.1176, 0.7693] | 0.1225 | 0.2403 | [-0.3484, 0.5935] |
| signal_e | 1 | 3 | 119 | 0.3704 | 0.2269 | [0.1609, 0.3100] | -0.1435 | 0.0383 | [-0.2186, -0.0684] |
| signal_e | 1 | 4 | 1145 | 0.4648 | 0.3799 | [0.3522, 0.4084] | -0.0849 | 0.0143 | [-0.1129, -0.0569] |
| signal_e | 1 | 5 | 2684 | 0.5460 | 0.5745 | [0.5557, 0.5931] | 0.0285 | 0.0095 | [0.0099, 0.0471] |
| signal_e | 1 | 6 | 925 | 0.6369 | 0.7459 | [0.7169, 0.7729] | 0.1091 | 0.0142 | [0.0811, 0.1370] |
| signal_e | 1 | 7 | 117 | 0.7287 | 0.8376 | [0.7602, 0.8935] | 0.1089 | 0.0342 | [0.0419, 0.1760] |
| signal_e | 1 | 8 | 4 | 0.8322 | 1.0000 | [0.5101, 1.0000] | 0.1678 | 0.0188 | [0.1309, 0.2047] |
| signal_e | 1 | 9 | 1 | 0.9171 | 1.0000 | [0.2065, 1.0000] | 0.0829 | NA | [NA, NA] |

Fixed label-permutation invariance checks:

| probability source | max metric discrepancy | max selected-log-mass discrepancy | overall maximum |
|---|---:|---:|---:|
| `cvae` | 0.000000000000 | 0.000000000000 | 0.000000000000 |
| `intercept_null` | 0.000000000000 | 0.000000000000 | 0.000000000000 |
| `oracle` | 0.000000000000 | 0.000000000000 | 0.000000000000 |

Joint predictive scores and paired CVAE comparisons:

| model | joint NLL | NLL SE | CVAE-minus-model log gain | paired SE | one-sided LCB |
|---|---:|---:|---:|---:|---:|
| `cvae` | 3.2196 | 0.0103 | NA | NA | NA |
| `independent_fitted_marginal` | 3.1810 | 0.0089 | -0.0386 | 0.0043 | -0.0457 |
| `intercept_null` | 3.4563 | 0.0020 | 0.2367 | 0.0105 | 0.2195 |
| `empirical_joint` | 3.4329 | 0.0040 | 0.2133 | 0.0095 | 0.1976 |
| `oracle` | 3.0998 | 0.0118 | -0.1198 | 0.0067 | -0.1307 |

Fitted-CVAE quadrature sensitivity (fixed subset, relative to primary order):

| check order | joint mean delta | joint RMSE | product RMSE | mean G change | p99 shared change | p99 G change |
|---:|---:|---:|---:|---:|---:|---:|
| 21 | -0.002053 | 0.069089 | 0.021484 | -0.001825 | 0.208441 | 0.222287 |
| 41 | -0.002390 | 0.035846 | 0.012618 | -0.002158 | 0.118264 | 0.099884 |

Secondary nested common-panel Monte Carlo diagnostic:

| M | joint NLL | product NLL | mean gain | joint RMSE vs GH | product RMSE vs GH |
|---:|---:|---:|---:|---:|---:|
| 16 | 3.5711 | 3.1497 | -0.4214 | 1.5176 | 0.4971 |
| 64 | 3.1898 | 3.0458 | -0.1440 | 0.5904 | 0.2247 |
| 256 | 3.1121 | 3.0517 | -0.0604 | 0.2356 | 0.1091 |
| 1024 | 3.0647 | 3.0402 | -0.0245 | 0.1107 | 0.0589 |

MC block gain SD: 0.0230; jackknife-corrected mean gain: -0.0209; median joint-integrand ESS: 126.4 (0.123 of M).

Conditional-independence oracle common-panel shared-vs-product maximum absolute differences: M=16: 8.882e-16, M=64: 8.882e-16, M=256: 1.776e-15, M=1024: 1.776e-15.

### Seed 9901 detail

Restored checkpoint: epoch 30 of 30; validation beta-ELBO 1.3955; training KL/latent [1.66037, 1.76990]; active latent units 2 / 2; effective beta 0.2000.

Marginal predictive metrics:

| model | NLL | multiclass Brier | classwise ECE | max class-bin gap |
|---|---:|---:|---:|---:|
| `cvae` | 0.6317 | 0.4416 | 0.0475 | 0.2847 |
| `intercept_null` | 0.6915 | 0.4984 | 0.0074 | 0.0104 |
| `oracle` | 0.6175 | 0.4292 | 0.0101 | 0.0596 |

Classwise calibration-in-the-large and uncertainty:

| model | outcome | level | mean predicted | observed | observed Wilson 95% | CIL gap | CIL SE | CIL 95% | ECE | max gap |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cvae` | signal_a | 0 | 0.4820 | 0.4856 | [0.4718, 0.4995] | 0.0036 | 0.0066 | [-0.0093, 0.0165] | 0.0292 | 0.0943 |
| `cvae` | signal_a | 1 | 0.5180 | 0.5144 | [0.5005, 0.5282] | -0.0036 | 0.0066 | [-0.0165, 0.0093] | 0.0292 | 0.0943 |
| `cvae` | signal_b | 0 | 0.6093 | 0.5390 | [0.5252, 0.5528] | -0.0703 | 0.0066 | [-0.0834, -0.0573] | 0.0908 | 0.2847 |
| `cvae` | signal_b | 1 | 0.3907 | 0.4610 | [0.4472, 0.4748] | 0.0703 | 0.0066 | [0.0573, 0.0834] | 0.0908 | 0.2847 |
| `cvae` | signal_c | 0 | 0.5082 | 0.4862 | [0.4724, 0.5001] | -0.0220 | 0.0066 | [-0.0350, -0.0090] | 0.0784 | 0.2104 |
| `cvae` | signal_c | 1 | 0.4918 | 0.5138 | [0.4999, 0.5276] | 0.0220 | 0.0066 | [0.0090, 0.0350] | 0.0784 | 0.2104 |
| `cvae` | signal_d | 0 | 0.5212 | 0.5154 | [0.5015, 0.5292] | -0.0058 | 0.0066 | [-0.0188, 0.0071] | 0.0263 | 0.0895 |
| `cvae` | signal_d | 1 | 0.4788 | 0.4846 | [0.4708, 0.4985] | 0.0058 | 0.0066 | [-0.0071, 0.0188] | 0.0263 | 0.0895 |
| `cvae` | signal_e | 0 | 0.4504 | 0.4534 | [0.4396, 0.4672] | 0.0030 | 0.0067 | [-0.0101, 0.0161] | 0.0126 | 0.0510 |
| `cvae` | signal_e | 1 | 0.5496 | 0.5466 | [0.5328, 0.5604] | -0.0030 | 0.0067 | [-0.0161, 0.0101] | 0.0126 | 0.0510 |
| `intercept_null` | signal_a | 0 | 0.4755 | 0.4856 | [0.4718, 0.4995] | 0.0101 | 0.0071 | [-0.0038, 0.0239] | 0.0101 | 0.0101 |
| `intercept_null` | signal_a | 1 | 0.5245 | 0.5144 | [0.5005, 0.5282] | -0.0101 | 0.0071 | [-0.0239, 0.0038] | 0.0101 | 0.0101 |
| `intercept_null` | signal_b | 0 | 0.5330 | 0.5390 | [0.5252, 0.5528] | 0.0060 | 0.0071 | [-0.0078, 0.0198] | 0.0060 | 0.0060 |
| `intercept_null` | signal_b | 1 | 0.4670 | 0.4610 | [0.4472, 0.4748] | -0.0060 | 0.0071 | [-0.0198, 0.0078] | 0.0060 | 0.0060 |
| `intercept_null` | signal_c | 0 | 0.4908 | 0.4862 | [0.4724, 0.5001] | -0.0046 | 0.0071 | [-0.0184, 0.0093] | 0.0046 | 0.0046 |
| `intercept_null` | signal_c | 1 | 0.5092 | 0.5138 | [0.4999, 0.5276] | 0.0046 | 0.0071 | [-0.0093, 0.0184] | 0.0046 | 0.0046 |
| `intercept_null` | signal_d | 0 | 0.5050 | 0.5154 | [0.5015, 0.5292] | 0.0104 | 0.0071 | [-0.0035, 0.0243] | 0.0104 | 0.0104 |
| `intercept_null` | signal_d | 1 | 0.4950 | 0.4846 | [0.4708, 0.4985] | -0.0104 | 0.0071 | [-0.0243, 0.0035] | 0.0104 | 0.0104 |
| `intercept_null` | signal_e | 0 | 0.4593 | 0.4534 | [0.4396, 0.4672] | -0.0059 | 0.0070 | [-0.0197, 0.0079] | 0.0059 | 0.0059 |
| `intercept_null` | signal_e | 1 | 0.5407 | 0.5466 | [0.5328, 0.5604] | 0.0059 | 0.0070 | [-0.0079, 0.0197] | 0.0059 | 0.0059 |
| `oracle` | signal_a | 0 | 0.4775 | 0.4856 | [0.4718, 0.4995] | 0.0081 | 0.0066 | [-0.0047, 0.0210] | 0.0128 | 0.0596 |
| `oracle` | signal_a | 1 | 0.5225 | 0.5144 | [0.5005, 0.5282] | -0.0081 | 0.0066 | [-0.0210, 0.0047] | 0.0128 | 0.0596 |
| `oracle` | signal_b | 0 | 0.5356 | 0.5390 | [0.5252, 0.5528] | 0.0034 | 0.0065 | [-0.0094, 0.0161] | 0.0067 | 0.0542 |
| `oracle` | signal_b | 1 | 0.4644 | 0.4610 | [0.4472, 0.4748] | -0.0034 | 0.0065 | [-0.0161, 0.0094] | 0.0067 | 0.0542 |
| `oracle` | signal_c | 0 | 0.4888 | 0.4862 | [0.4724, 0.5001] | -0.0026 | 0.0065 | [-0.0153, 0.0101] | 0.0111 | 0.0468 |
| `oracle` | signal_c | 1 | 0.5112 | 0.5138 | [0.4999, 0.5276] | 0.0026 | 0.0065 | [-0.0101, 0.0153] | 0.0111 | 0.0468 |
| `oracle` | signal_d | 0 | 0.5086 | 0.5154 | [0.5015, 0.5292] | 0.0068 | 0.0066 | [-0.0061, 0.0197] | 0.0149 | 0.0533 |
| `oracle` | signal_d | 1 | 0.4914 | 0.4846 | [0.4708, 0.4985] | -0.0068 | 0.0066 | [-0.0197, 0.0061] | 0.0149 | 0.0533 |
| `oracle` | signal_e | 0 | 0.4570 | 0.4534 | [0.4396, 0.4672] | -0.0036 | 0.0067 | [-0.0167, 0.0094] | 0.0052 | 0.0501 |
| `oracle` | signal_e | 1 | 0.5430 | 0.5466 | [0.5328, 0.5604] | 0.0036 | 0.0067 | [-0.0094, 0.0167] | 0.0052 | 0.0501 |

CVAE classwise reliability bins (signed gap = observed - predicted):

| outcome | level | bin | n | mean predicted | observed | observed Wilson 95% | signed gap | gap SE | gap 95% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| signal_a | 0 | 0 | 5 | 0.0943 | 0.0000 | [0.0000, 0.4345] | -0.0943 | 0.0006 | [-0.0955, -0.0930] |
| signal_a | 0 | 1 | 93 | 0.1730 | 0.0860 | [0.0442, 0.1607] | -0.0870 | 0.0290 | [-0.1438, -0.0302] |
| signal_a | 0 | 2 | 459 | 0.2582 | 0.2200 | [0.1846, 0.2602] | -0.0382 | 0.0193 | [-0.0759, -0.0004] |
| signal_a | 0 | 3 | 986 | 0.3553 | 0.3266 | [0.2980, 0.3565] | -0.0287 | 0.0149 | [-0.0578, 0.0004] |
| signal_a | 0 | 4 | 1254 | 0.4495 | 0.4418 | [0.4145, 0.4694] | -0.0077 | 0.0140 | [-0.0352, 0.0197] |
| signal_a | 0 | 5 | 1066 | 0.5475 | 0.5666 | [0.5367, 0.5961] | 0.0191 | 0.0152 | [-0.0107, 0.0488] |
| signal_a | 0 | 6 | 731 | 0.6430 | 0.6840 | [0.6494, 0.7167] | 0.0410 | 0.0172 | [0.0074, 0.0747] |
| signal_a | 0 | 7 | 330 | 0.7392 | 0.8242 | [0.7795, 0.8615] | 0.0851 | 0.0208 | [0.0444, 0.1258] |
| signal_a | 0 | 8 | 75 | 0.8315 | 0.8800 | [0.7874, 0.9356] | 0.0485 | 0.0375 | [-0.0250, 0.1219] |
| signal_a | 0 | 9 | 1 | 0.9224 | 1.0000 | [0.2065, 1.0000] | 0.0776 | NA | [NA, NA] |
| signal_a | 1 | 0 | 1 | 0.0776 | 0.0000 | [0.0000, 0.7935] | -0.0776 | NA | [NA, NA] |
| signal_a | 1 | 1 | 75 | 0.1685 | 0.1200 | [0.0644, 0.2126] | -0.0485 | 0.0375 | [-0.1219, 0.0250] |
| signal_a | 1 | 2 | 330 | 0.2608 | 0.1758 | [0.1385, 0.2205] | -0.0851 | 0.0208 | [-0.1258, -0.0444] |
| signal_a | 1 | 3 | 731 | 0.3570 | 0.3160 | [0.2833, 0.3506] | -0.0410 | 0.0172 | [-0.0747, -0.0074] |
| signal_a | 1 | 4 | 1066 | 0.4525 | 0.4334 | [0.4039, 0.4633] | -0.0191 | 0.0152 | [-0.0488, 0.0107] |
| signal_a | 1 | 5 | 1254 | 0.5505 | 0.5582 | [0.5306, 0.5855] | 0.0077 | 0.0140 | [-0.0197, 0.0352] |
| signal_a | 1 | 6 | 986 | 0.6447 | 0.6734 | [0.6435, 0.7020] | 0.0287 | 0.0149 | [-0.0004, 0.0578] |
| signal_a | 1 | 7 | 459 | 0.7418 | 0.7800 | [0.7398, 0.8154] | 0.0382 | 0.0193 | [0.0004, 0.0759] |
| signal_a | 1 | 8 | 93 | 0.8270 | 0.9140 | [0.8393, 0.9558] | 0.0870 | 0.0290 | [0.0302, 0.1438] |
| signal_a | 1 | 9 | 5 | 0.9057 | 1.0000 | [0.5655, 1.0000] | 0.0943 | 0.0006 | [0.0930, 0.0955] |
| signal_b | 0 | 2 | 6 | 0.2847 | 0.0000 | [0.0000, 0.3903] | -0.2847 | 0.0051 | [-0.2947, -0.2746] |
| signal_b | 0 | 3 | 86 | 0.3729 | 0.1279 | [0.0729, 0.2147] | -0.2450 | 0.0363 | [-0.3161, -0.1738] |
| signal_b | 0 | 4 | 550 | 0.4632 | 0.2455 | [0.2113, 0.2831] | -0.2178 | 0.0183 | [-0.2536, -0.1820] |
| signal_b | 0 | 5 | 1676 | 0.5540 | 0.4206 | [0.3972, 0.4444] | -0.1333 | 0.0120 | [-0.1568, -0.1099] |
| signal_b | 0 | 6 | 1756 | 0.6479 | 0.6270 | [0.6041, 0.6493] | -0.0209 | 0.0115 | [-0.0434, 0.0016] |
| signal_b | 0 | 7 | 834 | 0.7383 | 0.7914 | [0.7625, 0.8176] | 0.0530 | 0.0140 | [0.0255, 0.0806] |
| signal_b | 0 | 8 | 92 | 0.8280 | 0.9022 | [0.8244, 0.9477] | 0.0742 | 0.0311 | [0.0132, 0.1352] |
| signal_b | 1 | 1 | 92 | 0.1720 | 0.0978 | [0.0523, 0.1756] | -0.0742 | 0.0311 | [-0.1352, -0.0132] |
| signal_b | 1 | 2 | 834 | 0.2617 | 0.2086 | [0.1824, 0.2375] | -0.0530 | 0.0140 | [-0.0806, -0.0255] |
| signal_b | 1 | 3 | 1756 | 0.3521 | 0.3730 | [0.3507, 0.3959] | 0.0209 | 0.0115 | [-0.0016, 0.0434] |
| signal_b | 1 | 4 | 1676 | 0.4460 | 0.5794 | [0.5556, 0.6028] | 0.1333 | 0.0120 | [0.1099, 0.1568] |
| signal_b | 1 | 5 | 550 | 0.5368 | 0.7545 | [0.7169, 0.7887] | 0.2178 | 0.0183 | [0.1820, 0.2536] |
| signal_b | 1 | 6 | 86 | 0.6271 | 0.8721 | [0.7853, 0.9271] | 0.2450 | 0.0363 | [0.1738, 0.3161] |
| signal_b | 1 | 7 | 6 | 0.7153 | 1.0000 | [0.6097, 1.0000] | 0.2847 | 0.0051 | [0.2746, 0.2947] |
| signal_c | 0 | 1 | 9 | 0.1784 | 0.0000 | [0.0000, 0.2991] | -0.1784 | 0.0066 | [-0.1913, -0.1655] |
| signal_c | 0 | 2 | 116 | 0.2707 | 0.0603 | [0.0295, 0.1193] | -0.2104 | 0.0221 | [-0.2537, -0.1670] |
| signal_c | 0 | 3 | 569 | 0.3638 | 0.2214 | [0.1893, 0.2574] | -0.1424 | 0.0173 | [-0.1763, -0.1084] |
| signal_c | 0 | 4 | 1688 | 0.4538 | 0.3685 | [0.3458, 0.3918] | -0.0853 | 0.0117 | [-0.1082, -0.0624] |
| signal_c | 0 | 5 | 1715 | 0.5488 | 0.5732 | [0.5496, 0.5964] | 0.0244 | 0.0119 | [0.0011, 0.0476] |
| signal_c | 0 | 6 | 724 | 0.6363 | 0.7445 | [0.7115, 0.7749] | 0.1082 | 0.0161 | [0.0766, 0.1398] |
| signal_c | 0 | 7 | 169 | 0.7390 | 0.8521 | [0.7907, 0.8977] | 0.1131 | 0.0275 | [0.0592, 0.1670] |
| signal_c | 0 | 8 | 9 | 0.8253 | 1.0000 | [0.7009, 1.0000] | 0.1747 | 0.0064 | [0.1621, 0.1873] |
| signal_c | 0 | 9 | 1 | 0.9174 | 1.0000 | [0.2065, 1.0000] | 0.0826 | NA | [NA, NA] |
| signal_c | 1 | 0 | 1 | 0.0826 | 0.0000 | [0.0000, 0.7935] | -0.0826 | NA | [NA, NA] |
| signal_c | 1 | 1 | 9 | 0.1747 | 0.0000 | [0.0000, 0.2991] | -0.1747 | 0.0064 | [-0.1873, -0.1621] |
| signal_c | 1 | 2 | 169 | 0.2610 | 0.1479 | [0.1023, 0.2093] | -0.1131 | 0.0275 | [-0.1670, -0.0592] |
| signal_c | 1 | 3 | 724 | 0.3637 | 0.2555 | [0.2251, 0.2885] | -0.1082 | 0.0161 | [-0.1398, -0.0766] |
| signal_c | 1 | 4 | 1715 | 0.4512 | 0.4268 | [0.4036, 0.4504] | -0.0244 | 0.0119 | [-0.0476, -0.0011] |
| signal_c | 1 | 5 | 1688 | 0.5462 | 0.6315 | [0.6082, 0.6542] | 0.0853 | 0.0117 | [0.0624, 0.1082] |
| signal_c | 1 | 6 | 569 | 0.6362 | 0.7786 | [0.7426, 0.8107] | 0.1424 | 0.0173 | [0.1084, 0.1763] |
| signal_c | 1 | 7 | 116 | 0.7293 | 0.9397 | [0.8807, 0.9705] | 0.2104 | 0.0221 | [0.1670, 0.2537] |
| signal_c | 1 | 8 | 9 | 0.8216 | 1.0000 | [0.7009, 1.0000] | 0.1784 | 0.0066 | [0.1655, 0.1913] |
| signal_d | 0 | 0 | 3 | 0.0663 | 0.0000 | [0.0000, 0.5615] | -0.0663 | 0.0155 | [-0.0967, -0.0359] |
| signal_d | 0 | 1 | 68 | 0.1642 | 0.2059 | [0.1268, 0.3164] | 0.0417 | 0.0494 | [-0.0551, 0.1384] |
| signal_d | 0 | 2 | 374 | 0.2606 | 0.1711 | [0.1363, 0.2126] | -0.0895 | 0.0195 | [-0.1276, -0.0513] |
| signal_d | 0 | 3 | 714 | 0.3565 | 0.3291 | [0.2957, 0.3644] | -0.0274 | 0.0176 | [-0.0618, 0.0071] |
| signal_d | 0 | 4 | 1097 | 0.4551 | 0.4303 | [0.4013, 0.4598] | -0.0248 | 0.0150 | [-0.0542, 0.0045] |
| signal_d | 0 | 5 | 1162 | 0.5494 | 0.5559 | [0.5272, 0.5843] | 0.0066 | 0.0146 | [-0.0220, 0.0351] |
| signal_d | 0 | 6 | 902 | 0.6498 | 0.6674 | [0.6360, 0.6974] | 0.0176 | 0.0156 | [-0.0130, 0.0482] |
| signal_d | 0 | 7 | 516 | 0.7389 | 0.7771 | [0.7392, 0.8109] | 0.0382 | 0.0183 | [0.0023, 0.0742] |
| signal_d | 0 | 8 | 151 | 0.8336 | 0.8675 | [0.8043, 0.9126] | 0.0340 | 0.0275 | [-0.0199, 0.0879] |
| signal_d | 0 | 9 | 13 | 0.9184 | 0.9231 | [0.6669, 0.9863] | 0.0047 | 0.0761 | [-0.1445, 0.1539] |
| signal_d | 1 | 0 | 13 | 0.0816 | 0.0769 | [0.0137, 0.3331] | -0.0047 | 0.0761 | [-0.1539, 0.1445] |
| signal_d | 1 | 1 | 151 | 0.1664 | 0.1325 | [0.0874, 0.1957] | -0.0340 | 0.0275 | [-0.0879, 0.0199] |
| signal_d | 1 | 2 | 516 | 0.2611 | 0.2229 | [0.1891, 0.2608] | -0.0382 | 0.0183 | [-0.0742, -0.0023] |
| signal_d | 1 | 3 | 902 | 0.3502 | 0.3326 | [0.3026, 0.3640] | -0.0176 | 0.0156 | [-0.0482, 0.0130] |
| signal_d | 1 | 4 | 1162 | 0.4506 | 0.4441 | [0.4157, 0.4728] | -0.0066 | 0.0146 | [-0.0351, 0.0220] |
| signal_d | 1 | 5 | 1097 | 0.5449 | 0.5697 | [0.5402, 0.5987] | 0.0248 | 0.0150 | [-0.0045, 0.0542] |
| signal_d | 1 | 6 | 714 | 0.6435 | 0.6709 | [0.6356, 0.7043] | 0.0274 | 0.0176 | [-0.0071, 0.0618] |
| signal_d | 1 | 7 | 374 | 0.7394 | 0.8289 | [0.7874, 0.8637] | 0.0895 | 0.0195 | [0.0513, 0.1276] |
| signal_d | 1 | 8 | 68 | 0.8358 | 0.7941 | [0.6836, 0.8732] | -0.0417 | 0.0494 | [-0.1384, 0.0551] |
| signal_d | 1 | 9 | 3 | 0.9337 | 1.0000 | [0.4385, 1.0000] | 0.0663 | 0.0155 | [0.0359, 0.0967] |
| signal_e | 0 | 0 | 9 | 0.0843 | 0.1111 | [0.0199, 0.4350] | 0.0268 | 0.1136 | [-0.1959, 0.2494] |
| signal_e | 0 | 1 | 184 | 0.1648 | 0.1630 | [0.1167, 0.2232] | -0.0018 | 0.0271 | [-0.0550, 0.0514] |
| signal_e | 0 | 2 | 622 | 0.2565 | 0.2379 | [0.2062, 0.2730] | -0.0186 | 0.0171 | [-0.0520, 0.0149] |
| signal_e | 0 | 3 | 1118 | 0.3529 | 0.3435 | [0.3162, 0.3718] | -0.0094 | 0.0142 | [-0.0371, 0.0184] |
| signal_e | 0 | 4 | 1217 | 0.4491 | 0.4478 | [0.4201, 0.4759] | -0.0013 | 0.0143 | [-0.0293, 0.0267] |
| signal_e | 0 | 5 | 1008 | 0.5474 | 0.5734 | [0.5427, 0.6036] | 0.0260 | 0.0155 | [-0.0044, 0.0565] |
| signal_e | 0 | 6 | 595 | 0.6432 | 0.6471 | [0.6078, 0.6844] | 0.0039 | 0.0195 | [-0.0344, 0.0421] |
| signal_e | 0 | 7 | 215 | 0.7398 | 0.7814 | [0.7215, 0.8314] | 0.0416 | 0.0281 | [-0.0135, 0.0967] |
| signal_e | 0 | 8 | 31 | 0.8323 | 0.8710 | [0.7115, 0.9487] | 0.0387 | 0.0605 | [-0.0798, 0.1572] |
| signal_e | 0 | 9 | 1 | 0.9490 | 1.0000 | [0.2065, 1.0000] | 0.0510 | NA | [NA, NA] |
| signal_e | 1 | 0 | 1 | 0.0510 | 0.0000 | [0.0000, 0.7935] | -0.0510 | NA | [NA, NA] |
| signal_e | 1 | 1 | 31 | 0.1677 | 0.1290 | [0.0513, 0.2885] | -0.0387 | 0.0605 | [-0.1572, 0.0798] |
| signal_e | 1 | 2 | 215 | 0.2602 | 0.2186 | [0.1686, 0.2785] | -0.0416 | 0.0281 | [-0.0967, 0.0135] |
| signal_e | 1 | 3 | 595 | 0.3568 | 0.3529 | [0.3156, 0.3922] | -0.0039 | 0.0195 | [-0.0421, 0.0344] |
| signal_e | 1 | 4 | 1008 | 0.4526 | 0.4266 | [0.3964, 0.4573] | -0.0260 | 0.0155 | [-0.0565, 0.0044] |
| signal_e | 1 | 5 | 1217 | 0.5509 | 0.5522 | [0.5241, 0.5799] | 0.0013 | 0.0143 | [-0.0267, 0.0293] |
| signal_e | 1 | 6 | 1118 | 0.6471 | 0.6565 | [0.6282, 0.6838] | 0.0094 | 0.0142 | [-0.0184, 0.0371] |
| signal_e | 1 | 7 | 622 | 0.7435 | 0.7621 | [0.7270, 0.7938] | 0.0186 | 0.0171 | [-0.0149, 0.0520] |
| signal_e | 1 | 8 | 184 | 0.8352 | 0.8370 | [0.7768, 0.8833] | 0.0018 | 0.0271 | [-0.0514, 0.0550] |
| signal_e | 1 | 9 | 9 | 0.9157 | 0.8889 | [0.5650, 0.9801] | -0.0268 | 0.1136 | [-0.2494, 0.1959] |

Fixed label-permutation invariance checks:

| probability source | max metric discrepancy | max selected-log-mass discrepancy | overall maximum |
|---|---:|---:|---:|
| `cvae` | 0.000000000000 | 0.000000000000 | 0.000000000000 |
| `intercept_null` | 0.000000000000 | 0.000000000000 | 0.000000000000 |
| `oracle` | 0.000000000000 | 0.000000000000 | 0.000000000000 |

Joint predictive scores and paired CVAE comparisons:

| model | joint NLL | NLL SE | CVAE-minus-model log gain | paired SE | one-sided LCB |
|---|---:|---:|---:|---:|---:|
| `cvae` | 3.2117 | 0.0107 | NA | NA | NA |
| `independent_fitted_marginal` | 3.1587 | 0.0089 | -0.0531 | 0.0051 | -0.0615 |
| `intercept_null` | 3.4577 | 0.0018 | 0.2459 | 0.0103 | 0.2290 |
| `empirical_joint` | 3.4311 | 0.0038 | 0.2193 | 0.0093 | 0.2040 |
| `oracle` | 3.0873 | 0.0118 | -0.1244 | 0.0070 | -0.1358 |

Fitted-CVAE quadrature sensitivity (fixed subset, relative to primary order):

| check order | joint mean delta | joint RMSE | product RMSE | mean G change | p99 shared change | p99 G change |
|---:|---:|---:|---:|---:|---:|---:|
| 21 | -0.000673 | 0.105251 | 0.035089 | -0.000827 | 0.317698 | 0.295489 |
| 41 | 0.001037 | 0.055836 | 0.017622 | 0.003138 | 0.153647 | 0.134124 |

Secondary nested common-panel Monte Carlo diagnostic:

| M | joint NLL | product NLL | mean gain | joint RMSE vs GH | product RMSE vs GH |
|---:|---:|---:|---:|---:|---:|
| 16 | 3.7141 | 3.2341 | -0.4800 | 1.6630 | 0.4666 |
| 64 | 3.2567 | 3.1465 | -0.1102 | 0.4123 | 0.2455 |
| 256 | 3.1649 | 3.1481 | -0.0168 | 0.1920 | 0.1038 |
| 1024 | 3.1597 | 3.1376 | -0.0221 | 0.1060 | 0.0577 |

MC block gain SD: 0.0349; jackknife-corrected mean gain: -0.0184; median joint-integrand ESS: 110.9 (0.108 of M).

Conditional-independence oracle common-panel shared-vs-product maximum absolute differences: M=16: 8.882e-16, M=64: 8.882e-16, M=256: 1.776e-15, M=1024: 1.776e-15.

### Seed 31415 detail

Restored checkpoint: epoch 30 of 30; validation beta-ELBO 1.2239; training KL/latent [1.86174, 1.84563]; active latent units 2 / 2; effective beta 0.2000.

Marginal predictive metrics:

| model | NLL | multiclass Brier | classwise ECE | max class-bin gap |
|---|---:|---:|---:|---:|
| `cvae` | 0.6503 | 0.4587 | 0.0855 | 0.4159 |
| `intercept_null` | 0.6924 | 0.4993 | 0.0115 | 0.0182 |
| `oracle` | 0.6185 | 0.4300 | 0.0167 | 0.4172 |

Classwise calibration-in-the-large and uncertainty:

| model | outcome | level | mean predicted | observed | observed Wilson 95% | CIL gap | CIL SE | CIL 95% | ECE | max gap |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cvae` | signal_a | 0 | 0.5245 | 0.4896 | [0.4758, 0.5035] | -0.0349 | 0.0066 | [-0.0480, -0.0219] | 0.0511 | 0.1289 |
| `cvae` | signal_a | 1 | 0.4755 | 0.5104 | [0.4965, 0.5242] | 0.0349 | 0.0066 | [0.0219, 0.0480] | 0.0511 | 0.1289 |
| `cvae` | signal_b | 0 | 0.6456 | 0.5142 | [0.5003, 0.5280] | -0.1314 | 0.0067 | [-0.1445, -0.1182] | 0.1318 | 0.2622 |
| `cvae` | signal_b | 1 | 0.3544 | 0.4858 | [0.4720, 0.4997] | 0.1314 | 0.0067 | [0.1182, 0.1445] | 0.1318 | 0.2622 |
| `cvae` | signal_c | 0 | 0.4784 | 0.4882 | [0.4744, 0.5021] | 0.0098 | 0.0066 | [-0.0031, 0.0228] | 0.0481 | 0.1318 |
| `cvae` | signal_c | 1 | 0.5216 | 0.5118 | [0.4979, 0.5256] | -0.0098 | 0.0066 | [-0.0228, 0.0031] | 0.0481 | 0.1318 |
| `cvae` | signal_d | 0 | 0.4839 | 0.4990 | [0.4851, 0.5129] | 0.0151 | 0.0068 | [0.0017, 0.0284] | 0.0886 | 0.2968 |
| `cvae` | signal_d | 1 | 0.5161 | 0.5010 | [0.4871, 0.5149] | -0.0151 | 0.0068 | [-0.0284, -0.0017] | 0.0886 | 0.2968 |
| `cvae` | signal_e | 0 | 0.3454 | 0.4522 | [0.4384, 0.4660] | 0.1068 | 0.0067 | [0.0937, 0.1199] | 0.1077 | 0.4159 |
| `cvae` | signal_e | 1 | 0.6546 | 0.5478 | [0.5340, 0.5616] | -0.1068 | 0.0067 | [-0.1199, -0.0937] | 0.1077 | 0.4159 |
| `intercept_null` | signal_a | 0 | 0.4725 | 0.4896 | [0.4758, 0.5035] | 0.0171 | 0.0071 | [0.0032, 0.0310] | 0.0171 | 0.0171 |
| `intercept_null` | signal_a | 1 | 0.5275 | 0.5104 | [0.4965, 0.5242] | -0.0171 | 0.0071 | [-0.0310, -0.0032] | 0.0171 | 0.0171 |
| `intercept_null` | signal_b | 0 | 0.5317 | 0.5142 | [0.5003, 0.5280] | -0.0175 | 0.0071 | [-0.0314, -0.0037] | 0.0175 | 0.0175 |
| `intercept_null` | signal_b | 1 | 0.4683 | 0.4858 | [0.4720, 0.4997] | 0.0175 | 0.0071 | [0.0037, 0.0314] | 0.0175 | 0.0175 |
| `intercept_null` | signal_c | 0 | 0.4893 | 0.4882 | [0.4744, 0.5021] | -0.0011 | 0.0071 | [-0.0149, 0.0128] | 0.0011 | 0.0011 |
| `intercept_null` | signal_c | 1 | 0.5107 | 0.5118 | [0.4979, 0.5256] | 0.0011 | 0.0071 | [-0.0128, 0.0149] | 0.0011 | 0.0011 |
| `intercept_null` | signal_d | 0 | 0.5172 | 0.4990 | [0.4851, 0.5129] | -0.0182 | 0.0071 | [-0.0321, -0.0044] | 0.0182 | 0.0182 |
| `intercept_null` | signal_d | 1 | 0.4828 | 0.5010 | [0.4871, 0.5149] | 0.0182 | 0.0071 | [0.0044, 0.0321] | 0.0182 | 0.0182 |
| `intercept_null` | signal_e | 0 | 0.4560 | 0.4522 | [0.4384, 0.4660] | -0.0038 | 0.0070 | [-0.0176, 0.0100] | 0.0038 | 0.0038 |
| `intercept_null` | signal_e | 1 | 0.5440 | 0.5478 | [0.5340, 0.5616] | 0.0038 | 0.0070 | [-0.0100, 0.0176] | 0.0038 | 0.0038 |
| `oracle` | signal_a | 0 | 0.4805 | 0.4896 | [0.4758, 0.5035] | 0.0091 | 0.0065 | [-0.0038, 0.0219] | 0.0212 | 0.0422 |
| `oracle` | signal_a | 1 | 0.5195 | 0.5104 | [0.4965, 0.5242] | -0.0091 | 0.0065 | [-0.0219, 0.0038] | 0.0212 | 0.0422 |
| `oracle` | signal_b | 0 | 0.5286 | 0.5142 | [0.5003, 0.5280] | -0.0144 | 0.0065 | [-0.0273, -0.0016] | 0.0175 | 0.0376 |
| `oracle` | signal_b | 1 | 0.4714 | 0.4858 | [0.4720, 0.4997] | 0.0144 | 0.0065 | [0.0016, 0.0273] | 0.0175 | 0.0376 |
| `oracle` | signal_c | 0 | 0.4930 | 0.4882 | [0.4744, 0.5021] | -0.0048 | 0.0065 | [-0.0176, 0.0080] | 0.0170 | 0.0540 |
| `oracle` | signal_c | 1 | 0.5070 | 0.5118 | [0.4979, 0.5256] | 0.0048 | 0.0065 | [-0.0080, 0.0176] | 0.0170 | 0.0540 |
| `oracle` | signal_d | 0 | 0.5078 | 0.4990 | [0.4851, 0.5129] | -0.0088 | 0.0065 | [-0.0216, 0.0039] | 0.0155 | 0.0537 |
| `oracle` | signal_d | 1 | 0.4922 | 0.5010 | [0.4871, 0.5149] | 0.0088 | 0.0065 | [-0.0039, 0.0216] | 0.0155 | 0.0537 |
| `oracle` | signal_e | 0 | 0.4584 | 0.4522 | [0.4384, 0.4660] | -0.0062 | 0.0066 | [-0.0193, 0.0068] | 0.0123 | 0.4172 |
| `oracle` | signal_e | 1 | 0.5416 | 0.5478 | [0.5340, 0.5616] | 0.0062 | 0.0066 | [-0.0068, 0.0193] | 0.0123 | 0.4172 |

CVAE classwise reliability bins (signed gap = observed - predicted):

| outcome | level | bin | n | mean predicted | observed | observed Wilson 95% | signed gap | gap SE | gap 95% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| signal_a | 0 | 0 | 2 | 0.0864 | 0.0000 | [0.0000, 0.6576] | -0.0864 | 0.0027 | [-0.0916, -0.0812] |
| signal_a | 0 | 1 | 26 | 0.1744 | 0.0769 | [0.0214, 0.2414] | -0.0975 | 0.0534 | [-0.2021, 0.0072] |
| signal_a | 0 | 2 | 273 | 0.2718 | 0.1429 | [0.1063, 0.1893] | -0.1289 | 0.0211 | [-0.1703, -0.0875] |
| signal_a | 0 | 3 | 398 | 0.3588 | 0.2563 | [0.2159, 0.3014] | -0.1025 | 0.0219 | [-0.1453, -0.0596] |
| signal_a | 0 | 4 | 1318 | 0.4398 | 0.3596 | [0.3342, 0.3859] | -0.0802 | 0.0132 | [-0.1060, -0.0544] |
| signal_a | 0 | 5 | 1778 | 0.5623 | 0.5450 | [0.5218, 0.5680] | -0.0173 | 0.0118 | [-0.0403, 0.0058] |
| signal_a | 0 | 6 | 717 | 0.6443 | 0.6653 | [0.6299, 0.6989] | 0.0210 | 0.0176 | [-0.0135, 0.0555] |
| signal_a | 0 | 7 | 445 | 0.7285 | 0.7775 | [0.7366, 0.8137] | 0.0491 | 0.0196 | [0.0107, 0.0874] |
| signal_a | 0 | 8 | 43 | 0.8268 | 0.9070 | [0.7840, 0.9632] | 0.0802 | 0.0448 | [-0.0076, 0.1679] |
| signal_a | 1 | 1 | 43 | 0.1732 | 0.0930 | [0.0368, 0.2160] | -0.0802 | 0.0448 | [-0.1679, 0.0076] |
| signal_a | 1 | 2 | 445 | 0.2715 | 0.2225 | [0.1863, 0.2634] | -0.0491 | 0.0196 | [-0.0874, -0.0107] |
| signal_a | 1 | 3 | 717 | 0.3557 | 0.3347 | [0.3011, 0.3701] | -0.0210 | 0.0176 | [-0.0555, 0.0135] |
| signal_a | 1 | 4 | 1778 | 0.4377 | 0.4550 | [0.4320, 0.4782] | 0.0173 | 0.0118 | [-0.0058, 0.0403] |
| signal_a | 1 | 5 | 1318 | 0.5602 | 0.6404 | [0.6141, 0.6658] | 0.0802 | 0.0132 | [0.0544, 0.1060] |
| signal_a | 1 | 6 | 398 | 0.6412 | 0.7437 | [0.6986, 0.7841] | 0.1025 | 0.0219 | [0.0596, 0.1453] |
| signal_a | 1 | 7 | 273 | 0.7282 | 0.8571 | [0.8107, 0.8937] | 0.1289 | 0.0211 | [0.0875, 0.1703] |
| signal_a | 1 | 8 | 26 | 0.8256 | 0.9231 | [0.7586, 0.9786] | 0.0975 | 0.0534 | [-0.0072, 0.2021] |
| signal_a | 1 | 9 | 2 | 0.9136 | 1.0000 | [0.3424, 1.0000] | 0.0864 | 0.0027 | [0.0812, 0.0916] |
| signal_b | 0 | 1 | 1 | 0.1745 | 0.0000 | [0.0000, 0.7935] | -0.1745 | NA | [NA, NA] |
| signal_b | 0 | 2 | 7 | 0.2566 | 0.0000 | [0.0000, 0.3543] | -0.2566 | 0.0098 | [-0.2757, -0.2375] |
| signal_b | 0 | 3 | 50 | 0.3622 | 0.1000 | [0.0435, 0.2136] | -0.2622 | 0.0438 | [-0.3481, -0.1763] |
| signal_b | 0 | 4 | 336 | 0.4645 | 0.2054 | [0.1656, 0.2518] | -0.2592 | 0.0220 | [-0.3022, -0.2161] |
| signal_b | 0 | 5 | 1046 | 0.5570 | 0.3241 | [0.2964, 0.3531] | -0.2329 | 0.0144 | [-0.2612, -0.2046] |
| signal_b | 0 | 6 | 2000 | 0.6527 | 0.5145 | [0.4926, 0.5364] | -0.1382 | 0.0111 | [-0.1600, -0.1164] |
| signal_b | 0 | 7 | 1456 | 0.7409 | 0.7163 | [0.6926, 0.7389] | -0.0245 | 0.0117 | [-0.0475, -0.0015] |
| signal_b | 0 | 8 | 104 | 0.8174 | 0.8269 | [0.7429, 0.8876] | 0.0095 | 0.0371 | [-0.0631, 0.0821] |
| signal_b | 1 | 1 | 104 | 0.1826 | 0.1731 | [0.1124, 0.2571] | -0.0095 | 0.0371 | [-0.0821, 0.0631] |
| signal_b | 1 | 2 | 1456 | 0.2591 | 0.2837 | [0.2611, 0.3074] | 0.0245 | 0.0117 | [0.0015, 0.0475] |
| signal_b | 1 | 3 | 2000 | 0.3473 | 0.4855 | [0.4636, 0.5074] | 0.1382 | 0.0111 | [0.1164, 0.1600] |
| signal_b | 1 | 4 | 1046 | 0.4430 | 0.6759 | [0.6469, 0.7036] | 0.2329 | 0.0144 | [0.2046, 0.2612] |
| signal_b | 1 | 5 | 336 | 0.5355 | 0.7946 | [0.7482, 0.8344] | 0.2592 | 0.0220 | [0.2161, 0.3022] |
| signal_b | 1 | 6 | 50 | 0.6378 | 0.9000 | [0.7864, 0.9565] | 0.2622 | 0.0438 | [0.1763, 0.3481] |
| signal_b | 1 | 7 | 7 | 0.7434 | 1.0000 | [0.6457, 1.0000] | 0.2566 | 0.0098 | [0.2375, 0.2757] |
| signal_b | 1 | 8 | 1 | 0.8255 | 1.0000 | [0.2065, 1.0000] | 0.1745 | NA | [NA, NA] |
| signal_c | 0 | 0 | 2 | 0.0982 | 0.0000 | [0.0000, 0.6576] | -0.0982 | 0.0004 | [-0.0990, -0.0974] |
| signal_c | 0 | 1 | 50 | 0.1671 | 0.1000 | [0.0435, 0.2136] | -0.0671 | 0.0430 | [-0.1514, 0.0171] |
| signal_c | 0 | 2 | 501 | 0.2686 | 0.1876 | [0.1559, 0.2241] | -0.0810 | 0.0174 | [-0.1152, -0.0468] |
| signal_c | 0 | 3 | 714 | 0.3551 | 0.2829 | [0.2511, 0.3170] | -0.0722 | 0.0168 | [-0.1050, -0.0393] |
| signal_c | 0 | 4 | 1535 | 0.4374 | 0.4534 | [0.4287, 0.4784] | 0.0160 | 0.0127 | [-0.0089, 0.0408] |
| signal_c | 0 | 5 | 1446 | 0.5613 | 0.5858 | [0.5602, 0.6109] | 0.0245 | 0.0129 | [-0.0008, 0.0497] |
| signal_c | 0 | 6 | 440 | 0.6415 | 0.7477 | [0.7051, 0.7860] | 0.1062 | 0.0208 | [0.0654, 0.1470] |
| signal_c | 0 | 7 | 287 | 0.7287 | 0.8502 | [0.8043, 0.8868] | 0.1215 | 0.0210 | [0.0804, 0.1626] |
| signal_c | 0 | 8 | 25 | 0.8282 | 0.9600 | [0.8046, 0.9929] | 0.1318 | 0.0408 | [0.0518, 0.2117] |
| signal_c | 1 | 1 | 25 | 0.1718 | 0.0400 | [0.0071, 0.1954] | -0.1318 | 0.0408 | [-0.2117, -0.0518] |
| signal_c | 1 | 2 | 287 | 0.2713 | 0.1498 | [0.1132, 0.1957] | -0.1215 | 0.0210 | [-0.1626, -0.0804] |
| signal_c | 1 | 3 | 440 | 0.3585 | 0.2523 | [0.2140, 0.2949] | -0.1062 | 0.0208 | [-0.1470, -0.0654] |
| signal_c | 1 | 4 | 1446 | 0.4387 | 0.4142 | [0.3891, 0.4398] | -0.0245 | 0.0129 | [-0.0497, 0.0008] |
| signal_c | 1 | 5 | 1535 | 0.5626 | 0.5466 | [0.5216, 0.5713] | -0.0160 | 0.0127 | [-0.0408, 0.0089] |
| signal_c | 1 | 6 | 714 | 0.6449 | 0.7171 | [0.6830, 0.7489] | 0.0722 | 0.0168 | [0.0393, 0.1050] |
| signal_c | 1 | 7 | 501 | 0.7314 | 0.8124 | [0.7759, 0.8441] | 0.0810 | 0.0174 | [0.0468, 0.1152] |
| signal_c | 1 | 8 | 50 | 0.8329 | 0.9000 | [0.7864, 0.9565] | 0.0671 | 0.0430 | [-0.0171, 0.1514] |
| signal_c | 1 | 9 | 2 | 0.9018 | 1.0000 | [0.3424, 1.0000] | 0.0982 | 0.0004 | [0.0974, 0.0990] |
| signal_d | 0 | 2 | 2 | 0.2968 | 0.0000 | [0.0000, 0.6576] | -0.2968 | 0.0016 | [-0.2998, -0.2937] |
| signal_d | 0 | 3 | 340 | 0.3797 | 0.2059 | [0.1663, 0.2520] | -0.1738 | 0.0220 | [-0.2168, -0.1308] |
| signal_d | 0 | 4 | 2898 | 0.4525 | 0.4096 | [0.3918, 0.4276] | -0.0429 | 0.0090 | [-0.0606, -0.0252] |
| signal_d | 0 | 5 | 1471 | 0.5360 | 0.6791 | [0.6548, 0.7025] | 0.1432 | 0.0121 | [0.1195, 0.1669] |
| signal_d | 0 | 6 | 235 | 0.6383 | 0.8340 | [0.7812, 0.8762] | 0.1957 | 0.0243 | [0.1482, 0.2433] |
| signal_d | 0 | 7 | 46 | 0.7321 | 0.7609 | [0.6206, 0.8609] | 0.0288 | 0.0635 | [-0.0957, 0.1532] |
| signal_d | 0 | 8 | 7 | 0.8245 | 1.0000 | [0.6457, 1.0000] | 0.1755 | 0.0056 | [0.1645, 0.1865] |
| signal_d | 0 | 9 | 1 | 0.9242 | 1.0000 | [0.2065, 1.0000] | 0.0758 | NA | [NA, NA] |
| signal_d | 1 | 0 | 1 | 0.0758 | 0.0000 | [0.0000, 0.7935] | -0.0758 | NA | [NA, NA] |
| signal_d | 1 | 1 | 7 | 0.1755 | 0.0000 | [0.0000, 0.3543] | -0.1755 | 0.0056 | [-0.1865, -0.1645] |
| signal_d | 1 | 2 | 46 | 0.2679 | 0.2391 | [0.1391, 0.3794] | -0.0288 | 0.0635 | [-0.1532, 0.0957] |
| signal_d | 1 | 3 | 235 | 0.3617 | 0.1660 | [0.1238, 0.2188] | -0.1957 | 0.0243 | [-0.2433, -0.1482] |
| signal_d | 1 | 4 | 1471 | 0.4640 | 0.3209 | [0.2975, 0.3452] | -0.1432 | 0.0121 | [-0.1669, -0.1195] |
| signal_d | 1 | 5 | 2898 | 0.5475 | 0.5904 | [0.5724, 0.6082] | 0.0429 | 0.0090 | [0.0252, 0.0606] |
| signal_d | 1 | 6 | 340 | 0.6203 | 0.7941 | [0.7480, 0.8337] | 0.1738 | 0.0220 | [0.1308, 0.2168] |
| signal_d | 1 | 7 | 2 | 0.7032 | 1.0000 | [0.3424, 1.0000] | 0.2968 | 0.0016 | [0.2937, 0.2998] |
| signal_e | 0 | 0 | 14 | 0.0897 | 0.0714 | [0.0127, 0.3147] | -0.0182 | 0.0710 | [-0.1574, 0.1209] |
| signal_e | 0 | 1 | 662 | 0.1655 | 0.2190 | [0.1892, 0.2521] | 0.0535 | 0.0160 | [0.0223, 0.0848] |
| signal_e | 0 | 2 | 1525 | 0.2502 | 0.3502 | [0.3266, 0.3745] | 0.0999 | 0.0122 | [0.0760, 0.1239] |
| signal_e | 0 | 3 | 1252 | 0.3473 | 0.4720 | [0.4445, 0.4997] | 0.1248 | 0.0141 | [0.0972, 0.1523] |
| signal_e | 0 | 4 | 824 | 0.4463 | 0.5619 | [0.5278, 0.5954] | 0.1156 | 0.0173 | [0.0818, 0.1495] |
| signal_e | 0 | 5 | 451 | 0.5450 | 0.7184 | [0.6752, 0.7579] | 0.1734 | 0.0212 | [0.1318, 0.2149] |
| signal_e | 0 | 6 | 186 | 0.6454 | 0.7419 | [0.6746, 0.7995] | 0.0966 | 0.0322 | [0.0335, 0.1596] |
| signal_e | 0 | 7 | 61 | 0.7390 | 0.7213 | [0.5983, 0.8181] | -0.0177 | 0.0572 | [-0.1299, 0.0945] |
| signal_e | 0 | 8 | 23 | 0.8372 | 0.8696 | [0.6787, 0.9546] | 0.0323 | 0.0727 | [-0.1101, 0.1747] |
| signal_e | 0 | 9 | 2 | 0.9159 | 0.5000 | [0.0945, 0.9055] | -0.4159 | 0.5033 | [-1.4023, 0.5705] |
| signal_e | 1 | 0 | 2 | 0.0841 | 0.5000 | [0.0945, 0.9055] | 0.4159 | 0.5033 | [-0.5705, 1.4023] |
| signal_e | 1 | 1 | 23 | 0.1628 | 0.1304 | [0.0454, 0.3213] | -0.0323 | 0.0727 | [-0.1747, 0.1101] |
| signal_e | 1 | 2 | 61 | 0.2610 | 0.2787 | [0.1819, 0.4017] | 0.0177 | 0.0572 | [-0.0945, 0.1299] |
| signal_e | 1 | 3 | 186 | 0.3546 | 0.2581 | [0.2005, 0.3254] | -0.0966 | 0.0322 | [-0.1596, -0.0335] |
| signal_e | 1 | 4 | 451 | 0.4550 | 0.2816 | [0.2421, 0.3248] | -0.1734 | 0.0212 | [-0.2149, -0.1318] |
| signal_e | 1 | 5 | 824 | 0.5537 | 0.4381 | [0.4046, 0.4722] | -0.1156 | 0.0173 | [-0.1495, -0.0818] |
| signal_e | 1 | 6 | 1252 | 0.6527 | 0.5280 | [0.5003, 0.5555] | -0.1248 | 0.0141 | [-0.1523, -0.0972] |
| signal_e | 1 | 7 | 1525 | 0.7498 | 0.6498 | [0.6255, 0.6734] | -0.0999 | 0.0122 | [-0.1239, -0.0760] |
| signal_e | 1 | 8 | 662 | 0.8345 | 0.7810 | [0.7479, 0.8108] | -0.0535 | 0.0160 | [-0.0848, -0.0223] |
| signal_e | 1 | 9 | 14 | 0.9103 | 0.9286 | [0.6853, 0.9873] | 0.0182 | 0.0710 | [-0.1209, 0.1574] |

Fixed label-permutation invariance checks:

| probability source | max metric discrepancy | max selected-log-mass discrepancy | overall maximum |
|---|---:|---:|---:|
| `cvae` | 0.000000000000 | 0.000000000000 | 0.000000000000 |
| `intercept_null` | 0.000000000000 | 0.000000000000 | 0.000000000000 |
| `oracle` | 0.000000000000 | 0.000000000000 | 0.000000000000 |

Joint predictive scores and paired CVAE comparisons:

| model | joint NLL | NLL SE | CVAE-minus-model log gain | paired SE | one-sided LCB |
|---|---:|---:|---:|---:|---:|
| `cvae` | 3.2995 | 0.0109 | NA | NA | NA |
| `independent_fitted_marginal` | 3.2514 | 0.0095 | -0.0481 | 0.0051 | -0.0565 |
| `intercept_null` | 3.4622 | 0.0019 | 0.1627 | 0.0103 | 0.1457 |
| `empirical_joint` | 3.4384 | 0.0038 | 0.1389 | 0.0095 | 0.1233 |
| `oracle` | 3.0925 | 0.0116 | -0.2070 | 0.0086 | -0.2212 |

Fitted-CVAE quadrature sensitivity (fixed subset, relative to primary order):

| check order | joint mean delta | joint RMSE | product RMSE | mean G change | p99 shared change | p99 G change |
|---:|---:|---:|---:|---:|---:|---:|
| 21 | -0.004180 | 0.130715 | 0.052949 | -0.003396 | 0.430950 | 0.430264 |
| 41 | 0.011902 | 0.062721 | 0.029860 | 0.010698 | 0.202981 | 0.172902 |

Secondary nested common-panel Monte Carlo diagnostic:

| M | joint NLL | product NLL | mean gain | joint RMSE vs GH | product RMSE vs GH |
|---:|---:|---:|---:|---:|---:|
| 16 | 4.3220 | 3.4558 | -0.8662 | 2.0705 | 0.5482 |
| 64 | 3.5905 | 3.3344 | -0.2561 | 0.6483 | 0.2507 |
| 256 | 3.4326 | 3.3172 | -0.1154 | 0.2562 | 0.1354 |
| 1024 | 3.3977 | 3.3155 | -0.0822 | 0.1475 | 0.0694 |

MC block gain SD: 0.0386; jackknife-corrected mean gain: -0.0766; median joint-integrand ESS: 82.3 (0.080 of M).

Conditional-independence oracle common-panel shared-vs-product maximum absolute differences: M=16: 8.882e-16, M=64: 8.882e-16, M=256: 1.332e-15, M=1024: 1.776e-15.

## Interpretation constraints

- Joint log predictive mass is the primary dependence metric; integer-code covariance is not used.
- The independent fitted-marginal baseline is the product of the same fitted CVAE marginals, integrated with the same deterministic quadrature rule. This isolates the shared-latent joint contribution.
- The intercept-null and empirical-joint baselines are fitted on training data only. Validation rows are used only for early stopping, and test rows are touched only after fitting.
- Early stopping selects the validation beta-ELBO after warm-up, not held-out prior-predictive log score. Alternative selection criteria are a Phase B sensitivity axis.
- Gauss-Hermite scores, not finite-M log-mean estimates, determine the gates. The nested Monte Carlo panel is diagnostic because finite-M log estimates can have unequal downward bias; M=1 is intentionally excluded.
- Oracle results are a population-optimal reference under the specified DGP, not a finite-sample ceiling: a fitted model can score better on a realized test sample by chance.
- A top-level seed jointly indexes one data replicate and one initialization/training stream. The three-seed bounds measure replicate-level variability but do not separately identify initialization sensitivity.
- Per-row paired standard errors condition on the fitted checkpoint. Only the three-seed summaries incorporate replicate-level data and training variation.
- Fixed-probability label permutation is a scoring and representation sanity check, not a refit-under-relabeling stability experiment; refitted equivalence remains Phase B work.
- The nested Monte Carlo results are descriptive and have no acceptance gate in Phase A.
- Pairwise contingency recovery is covered by the existing synthetic recovery regression test; richer held-out contingency diagnostics and stress distributions remain Phase B work.
- These core DGPs share the fitted model's Gaussian-latent structure. Nonlinear conditional-independence, rare-level, non-Gaussian/Potts, sample-size, latent-capacity, and KL sensitivity arms remain required before Issue #2 can be closed.
- Results apply to the fixed two-dimensional latent configuration only. Other latent dimensions require their own integration and validation evidence.

Total runtime: 63.1 seconds.
