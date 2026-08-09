# MLA / MoE Architecture Modeling Study

Independent GlacierEQ portfolio work exploring latent KV-storage arithmetic and expert-activation ratios inspired by publicly described MLA/MoE design patterns.

**Status:** local architecture model + C reference sources.  
**Evidence token:** `MODELED_MLA_MOE_SCENARIO_NOT_MODEL_EXECUTION`

This repository is **not affiliated with, endorsed by, or operated by DeepSeek**. It does not claim proprietary model access, execution of DeepSeek V3/R1, production inference integration, measured model quality, or measured serving performance.

## Verified capability

The canonical tested Python surface is `src/deepseek_mla_moe_sentinel.py`.

Given explicit dimensions, it deterministically models:

- FP16 K+V storage size;
- a latent-vector storage representation;
- arithmetic storage reduction from those dimensions;
- active-expert fraction for a configured expert pool;
- fail-closed invalid dimension/routing inputs.

The result is **architecture arithmetic**, not a trained-model benchmark or MLA kernel execution.

## Engineering anatomy

| Surface | Evidence-bound role |
|---|---|
| `src/deepseek_mla_moe_sentinel.py` | Canonical tested storage/routing scenario model |
| `tests/test_deepseek.py` | Model semantics, compatibility API, and fail-closed regression |
| `tests/test_mla.py` | Independent storage-ratio arithmetic check; not C execution |
| `src/mla_compression.c` | C reference implementation compiled by CI for build correctness |
| `src/mla_moe_solver.c` | C reference implementation compiled by CI for build correctness |
| `mastermind_sidecar.py` | Local status helper; not proof of APEX/Mastermind integration |

## Native proof

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py' -v
gcc -std=c11 -Wall -Wextra -c src/mla_compression.c -o /tmp/mla_compression.o
gcc -std=c11 -Wall -Wextra -c src/mla_moe_solver.c -o /tmp/mla_moe_solver.o
```

The repository-owned Public Truth Gate runs Python proof on 3.11 and 3.13, compiles both C sources, and verifies the modeled-evidence/non-affiliation boundary.

## Explicit nonclaims

Current evidence does **not** establish:

- execution of DeepSeek V3 or R1;
- a faithful implementation of DeepSeek's production MLA or MoE stack;
- 93% measured KV-cache savings in a deployed model;
- model accuracy, perplexity, throughput, or latency preservation;
- auxiliary-loss-free training behavior;
- CPU cache-hit improvements;
- MCP registration;
- live APEX/AKOS/Mastermind connectivity;
- DeepSeek employment, endorsement, affiliation, or proprietary access.

Those are higher evidence states requiring separate model/runtime measurements.

## Why the capability matters

The useful mechanism is the separation of **dimension-driven compression/routing arithmetic from model-performance claims**. That makes the assumptions inspectable and reusable while leaving real kernel/model validation as an explicit future evidence gate.
