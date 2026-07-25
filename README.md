# DeepSeek MLA MoE Sentinel

> **Production Solution for DeepSeek V3/R1 Multi-Head Latent Attention & MoE Routing**

## Overview
Low-rank latent KV-cache compressor and auxiliary-loss-free MoE load balancer for DeepSeek architectures.

## Verification
```bash
PYTHONPATH=src python3 tests/test_deepseek.py
python3 mastermind_sidecar.py
```
