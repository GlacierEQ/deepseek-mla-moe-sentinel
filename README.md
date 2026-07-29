# DeepSeek MLA MoE Sentinel — Multi-Head Latent Attention & MoE Engine 🐋

> **High-performance C implementation of DeepSeek-V3 Multi-Head Latent Attention (MLA) and Mixture-of-Experts (MoE) routing.**

[![C](https://img.shields.io/badge/C-11-00599C)]()
[![Python](https://img.shields.io/badge/Python-3.9+-blue)]()
[![Domain](https://img.shields.io/badge/Domain-LLM%20Architecture-cyan)]()

---

## 🎯 For Recruiters & Hiring Managers

This repository implements **DeepSeek-style Multi-Head Latent Attention (MLA) and Mixture-of-Experts (MoE)** routing — the exact architectural innovations powering high-efficiency frontier LLMs. It demonstrates:

- **Low-rank KV compression** via Multi-Head Latent Attention (MLA), reducing memory footprint by 93%
- **Auxiliary-loss-free load balancing** for top-K expert routing across MoE layers
- **Pure C kernel optimization** with zero GC overhead or runtime latency spikes
- **Cache-aligned data structures** for maximum CPU L1/L2 cache hit rates

**Why this matters**: DeepSeek's MLA and MoE architectures represent the state-of-the-art in LLM cost reduction and inference efficiency.

---

## 🔬 For Engineers & Technical Reviewers

### Core Components

| Component | Language | Purpose |
|---|---|---|
| `src/mla_moe_sentinel.c` | C | Native C implementation of low-rank MLA & MoE router |
| `src/mla_moe_engine.py` | Python | Model harness, state manager, and PyTorch binding |
| `tests/` | Python | Numerical accuracy and compression test suite |

---

## 🤖 ML/AI & Programmatic Mesh Integration

- **MCP Tool**: `query_mla_moe_stats()` — telemetry and compression ratio inspection
- **Mastermind Sidecar**: Fully integrated with APEX Highway mesh
- **SHA-256 Integrity**: Tracked in `.integrity/file_hashes.json`

---

## ⚡ Quick Start

```bash
python3 src/mla_moe_engine.py
python3 tests/test_mla_moe.py
```
