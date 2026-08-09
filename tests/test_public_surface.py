from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"


def test_public_surface_uses_real_paths_and_modeled_token() -> None:
    text = README.read_text(encoding="utf-8")
    assert "src/deepseek_mla_moe_sentinel.py" in text
    assert "src/mla_compression.c" in text
    assert "src/mla_moe_solver.c" in text
    assert "MODELED_MLA_MOE_SCENARIO_NOT_MODEL_EXECUTION" in text


def test_public_surface_excludes_stale_production_claims() -> None:
    text = README.read_text(encoding="utf-8").casefold()
    forbidden = (
        "src/mla_moe_sentinel.c",
        "src/mla_moe_engine.py",
        "reducing memory footprint by 93%",
        "mcp tool: `query_mla_moe_stats()`",
        "fully integrated with apex highway",
    )
    assert all(marker not in text for marker in forbidden)


def test_public_surface_declares_deepseek_non_affiliation_and_model_boundary() -> None:
    text = README.read_text(encoding="utf-8").casefold()
    assert "not affiliated with, endorsed by, or operated by deepseek" in text
    assert "does not claim proprietary model access" in text
    assert "architecture arithmetic" in text
    assert "not a trained-model benchmark" in text
