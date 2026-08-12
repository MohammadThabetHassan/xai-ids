"""Regression tests for canonical XAI-IDS project references."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_REPOSITORY = "MohammadThabetHassan/xai-ids"
CANONICAL_PAGES_URL = "https://mohammadthabethassan.github.io/xai-ids/"
LEGACY_IDENTIFIER = "xai-ids-cicids2017"
CANONICAL_SOURCES = (
    "README.md",
    "CONTRIBUTING.md",
    "MODEL_CARD.md",
    "docs/index.md",
    "docs/site/index.html",
    "model_metadata.json",
)


def test_canonical_project_sources_do_not_reference_legacy_identity():
    """Keep published project sources aligned with the current repository name."""
    for relative_path in CANONICAL_SOURCES:
        content = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
        assert LEGACY_IDENTIFIER not in content, relative_path


def test_canonical_project_identity_is_present_in_published_sources():
    """Ensure the repository and Pages source point users to the current project."""
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    site_source = (PROJECT_ROOT / "docs/site/index.html").read_text(encoding="utf-8")
    metadata = (PROJECT_ROOT / "model_metadata.json").read_text(encoding="utf-8")

    assert CANONICAL_REPOSITORY in readme
    assert CANONICAL_REPOSITORY in site_source
    assert CANONICAL_REPOSITORY in metadata
    assert CANONICAL_PAGES_URL in readme
    assert CANONICAL_PAGES_URL in site_source
