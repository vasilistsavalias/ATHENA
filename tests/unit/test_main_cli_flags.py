from pathlib import Path


def test_main_cli_no_longer_exposes_clean_or_strict_flags():
    main_path = Path("src/thesis_pipeline/main.py")
    text = main_path.read_text(encoding="utf-8")

    assert 'parser.add_argument("--clean"' not in text
    assert 'parser.add_argument("--strict"' not in text
