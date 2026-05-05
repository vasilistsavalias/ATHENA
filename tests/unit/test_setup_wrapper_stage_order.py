from pathlib import Path


def _extract_assignment_lines(script_text: str, prefix: str) -> list[str]:
    return [
        line.strip()
        for line in script_text.splitlines()
        if line.strip().startswith(prefix)
    ]


def test_wrapper_uses_canonical_stage_groups():
    script_path = Path("scripts/pipeline/setup_and_run.sh")
    text = script_path.read_text(encoding="utf-8")
    pre_lines = _extract_assignment_lines(text, "STAGES_PRE=")
    post_lines = _extract_assignment_lines(text, "STAGES_POST=")
    assert pre_lines, "Expected STAGES_PRE declarations in setup wrapper."
    assert post_lines, "Expected STAGES_POST declarations in setup wrapper."

    pre_order = pre_lines[0].split("=", 1)[1].strip().strip('"').split()
    post_order = post_lines[0].split("=", 1)[1].strip().strip('"').split()

    assert pre_order == [f"S{i:02d}" for i in range(0, 13)]
    assert post_order == [f"S{i:02d}" for i in range(14, 19)]


def test_wrapper_treats_smoke_test_as_full_run():
    script_path = Path("scripts/pipeline/setup_and_run.sh")
    text = script_path.read_text(encoding="utf-8")
    assert '--smoke-test' in text
    assert '[[ "$joined" == *" --full "* ]] || [[ "$joined" == *" --smoke-test "* ]]' in text
