from modules.utils.analysis_id import (
    build_test_output_name,
    ensure_test_output_folders,
    find_test_images,
    find_test_source_folders,
    get_test_output_folders,
    is_test_analysis_id,
)


def test_test_analysis_id_pattern() -> None:
    assert is_test_analysis_id("TEST")
    assert is_test_analysis_id("TEST01")
    assert is_test_analysis_id("TEST43")
    assert is_test_analysis_id("test99")
    assert not is_test_analysis_id("TEST1")
    assert not is_test_analysis_id("AHG06113")

def test_find_test_images_recurses_and_ignores_outputs(tmp_path) -> None:
    analysis_path = tmp_path / "TEST"
    nested_path = analysis_path / "any" / "structure"
    nested_path.mkdir(parents=True)
    (analysis_path / "root.jpg").write_bytes(b"")
    (nested_path / "nested.png").write_bytes(b"")
    (nested_path / "ignore.txt").write_bytes(b"")
    pass_path = analysis_path / "_pass"
    pass_path.mkdir()
    (pass_path / "generated.png").write_bytes(b"")

    images = find_test_images(analysis_path, {".jpg", ".png"})

    assert images == [nested_path / "nested.png", analysis_path / "root.jpg"]
    assert build_test_output_name(analysis_path, analysis_path / "root.jpg") == "root.png"
    assert build_test_output_name(analysis_path, nested_path / "nested.png").startswith(
        "nested__"
    )


def test_arbitrary_test_folders_have_sibling_output_folders(tmp_path) -> None:
    analysis_path = tmp_path / "TEST"
    source_02 = analysis_path / "02(OLD)"
    source_01 = analysis_path / "01(NEW)"
    source_02.mkdir(parents=True)
    source_01.mkdir()

    assert find_test_source_folders(analysis_path) == [
        (1, source_01),
        (2, source_02),
    ]
    assert get_test_output_folders(analysis_path, source_01) == (
        analysis_path / "01(NEW)_pass",
        analysis_path / "01(NEW)_mask",
    )

    ensure_test_output_folders(analysis_path)

    assert (analysis_path / "01(NEW)_pass").is_dir()
    assert (analysis_path / "01(NEW)_mask").is_dir()
    assert (analysis_path / "02(OLD)_pass").is_dir()
    assert (analysis_path / "02(OLD)_mask").is_dir()
    assert find_test_source_folders(analysis_path) == [
        (1, source_01),
        (2, source_02),
    ]


