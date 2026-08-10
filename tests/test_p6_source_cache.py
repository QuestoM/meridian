"""P6 source-shape cache contracts."""

from pathlib import Path

import kairos_api.uploads_status as uploads_status
from kairos_api import read_cache


def test_the_row_counts_are_served_from_the_files_own_signature(tmp_path) -> None:
    path = tmp_path / "sample.csv"
    path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    read_cache.invalidate(uploads_status.SHAPE_NAMESPACE)
    reads: list[Path] = []

    def counting_reader(target: Path):
        reads.append(target)
        return (["a", "b"], 2, [])

    for _ in range(5):
        columns, rows, _ = uploads_status.file_shape(path, counting_reader)
    assert (columns, rows) == (["a", "b"], 2)
    assert len(reads) == 1, "the file was re-read while its signature had not moved"

    path.write_text("a,b\n1,2\n3,4\n5,6\n", encoding="utf-8")
    uploads_status.file_shape(path, counting_reader)
    assert len(reads) == 2, "a changed file must be re-read"


def test_the_cached_shape_cannot_be_edited_through_a_caller(tmp_path) -> None:
    path = tmp_path / "sample.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    read_cache.invalidate(uploads_status.SHAPE_NAMESPACE)
    columns, _, _ = uploads_status.file_shape(path, lambda target: (["a", "b"], 1, []))
    columns.append("c")
    again, _, _ = uploads_status.file_shape(path, lambda target: (["a", "b"], 1, []))
    assert again == ["a", "b"], "a caller edited the cached value"
