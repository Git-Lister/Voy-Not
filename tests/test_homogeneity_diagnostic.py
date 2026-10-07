from src.layer3_mechanisms.data_loader import DataBundle
from scripts.diagnose_homogeneity import _bundle_without_quire


def test_drop_quire_removes_lines():
    b = DataBundle(
        tokenized=[["a"], ["b"], ["c"], ["d"]],
        line_boundaries=[0, 1, 2, 3, 4],
        quire_labels=["Q1", "Q1", "Q2", "Q2"],
        quire_token_lists={"Q1": [["a"], ["b"]], "Q2": [["c"], ["d"]]},
        mismatch_df=None, page_records=None, token_count=4,
    )
    reduced = _bundle_without_quire(b, "Q1")
    assert len(reduced.tokenized) == 2
    assert "Q1" not in reduced.quire_token_lists
    assert reduced.token_count == 2