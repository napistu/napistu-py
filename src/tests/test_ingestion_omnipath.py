from napistu.ingestion import omnipath
from napistu.ingestion.constants import OMNIPATH_ANNOTATIONS


def test_parse_omnipath_named_annotation():
    """Test the parse_omnipath_named_annotation function."""
    annotation_str = "CORUM:4478;Compleat:HC1449;PDB:4awl;PDB:6qmp;PDB:6qmq;PDB:6qms;SIGNOR:SIGNOR-C1"
    df = omnipath._parse_omnipath_named_annotation(annotation_str)

    assert df.shape == (7, 3)
    assert list(df.columns) == [
        OMNIPATH_ANNOTATIONS.NAME,
        OMNIPATH_ANNOTATIONS.ANNOTATION,
        OMNIPATH_ANNOTATIONS.ANNOTATION_STR,
    ]
    assert df.iloc[0][OMNIPATH_ANNOTATIONS.NAME] == "CORUM"
    assert df.iloc[0][OMNIPATH_ANNOTATIONS.ANNOTATION] == "4478"
    assert all(df[OMNIPATH_ANNOTATIONS.ANNOTATION_STR] == annotation_str)


def test_parse_omnipath_annotation():
    """Test the parse_omnipath_annotation function."""
    annotation_str = "11290752;11983166;12601176"
    df = omnipath._parse_omnipath_annotation(annotation_str)

    assert df.shape == (3, 2)
    assert list(df.columns) == [
        OMNIPATH_ANNOTATIONS.ANNOTATION,
        OMNIPATH_ANNOTATIONS.ANNOTATION_STR,
    ]
    assert df.iloc[0][OMNIPATH_ANNOTATIONS.ANNOTATION] == "11290752"
    assert all(df[OMNIPATH_ANNOTATIONS.ANNOTATION_STR] == annotation_str)
