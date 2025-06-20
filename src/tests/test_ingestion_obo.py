from __future__ import annotations

from napistu.ingestion import obo


def test_formatting_obo_attributes():
    assert obo._format_entry_tuple("foo: bar: baz") == ("foo", "bar: baz")
    assert obo._format_entry_tuple("foo") is None


def test_formatting_go_isa_attributes():
    go_parents_test_entries = [
        ([], list()),
        (["foo ! bar"], [{"parent_id": "foo", "parent_name": "bar"}]),
        (
            ["foo ! bar", "fun ! baz"],
            [
                {"parent_id": "foo", "parent_name": "bar"},
                {"parent_id": "fun", "parent_name": "baz"},
            ],
        ),
    ]

    for val_list, expected in go_parents_test_entries:
        assert obo._isa_str_list_to_dict_list(val_list) == expected


################################################
# __main__
################################################

if __name__ == "__main__":
    test_formatting_obo_attributes()
    test_formatting_go_isa_attributes()
