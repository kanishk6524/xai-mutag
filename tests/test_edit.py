"""Tests for the molecule editor's core (src/edit.py): edits, renumbering, layout, warnings and the PyG conversion."""
import math

import pytest

from src.edit import ATOMS, BONDS, PRESETS, MolEdit


def chain(n=4, element="C", bond="single"):
    return MolEdit([element] * n, {(i, i + 1): bond for i in range(n - 1)})


def test_constants_match_the_dataset_encoding():
    pytest.importorskip("torch_geometric")
    from src.data import ATOM_NAMES, BOND_NAMES

    assert ATOMS == ATOM_NAMES
    assert BONDS == BOND_NAMES


def test_presets_have_the_expected_shape():
    benzene = PRESETS["Benzene (6 carbon ring, no nitro group)"]()
    assert benzene.n == 6 and len(benzene.bonds) == 6 and benzene.components() == 1
    assert benzene.nitro_groups() == []
    nitro = PRESETS["Nitrobenzene (benzene ring + nitro group)"]()
    assert nitro.n == 9 and len(nitro.bonds) == 9 and nitro.nitro_groups() == [6]
    naph = PRESETS["Nitronaphthalene (two rings + nitro group)"]()
    assert naph.n == 13 and len(naph.bonds) == 14 and naph.nitro_groups() == [10]
    for make in PRESETS.values():
        assert make().components() == 1


def test_set_element_and_bond_and_remove_bond():
    m = chain()
    m.set_element(1, "N")
    assert m.elements[1] == "N" and m.label(1) == "N1"
    m.set_bond(0, 3, "double")
    assert m.bond_between(3, 0) == "double"        # order of the two atoms does not matter
    m.set_bond(0, 3, "triple")                      # changing an existing bond
    assert m.bond_between(0, 3) == "triple" and len(m.bonds) == 4
    m.remove_bond(3, 0)
    assert m.bond_between(0, 3) is None


def test_bad_edits_raise_value_error_and_change_nothing():
    m = chain()
    before = (list(m.elements), dict(m.bonds))
    for bad in (lambda: m.set_element(9, "C"), lambda: m.set_element(0, "Xe"), lambda: m.set_bond(0, 0, "single"),
                lambda: m.set_bond(0, 1, "quadruple"), lambda: m.remove_bond(0, 2), lambda: m.add_atom("Q"),
                lambda: m.add_atom("C", attach_to=7)):
        with pytest.raises(ValueError):
            bad()
    assert (m.elements, m.bonds) == before
    with pytest.raises(ValueError):
        MolEdit([])
    with pytest.raises(ValueError):
        MolEdit(["C", "C"], {(0, 5): "single"})


def test_add_atom_attached_and_separate():
    m = chain(3)
    k = m.add_atom("O", attach_to=2, bond="double")
    assert k == 3 and m.n == 4 and m.bond_between(2, 3) == "double" and m.neighbors(3) == [2]
    k = m.add_atom("Cl")
    assert k == 4 and m.neighbors(4) == [] and m.components() == 2


def test_add_atom_keeps_existing_atoms_where_they_were_and_places_the_new_one_close():
    m = chain(5)
    before = m.positions()
    m.add_atom("N", attach_to=4)
    after = m.positions()
    assert set(after) == set(range(6))
    for i in range(5):
        assert after[i] == pytest.approx(before[i])
    assert all(math.isfinite(c) for p in after.values() for c in p)
    dx, dy = after[5][0] - after[4][0], after[5][1] - after[4][1]
    assert 0 < math.hypot(dx, dy) < 1.5


def test_remove_atom_renumbers_bonds_elements_and_positions():
    m = MolEdit(["C", "N", "O", "Cl"], {(0, 1): "single", (1, 2): "double", (2, 3): "single", (0, 3): "aromatic"})
    pos = m.positions()
    m.remove_atom(1)
    assert m.elements == ["C", "O", "Cl"]
    assert m.bonds == {(1, 2): "single", (0, 2): "aromatic"}   # old (2,3) -> (1,2); old (0,3) -> (0,2); bonds of atom 1 gone
    after = m.positions()
    assert set(after) == {0, 1, 2}
    assert after[1] == pytest.approx(pos[2]) and after[2] == pytest.approx(pos[3])


def test_cannot_remove_the_last_atom():
    m = MolEdit(["C"])
    with pytest.raises(ValueError):
        m.remove_atom(0)


def test_copy_is_independent():
    m = chain()
    c = m.copy()
    c.set_element(0, "O")
    c.remove_atom(3)
    assert m.elements == ["C"] * 4 and len(m.bonds) == 3


def test_nitro_detection_needs_two_terminal_oxygens():
    m = PRESETS["Nitrobenzene (benzene ring + nitro group)"]()
    assert m.nitro_groups() == [6]
    m.set_element(8, "C")                       # one oxygen left
    assert m.nitro_groups() == []
    m2 = PRESETS["Nitrobenzene (benzene ring + nitro group)"]()
    m2.add_atom("C", attach_to=7, bond="single")  # an oxygen with another neighbour is no longer terminal
    assert m2.nitro_groups() == []


def test_warnings_cover_size_pieces_valence_and_missing_nitro():
    nitro = PRESETS["Nitrobenzene (benzene ring + nitro group)"]()
    text = " ".join(nitro.warnings((10, 28)))
    assert "outside the sizes" in text and "nitro" not in text      # 9 atoms is too small, but the nitro group is present
    assert nitro.warnings(None) == []
    benzene = PRESETS["Benzene (6 carbon ring, no nitro group)"]()
    assert any("no nitro group" in w for w in benzene.warnings())
    split = MolEdit(["C", "C", "C"], {(0, 1): "single"})
    assert any("separate pieces" in w for w in split.warnings())
    over = MolEdit(["C"] * 6, {(0, k): "single" for k in range(1, 6)})   # carbon with five bonds
    assert any("C0" in w for w in over.warnings())
    halogen = MolEdit(["Cl", "C", "C"], {(0, 1): "single", (0, 2): "single"})
    assert any("Cl0" in w for w in halogen.warnings())


def test_to_data_and_from_data_round_trip():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    m = PRESETS["Nitrobenzene (benzene ring + nitro group)"]()
    d = m.to_data()
    assert tuple(d.x.shape) == (9, 7) and d.x.sum().item() == 9
    assert tuple(d.edge_index.shape) == (2, 18)                    # every bond in both directions
    assert tuple(d.edge_attr.shape) == (18, 4) and d.edge_attr.sum().item() == 18
    back = MolEdit.from_data(d)
    assert back.elements == m.elements and back.bonds == m.bonds


def test_to_data_handles_a_molecule_without_bonds():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    d = MolEdit(["C", "N"]).to_data()
    assert tuple(d.edge_index.shape) == (2, 0) and tuple(d.edge_attr.shape) == (0, 4) and d.num_nodes == 2


def test_an_edited_molecule_goes_through_the_real_model():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    import torch
    from src.explain.subgraphx import predict_proba
    from src.models import build_model

    torch.manual_seed(0)
    model = build_model("gin", 7, hidden=16, num_layers=2).eval()
    m = PRESETS["Nitrobenzene (benzene ring + nitro group)"]()
    p = predict_proba(model, m.to_data())
    assert p.shape == (2,) and abs(p.sum() - 1) < 1e-5
    m.add_atom("Cl", attach_to=3)
    m.remove_atom(8)
    p2 = predict_proba(model, m.to_data())
    assert p2.shape == (2,) and abs(p2.sum() - 1) < 1e-5

