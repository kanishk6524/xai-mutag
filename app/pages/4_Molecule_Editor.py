import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `import ui` finds app/ui.py

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
from src.edit import ATOMS, BONDS, PRESETS, MolEdit

ui.setup("Molecule editor", "✏️")
ui.header("Molecule editor",
          "Start from a molecule, then change atoms and bonds and watch the model's prediction move after every edit. "
          "SubgraphX can then show which atoms the new prediction depends on.")
ui.note("<b>Read this before presenting:</b> the model learned from 188 MUTAG molecules only. For an edited molecule that "
        "looks unlike them, the output shows how the <i>model</i> behaves, not what the chemistry would do. "
        "This is a teaching tool, not a way to judge real compounds.")
st.write("")

if not (ui.CHECKPOINTS / ui.MAIN_CKPT).exists():
    ui.missing("The model checkpoint", "python -m src.train --final --drop-node 0.1 --tag dn10  (then copy it to checkpoints/gin_best.pt)")
    st.stop()

model, ck = ui.get_model()
ds = ui.get_dataset()
SS = st.session_state


# ---------------------------------------------------------------------------------------------------- state helpers
def p_of(mol):
    """P(mutagenic) of an edited molecule."""
    return float(ui.class_probs(model, mol.to_data())[1])


def clear_widgets():
    """Forget the editing widgets' values: atom numbers change after an edit, so old selections could be invalid."""
    for k in [k for k in SS.keys() if str(k).startswith("ed_")]:
        del SS[k]


def load(mol, label, base_idx=None):
    SS["mol"] = mol
    SS["snaps"] = [mol.copy()]
    SS["base_label"] = label
    SS["base_idx"] = base_idx
    SS["hist"] = [{"Step": 0, "Edit": f"start: {label}", "P(mutagenic)": p_of(mol)}]
    SS["version"] = SS.get("version", 0) + 1
    SS.pop("expl", None)
    SS.pop("flash", None)
    clear_widgets()


def edit(description, fn):
    new = SS["mol"].copy()
    try:
        fn(new)
        p = p_of(new)
    except ValueError as err:
        SS["flash"] = str(err)
        return
    SS["mol"] = new
    SS["snaps"].append(new.copy())
    SS["hist"].append({"Step": len(SS["hist"]), "Edit": description, "P(mutagenic)": p})
    SS["version"] += 1
    SS.pop("expl", None)
    SS.pop("flash", None)
    clear_widgets()


def undo():
    if len(SS["snaps"]) <= 1:
        return
    SS["snaps"].pop()
    SS["hist"].pop()
    SS["mol"] = SS["snaps"][-1].copy()
    SS["version"] += 1
    SS.pop("expl", None)
    SS.pop("flash", None)
    clear_widgets()


def restart():
    first = SS["snaps"][0]
    SS["mol"] = first.copy()
    SS["snaps"] = [first.copy()]
    SS["hist"] = SS["hist"][:1]
    SS["version"] += 1
    SS.pop("expl", None)
    SS.pop("flash", None)
    clear_widgets()


# ---- button callbacks: read the widget values at click time ------------------------------------------------------------
def do_load_dataset():
    i = int(SS["ld_idx"])
    load(MolEdit.from_data(ds[i]), f"MUTAG molecule #{i}", base_idx=i)


def do_load_preset():
    name = SS["ld_preset"]
    load(PRESETS[name](), name.split(" (")[0])


def do_change_atom():
    i, el, mol = SS["ed_ca_atom"], SS["ed_ca_el"], SS["mol"]
    edit(f"change {mol.label(i)} to {el}", lambda m: m.set_element(i, el))


def do_add_atom():
    el, att, bond, mol = SS["ed_aa_el"], SS["ed_aa_att"], SS["ed_aa_bond"], SS["mol"]
    where = f"bonded to {mol.label(att)} ({bond})" if att is not None else "as a separate atom"
    edit(f"add {el} {where}", lambda m: m.add_atom(el, att, bond))


def do_set_bond():
    a, b, kind, mol = SS["ed_b_a"], SS["ed_b_b"], SS["ed_b_kind"], SS["mol"]
    if kind == "no bond":
        edit(f"remove bond {mol.label(a)}-{mol.label(b)}", lambda m: m.remove_bond(a, b))
    else:
        edit(f"set bond {mol.label(a)}-{mol.label(b)} to {kind}", lambda m: m.set_bond(a, b, kind))


def do_remove_atom():
    i, mol = SS["ed_rm_atom"], SS["mol"]
    edit(f"remove {mol.label(i)}", lambda m: m.remove_atom(i))


# ---- first visit: start from the first held-out molecule --------------------------------------------------------------
if "mol" not in SS:
    first_test = int(ck["split"]["test"][0])
    load(MolEdit.from_data(ds[first_test]), f"MUTAG molecule #{first_test}", base_idx=first_test)

# ---- 1. starting point ------------------------------------------------------------------------------------------------
st.subheader("1. Start from")
src = st.radio("Starting point", ["A MUTAG molecule", "A hand-built example"], horizontal=True,
               key="ld_src", label_visibility="collapsed")
splits = ui.split_map(ck)
if src == "A MUTAG molecule":
    c1, c2 = st.columns([3, 1])
    c1.selectbox("Molecule", list(range(len(ds))), key="ld_idx",
                 format_func=lambda i: f"#{i}  ·  {ds[i].num_nodes} atoms  ·  true label: {ui.CLASS_NAMES[int(ds[i].y)]}  ·  "
                                       + ("held-out" if splits[i] == "test" else "seen in training"))
    c2.write("")
    c2.button("Load molecule", on_click=do_load_dataset, key="ld_go_ds")
else:
    c1, c2 = st.columns([3, 1])
    c1.selectbox("Example", list(PRESETS), key="ld_preset")
    c2.write("")
    c2.button("Load example", on_click=do_load_preset, key="ld_go_pre")
    st.caption("Examples are built by hand in MUTAG's atom and bond encoding. They are not entries of the dataset.")

mol = SS["mol"]
data = mol.to_data()
probs = ui.class_probs(model, data)
pred = int(probs.argmax())
p0 = SS["hist"][0]["P(mutagenic)"]
expl = SS.get("expl")
expl = expl["res"] if expl and expl["version"] == SS["version"] else None

# ---- 2. the molecule and the verdict ----------------------------------------------------------------------------------
st.subheader("2. The molecule now")
left, right = st.columns([1.35, 1])
with left:
    title = f"{SS['base_label']}, after {len(SS['hist']) - 1} edit(s)" + ("   (highlighted: SubgraphX)" if expl else "")
    ui.show_mpl(ui.mol_fig(data, expl["nodes"] if expl else None, title=title, size=5.4, label_index=True,
                           pos=mol.positions()))
with right:
    st.markdown(ui.pill("edited molecule" if len(SS["hist"]) > 1 else "unedited start", "orange" if len(SS["hist"]) > 1 else "grey"),
                unsafe_allow_html=True)
    st.write("")
    st.metric("Prediction", ui.CLASS_NAMES[pred])
    st.progress(float(probs[1]), text=f"P(mutagenic) = {probs[1]:.2f}")
    st.metric("Change since the start", f"{probs[1] - p0:+.2f}", help="P(mutagenic) now minus P(mutagenic) at step 0.")
    if SS["base_idx"] is not None and len(SS["hist"]) == 1:
        true = int(ds[SS["base_idx"]].y)
        st.markdown(f"True label of this dataset molecule: **{ui.CLASS_NAMES[true]}**  ·  " +
                    (ui.pill("model correct", "green") if pred == true else ui.pill("model wrong", "orange")),
                    unsafe_allow_html=True)
    elif SS["base_idx"] is not None:
        st.caption("The dataset's label belonged to the original molecule, so it no longer applies after an edit.")
    nitro = mol.nitro_groups()
    st.markdown(f"Nitro group (N with two terminal O): **{'yes' if nitro else 'no'}**  ·  atoms: **{mol.n}**  ·  "
                f"bonds: **{len(mol.bonds)}**")
    warns = mol.warnings(ui.size_range())
    if warns:
        st.warning("**Be careful with this prediction**\n\n" + "\n".join(f"- {w}" for w in warns))

# ---- 3. edit ----------------------------------------------------------------------------------------------------------
st.subheader("3. Edit it")
flash = SS.pop("flash", None)
if flash:
    st.error(flash)

atoms = list(range(mol.n))
fmt = lambda i: mol.label(i)  # noqa: E731
t_atom, t_add, t_bond, t_rm = st.tabs(["Change an atom", "Add an atom", "Add / change / remove a bond", "Remove an atom"])

with t_atom:
    a1, a2, a3 = st.columns([1, 1, 1])
    a1.selectbox("Atom", atoms, format_func=fmt, key="ed_ca_atom")
    a2.selectbox("Becomes", ATOMS, key="ed_ca_el")
    a3.write("")
    a3.button("Apply", on_click=do_change_atom, key="bt_ca_go", type="primary")
    st.caption("Example: turn a carbon into a nitrogen, or an oxygen of the nitro group into a chlorine.")

with t_add:
    b1, b2, b3, b4 = st.columns([1, 1.4, 1, 1])
    b1.selectbox("New atom", ATOMS, key="ed_aa_el")
    b2.selectbox("Bonded to", [None] + atoms, key="ed_aa_att",
                 format_func=lambda i: "nothing (separate atom)" if i is None else mol.label(i))
    b3.selectbox("Bond type", BONDS, index=1, key="ed_aa_bond")
    b4.write("")
    b4.button("Add atom", on_click=do_add_atom, key="bt_aa_go", type="primary")
    st.caption("The new atom is drawn next to the atom it is bonded to. The bond type is ignored for a separate atom.")

with t_bond:
    c1, c2, c3, c4 = st.columns([1, 1, 1.2, 1])
    ba = c1.selectbox("Atom A", atoms, format_func=fmt, key="ed_b_a")
    bb = c2.selectbox("Atom B", atoms, format_func=fmt, index=min(1, mol.n - 1), key="ed_b_b")
    c3.selectbox("Bond", BONDS + ["no bond"], index=1, key="ed_b_kind")
    c4.write("")
    c4.button("Apply", on_click=do_set_bond, key="bt_b_go", type="primary")
    if ba == bb:
        st.caption("Pick two different atoms.")
    else:
        cur = mol.bond_between(ba, bb)
        st.caption(f"Right now {mol.label(ba)} and {mol.label(bb)} " + (f"have a {cur} bond." if cur else "are not bonded."))

with t_rm:
    r1, r2 = st.columns([2, 1])
    r1.selectbox("Atom to remove", atoms, format_func=fmt, key="ed_rm_atom")
    r2.write("")
    r2.button("Remove atom", on_click=do_remove_atom, key="bt_rm_go", type="primary")
    st.caption("Its bonds go with it, and the atoms after it are renumbered.")

u1, u2, _ = st.columns([1, 1.4, 3])
u1.button("Undo last edit", on_click=undo, disabled=len(SS["snaps"]) <= 1, key="bt_undo")
u2.button("Back to the start", on_click=restart, disabled=len(SS["snaps"]) <= 1, key="bt_restart")

# ---- 4. explain -------------------------------------------------------------------------------------------------------
st.subheader("4. Which atoms does the model rely on?")
with st.expander("Search settings"):
    s1, s2, s3, s4 = st.columns(4)
    rollouts = s1.slider("MCTS rollouts", 5, 60, 20, key="ex_roll")
    max_size = s2.slider("Largest explanation (atoms)", 3, 12, 7, key="ex_max")
    samples = s3.slider("Shapley samples", 10, 150, 50, step=10, key="ex_samp")
    seed = s4.number_input("Random seed", 0, 999, 0, key="ex_seed")
if st.button("Run SubgraphX on this molecule", type="primary", key="ex_run"):
    if mol.n < 3:
        st.warning("SubgraphX needs a molecule with at least 3 atoms.")
    else:
        with st.spinner("Searching..."):
            res = ui.explain_data(data, rollouts, max_size, min(3, max_size), samples, int(seed))
        SS["expl"] = {"version": SS["version"], "res": res}
        st.rerun()
if expl:
    e1, e2, e3 = st.columns(3)
    e1.metric("Fidelity+", f"{expl['fidelity_plus']:.2f}",
              help="P(class) on the whole molecule minus P(class) after removing the highlighted atoms.")
    e2.metric("Fidelity−", f"{expl['fidelity_minus']:.2f}",
              help="P(class) on the whole molecule minus P(class) on the highlighted atoms alone.")
    e3.metric("Explanation size", f"{len(expl['nodes'])} of {mol.n} atoms")
    st.markdown("**Highlighted atoms:** " + ", ".join(mol.label(i) for i in expl["nodes"]) +
                f"  ·  explaining the predicted class: {ui.CLASS_NAMES[expl['target']]}  ·  {expl['seconds']:.1f} s")
    st.caption("The highlighted group is drawn in the picture above. Any edit clears it, because it described the old molecule.")
else:
    st.caption("Press the button to highlight the connected group of atoms the current prediction depends on most. "
               "It takes a few seconds.")

# ---- 5. history -------------------------------------------------------------------------------------------------------
st.subheader("5. What each edit did")
hist = pd.DataFrame(SS["hist"])
fig = go.Figure(go.Scatter(x=hist["Step"], y=hist["P(mutagenic)"], mode="lines+markers", text=hist["Edit"],
                           hovertemplate="step %{x}: %{text}<br>P(mutagenic) %{y:.2f}<extra></extra>",
                           line=dict(color=ui.BLUE, width=3), marker=dict(size=9)))
fig.add_hline(y=0.5, line_dash="dot", line_color=ui.GREY)
fig.update_yaxes(range=[0, 1.02], title="P(mutagenic)")
fig.update_xaxes(title="edit number", dtick=1)
ui.plotly(ui.style(fig, 280, legend=False))
shown = hist.copy()
shown["P(mutagenic)"] = shown["P(mutagenic)"].map(lambda v: f"{v:.2f}")
ui.table(shown)
st.caption("Above the dotted line the model says mutagenic, below it non-mutagenic. A big jump after a small edit means the "
           "model is sensitive to that part of the molecule; it does not by itself say the real chemistry would change.")
