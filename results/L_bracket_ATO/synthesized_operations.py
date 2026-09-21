# Operations written by the synthesis stage during this run.
# Recorded verbatim for review. NOT registered anywhere: a synthesized
# operation lives only for the run that wrote it.


# --- H1 -> add_out_of_plane_hand_load_case  admitted=True  slot='load cases'  tier=T2  (writer proposed T1) ---
def apply(spec, prm):
    magnitude = float(prm["hand_load_magnitude"])
    distribute_nodes = int(prm["distribute_nodes"])
    spec.load_cases.append(
        LoadCase(
            magnitude=magnitude,
            distribute_nodes=distribute_nodes,
            direction="z",
            name="hand_load_out_of_plane",
        )
    )


# --- H2 -> switch_to_extruded_3d_model  admitted=True  slot='discretisation (analysis model)'  tier=T1  (writer proposed T3) ---
def apply(spec, prm):
    spec.geometry.model = str("extruded_3d")
    spec.geometry.n_layers = int(prm["n_layers"])


# --- H3 -> add_out_of_plane_hand_load_case  admitted=True  slot='load cases'  tier=T2  (writer proposed T1) ---
def apply(spec, prm):
    spec.geometry.model = "extruded_3d"
    spec.load_cases.append(
        LoadCase(
            magnitude=float(prm["hand_load_magnitude"]),
            distribute_nodes=int(prm["distribute_nodes"]),
            direction="z",
            name="hand_load_out_of_plane",
        )
    )


# --- H4 -> set_thickness_stock_bounds  admitted=False  rejected_because="unexpected KeyError: 'apply_source'" ---
# no source returned


# --- H2 -> set_initial_thickness  admitted=True  slot='design variables'  tier=T2  (writer proposed None) ---
def apply(spec, prm):
    t = float(prm["thickness"])
    spec.geometry.thickness = t


# --- H5 -> set_thickness_to_stock_bound  admitted=True  slot='design domain (geometry)'  tier=T3  (writer proposed None) ---
def apply(spec, prm):
    thickness = float(prm["thickness"])
    spec.geometry.thickness = thickness
