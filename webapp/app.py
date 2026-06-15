"""Streamlit explorer for the 3GPP CSI codebook framework.

Run:  streamlit run webapp/app.py

Two tabs:
* **Figure gallery** — pick a channel, codebook families and figures; the app
  runs the existing figure scripts and shows each PNG with its numbers.
* **Compare channels** — overlay the delay / Doppler / spatial diagnostics of
  several channel models on one grid, with a measured-vs-configured validation
  table (see ``nr_csi.channel.diagnostics`` and ``scripts/figures/channel_compare.py``).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from registry import CDL_MODELS, FAMILIES, FIGURES, run_channel_compare, run_figure

from nr_csi.config import SUPPORTED_N1N2

st.set_page_config(page_title="CSI Codebook Explorer", page_icon="📡", layout="wide")

ANTENNA_PAIRS = sorted(SUPPORTED_N1N2, key=lambda k: (2 * k[0] * k[1], k))


def _ant_label(pair: tuple[int, int]) -> str:
    n1, n2 = pair
    return f"{n1}×{n2}  (P = {2 * n1 * n2} ports)"


def _as_table(data) -> pd.DataFrame | None:
    """Best-effort flat table from a figure's JSON, else None (fall back to JSON)."""
    if not isinstance(data, dict):
        return None
    lists = {k: v for k, v in data.items() if isinstance(v, list) and v
             and all(isinstance(x, (int, float)) for x in v)}
    if lists and len({len(v) for v in lists.values()}) == 1:
        return pd.DataFrame(lists)
    if isinstance(data.get("rows"), list) and data["rows"] and isinstance(data["rows"][0], dict):
        return pd.DataFrame(data["rows"])
    return None


# ----------------------------------------------------------------- shared sidebar
with st.sidebar:
    st.header("Shared grid")
    st.caption("Antenna geometry and frequency granularity used by **both** tabs.")
    pair = st.selectbox("Array (N1×N2)", ANTENNA_PAIRS,
                        index=ANTENNA_PAIRS.index((4, 2)), format_func=_ant_label)
    n3 = st.number_input("Frequency units (N3)", 1, 64, 8, 1)
    n_rx = st.number_input("RX antennas (n_rx)", 1, 8, 2, 1)

    st.header("Run settings")
    fast = st.checkbox("Fast preview (few drops)", value=True,
                       help="Smoke-test sizes — fast but noisy. Uncheck for quality.")
    seed = st.number_input("Seed", 0, 10_000, 0, 1)

st.title("📡 3GPP CSI Codebook Explorer")
tab_fig, tab_chan = st.tabs(["📊 Figure gallery", "📡 Compare channels"])

# ===================================================================== FIGURES
with tab_fig:
    st.caption("Configure a channel, choose codebook families and figures, then "
               "inspect each result with its underlying numbers.")
    c1, c2 = st.columns(2)
    with c1:
        channel = st.selectbox(
            "Channel model", ["synthetic", "cdl"],
            format_func=lambda c: "Synthetic (random multipath)" if c == "synthetic"
            else "Sionna CDL (3GPP TR 38.901)",
        )
        if channel == "synthetic":
            n_paths = st.slider("Multipath rays (n_paths)", 1, 16, 4)
            max_delay = st.slider("Max ray delay (DFT taps)", 0.0, 8.0, 3.0, 0.5)
            cdl_model = cdl_speed = cdl_ds = None
        else:
            cdl_model = st.selectbox("CDL model", CDL_MODELS, index=2)
            cdl_speed = st.number_input("UE speed (km/h)", 0.0, 500.0, 3.0, 1.0)
            cdl_ds = st.number_input("Delay spread (ns)", 1.0, 1000.0, 100.0, 10.0)
            n_paths = max_delay = None
            st.caption("Needs the optional `sionna` extra; first run loads TensorFlow.")
    with c2:
        families = st.multiselect("Codebook families", list(FAMILIES), default=list(FAMILIES))
        figures = st.multiselect("Figures", list(FIGURES), default=["fig_01_se_vs_snr"],
                                 format_func=lambda s: FIGURES[s].title)
        drops = st.number_input("Monte-Carlo drops", 1, 1000, 100, 10, disabled=fast)

    go = st.button("Generate figures", type="primary", width="stretch")
    st.caption("Antenna/N3 apply to every figure except *Array scaling* (sweeps the "
               "array) and *Frequency granularity* (sweeps N3); *Overhead scaling* is "
               "analytic. Figures flagged below ignore the family selection.")

    cfg = {"channel": channel, "n_rx": int(n_rx), "n1": pair[0], "n2": pair[1],
           "n3": int(n3), "families": families, "n_paths": n_paths, "max_delay": max_delay,
           "cdl_model": cdl_model, "cdl_speed": cdl_speed, "cdl_delay_spread_ns": cdl_ds}
    run = {"drops": int(drops), "seed": int(seed), "fast": bool(fast),
           "timeout": 600 if fast else 2400}

    if go:
        if not figures:
            st.warning("Select at least one figure.")
        elif not families:
            st.warning("Select at least one codebook family.")
        else:
            out = []
            progress = st.progress(0.0, text="Starting…")
            for i, slug in enumerate(figures):
                f = FIGURES[slug]
                progress.progress(i / len(figures), text=f"Running {f.title}…")
                with st.spinner(f"Generating {f.title} (~{f.est_seconds}s full quality)…"):
                    out.append((slug, run_figure(slug, cfg, run)))
            progress.progress(1.0, text="Done")
            st.session_state["fig_results"] = out
            st.session_state["fig_fast"] = fast

    fig_results = st.session_state.get("fig_results")
    if fig_results:
        st.divider()
        if st.session_state.get("fig_fast"):
            st.caption("⚡ Fast-preview mode — uncheck *Fast preview* for quality drop counts.")
        for slug, res in fig_results:
            f = FIGURES[slug]
            st.subheader(f.title)
            st.write(f.blurb)
            if not f.honors_families:
                st.info("Shows its own built-in codebook set — the family selection "
                        "does not apply.", icon="ℹ️")
            if res["ok"] and res["png"]:
                st.image(str(res["png"]), width="stretch")
                st.caption(f"Rendered in {res['seconds']:.1f}s · `{res['png'].name}`")
                if res["data"] is not None:
                    with st.expander("Investigate data"):
                        tbl = _as_table(res["data"])
                        if tbl is not None:
                            st.dataframe(tbl, width="stretch")
                        st.json(res["data"], expanded=False)
            else:
                st.error(f"{f.title} failed to generate.")
                with st.expander("Show log"):
                    st.code(res["log"] or "(no output)", language="text")
            st.divider()
    else:
        st.info("Configure above and click **Generate figures**.")

# ============================================================ COMPARE CHANNELS
with tab_chan:
    st.caption("Overlay the channel-domain diagnostics of several models on the shared "
               "grid. All channels use the sidebar's array / N3 / n_rx so the curves are "
               "directly comparable.")

    default_rows = pd.DataFrame([
        {"label": "CDL-C 100ns 3km/h", "type": "cdl", "model": "C", "speed_kmh": 3.0,
         "delay_spread_ns": 100.0, "n_paths": 4, "max_delay": 3.0, "max_doppler": 0.0},
        {"label": "CDL-C 300ns 3km/h", "type": "cdl", "model": "C", "speed_kmh": 3.0,
         "delay_spread_ns": 300.0, "n_paths": 4, "max_delay": 3.0, "max_doppler": 0.0},
        {"label": "CDL-D 30ns (LOS)", "type": "cdl", "model": "D", "speed_kmh": 3.0,
         "delay_spread_ns": 30.0, "n_paths": 4, "max_delay": 3.0, "max_doppler": 0.0},
        {"label": "synthetic 4-ray", "type": "synthetic", "model": "C", "speed_kmh": 3.0,
         "delay_spread_ns": 100.0, "n_paths": 4, "max_delay": 3.0, "max_doppler": 0.5},
    ])
    edited = st.data_editor(
        default_rows, num_rows="dynamic", width="stretch", key="chan_editor",
        column_config={
            "label": st.column_config.TextColumn("label", width="medium"),
            "type": st.column_config.SelectboxColumn("type", options=["cdl", "synthetic"]),
            "model": st.column_config.SelectboxColumn("CDL model", options=CDL_MODELS),
            "speed_kmh": st.column_config.NumberColumn("speed km/h", min_value=0.0, step=1.0),
            "delay_spread_ns": st.column_config.NumberColumn("DS ns", min_value=1.0, step=10.0),
            "n_paths": st.column_config.NumberColumn("syn n_paths", min_value=1, step=1),
            "max_delay": st.column_config.NumberColumn("syn max_delay", min_value=0.0, step=0.5),
            "max_doppler": st.column_config.NumberColumn("syn max_doppler", min_value=0.0,
                                                         step=0.1),
        },
    )

    cc1, cc2, cc3 = st.columns(3)
    n_slots = cc1.number_input("Time slots", 1, 64, 12, 1,
                               help="Slots per drop — the temporal-correlation horizon.")
    interval_ms = cc2.number_input("Slot interval (ms)", 0.1, 20.0, 2.0, 0.1,
                                   help="CDL slot spacing; sets the Doppler time axis.")
    chan_drops = cc3.number_input("Drops per channel", 1, 200, 20, 1, disabled=fast)

    go_chan = st.button("Compare channels", type="primary", width="stretch")
    st.caption("Validation: for CDL, *RMS delay spread* is measured from the PDP and "
               "compared to the configured value (resolution ≈ 130 ns per tap at the "
               "default 256-pt / 30 kHz grid, so very small spreads read high); "
               "temporal coherence tracks UE speed; λ₁ is the dominant spatial "
               "eigenvalue (high ⇒ line-of-sight).")

    if go_chan:
        rows = []
        for _, r in edited.iterrows():
            label = str(r.get("label") or "").strip()
            ctype = str(r.get("type") or "cdl")
            if not label:
                continue
            if ctype == "cdl":
                rows.append({"label": label, "type": "cdl", "model": str(r["model"]),
                             "speed": float(r["speed_kmh"]),
                             "delay_spread_ns": float(r["delay_spread_ns"])})
            else:
                rows.append({"label": label, "type": "synthetic",
                             "n_paths": int(r["n_paths"]), "max_delay": float(r["max_delay"]),
                             "max_doppler": float(r["max_doppler"])})
        if not rows:
            st.warning("Add at least one channel (with a label).")
        else:
            spec = {"antenna": {"n1": pair[0], "n2": pair[1]}, "n3": int(n3),
                    "n_rx": int(n_rx), "n_slots": int(n_slots),
                    "interval_ms": float(interval_ms), "drops": int(chan_drops),
                    "channels": rows}
            crun = {"seed": int(seed), "fast": bool(fast), "timeout": 600 if fast else 2400}
            with st.spinner(f"Generating channels for {len(rows)} model(s)…"):
                st.session_state["chan_result"] = run_channel_compare(spec, crun)
                st.session_state["chan_fast"] = fast

    cres = st.session_state.get("chan_result")
    if cres:
        st.divider()
        if st.session_state.get("chan_fast"):
            st.caption("⚡ Fast-preview mode — uncheck *Fast preview* for more drops.")
        if cres["ok"] and cres["png"]:
            st.image(str(cres["png"]), width="stretch")
            st.caption(f"Rendered in {cres['seconds']:.1f}s")
            if cres["data"] and cres["data"].get("validation"):
                st.subheader("Validation / diagnostics")
                st.dataframe(pd.DataFrame(cres["data"]["validation"]), width="stretch")
            if cres["data"] is not None:
                with st.expander("Raw diagnostics JSON"):
                    st.json(cres["data"], expanded=False)
        else:
            st.error("Channel comparison failed.")
            with st.expander("Show log"):
                st.code(cres["log"] or "(no output)", language="text")
    else:
        st.info("Edit the channel table and click **Compare channels**.")
