"""Codebook registry: metadata, param schemas, antenna specs, and factories.

Twelve locked entries per ``webapp/SPEC.md`` §2. Each entry is a
``CatalogEntry`` with a ``factory(antenna_cfg, n3, params) -> CodebookScheme``
that translates the resolved playground request into a real constructor
call. Constructor ``ValueError``s are already spec-quoting; callers (runner.py,
main.py) catch and translate them, this module does not swallow them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable

from nr_csi.codebooks import (
    R15Type2Codebook,
    R16Type2Codebook,
    R17Type2Codebook,
    R18CJTCodebook,
    R18CJTPortSelectionCodebook,
    R18PredictedPortSelectionCodebook,
    R18Type2Codebook,
    RefinedEType2Codebook,
    RefinedFeType2PortSelectionCodebook,
    RefinedPredictedEType2Codebook,
    RefinedType1MultiPanelCodebook,
    RefinedType1SinglePanelCodebook,
    TwoPortType1Codebook,
    Type1Codebook,
    Type1MultiPanelCodebook,
)
from nr_csi.codebooks.cjt_r18 import (
    CJT_L_COMBOS,
    CJT_PS_ALPHA_COMBOS,
)
from nr_csi.config import (
    R16_PARAM_COMBOS,
    R16_PS_PARAM_COMBOS,
    R17_PARAM_COMBOS,
    R18_PARAM_COMBOS,
    SUPPORTED_N1N2,
    SUPPORTED_N1N2_R19,
    SUPPORTED_NG_N1N2,
    SUPPORTED_NG_N1N2_R19,
    AntennaConfig,
)


def _frac(f: Fraction) -> str:
    return str(f) if f.denominator != 1 else str(f.numerator)


# --------------------------------------------------------------------- specs


@dataclass(frozen=True)
class ParamChoice:
    value: Any
    label: str
    description: str | None = None

    def to_dict(self) -> dict:
        d = {"value": self.value, "label": self.label}
        if self.description is not None:
            d["description"] = self.description
        return d


@dataclass(frozen=True)
class ParamSpec:
    key: str
    label: str
    type: str  # "choice" | "int" | "float" | "bool"
    default: Any
    choices: list[ParamChoice] = field(default_factory=list)
    min: float | None = None
    max: float | None = None
    step: float | None = None
    visible_if: dict | None = None  # {"key": ..., "value": ...}
    help: str = ""

    def to_dict(self) -> dict:
        d = {
            "key": self.key,
            "label": self.label,
            "type": self.type,
            "default": self.default,
            "choices": [c.to_dict() for c in self.choices],
            "min": self.min,
            "max": self.max,
            "step": self.step,
            "help": self.help,
        }
        if self.visible_if is not None:
            d["visibleIf"] = self.visible_if
        return d


@dataclass(frozen=True)
class AntennaPair:
    n1: int
    n2: int
    ports: int
    ng: int = 1

    def to_dict(self) -> dict:
        d = {"n1": self.n1, "n2": self.n2, "ports": self.ports}
        if self.ng != 1:
            d["ng"] = self.ng
        return d


@dataclass(frozen=True)
class AntennaSpec:
    mode: str  # "single" | "multi" | "fixed"
    pairs: list[AntennaPair]

    def to_dict(self) -> dict:
        return {"mode": self.mode, "pairs": [p.to_dict() for p in self.pairs]}


@dataclass(frozen=True)
class CatalogEntry:
    id: str
    name: str
    short_name: str
    release: str
    spec_clause: str
    tagline: str
    ranks: tuple[int, int]
    port_range: tuple[int, int]
    position: int
    lineage: dict | None
    doc_file: str
    params: list[ParamSpec]
    antenna: AntennaSpec
    factory: Callable[[AntennaConfig, int, dict], Any]

    def summary_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "shortName": self.short_name,
            "release": self.release,
            "specClause": self.spec_clause,
            "tagline": self.tagline,
            "ranks": list(self.ranks),
            "portRange": list(self.port_range),
            "position": self.position,
            "lineage": self.lineage,
            "docFile": self.doc_file,
        }


# ------------------------------------------------------------- antenna specs

_STANDARD_PAIRS = [
    AntennaPair(n1, n2, 2 * n1 * n2)
    for n1, n2 in sorted(SUPPORTED_N1N2, key=lambda k: (2 * k[0] * k[1], k))
]
_R19_PAIRS = [
    AntennaPair(n1, n2, 2 * n1 * n2)
    for n1, n2 in sorted(SUPPORTED_N1N2_R19, key=lambda k: (2 * k[0] * k[1], k))
]
_MP_PAIRS = [
    AntennaPair(n1, n2, 2 * ng * n1 * n2, ng=ng)
    for ng, n1, n2 in sorted(SUPPORTED_NG_N1N2, key=lambda k: (2 * k[0] * k[1] * k[2], k))
]
_MP_R19_PAIRS = [
    AntennaPair(n1, n2, 2 * ng * n1 * n2, ng=ng)
    for ng, n1, n2 in sorted(SUPPORTED_NG_N1N2_R19, key=lambda k: (2 * k[0] * k[1] * k[2], k))
]


def _antenna_from_request(mode: str, req: dict) -> AntennaConfig:
    """Build an ``AntennaConfig`` from the RunRequest ``antenna`` dict."""
    n1 = int(req.get("n1", 4))
    n2 = int(req.get("n2", 2))
    ng = int(req.get("ng", 1))
    if mode == "fixed":
        return AntennaConfig(N1=1, N2=1, O1=1, O2=1, strict=False)  # P=2
    return AntennaConfig.standard(n1, n2, ng)


# ------------------------------------------------------------------ helpers


def _bool_param(key: str, label: str, default: bool, help_text: str, visible_if=None) -> ParamSpec:
    return ParamSpec(
        key=key, label=label, type="bool", default=default,
        choices=[], help=help_text, visible_if=visible_if,
    )


def _rank_stepper_choices(lo: int, hi: int) -> None:
    return None  # ranks are a shared Evaluation control, not a ParamSpec


def _r16_combo_choices(combos: dict, ps: bool = False) -> list[ParamChoice]:
    out = []
    for idx, c in sorted(combos.items()):
        rank34 = f", p(3,4)={_frac(c.p_v34)}" if c.p_v34 is not None else " (ranks 1-2 only)"
        label = f"{idx} — L={c.L}, p(1,2)={_frac(c.p_v12)}{rank34}, β={_frac(c.beta)}"
        out.append(ParamChoice(idx, label))
    return out


def _r17_combo_choices() -> list[ParamChoice]:
    out = []
    for idx, c in sorted(R17_PARAM_COMBOS.items()):
        label = f"{idx} — M={c.M}, α={_frac(c.alpha)}, β={_frac(c.beta)}"
        out.append(ParamChoice(idx, label))
    return out


def _r18_combo_choices() -> list[ParamChoice]:
    return _r16_combo_choices(R18_PARAM_COMBOS)


# ============================================================== 1. type1-2port


def _factory_type1_2port(antenna: AntennaConfig, n3: int, params: dict):
    return TwoPortType1Codebook(N3=n3)


ENTRY_TYPE1_2PORT = CatalogEntry(
    id="type1-2port",
    name="Type I Single-Panel (2-port)",
    short_name="Type I (2-port)",
    release="R15",
    spec_clause="TS 38.214 §5.2.2.2.1, Table 5.2.2.2.1-1",
    tagline="The fixed 6-precoder codebook for the smallest 2-antenna-port deployments.",
    ranks=(1, 2),
    port_range=(2, 2),
    position=1,
    lineage=None,
    doc_file="01-type1-single-panel.md",
    params=[],
    antenna=AntennaSpec(mode="fixed", pairs=[AntennaPair(1, 1, 2)]),
    factory=_factory_type1_2port,
)


# ================================================================= 2. type1-sp


def _factory_type1_sp(antenna: AntennaConfig, n3: int, params: dict):
    mode = int(params.get("codebook_mode", 1))
    return Type1Codebook(antenna, N3=n3, mode=mode)


ENTRY_TYPE1_SP = CatalogEntry(
    id="type1-sp",
    name="Type I Single-Panel",
    short_name="Type I",
    release="R15",
    spec_clause="TS 38.214 §5.2.2.2.1",
    tagline="The baseline DFT-beam codebook: one beam, one phase, per report.",
    ranks=(1, 8),
    port_range=(4, 32),
    position=2,
    lineage=None,
    doc_file="01-type1-single-panel.md",
    params=[
        ParamSpec(
            key="codebook_mode", label="Codebook mode", type="choice", default=1,
            choices=[
                ParamChoice(1, "Mode 1 — standard beam grid"),
                ParamChoice(2, "Mode 2 — finer beam grid, ranks 1-2 only"),
            ],
            help="Mode 2 halves the reported beam-grid granularity for ranks 1-2, "
                 "trading a slightly larger index for finer beam resolution.",
        ),
    ],
    antenna=AntennaSpec(mode="single", pairs=_STANDARD_PAIRS),
    factory=_factory_type1_sp,
)


# ================================================================= 3. type1-mp


def _factory_type1_mp(antenna: AntennaConfig, n3: int, params: dict):
    mode = int(params.get("codebook_mode", 1))
    return Type1MultiPanelCodebook(antenna, N3=n3, mode=mode)


ENTRY_TYPE1_MP = CatalogEntry(
    id="type1-mp",
    name="Type I Multi-Panel",
    short_name="Type I MP",
    release="R15",
    spec_clause="TS 38.214 §5.2.2.2.2",
    tagline="Type I extended to several co-located antenna panels with inter-panel phase.",
    ranks=(1, 4),
    port_range=(8, 32),
    position=3,
    lineage={"parent": "type1-sp", "adds": "multiple antenna panels with inter-panel co-phasing"},
    doc_file="02-type1-multi-panel.md",
    params=[
        ParamSpec(
            key="codebook_mode", label="Codebook mode", type="choice", default=1,
            choices=[
                ParamChoice(1, "Mode 1 — common per-subband phase across panels"),
                ParamChoice(2, "Mode 2 — richer per-subband phase, Ng=2 only"),
            ],
            help="Mode 2 reports a richer per-subband phase state but only exists "
                 "for 2-panel arrays.",
        ),
    ],
    antenna=AntennaSpec(mode="multi", pairs=_MP_PAIRS),
    factory=_factory_type1_mp,
)


# ================================================================ 4. type2-r15


def _factory_type2_r15(antenna: AntennaConfig, n3: int, params: dict):
    L = int(params.get("L", 4))
    subband_amplitude = bool(params.get("subband_amplitude", False))
    port_selection = bool(params.get("port_selection", False))
    d = int(params.get("d", 1))
    return R15Type2Codebook(
        antenna, N3=n3, L=L, subband_amplitude=subband_amplitude,
        port_selection=port_selection, d=d if port_selection else 1,
    )


ENTRY_TYPE2_R15 = CatalogEntry(
    id="type2-r15",
    name="Type II",
    short_name="Type II",
    release="R15",
    spec_clause="TS 38.214 §5.2.2.2.3–4",
    tagline="Linear-combination beamforming: a weighted sum of L beams instead of one.",
    ranks=(1, 2),
    port_range=(4, 32),
    position=4,
    lineage={"parent": "type1-sp", "adds": "linear combination of L DFT beams per polarization"},
    doc_file="03-type2-r15.md",
    params=[
        ParamSpec(
            key="L", label="Number of beams (L)", type="choice", default=4,
            choices=[ParamChoice(v, f"L = {v}") for v in (2, 3, 4)],
            help="How many DFT beams are linearly combined per polarization; more "
                 "beams track richer channels at higher feedback cost.",
        ),
        _bool_param(
            "subband_amplitude", "Subband amplitude", False,
            "Report per-subband amplitude refinements on top of the wideband "
            "amplitude, at extra bit cost.",
        ),
        _bool_param(
            "port_selection", "Port-selection variant", False,
            "Select beams directly on CSI-RS ports instead of DFT beams "
            "(cheaper UE search, assumes the gNB already beamformed the ports).",
        ),
        ParamSpec(
            key="d", label="Port sampling size (d)", type="choice", default=1,
            choices=[ParamChoice(v, str(v)) for v in (1, 2, 3, 4)],
            visible_if={"key": "port_selection", "value": True},
            help="Spacing between candidate starting ports for the port-selection "
                 "window search.",
        ),
    ],
    antenna=AntennaSpec(mode="single", pairs=_STANDARD_PAIRS),
    factory=_factory_type2_r15,
)


# =============================================================== 5. etype2-r16


def _factory_etype2_r16(antenna: AntennaConfig, n3: int, params: dict):
    port_selection = bool(params.get("port_selection", False))
    combo = int(params.get("param_combination", 4))
    R = int(params.get("R", 1))
    d = int(params.get("d", 1))
    return R16Type2Codebook(
        antenna, N3=n3, param_combination=combo, R=R,
        port_selection=port_selection, d=d if port_selection else 1,
    )


def _r16_combo_choices_dynamic(params: dict) -> list[ParamChoice]:
    ps = bool(params.get("port_selection", False))
    return _r16_combo_choices(R16_PS_PARAM_COMBOS if ps else R16_PARAM_COMBOS)


ENTRY_ETYPE2_R16 = CatalogEntry(
    id="etype2-r16",
    name="Enhanced Type II (eType II)",
    short_name="eType II",
    release="R16",
    spec_clause="TS 38.214 §5.2.2.2.5–6",
    tagline="Type II with a frequency (delay) domain compression layer on top of the beams.",
    ranks=(1, 4),
    port_range=(4, 32),
    position=6,
    lineage={"parent": "type2-r15", "adds": "delay-domain (frequency) compression"},
    doc_file="04-etype2-r16.md",
    params=[
        _bool_param(
            "port_selection", "Port-selection variant", False,
            "Select beams directly on CSI-RS ports instead of DFT beams.",
        ),
        ParamSpec(
            key="param_combination", label="Parameter combination", type="choice", default=4,
            choices=_r16_combo_choices(R16_PARAM_COMBOS),
            help="Standardized (L, tap density, amplitude budget) bundle; higher "
                 "indices generally cost more bits for more fidelity. Port-selection "
                 "restricts this to combinations 1-6 (Table 5.2.2.2.6-1).",
        ),
        ParamSpec(
            key="R", label="Subbands per CQI subband (R)", type="choice", default=1,
            choices=[ParamChoice(1, "R = 1"), ParamChoice(2, "R = 2")],
            help="Number of PMI frequency units reported per configured CQI subband.",
        ),
        ParamSpec(
            key="d", label="Port sampling size (d)", type="choice", default=1,
            choices=[ParamChoice(v, str(v)) for v in (1, 2, 3, 4)],
            visible_if={"key": "port_selection", "value": True},
            help="Spacing between candidate starting ports for the port-selection "
                 "window search.",
        ),
    ],
    antenna=AntennaSpec(mode="single", pairs=_STANDARD_PAIRS),
    factory=_factory_etype2_r16,
)


# ============================================================== 6. fetype2-r17


def _factory_fetype2_r17(antenna: AntennaConfig, n3: int, params: dict):
    combo = int(params.get("param_combination", 7))
    n_window = int(params.get("N_window", 4))
    return R17Type2Codebook(antenna, N3=n3, param_combination=combo, N_window=n_window)


ENTRY_FETYPE2_R17 = CatalogEntry(
    id="fetype2-r17",
    name="Further Enhanced Type II Port Selection (FeType II PS)",
    short_name="FeType II PS",
    release="R17",
    spec_clause="TS 38.214 §5.2.2.2.7",
    tagline="Port-selection eType II with free port choice and a compact delay-tap window.",
    ranks=(1, 4),
    port_range=(4, 32),
    position=7,
    lineage={"parent": "etype2-r16",
             "adds": "free port selection + windowed delay taps (always port-selection)"},
    doc_file="05-fetype2-r17.md",
    params=[
        ParamSpec(
            key="param_combination", label="Parameter combination", type="choice", default=7,
            choices=_r17_combo_choices(),
            help="Standardized (delay taps M, port fraction α, amplitude budget "
                 "β) bundle.",
        ),
        ParamSpec(
            key="N_window", label="Delay tap window (N)", type="choice", default=4,
            choices=[ParamChoice(2, "N = 2"), ParamChoice(4, "N = 4")],
            help="Width of the window the second delay tap is searched in "
                 "(only relevant when the parameter combination selects 2 taps).",
        ),
    ],
    antenna=AntennaSpec(mode="single", pairs=_STANDARD_PAIRS),
    factory=_factory_fetype2_r17,
)


# ============================================================ 7. predicted-ps-r18


def _factory_predicted_ps_r18(antenna: AntennaConfig, n3: int, params: dict):
    combo = int(params.get("param_combination", 7))
    n_window = int(params.get("N_window", 4))
    R = int(params.get("R", 1))
    return R18PredictedPortSelectionCodebook(
        antenna, N3=n3, param_combination=combo, N_window=n_window, R=R,
    )


ENTRY_PREDICTED_PS_R18 = CatalogEntry(
    id="predicted-ps-r18",
    name="Predicted Port-Selection PMI",
    short_name="Predicted PS",
    release="R18",
    spec_clause="TS 38.214 §5.2.2.2.11",
    tagline="FeType II PS reconstructed from a single predicted (N4=1) interval.",
    ranks=(1, 4),
    port_range=(4, 32),
    position=8,
    lineage={"parent": "fetype2-r17",
             "adds": "reconstruction aimed at a UE-side predicted channel"},
    doc_file="05-fetype2-r17.md",
    params=[
        ParamSpec(
            key="param_combination", label="Parameter combination", type="choice", default=7,
            choices=_r17_combo_choices(),
            help="Standardized (delay taps M, port fraction α, amplitude budget "
                 "β) bundle, reused from the R17 table.",
        ),
        ParamSpec(
            key="N_window", label="Delay tap window (N)", type="choice", default=4,
            choices=[ParamChoice(2, "N = 2"), ParamChoice(4, "N = 4")],
            help="Width of the window the second delay tap is searched in.",
        ),
        ParamSpec(
            key="R", label="Subbands per CQI subband (R)", type="choice", default=1,
            choices=[ParamChoice(1, "R = 1"), ParamChoice(2, "R = 2")],
            help="R = 2 is only configurable when the parameter combination selects "
                 "2 delay taps.",
        ),
    ],
    antenna=AntennaSpec(mode="single", pairs=_STANDARD_PAIRS),
    factory=_factory_predicted_ps_r18,
)


# ======================================================= 8. etype2-doppler-r18


def _factory_etype2_doppler_r18(antenna: AntennaConfig, n3: int, params: dict):
    combo = int(params.get("param_combination", 3))
    n4 = int(params.get("N4", 4))
    R = int(params.get("R", 1))
    return R18Type2Codebook(antenna, N3=n3, N4=n4, param_combination=combo, R=R)


ENTRY_ETYPE2_DOPPLER_R18 = CatalogEntry(
    id="etype2-doppler-r18",
    name="Enhanced Type II Doppler (predicted PMI)",
    short_name="eType II Doppler",
    release="R18",
    spec_clause="TS 38.214 §5.2.2.2.10",
    tagline="eType II plus a temporal (Doppler) basis: one report predicts N4 future slots.",
    ranks=(1, 4),
    port_range=(4, 32),
    position=9,
    lineage={"parent": "etype2-r16",
             "adds": "temporal (Doppler) domain compression across N4 slot intervals"},
    doc_file="06-etype2-doppler-r18.md",
    params=[
        ParamSpec(
            key="param_combination", label="Parameter combination-Doppler", type="choice",
            default=3,
            choices=_r18_combo_choices(),
            help="Standardized (L, tap density, amplitude budget) bundle for the "
                 "Doppler codebook.",
        ),
        ParamSpec(
            key="N4", label="Predicted intervals (N4)", type="choice", default=4,
            choices=[ParamChoice(v, f"N4 = {v}") for v in (1, 2, 4, 8)],
            help="Number of future slot intervals covered by one predicted report; "
                 "N4=1 degenerates exactly to R16 eType II.",
        ),
        ParamSpec(
            key="R", label="Subbands per CQI subband (R)", type="choice", default=1,
            choices=[ParamChoice(1, "R = 1"), ParamChoice(2, "R = 2")],
            help="Number of PMI frequency units reported per configured CQI subband.",
        ),
    ],
    antenna=AntennaSpec(mode="single", pairs=_STANDARD_PAIRS),
    factory=_factory_etype2_doppler_r18,
)


# ===================================================================== 9. cjt-r18


def _cjt_l_choices(n_trp: int) -> list[ParamChoice]:
    out = []
    for idx, combo in sorted(CJT_L_COMBOS.get(n_trp, {}).items()):
        out.append(ParamChoice(idx, f"{idx} — L = {list(combo)}"))
    return out


def _cjt_alpha_choices(n_trp: int) -> list[ParamChoice]:
    out = []
    for idx, combo in sorted(CJT_PS_ALPHA_COMBOS.get(n_trp, {}).items()):
        out.append(ParamChoice(idx, f"{idx} — α = {[_frac(a) for a in combo]}"))
    return out


def _factory_cjt_r18(antenna: AntennaConfig, n3: int, params: dict):
    n_trp = int(params.get("n_trp", 2))
    mode = str(params.get("mode", "mode1"))
    port_selection = bool(params.get("port_selection", False))
    restricted_cmr = bool(params.get("restricted_cmr", True))
    O3 = int(params.get("O3", 4))
    R = int(params.get("R", 1))
    if port_selection:
        pca = int(params.get("param_combination_alpha", 1))
        pc = int(params.get("param_combination", 1))
        return R18CJTPortSelectionCodebook(
            antenna, N3=n3, n_trp=n_trp, param_combination_alpha=pca,
            param_combination=pc, mode=mode, O3=O3, R=R,
            restricted_cmr=restricted_cmr,
        )
    pcl = int(params.get("param_combination_L", 1))
    pc = int(params.get("param_combination", 1))
    return R18CJTCodebook(
        antenna, N3=n3, n_trp=n_trp, param_combination_L=pcl,
        param_combination=pc, mode=mode, O3=O3, R=R,
        restricted_cmr=restricted_cmr,
    )


ENTRY_CJT_R18 = CatalogEntry(
    id="cjt-r18",
    name="Coherent Joint Transmission (CJT) eType II",
    short_name="CJT eType II",
    release="R18",
    spec_clause="TS 38.214 §5.2.2.2.8–9",
    tagline="One eType II block per transmission point, jointly precoded and co-phased.",
    ranks=(1, 4),
    port_range=(4, 32),
    position=10,
    lineage={"parent": "etype2-r16",
             "adds": "multiple transmission points (TRPs) jointly precoded"},
    doc_file="08-cjt.md",
    params=[
        ParamSpec(
            key="n_trp", label="Transmission points (N_TRP)", type="choice", default=2,
            choices=[ParamChoice(v, str(v)) for v in (1, 2, 3, 4)],
            help="Number of coherently-combined CSI-RS resources (one per TRP).",
        ),
        _bool_param(
            "port_selection", "Port-selection variant", False,
            "Use free per-TRP port selection (feType II style) instead of DFT beams "
            "per TRP.",
        ),
        ParamSpec(
            key="param_combination_L", label="L combination (paramCombination-CJT-L)",
            type="choice", default=1, choices=_cjt_l_choices(2),
            visible_if={"key": "port_selection", "value": False},
            help="Per-TRP beam count combination; choices depend on N_TRP.",
        ),
        ParamSpec(
            key="param_combination_alpha", label="α combination (paramCombination-CJT-PS-alpha)",
            type="choice", default=1, choices=_cjt_alpha_choices(2),
            visible_if={"key": "port_selection", "value": True},
            help="Per-TRP port-fraction combination; choices depend on N_TRP.",
        ),
        ParamSpec(
            key="param_combination", label="Coefficient combination", type="choice", default=1,
            choices=[ParamChoice(v, str(v)) for v in range(1, 8)],
            help="Selects the (tap density, amplitude budget) row; the allowed "
                 "values depend on N_TRP and the L/α combination above "
                 "(Table 5.2.2.2.8-3 / 5.2.2.2.9-3).",
        ),
        ParamSpec(
            key="mode", label="Codebook mode", type="choice", default="mode1",
            choices=[
                ParamChoice("mode1", "mode1 — reports an inter-TRP delay ramp"),
                ParamChoice("mode2", "mode2 — no inter-TRP delay ramp"),
            ],
            help="mode1 pre-compensates inter-TRP propagation-delay offsets; "
                 "mode2 assumes they are already aligned.",
        ),
        ParamSpec(
            key="O3", label="Delay ramp oversampling (numberOfO3)", type="choice", default=4,
            choices=[ParamChoice(1, "O3 = 1"), ParamChoice(4, "O3 = 4")],
            visible_if={"key": "mode", "value": "mode1"},
            help="Oversampling factor of the reported inter-TRP delay ramp.",
        ),
        ParamSpec(
            key="R", label="Subbands per CQI subband (R)", type="choice", default=1,
            choices=[ParamChoice(1, "R = 1"), ParamChoice(2, "R = 2")],
            help="Number of PMI frequency units reported per configured CQI subband.",
        ),
        _bool_param(
            "restricted_cmr", "Use all TRP resources (restrictedCMR-Selection)", True,
            "When off, the UE may select a subset of the configured TRPs and "
            "reports which ones it used.",
        ),
    ],
    antenna=AntennaSpec(mode="single", pairs=_STANDARD_PAIRS),
    factory=_factory_cjt_r18,
)


# ============================================================ 10. refined-type1-r19


def _factory_refined_type1_r19(antenna: AntennaConfig, n3: int, params: dict):
    mode = str(params.get("mode", "modeA"))
    return RefinedType1SinglePanelCodebook(antenna, N3=n3, mode=mode)


ENTRY_REFINED_TYPE1_R19 = CatalogEntry(
    id="refined-type1-r19",
    name="Refined Type I Single-Panel",
    short_name="Refined Type I",
    release="R19",
    spec_clause="TS 38.214 §5.2.2.2.1a",
    tagline="Type I extended to 48/64/128-port arrays with independently-selected companion beams.",
    ranks=(1, 8),
    port_range=(48, 128),
    position=11,
    lineage={"parent": "type1-sp",
             "adds": "large-array (48-128 port) beam selection, two report modes"},
    doc_file="07-refined-r19.md",
    params=[
        ParamSpec(
            key="mode", label="Codebook mode", type="choice", default="modeA",
            choices=[
                ParamChoice("modeA", "modeA — R15-style shared companion-beam pattern"),
                ParamChoice("modeB", "modeB — independent per-layer beams, more bits"),
            ],
            help="modeB lets every layer (or beam group) pick its own beam, at the "
                 "cost of more reported indices.",
        ),
    ],
    antenna=AntennaSpec(mode="single", pairs=_R19_PAIRS),
    factory=_factory_refined_type1_r19,
)


# ========================================================= 11. refined-type1mp-r19


def _factory_refined_type1mp_r19(antenna: AntennaConfig, n3: int, params: dict):
    return RefinedType1MultiPanelCodebook(antenna, N3=n3)


ENTRY_REFINED_TYPE1MP_R19 = CatalogEntry(
    id="refined-type1mp-r19",
    name="Refined Type I Multi-Panel",
    short_name="Refined Type I MP",
    release="R19",
    spec_clause="TS 38.214 §5.2.2.2.2a",
    tagline="Type I Multi-Panel extended to large aggregated arrays, one beam per panel.",
    ranks=(1, 4),
    port_range=(48, 128),
    position=12,
    lineage={"parent": "type1-mp",
             "adds": "large aggregated arrays (48-128 ports) with per-panel beam selection"},
    doc_file="07-refined-r19.md",
    params=[],
    antenna=AntennaSpec(mode="multi", pairs=_MP_R19_PAIRS),
    factory=_factory_refined_type1mp_r19,
)


# =========================================================== 12. refined-type2-r19


def _factory_refined_type2_r19(antenna: AntennaConfig, n3: int, params: dict):
    variant = str(params.get("variant", "regular"))
    if variant == "ps":
        combo = int(params.get("param_combination", 7))
        n_window = int(params.get("N_window", 4))
        return RefinedFeType2PortSelectionCodebook(
            antenna, N3=n3, param_combination=combo, N_window=n_window,
        )
    if variant == "predicted":
        combo = int(params.get("param_combination", 3))
        n4 = int(params.get("N4", 4))
        R = int(params.get("R", 1))
        return RefinedPredictedEType2Codebook(
            antenna, N3=n3, N4=n4, param_combination=combo, R=R,
        )
    combo = int(params.get("param_combination", 4))
    R = int(params.get("R", 1))
    return RefinedEType2Codebook(antenna, N3=n3, param_combination=combo, R=R)


ENTRY_REFINED_TYPE2_R19 = CatalogEntry(
    id="refined-type2-r19",
    name="Refined Type II",
    short_name="Refined Type II",
    release="R19",
    spec_clause="TS 38.214 §5.2.2.2.5a, §5.2.2.2.9a, §5.2.2.2.11a",
    tagline="eType II / FeType II PS / Doppler eType II reused verbatim on 48–128 port arrays.",
    ranks=(1, 4),
    port_range=(48, 128),
    position=13,
    lineage={"parent": "etype2-r16",
             "adds": "large-array (48-128 port) reuse of the R16/R17/R18 reconstructions"},
    doc_file="07-refined-r19.md",
    params=[
        ParamSpec(
            key="variant", label="Variant", type="choice", default="regular",
            choices=[
                ParamChoice("regular", "Regular — R16 eType II reconstruction (48/64/128 ports)"),
                ParamChoice("ps", "Port-selection — R17 FeType II PS reconstruction (48/64 ports)"),
                ParamChoice("predicted",
                            "Predicted — R18 Doppler reconstruction (48/64/128 ports)"),
            ],
            help="Which R16/R17/R18 reconstruction the refined codebook reuses on "
                 "the large array.",
        ),
        ParamSpec(
            key="param_combination", label="Parameter combination", type="choice", default=4,
            choices=_r16_combo_choices(R16_PARAM_COMBOS),
            visible_if={"key": "variant", "value": "regular"},
            help="Standardized (L, tap density, amplitude budget) bundle, as R16 eType II.",
        ),
        ParamSpec(
            key="R", label="Subbands per CQI subband (R)", type="choice", default=1,
            choices=[ParamChoice(1, "R = 1"), ParamChoice(2, "R = 2")],
            visible_if={"key": "variant", "value": "regular"},
            help="Number of PMI frequency units reported per configured CQI subband.",
        ),
        ParamSpec(
            key="N4", label="Predicted intervals (N4)", type="choice", default=4,
            choices=[ParamChoice(v, f"N4 = {v}") for v in (1, 2, 4, 8)],
            visible_if={"key": "variant", "value": "predicted"},
            help="Number of future slot intervals covered by one predicted report.",
        ),
        ParamSpec(
            key="N_window", label="Delay tap window (N)", type="choice", default=4,
            choices=[ParamChoice(2, "N = 2"), ParamChoice(4, "N = 4")],
            visible_if={"key": "variant", "value": "ps"},
            help="Width of the window the second delay tap is searched in.",
        ),
    ],
    antenna=AntennaSpec(mode="single", pairs=_R19_PAIRS),
    factory=_factory_refined_type2_r19,
)


# NOTE: the "predicted" variant of refined-type2-r19 reuses the same
# param_combination key as "regular" (both are choice[int]); the frontend
# should key on `variant` to decide which choices/help text is current. Both
# use the R16/R18 combo tables (1..9); since ParamSpec cannot depend on two
# keys at once, `param_combination` for the predicted variant is intentionally
# left to accept the same numeric domain as etype2-doppler-r18 (validated by
# the constructor itself, which raises a friendly ValueError on a bad index).


CATALOG: list[CatalogEntry] = [
    ENTRY_TYPE1_2PORT,
    ENTRY_TYPE1_SP,
    ENTRY_TYPE1_MP,
    ENTRY_TYPE2_R15,
    ENTRY_ETYPE2_R16,
    ENTRY_FETYPE2_R17,
    ENTRY_PREDICTED_PS_R18,
    ENTRY_ETYPE2_DOPPLER_R18,
    ENTRY_CJT_R18,
    ENTRY_REFINED_TYPE1_R19,
    ENTRY_REFINED_TYPE1MP_R19,
    ENTRY_REFINED_TYPE2_R19,
]

CATALOG_BY_ID: dict[str, CatalogEntry] = {e.id: e for e in CATALOG}


def get_entry(codebook_id: str) -> CatalogEntry | None:
    return CATALOG_BY_ID.get(codebook_id)


def build_antenna(entry: CatalogEntry, req: dict) -> AntennaConfig:
    """Resolve the RunRequest ``antenna`` dict into a concrete ``AntennaConfig``
    for this catalog entry (honoring "fixed"-mode 2-port entries)."""
    return _antenna_from_request(entry.antenna.mode, req or {})
