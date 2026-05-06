"""
LoadLimiter Energiemodel GEAVANCEERD — 15-min resolutie
=======================================================
Verbeteringen t.o.v. app.py (deterministisch, uurmodel):

  1. Tijdresolutie: 15 min (96 stappen/dag)
     → Kortere pieken en aanloopstromen beter zichtbaar
  2. Thermisch huismodel (RC-model, 1e orde)
     → WP-onderbreking leidt tot meetbare T_in-daling (warmteschuld zichtbaar)
     → Na herstart draait WP harder/langer om temperatuur te herstellen
     → Parameters: C_th (thermische massa) en UA (warmteverlies) instelbaar
  3. Monte Carlo
     → N random dagprofielen → P10/P50/P90 bandgrafiek per tijdstap
     → Stochastische variabelen: buiten-T (±2°C), kookmoment (±2u),
       douche-intensiteit (50–150%), was/droger (kans 50%/40%),
       EV-aankomst-SOC (variabel)
     → Statistieken: % dagen boven 35A, P90 piek, % EV vol bij vertrek
  4. Seizoensprofielen
     → Winter / Lente-Herfst / Zomer — multiplier op zonopbrengst; WP-last via temperatuurmodel

Origineel deterministisch uurmodel: zie app.py (ongewijzigd bewaard).

Gebruik:
    pip install streamlit plotly numpy
    streamlit run app_advanced.py
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="LoadLimiter — Geavanceerd Model",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════
# CONSTANTEN
# ═══════════════════════════════════════════════════════════════════
STEPS       = 96                   # tijdstappen per dag (elke 15 min)
DT          = 0.25                 # uur per stap
STEP_LABELS = [f"{s//4:02d}:{(s%4)*15:02d}" for s in range(STEPS)]

BASE_T_IN, BASE_T_OUT, BASE_COP = 20.0, 8.0, 3.5
L35       = 8.05                   # kW bij 35A × 230V
EV_MIN_KW = 6 * 0.23              # IEC 61851 minimumlaadstroom (6A × 230V)

# Nacht-stappen: 00:00–05:45 (stap 0–23) en 22:00–23:45 (stap 88–95)
NIGHT_STEPS = set(range(0, 24)) | set(range(88, 96))

# EV laadvolgorde: aankomst 18:00 (stap 72) → nacht → vertrek 07:00 (stap 27)
EV_CHARGE_STEPS = list(range(72, 96)) + list(range(0, 28))

# Thermisch model — standaardwaarden
#
# UA = warmtedoorgangscoëfficiënt (W/°C of kW/°C):
#   Vóór renovatie, typisch NL tussenwoning ~300 W/°C
#   Na volledige renovatie (isolatie + HR-glas) ~80 W/°C
#   Kalibratie: bij UA=0.30 en ΔT=12°C is steady-state WP-elektrisch ≈1.0 kW,
#   wat overeenkomt met de basisprofielen in app.py (gekalibreerd op 20°C/8°C).
#
# C_th = effectieve thermische massa op uur-tijdschaal (kWh/°C):
#   Hoge waarde = traag reagerend gebouw, kleine T-daling bij WP-stop
#   Lage waarde = snel reagerend, grote T-daling (nuttig voor demonstratie warmteschuld)
#   Bij C_th=5 en UA=0.30: T-daling bij 1u WP-stop ≈ 0.72°C — zichtbaar en realistisch
#
C_TH_DEFAULT    = 5.0   # kWh/°C  (matig zware constructie)
UA_DEFAULT      = 0.30  # kW/°C   (= 300 W/°C, vóór renovatie — past bij de basisprofielen)
TAU_RECOVERY    = 4.0   # uur — tijdconstante temperatuurherstel na setpuntsprong
                        # τ=4u: bij ΔT=3°C en C_th=5 → q_extra=3.75 kW th → realistisch

# ═══════════════════════════════════════════════════════════════════
# SEIZOENSPROFIELEN
# ═══════════════════════════════════════════════════════════════════
SEASONS = {
    "❄️  Winter":       {"solar_scale": 0.15, "t_out_default": -2,
                         "desc": "Weinig zon · hoge WP-last · typisch januari"},
    "🌿  Lente/Herfst": {"solar_scale": 1.00, "t_out_default":  8,
                         "desc": "Referentieprofiel — gematigde voor- of najaarscondities"},
    "☀️  Zomer":        {"solar_scale": 1.60, "t_out_default": 18,
                         "desc": "Veel zon · lage WP-last · EV laadt grotendeels op zon"},
}

# ═══════════════════════════════════════════════════════════════════
# BASISPROFIELEN (uurwaarden; worden opgeschaald naar 15 min)
# Identiek aan app.py — niet aanpassen
# ═══════════════════════════════════════════════════════════════════
PROFILES = {
    "Senioren": {
        "wp":  [.8,.8,.8,.8,.8,.8,1.4,1.8,1.5,1.0,.7,.5,.4,.4,.5,.6,.8,1.2,1.4,1.3,1.1,1.0,.9,.8],
        "ww":  [0,0,0,0,0,0,0,.5,.4,0,0,0,0,0,0,0,0,0,.3,.3,0,0,0,0],
        "ind": [0,0,0,0,0,0,0,0,0,0,0,0,1.8,0,0,0,0,1.5,2.2,.5,0,0,0,0],
        "was": [0,0,0,0,0,0,0,0,0,2.0,1.5,0,0,0,0,0,0,0,0,0,0,0,0,0],
        "dro": [0]*24,
        "ov":  [0,0,0,0,0,0,0,.1,.1,.1,.1,.1,.1,.1,.1,.1,.15,.2,.2,.25,.25,.2,.15,0],
        "bas": 0.35,
        "t_dag_default": 23, "t_nacht_default": 19,
        "omschrijving": "Thuis overdag, continue warmtebehoefte, hoog comfort-setpoint, 1 douche 07:00.",
    },
    "Jonge kinderen": {
        "wp":  [.8,.8,.8,.8,.8,.8,2.0,2.2,1.8,.8,.6,.5,.4,.4,.5,1.0,1.2,1.5,1.8,1.6,1.3,1.0,.9,.8],
        "ww":  [0,0,0,0,0,0,0,1.0,.8,0,0,0,0,0,0,0,0,0,.8,.7,0,0,0,0],
        "ind": [0,0,0,0,0,0,0,.8,0,0,0,0,1.0,0,0,.5,0,1.0,3.0,.5,0,0,0,0],
        "was": [0,0,0,0,0,0,0,2.2,2.0,0,0,0,0,0,2.2,2.0,0,0,0,0,0,0,0,0],
        "dro": [0,0,0,0,0,0,0,0,0,2.2,2.0,0,0,0,0,0,2.2,2.0,0,0,0,0,0,0],
        "ov":  [0,0,0,0,0,0,0,.5,.3,.2,.1,.1,.2,.1,.1,.3,.3,.2,.2,.3,.2,.15,.1,0],
        "bas": 0.4,
        "t_dag_default": 20, "t_nacht_default": 16,
        "omschrijving": "Setback 09–15u. Ochtendpiek 06:30–08:30, badtijd 19–20u.",
    },
    "Tieners": {
        "wp":  [.9,.9,.9,.9,.9,.9,1.5,2.2,2.2,1.0,.6,.5,.5,.5,.6,1.2,1.5,1.8,2.0,1.8,1.5,1.2,1.0,.9],
        "ww":  [0,0,0,0,0,.5,1.0,2.0,1.5,.5,0,0,0,0,0,0,0,0,.5,.3,0,0,0,0],
        "ind": [0,0,0,0,0,0,0,.5,0,0,0,0,.5,0,0,.5,0,.5,3.5,.5,0,1.5,.5,0],
        "was": [0,0,0,0,0,0,0,0,0,2.2,2.0,0,0,0,2.2,2.0,0,0,0,0,0,0,0,0],
        "dro": [0,0,0,0,0,0,0,0,0,0,2.2,2.0,0,0,0,2.2,2.0,0,0,0,0,0,0,0],
        "ov":  [0,0,0,0,0,0,0,.8,.8,.3,.2,.2,.3,.3,.3,.6,.6,.6,.7,.7,.7,.7,.5,.2],
        "bas": 0.5,
        "t_dag_default": 20, "t_nacht_default": 17,
        "omschrijving": "Setback 08–15u. 5 douches 06–08u (backup boilerelement actief). Gaming/TV structureel 15–23u.",
    },
}

SOLAR_PER_KWP = [0,0,0,0,0,0,.033,.133,.267,.433,.633,.8,.9,.933,.867,.733,.533,.333,.167,.067,.017,0,0,0]

# ═══════════════════════════════════════════════════════════════════
# HULPFUNCTIES
# ═══════════════════════════════════════════════════════════════════
def cop(t_out: float) -> float:
    """Lineaire COP-benadering: daalt ~0,08 per graad koeler. Min. 1,5."""
    return max(1.5, BASE_COP + 0.08 * (t_out - BASE_T_OUT))


def wp_base_factor(t_in: float, t_out: float) -> float:
    """Schaalfactor WP-elektrisch t.o.v. basismodel (20°C binnen, 8°C buiten)."""
    base   = (BASE_T_IN - BASE_T_OUT) / BASE_COP
    actual = max(0.0, t_in - t_out) / cop(t_out)
    return actual / base if base > 0 else 0.0


def upscale(hourly: list) -> np.ndarray:
    """Zet 24 uurwaarden om naar 96 kwartierwaarden (elke waarde 4× herhalen)."""
    return np.repeat(hourly, 4).astype(float)


def shift_arr(arr: np.ndarray, steps: int) -> np.ndarray:
    """Verschuif array circulair met `steps` kwartier-stappen."""
    return np.roll(arr, steps)


# ═══════════════════════════════════════════════════════════════════
# KERN-SIMULATIE — 15 MINUTEN + THERMISCH MODEL
# ═══════════════════════════════════════════════════════════════════
def simulate_15min(
    prof, t_dag, t_nacht, t_out, ckw,
    ev_on, ev_max_kw, ev_ll_on, ev_ll_a, ev_needed_kwh,
    sol_on, sol_kwp, season_solar_scale,
    ll_on, trigger_a, resume_a,
    c_th, ua, wp_elec_max_kw=3.5,
    # Monte Carlo overrides
    cook_shift_steps=0,
    ww_scale=1.0,
    was_active=True,
    dro_active=True,
    ev_needed_kwh_override=None,
    t_out_offset=0.0,
):
    """
    Simuleert één dag op 15-minuten resolutie met thermisch huismodel.

    Thermisch model (RC, 1e orde):
      q_loss[t]    = UA × (T_in[t] − T_out)                        [kW warmteverlies]
      q_need[t]    = UA × max(0, T_set[t] − T_out)                 [kW steady-state vraag]
      q_extra[t]   = max(0, T_set[t] − T_in[t]) × C_th / TAU_RECOVERY  [kW herstel]
      wp_want[t]   = 0  als T_in[t] ≥ T_set[t]  (WP niet nodig)
                   = min(wp_elec_max, (q_need + q_extra) / COP)  anders
      T_in[t+1]    = T_in[t] + (wp_want[t] × COP − q_loss[t]) × DT / C_th

    Wanneer de LoadLimiter ingrijpt (suspended=True): wp_actual[t]=0,
    de woning koelt af, en de gemiste warmte accumuleert als warmteschuld (kWh thermisch).

    Monte Carlo parameters (optioneel):
      cook_shift_steps  : verschuif kookprofiel circulair (stappen van 15 min)
      ww_scale          : schaalfactor warm-water profiel (douche-variatie)
      was_active        : wasmachine actief vandaag (bool)
      dro_active        : droger actief vandaag (bool)
      ev_needed_kwh_override : overschrijf EV-laadvraag
      t_out_offset      : offset op buitentemperatuur (°C)
    """
    t_out_eff  = t_out + t_out_offset
    trigger_kw = trigger_a * 0.23
    resume_kw  = resume_a  * 0.23
    ev_ll_kw   = ev_ll_a   * 0.23
    ev_kwh     = ev_needed_kwh_override if ev_needed_kwh_override is not None else ev_needed_kwh

    # ── Basisprofielen → 96 kwartier-stappen ────────────────────────
    # wp_base_factor schaalt het WP-profiel op basis van temperatuurfysica:
    #   hogere ΔT (binnen−buiten) → meer warmtevraag → hogere elektrische opname.
    # De seizoensinvloed loopt volledig via t_out (ingesteld per seizoen) + COP-model.
    wp_base_dag   = upscale(prof["wp"]) * wp_base_factor(t_dag,   t_out_eff)
    wp_base_nacht = upscale(prof["wp"]) * wp_base_factor(t_nacht, t_out_eff)
    # WP-referentieprofiel per stap (dag/nacht gescheiden)
    wp_ref = np.where([s in NIGHT_STEPS for s in range(STEPS)], wp_base_nacht, wp_base_dag)
    # Maximaal elektrisch WP-vermogen: gebaseerd op WP-spec, niet op profielpiek × factor.
    # wp_ref.max() × 1.4 gaf bij -8°C onrealistische waarden van 12+ kW.
    # wp_elec_max_kw is instelbaar via sidebar (default 3.5 kW voor 8 kW WP).
    wp_max_kw = wp_elec_max_kw

    ww  = upscale(prof["ww"]) * ww_scale
    ind_base = [min(prof["ind"][h] * ckw / 7.4, ckw) for h in range(24)]
    ind = shift_arr(upscale(ind_base), cook_shift_steps)
    was = upscale(prof["was"]) if was_active else np.zeros(STEPS)
    dro = upscale(prof["dro"]) if dro_active else np.zeros(STEPS)
    ov  = upscale([prof["ov"][h] + prof["bas"] for h in range(24)])
    sol = upscale([SOLAR_PER_KWP[h] * sol_kwp * season_solar_scale
                   if sol_on else 0.0 for h in range(24)])

    # ── EV cumulatief laden ──────────────────────────────────────────
    # Laadvolgorde: 18:00–23:45 (avond), dan 00:00–06:45 (nacht)
    # Stopt zodra ev_kwh bereikt of laadvenster voorbij (07:00)
    ev = np.zeros(STEPS)
    ev_charged = 0.0
    if ev_on:
        for s in EV_CHARGE_STEPS:
            if ev_charged >= ev_kwh:
                break
            remaining = ev_kwh - ev_charged          # kWh nog te laden
            if ev_ll_on:
                other    = wp_ref[s] + ww[s] + ind[s] + was[s] + dro[s] + ov[s]
                headroom = ev_ll_kw - (other - sol[s])
                ev_h     = max(0.0, min(ev_max_kw, headroom))
                ev_h     = ev_h if ev_h >= EV_MIN_KW else 0.0
            else:
                ev_h = ev_max_kw
            ev_h        = min(ev_h, remaining / DT)  # kW-cap zodat we kWh-doel niet overschrijden
            ev[s]       = ev_h                       # kW (vermogen dit kwartier)
            ev_charged += ev_h * DT                  # kWh (energie = vermogen × 0.25 u)

    # ── Thermisch model + WP LoadLimiter ────────────────────────────
    T_in = np.zeros(STEPS + 1)
    # Startconditie: huis begint op nachtsetpoint (00:00 = begin nachtperiode).
    # t_dag als beginwaarde was onrealistisch — huis is 's avonds al naar nacht-instelling gegaan.
    T_in[0] = t_nacht
    wp_actual    = np.zeros(STEPS)
    wp_want_arr  = np.zeros(STEPS)   # wat het thermisch model wilde (zonder LL)
    ll_active    = np.zeros(STEPS, dtype=bool)  # True = LL heeft WP geblokkeerd
    warmteschuld = 0.0       # kWh WARMTE die WP verschuldigd is door LL (niet profiel-diff)
    suspended    = False
    # Thermische energiebalans (kW warmte, niet kW elektrisch)
    q_delivered_arr = np.zeros(STEPS)   # warmte die WP daadwerkelijk levert
    q_want_th_arr   = np.zeros(STEPS)   # warmte die WP zou willen leveren (zonder LL)
    q_loss_arr      = np.zeros(STEPS)   # warmteverlies woning naar buiten

    for s in range(STEPS):
        t_set  = t_nacht if s in NIGHT_STEPS else t_dag

        # Warmteverlies van de woning naar buiten
        q_loss = ua * (T_in[s] - t_out_eff)            # kW (positief = verlies)
        q_loss_arr[s] = q_loss

        # WP-vermogen berekenen (thermostaat-logica):
        # - Als T_in >= t_set: WP staat uit. Huis is al warm genoeg; koelt vanzelf af naar setpoint.
        # - Als T_in < t_set: WP draait om verlies te compenseren én temperatuur te herstellen.
        #
        if T_in[s] >= t_set:
            # Huis is op of boven setpoint: WP niet nodig
            wp_want = 0.0
        else:
            q_heat_need  = ua * max(0.0, t_set - t_out_eff)        # steady-state warmtevraag
            q_temp_error = (t_set - T_in[s]) * c_th / TAU_RECOVERY  # temperatuurherstel
            wp_want      = min(wp_max_kw, (q_heat_need + q_temp_error) / cop(t_out_eff))
        wp_want_arr[s]  = wp_want
        q_want_th_arr[s] = wp_want * cop(t_out_eff)    # kW warmte gewenst (zonder LL)

        # ── LoadLimiter logica ────────────────────────────────────
        other_load = ww[s] + ind[s] + was[s] + dro[s] + ov[s] + ev[s] - sol[s]
        if ll_on:
            if suspended and other_load < resume_kw:
                suspended = False
            if not suspended and (wp_want + other_load) > trigger_kw:
                suspended = True

        if ll_on and suspended:
            wp_s = 0.0
            ll_active[s] = True
            # Warmteschuld = warmte die de WP had geleverd als hij NIET geblokkeerd was
            warmteschuld += wp_want * cop(t_out_eff) * DT
        else:
            wp_s = wp_want

        wp_actual[s] = wp_s
        q_delivered_arr[s] = wp_s * cop(t_out_eff)    # kW warmte daadwerkelijk geleverd

        # Thermische integratie (Euler voorwaarts, stap DT = 0.25u)
        T_in[s + 1] = T_in[s] + (wp_s * cop(t_out_eff) - q_loss) * DT / c_th

    totals = wp_actual + ww + ind + was + dro + ov + ev - sol
    # wp_cut = alleen de LL-geblokkeerde WP-stappen (niet het verschil met profiel)
    wp_cut = np.where(ll_active, wp_want_arr, 0.0)

    return dict(
        wp=wp_actual, wp_ref=wp_ref, wp_cut=wp_cut, wp_want=wp_want_arr,
        ww=ww, ind=ind, was=was, dro=dro, ov=ov, ev=ev, sol=sol,
        totals=totals,
        T_in=T_in[:STEPS],
        warmteschuld=warmteschuld,    # kWh warmte (thermisch), niet kWh elektrisch
        ev_charged=ev_charged if ev_on else 0.0,
        ll_active=ll_active,
        # Thermische energiebalans (kW warmte per tijdstap)
        q_delivered=q_delivered_arr,  # warmte die WP daadwerkelijk levert [kW th]
        q_want_th=q_want_th_arr,       # warmte die WP zou willen leveren (zonder LL) [kW th]
        q_loss=q_loss_arr,             # warmteverlies woning naar buiten [kW th]
    )


# ═══════════════════════════════════════════════════════════════════
# MONTE CARLO
# ═══════════════════════════════════════════════════════════════════
def run_monte_carlo(n_runs: int, base_kwargs: dict, rng_seed: int = 42) -> dict:
    """
    Voert n_runs simulaties uit met random perturbaties op de basisparameters.

    Stochastische variabelen per run:
      t_out_offset          ~ Normaal(μ=0, σ=2°C)
          Dagelijkse weersvariatie rondom het ingestelde gemiddelde.
      cook_shift_steps      ~ Uniform_int(−8, +8) stappen = ±2 uur
          Kookmoment varieert: vroeg eten vs. laat eten.
      ww_scale              ~ Uniform(0.5, 1.5)
          Douche-intensiteit: weinig douches (0.5×) tot extra lang/veel (1.5×).
      was_active            ~ Bernoulli(p=0.50)
          Wasdag of niet — gemiddeld 3–4 wasbeurten per week.
      dro_active            ~ Bernoulli(p=0.40)
          Droger aan of niet — iets minder frequent dan wasmachine.
      ev_needed_kwh_override ~ Uniform(10, min(80, ev_needed × 1.3))
          Auto komt niet altijd leeg thuis — variabele aankomst-SOC.

    Retourneert dict met:
      p10/p50/p90_totals    : percentielband netafname per tijdstap [kW]
      p10/p50/p90_t_in      : percentielband binnentemperatuur per tijdstap [°C]
      ev_charged            : array van geladen kWh per run
      peaks                 : array van dagelijkse piekbelasting per run [kW]
      pct_over35            : percentage runs met piek > 35A
    """
    rng = np.random.default_rng(rng_seed)
    all_totals     = np.zeros((n_runs, STEPS))
    all_t_in       = np.zeros((n_runs, STEPS))
    all_ev_charged = np.zeros(n_runs)
    all_peaks      = np.zeros(n_runs)

    for i in range(n_runs):
        kwargs = dict(base_kwargs)
        kwargs["t_out_offset"]       = float(rng.normal(0, 2))
        kwargs["cook_shift_steps"]   = int(rng.integers(-8, 9))
        kwargs["ww_scale"]           = float(rng.uniform(0.5, 1.5))
        kwargs["was_active"]         = bool(rng.random() < 0.5)
        kwargs["dro_active"]         = bool(rng.random() < 0.4)
        if kwargs.get("ev_on"):
            ev_max = kwargs.get("ev_needed_kwh", 50)
            kwargs["ev_needed_kwh_override"] = float(rng.uniform(10, min(80, ev_max * 1.3)))

        result = simulate_15min(**kwargs)
        all_totals[i]     = result["totals"]
        all_t_in[i]       = result["T_in"]
        all_ev_charged[i] = result["ev_charged"]
        all_peaks[i]      = float(result["totals"].max())

    return dict(
        p10_totals = np.percentile(all_totals, 10, axis=0),
        p50_totals = np.percentile(all_totals, 50, axis=0),
        p90_totals = np.percentile(all_totals, 90, axis=0),
        p10_t_in   = np.percentile(all_t_in,   10, axis=0),
        p50_t_in   = np.percentile(all_t_in,   50, axis=0),
        p90_t_in   = np.percentile(all_t_in,   90, axis=0),
        ev_charged = all_ev_charged,
        peaks      = all_peaks,
        pct_over35 = 100.0 * float(np.mean(all_peaks > L35)),
    )


# ═══════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("Huishoudtype")
    hh   = st.selectbox("Type gezin", list(PROFILES.keys()), label_visibility="collapsed")
    prof = PROFILES[hh]
    st.caption(prof["omschrijving"])

    st.divider()
    st.header("Seizoen")
    season_name = st.selectbox("Seizoen", list(SEASONS.keys()), index=1,
                               label_visibility="collapsed")
    season = SEASONS[season_name]
    st.caption(season["desc"])

    st.divider()
    st.header("Temperatuur")
    t_dag   = st.slider("Setpunt dag (°C)",   16, 26, prof["t_dag_default"])
    t_nacht = st.slider("Setpunt nacht (°C)", 13, 22, prof["t_nacht_default"])
    t_out   = st.slider("Buiten (°C)", -12, 20, season["t_out_default"])
    st.caption(f"COP ≈ {cop(t_out):.2f}  ·  WP-factor t.o.v. 20°C/8°C referentie: {wp_base_factor(t_dag, t_out):.2f}×")

    st.divider()
    st.header("Thermisch huismodel")
    c_th = st.slider("Thermische massa C_th (kWh/°C)", 0.5, 10.0,
                     C_TH_DEFAULT, step=0.5,
                     help="Effectieve warmtecapaciteit op uur-tijdschaal. "
                          "Laag (1–2) = snel reagerend, grote T-daling bij WP-stop. "
                          "Hoog (5–10) = traag, zware constructie (beton, steen).")
    ua_wk = st.slider("Warmtedoorgang UA (W/°C)", 50, 400,
                      int(UA_DEFAULT * 1000), step=10,
                      help="Warmteverlies per °C temperatuurverschil. "
                           "50–100 W/°C = goed gerenoveerd. "
                           "200–400 W/°C = vóór renovatie. "
                           "300 W/°C past bij de gecalibreerde basisprofielen.")
    ua = ua_wk / 1000.0
    wp_elec_max_kw = st.slider(
        "Max. WP elektrisch vermogen (kW)", 1.0, 6.0, 3.5, step=0.1, format="%.1f",
        help="Maximale elektrische opname van de warmtepomp. "
             "Bijv. Daikin Altherma 3 M 8kW: ~3,5 kW elektrisch bij -7°C ontwerpcondities. "
             "Wordt bepaald door WP-specificatie (niet de groepszekering). "
             "Zonder deze begrenzing kan het model onrealistische pieken berekenen bij extreme kou."
    )

    st.divider()
    st.header("Kookplaat")
    ckw = st.select_slider("Max. vermogen (kW)", [3.7, 4.6, 6.6, 7.4], value=7.4)

    st.divider()
    st.header("EV-lader")
    ev_on         = st.checkbox("EV aangesloten (aankomst 18:00)", value=False)
    ev_needed_kwh = st.slider("Benodigde laadcapaciteit (kWh)", 10, 80, 50, step=5,
                              disabled=not ev_on,
                              help="Laadt door van 18u t/m 06u (vertrek 07u).")
    ev_max_kw     = st.slider("Max. laadvermogen (kW)", 1.4, 11.0, 7.4, step=0.1,
                              disabled=not ev_on, format="%.1f")
    ev_ll_on      = st.checkbox("EV load limiter (CT-sturing)", value=False,
                                disabled=not ev_on)
    ev_ll_a       = st.slider("Fasegrens bij EV-laden (A)", 16, 35, 25,
                              disabled=not (ev_on and ev_ll_on))

    st.divider()
    st.header("Zonnepanelen")
    sol_on  = st.checkbox("Zonnepanelen aan", value=True)
    sol_kwp = st.slider("Vermogen (kWp)", 0.5, 12.0, 3.0, step=0.5,
                        disabled=not sol_on, format="%.1f")

    st.divider()
    st.header("WP LoadLimiter")
    ll_on     = st.checkbox("LoadLimiter actief", value=False)
    trigger_a = st.slider("Triggerdrempel (A)", 20, 35, 30, disabled=not ll_on)
    resume_a  = st.slider("Hersteldrempel (A)", 15, 29, 25, disabled=not ll_on)

    st.divider()
    st.header("Monte Carlo")
    mc_runs = st.slider("Aantal simulaties", 50, 500, 200, step=50,
                        help="Meer runs = betrouwbaarder percentielband, maar langzamer.")
    mc_seed = int(st.number_input("Random seed", value=42, step=1,
                                  help="Zelfde seed → zelfde resultaat. "
                                       "Wissel seed om te controleren of resultaten stabiel zijn."))


# ═══════════════════════════════════════════════════════════════════
# SIMULATIE UITVOEREN
# ═══════════════════════════════════════════════════════════════════
sim_kwargs = dict(
    prof=prof, t_dag=t_dag, t_nacht=t_nacht, t_out=t_out, ckw=ckw,
    ev_on=ev_on, ev_max_kw=ev_max_kw, ev_ll_on=ev_ll_on,
    ev_ll_a=ev_ll_a, ev_needed_kwh=ev_needed_kwh,
    sol_on=sol_on, sol_kwp=sol_kwp,
    season_solar_scale=season["solar_scale"],
    ll_on=ll_on, trigger_a=trigger_a, resume_a=resume_a,
    c_th=c_th, ua=ua, wp_elec_max_kw=wp_elec_max_kw,
)

d        = simulate_15min(**sim_kwargs)
trig_kw  = trigger_a * 0.23
peak     = float(d["totals"].max())
over35   = int(np.sum(d["totals"] > L35))
over_trig = int(np.sum(d["totals"] > trig_kw))
wp_cut_kwh   = float(np.sum(d["wp_cut"]) * DT)      # kWh WP-elektrisch geblokkeerd door LL
wp_int_steps = int(np.sum(d["ll_active"]))           # aantal 15-min stappen LL actief
t_drop   = float(d["T_in"].min() - t_dag)
net_kwh  = float(np.sum(np.maximum(0, d["totals"])) * DT)

tick_vals   = list(range(0, STEPS, 4))
tick_labels = [f"{v//4:02d}:00" for v in tick_vals]


# ═══════════════════════════════════════════════════════════════════
# PAGINA-HEADER & STATISTIEKEN
# ═══════════════════════════════════════════════════════════════════
st.title("1×35A Energiemodel — Geavanceerd · 15 min · Thermisch · Monte Carlo")
st.caption(
    f"{hh} · {season_name.strip()} · {t_out}°C buiten · "
    f"C_th = {c_th} kWh/°C · UA = {ua_wk} W/°C"
)


def stat(col, label, value, sub="", color=None):
    _colors = {"red": "#E24B4A", "orange": "#F97316", "green": "#1D9E75"}
    with col:
        st.markdown(f"**{label}**")
        style = f"color:{_colors[color]};" if color in _colors else ""
        st.markdown(
            f"<span style='font-size:1.5rem;font-weight:500;{style}'>{value}</span>",
            unsafe_allow_html=True,
        )
        if sub:
            st.caption(sub)


c1, c2, c3, c4, c5 = st.columns(5)
stat(c1, "Piekbelasting", f"{peak:.1f} kW", f"{peak/0.23:.0f} A op 1 fase",
     color="red" if peak > L35 else ("orange" if peak > trig_kw else None))
stat(c2, "Kwartieren boven 35A", f"{over35}×",
     f"= {over35/4:.1f} u" if over35 else "OK",
     color="red" if over35 else None)
stat(c3, f"Kwartieren boven {trigger_a}A", f"{over_trig}×",
     "drempel overschreden" if over_trig else "OK",
     color="orange" if over_trig else None)
if ll_on:
    stat(c4, "WP uitgesteld", f"{wp_cut_kwh:.2f} kWh",
         f"{wp_int_steps/4:.1f} u warmtevraag verschoven",
         color="orange" if wp_cut_kwh > 0 else None)
    t_color = "red" if t_drop < -1.0 else ("orange" if t_drop < -0.3 else None)
    stat(c5, "Min. binnentemp.", f"{d['T_in'].min():.2f}°C",
         f"daling {abs(t_drop):.2f}°C t.o.v. setpoint", color=t_color)
else:
    stat(c4, "Dagverbruik netto", f"{net_kwh:.1f} kWh",
         "na zon-opbrengst" if sol_on else "")
    stat(c5, "Warmteschuld", "n.v.t.", "LoadLimiter uit")

if ev_on:
    ev_tekort = max(0.0, ev_needed_kwh - d["ev_charged"])
    st.divider()
    e1, e2, e3 = st.columns(3)
    stat(e1, "EV geladen", f"{d['ev_charged']:.1f} kWh",
         f"van {ev_needed_kwh} kWh nodig",
         color=None if ev_tekort < 0.5 else "orange")
    stat(e2, "EV tekort bij vertrek (07:00)",
         f"{ev_tekort:.0f} kWh" if ev_tekort > 0.5 else "✓ Vol",
         "auto niet vol" if ev_tekort > 0.5 else "op tijd klaar",
         color="red" if ev_tekort > 10 else ("orange" if ev_tekort > 0.5 else "green"))
    stat(e3, "Min. laadtijd (onbeperkt)",
         f"{ev_needed_kwh/ev_max_kw:.1f} u",
         f"bij {ev_max_kw:.1f} kW → klaar ~{18 + ev_needed_kwh/ev_max_kw:.1f}u")


# ═══════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════
tab1, tab2 = st.tabs(["📊  Deterministisch dagprofiel + temperatuur", "🎲  Monte Carlo bandgrafiek"])

KLEUREN = {
    "Warmtepomp":  "#185FA5",
    "Warm water":  "#85B7EB",
    "Inductie":    "#EF9F27",
    "Wasmachine":  "#7F77DD",
    "Droger":      "#D85A30",
    "Overig":      "#888780",
    "EV":          "#1D9E75",
    "Zon-export":  "rgba(99,153,34,0.6)",
}

# ── TAB 1: Deterministisch dagprofiel ─────────────────────────────
with tab1:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.68, 0.32],
        vertical_spacing=0.05,
        subplot_titles=("Elektriciteitsbelasting (kW)", "Binnentemperatuur (°C)"),
    )

    sx = list(range(STEPS))

    def add_bar(name, data, row=1):
        fig.add_trace(go.Bar(
            name=name, x=sx, y=list(data),
            marker_color=KLEUREN.get(name, "#aaa"),
            hovertemplate=f"{name}: %{{y:.2f}} kW<extra></extra>",
        ), row=row, col=1)

    add_bar("Warmtepomp", d["wp"])
    add_bar("Warm water", d["ww"])
    add_bar("Inductie",   d["ind"])
    add_bar("Wasmachine", d["was"])
    add_bar("Droger",     d["dro"])
    add_bar("Overig",     d["ov"])
    if ev_on:
        add_bar("EV", d["ev"])

    # Zon-export: overproductie als negatieve balk
    all_loads = d["wp"] + d["ww"] + d["ind"] + d["was"] + d["dro"] + d["ov"] + d["ev"]
    export = np.where(d["sol"] > all_loads, -(d["sol"] - all_loads), 0.0)
    if np.any(export < -0.01):
        add_bar("Zon-export", export)

    # Totaal netafname (gele lijn)
    fig.add_trace(go.Scatter(
        name="Totaal netafname", x=sx, y=list(d["totals"]),
        mode="lines", line=dict(color="#E8C547", width=2.5),
        hovertemplate="Totaal: %{y:.2f} kW<extra></extra>",
    ), row=1, col=1)

    # Referentielijn zonder LL (oranje stippel)
    # Gebruikt wp_want (thermisch model, ongeblokkeerd) — identiek aan wat de simulatie toont
    # als de LoadLimiter uitstaat. wp_ref (profielbased) zou inconsistent zijn.
    if ll_on:
        ref_totals = d["wp_want"] + d["ww"] + d["ind"] + d["was"] + d["dro"] + d["ov"] + d["ev"] - d["sol"]
        fig.add_trace(go.Scatter(
            name="Zonder LoadLimiter", x=sx, y=list(ref_totals),
            mode="lines", line=dict(color="#F97316", width=1.5, dash="dot"),
            opacity=0.75, hovertemplate="Zonder LL: %{y:.2f} kW<extra></extra>",
        ), row=1, col=1)

    # Drempellijnen (row=1)
    for y_val, color, dash, label in [
        (L35,            "#E24B4A", "dash", f"35A limiet ({L35:.2f} kW)"),
        (trig_kw,        "#F97316", "dot",  f"{trigger_a}A drempel ({trig_kw:.2f} kW)"),
    ]:
        fig.add_hline(y=y_val, line_dash=dash, line_color=color, line_width=1.8,
                      annotation_text=label, annotation_position="top right",
                      annotation_font=dict(color=color, size=10), row=1, col=1)
    if ll_on:
        rv = resume_a * 0.23
        fig.add_hline(y=rv, line_dash="dot", line_color="#1D9E75", line_width=1.5,
                      annotation_text=f"{resume_a}A herstel",
                      annotation_position="bottom right",
                      annotation_font=dict(color="#1D9E75", size=10), row=1, col=1)

    # Oranje achtergrond bij LL-interventie (alleen stappen waar LL daadwerkelijk ingreep)
    if ll_on:
        for s in range(STEPS):
            if d["ll_active"][s]:
                fig.add_vrect(x0=s - 0.5, x1=s + 0.5,
                              fillcolor="rgba(249,115,22,0.12)",
                              layer="below", line_width=0, row=1, col=1)

    # Rij 2: binnentemperatuur
    fig.add_trace(go.Scatter(
        name="Binnentemperatuur", x=sx, y=list(d["T_in"]),
        mode="lines", line=dict(color="#A855F7", width=2),
        fill="tozeroy", fillcolor="rgba(168,85,247,0.08)",
        hovertemplate="Binnentemperatuur: %{y:.2f}°C<extra></extra>",
    ), row=2, col=1)
    setpoints = [t_nacht if s in NIGHT_STEPS else t_dag for s in range(STEPS)]
    fig.add_trace(go.Scatter(
        name="Setpoint", x=sx, y=setpoints,
        mode="lines", line=dict(color="#999", width=1, dash="dot"),
        hovertemplate="Setpoint: %{y:.1f}°C<extra></extra>",
    ), row=2, col=1)

    fig.update_layout(
        barmode="relative", height=640,
        xaxis2=dict(tickmode="array", tickvals=tick_vals, ticktext=tick_labels,
                    showgrid=False, title="Tijdstip"),
        xaxis=dict(showgrid=False),
        yaxis=dict(title="Vermogen (kW)",
                   range=[min(-1.5, float(export.min()) - 0.5), 13.5],
                   dtick=1, gridcolor="rgba(128,128,128,0.15)"),
        yaxis2=dict(title="°C", gridcolor="rgba(128,128,128,0.15)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="left", x=0, font=dict(size=11),
                    bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=60, r=20, t=90, b=50),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified", font=dict(size=12),
    )
    fig.update_traces(marker_line_width=0)
    st.plotly_chart(fig, use_container_width=True)

    if ll_on and wp_cut_kwh > 0:
        st.info(
            f"🌡️  **Warmteschuld:** {d['warmteschuld']:.2f} kWh warmte die de WP verschuldigd is "
            f"door {wp_int_steps/4:.1f} u onderbreking. "
            f"Binnentemperatuur daalde maximaal **{abs(t_drop):.2f}°C** t.o.v. setpoint. "
            f"De paarse lijn toont de binnentemperatuur: de daling tijdens onderbreking en het herstel daarna."
        )

    with st.expander("🌡️  Thermische energiebalans (kW warmte)", expanded=False):
        st.caption(
            "Onderstaande grafiek toont het thermische perspectief: hoeveel warmte de woning "
            "verliest naar buiten (q_verlies), hoeveel de warmtepomp wil leveren (q_vraag), "
            "en hoeveel er daadwerkelijk geleverd wordt (q_geleverd). "
            "Het verschil tussen q_vraag en q_geleverd is de warmteschuld door LL-ingrepen."
        )
        cop_val = cop(t_out)
        q_delivered = d["q_delivered"]
        q_want_th   = d["q_want_th"]
        q_loss      = d["q_loss"]
        q_net       = q_delivered - q_loss   # netto warmtetoevoer (>0 = huis warmt op)

        fig_th = go.Figure()
        fig_th.add_trace(go.Scatter(
            name="Warmteverlies naar buiten",
            x=sx, y=list(q_loss),
            mode="lines", line=dict(color="#E24B4A", width=2),
            hovertemplate="q_verlies: %{y:.2f} kW<extra></extra>",
        ))
        fig_th.add_trace(go.Scatter(
            name="WP-warmtevraag (zonder LL)",
            x=sx, y=list(q_want_th),
            mode="lines", line=dict(color="#185FA5", width=2, dash="dot"),
            hovertemplate="q_vraag: %{y:.2f} kW<extra></extra>",
        ))
        fig_th.add_trace(go.Scatter(
            name="WP-warmtelevering (werkelijk)",
            x=sx, y=list(q_delivered),
            mode="lines", line=dict(color="#1D9E75", width=2.5),
            fill="tozeroy", fillcolor="rgba(29,158,117,0.10)",
            hovertemplate="q_geleverd: %{y:.2f} kW<extra></extra>",
        ))
        # Netto warmtebalans (vulling groen/rood afhankelijk van teken)
        fig_th.add_trace(go.Bar(
            name="Netto warmte (levering − verlies)",
            x=sx, y=list(q_net),
            marker_color=["rgba(29,158,117,0.45)" if v >= 0 else "rgba(226,75,74,0.45)" for v in q_net],
            hovertemplate="Netto: %{y:.2f} kW<extra></extra>",
        ))
        fig_th.update_layout(
            barmode="relative", height=320,
            xaxis=dict(tickmode="array", tickvals=tick_vals, ticktext=tick_labels,
                       showgrid=False, title="Tijdstip"),
            yaxis=dict(title="kW thermisch", gridcolor="rgba(128,128,128,0.15)", zeroline=True,
                       zerolinecolor="rgba(128,128,128,0.4)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
                        font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=60, r=20, t=60, b=50),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            hovermode="x unified", font=dict(size=12),
        )
        # LL-interventie achtergrond ook in thermische grafiek
        if ll_on:
            for s in range(STEPS):
                if d["ll_active"][s]:
                    fig_th.add_vrect(x0=s - 0.5, x1=s + 0.5,
                                     fillcolor="rgba(249,115,22,0.15)",
                                     layer="below", line_width=0)
        st.plotly_chart(fig_th, use_container_width=True)

        # Thermische statistieken
        th_col1, th_col2, th_col3, th_col4 = st.columns(4)
        total_q_del  = float(np.sum(q_delivered) * DT)
        total_q_loss = float(np.sum(q_loss) * DT)
        total_q_want = float(np.sum(q_want_th) * DT)
        th_deficit   = total_q_want - total_q_del
        stat(th_col1, "Warmte geleverd",      f"{total_q_del:.1f} kWh",  "door WP (thermisch)")
        stat(th_col2, "Warmteverlies",        f"{total_q_loss:.1f} kWh", "woning → buiten")
        stat(th_col3, "Warmtevraag (gewenst)", f"{total_q_want:.1f} kWh", "zonder LL-ingreep",
             color="orange" if th_deficit > 0.5 else None)
        stat(th_col4, "Thermisch tekort",     f"{th_deficit:.1f} kWh",
             "door LL-onderbreking" if th_deficit > 0.1 else "geen tekort",
             color="red" if th_deficit > 2.0 else ("orange" if th_deficit > 0.5 else "green"))

    with st.expander("📐  Modelaannames en bekende begrenzingen", expanded=False):
        st.markdown(f"""
**Thermisch huismodel (RC, 1e orde)**
- Warmtecapaciteit C\_th = {c_th} kWh/°C · Warmtedoorgang UA = {ua_wk} W/°C
- Tijdconstante τ = C\_th / UA = {c_th / ua:.0f} h · Herstel-τ = {TAU_RECOVERY:.0f} h
- WP draait alleen als T\_in < setpoint (thermostaat-logica); stopt bij T\_in ≥ setpoint
- q\_verlies = UA × (T\_in − T\_buiten) · q\_herstel = (T\_set − T\_in) × C\_th / τ\_herstel
- Warmteschuld = gemiste warmtelevering (kWh thermisch) tijdens LL-onderbrekingen
- **Beginconditie:** woning start op nachtsetpoint ({t_nacht}°C) om 00:00

**EV-laadvolgorde**
- Aankomst 18:00, laadvenster 18:00–23:45 en 00:00–06:45, vertrek 07:00
- Zonder fasebegrenzing: vaste last op {ev_max_kw:.1f} kW
- Met fasebegrenzing (CT-sturing): beschikbare ruimte = {ev_ll_a}A grens − (overige last − zon), min. 6A (IEC 61851)

**Tijdresolutie**
- 96 kwartier-stappen per dag · uurprofielen worden 4× herhaald (geen interpolatie)
- Aanloopstromen (WP-compressorstart, inductiekookplaat) zijn **niet** gemodelleerd — werkelijke pieken kunnen kortstondig hoger liggen

**Seizoensinvloed**
- Zonopbrengst: {season["solar_scale"]:.2f}× t.o.v. referentie (lente/herfst)
- WP-last: via buitentemperatuur + lineair COP-model — geen fabrikants-correctiecurve

**Monte Carlo** (zie tab 2)
- Stochastisch: T\_buiten ±2°C, kookmoment ±2u, doucheintensiteit 50–150%, was/droger kans, EV-aankomst-SOC variabel
- Vaste seed {mc_seed} → reproduceerbaar · wissel seed voor stabiliteitscheck

**Bekende vereenvoudigingen**
- Simulatie beslaat **één dag** — geen carry-over van binnentemperatuur of EV-laadniveau tussen dagen
- COP-model is lineair (−0,08 per °C); werkelijke WP-prestaties wijken af per fabrikant en model
- EV-fasebegrenzing gebruikt een profielbased WP-referentie, niet de thermisch berekende waarde — bij extreme kou kan de EV iets meer laadruimte krijgen dan het model berekent
- UA en C\_th zijn woningspecifiek; kalibratie op meetdata is nodig voor exacte uitkomsten
""")


# ── TAB 2: Monte Carlo ─────────────────────────────────────────────
with tab2:
    st.markdown(
        f"**{mc_runs} gesimuleerde dagen** met dagelijkse variatie in weersomstandigheden, "
        "gedrag (kookmoment, doucheintensiteit, was/droger) en EV-aankomst-laadniveau. "
        "Alle overige instellingen zijn identiek aan het deterministisch profiel."
    )

    with st.spinner(f"Monte Carlo: {mc_runs} simulaties..."):
        mc = run_monte_carlo(mc_runs, sim_kwargs, rng_seed=mc_seed)

    # ── Bandgrafiek ──────────────────────────────────────────────────
    sx = list(range(STEPS))
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        name="P10–P90 band (80% van de dagen)",
        x=sx + sx[::-1],
        y=list(mc["p90_totals"]) + list(mc["p10_totals"])[::-1],
        fill="toself", fillcolor="rgba(29,158,117,0.18)",
        line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
    ))
    fig2.add_trace(go.Scatter(
        name="P50 — mediaan dag",
        x=sx, y=list(mc["p50_totals"]),
        mode="lines", line=dict(color="#1D9E75", width=2.5),
        hovertemplate="P50: %{y:.2f} kW<extra></extra>",
    ))
    fig2.add_trace(go.Scatter(
        name="Deterministisch (referentie)",
        x=sx, y=list(d["totals"]),
        mode="lines", line=dict(color="#E8C547", width=1.5, dash="dot"),
        opacity=0.85, hovertemplate="Det.: %{y:.2f} kW<extra></extra>",
    ))
    fig2.add_hline(y=L35, line_dash="dash", line_color="#E24B4A", line_width=2,
                   annotation_text=f"35A limiet ({L35:.2f} kW)",
                   annotation_position="top right",
                   annotation_font=dict(color="#E24B4A", size=10))
    fig2.add_hline(y=trig_kw, line_dash="dot", line_color="#F97316", line_width=1.8,
                   annotation_text=f"{trigger_a}A triggerdrempel",
                   annotation_position="bottom right",
                   annotation_font=dict(color="#F97316", size=10))

    fig2.update_layout(
        height=420, barmode="relative",
        xaxis=dict(tickmode="array", tickvals=tick_vals, ticktext=tick_labels,
                   showgrid=False, title="Tijdstip"),
        yaxis=dict(title="Netafname (kW)", range=[-1, 14], dtick=1,
                   gridcolor="rgba(128,128,128,0.15)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        margin=dict(l=60, r=20, t=80, b=50),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified", font=dict(size=12),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── MC statistieken ──────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    stat(m1, "Dagen boven 35A", f"{mc['pct_over35']:.0f}%",
         f"van {mc_runs} gesimuleerde dagen",
         color="red" if mc["pct_over35"] > 50 else ("orange" if mc["pct_over35"] > 20 else None))
    stat(m2, "P90 piekbelasting", f"{np.percentile(mc['peaks'], 90):.1f} kW",
         "hoogste 10% van de dagen")
    stat(m3, "P50 piekbelasting", f"{np.percentile(mc['peaks'], 50):.1f} kW",
         "mediane dag")
    if ev_on:
        ev_pct_full = 100.0 * float(np.mean(mc["ev_charged"] >= ev_needed_kwh - 0.5))
        stat(m4, "EV vol bij vertrek", f"{ev_pct_full:.0f}%",
             f"van {mc_runs} dagen volledig geladen",
             color="green" if ev_pct_full > 80 else ("orange" if ev_pct_full > 40 else "red"))
    else:
        stat(m4, "Seed", str(mc_seed), "wissel voor stabiliteitscheck")

    # ── Histogram piekbelasting ──────────────────────────────────────
    fig3 = go.Figure()
    fig3.add_trace(go.Histogram(
        x=list(mc["peaks"]), nbinsx=30,
        marker_color="#185FA5", opacity=0.82,
        name="Dagelijkse piekbelasting",
    ))
    fig3.add_vline(x=L35, line_dash="dash", line_color="#E24B4A", line_width=2,
                   annotation_text="35A", annotation_position="top right")
    fig3.add_vline(x=trig_kw, line_dash="dot", line_color="#F97316", line_width=1.8,
                   annotation_text=f"{trigger_a}A", annotation_position="top left")
    p50v = float(np.percentile(mc["peaks"], 50))
    p90v = float(np.percentile(mc["peaks"], 90))
    fig3.add_vline(x=p50v, line_dash="dot", line_color="#1D9E75", line_width=1.5,
                   annotation_text=f"P50={p50v:.1f}", annotation_position="top left")
    fig3.add_vline(x=p90v, line_dash="dot", line_color="#E8C547", line_width=1.5,
                   annotation_text=f"P90={p90v:.1f}", annotation_position="top right")
    fig3.update_layout(
        height=300, showlegend=False,
        xaxis=dict(title="Piekbelasting (kW)", showgrid=False),
        yaxis=dict(title="Aantal dagen", gridcolor="rgba(128,128,128,0.15)"),
        margin=dict(l=60, r=20, t=40, b=50),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
    )
    st.plotly_chart(fig3, use_container_width=True)
