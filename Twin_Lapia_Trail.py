# ==========================================================
# TWIN-LAPIA DIGITAL TWIN v3
# Clean Farmer Mode + Research Mode Architecture
# ==========================================================

# ================== IMPORTS ==================
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json, os
from datetime import date
import uuid
import qrcode
from io import BytesIO

# ================== MEMORY FILES ==================
MEMORY_FILE = "farm_memory.json"
TRACE_FILE = "traceability.json"

farm_memory = json.load(open(MEMORY_FILE)) if os.path.exists(MEMORY_FILE) else {}
trace_db = json.load(open(TRACE_FILE)) if os.path.exists(TRACE_FILE) else {}

# ================== GLOBAL CONSTANTS ==================
FEED_ENERGY = 18000        # J/g
ASSIM_EFF = 0.8
FEED_WASTE_BASE = 0.05

mu_E = 5.5e5
w_E  = 23.9
d_V  = 1.0

N_CONTENT_FEED = 0.08
N_RETENTION = 0.35

# FCR realistic constraints
FCR_MIN_REALISTIC = 0.8
FCR_MAX_REALISTIC = 7.0

# ================== BASE DEB PARAMETERS ==================
BASE_DEB = {
    "p_Am": 320.0,
    "v": 0.035,
    "kap": 0.85,
    "p_M": 12.0,
    "E_G": 4200.0,
    "k_J": 0.002,
    "T_ref": 293.15,
    "T_A": 8000.0
}

# ================== APP TITLE ==================
st.title("Twin-Lapia")
st.markdown("Your digital assistant for Tilapia farming and research")

# ================== MODE SELECTION ==================
st.sidebar.header("System Mode")

mode = st.sidebar.selectbox(
    "Select Mode",
    ["Farmer Mode", "Research Mode"]
)

# ================== FARM IDENTIFICATION ==================
st.sidebar.header("Farm Identification")

farm_id = st.sidebar.text_input("Farm ID", "default_farm")
pond_id = st.sidebar.text_input("Pond ID", "pond_1")
MEM_KEY = f"{farm_id}::{pond_id}"

DEB = BASE_DEB.copy()
if MEM_KEY in farm_memory:
    DEB.update(farm_memory[MEM_KEY].get("params", {}))

# ================== COMMON STOCKING INPUTS ==================
st.sidebar.header("Stocking")

init_w = st.sidebar.number_input("Initial Weight (g)", 1.0, 2000.0, 50.0)
n_fish = st.sidebar.number_input("Number of Fish", 1, 200000, 1000)
pond_vol = st.sidebar.number_input("Pond Volume (m³)", 1.0, 20000.0, 100.0)
days = st.sidebar.slider("Production Days", 1, 365, 120)

# ================== ENVIRONMENT ==================
st.sidebar.header("Environment")

T_mean = st.sidebar.slider("Mean Temperature (°C)", 20.0, 35.0, 28.0)
O2_init = st.sidebar.slider("Initial DO (mg/L)", 2.0, 8.0, 6.0)
TAN_init = st.sidebar.slider("Initial TAN (mg/L)", 0.0, 3.0, 0.5)
NO2 = st.sidebar.slider("NO2 (mg/L)", 0.0, 2.0, 0.2)

disease = st.sidebar.slider("Disease Level", 0.0, 1.0, 0.0)

# ================== FEEDING ==================
st.sidebar.header("Feeding")

feeding_mode = st.sidebar.selectbox(
    "Feeding Mode",
    ["Manual (Total Feed per Day)", "Auto (Biomass %)"]
)

manual_total_feed = st.sidebar.number_input(
    "Manual Total Feed per Day (kg)",
    0.0, 50000.0, 100.0
)

feeding_frequency = st.sidebar.slider("Feedings per Day", 1, 6, 3)

# ================== AERATION ==================
st.sidebar.header("Aeration")

aerator_mode = st.sidebar.selectbox(
    "Aerator Mode",
    ["Manual", "Automatic"]
)

aerator_power_manual = st.sidebar.slider("Manual Aerator Power", 0.0, 1.0, 0.5)
water_exchange = st.sidebar.slider("Daily Water Exchange (%)", 0.0, 50.0, 5.0)

# ================== ECONOMICS ==================
st.sidebar.header("Economics")

price = st.sidebar.number_input("Fish Price ($/kg)", 0.0, 10.0, 2.5)
feed_cost = st.sidebar.number_input("Feed Cost ($/kg)", 0.0, 5.0, 1.2)
aerator_cost_per_day = st.sidebar.number_input("Aerator Cost ($/day)", 0.0, 50.0, 5.0)

# ==========================================================
# MODE-SPECIFIC SETTINGS
# ==========================================================

if mode == "Research Mode":

    st.sidebar.header("Research Controls")

    logistic_K_research = st.sidebar.number_input(
        "Asymptotic Weight for Logistic (g)",
        100.0, 2000.0, 800.0
    )

    run_monte_carlo = st.sidebar.checkbox("Enable Monte Carlo")
    mc_runs = st.sidebar.slider("Monte Carlo Runs", 10, 200, 50)

    parameter_variation = st.sidebar.slider(
        "Biological Variation (%)", 0.0, 30.0, 10.0
    )

    environment_variation = st.sidebar.slider(
        "Environmental Variation (%)", 0.0, 30.0, 10.0
    )

else:
    # Farmer Mode — no academic clutter
    logistic_K_research = None
    run_monte_carlo = False
    mc_runs = 0
    parameter_variation = 0
    environment_variation = 0
# ==========================================================
# SECTION 2 — STABILIZED BIOLOGICAL CORE
# ==========================================================

# ---------------- TEMPERATURE CORRECTION ----------------
def temp_corr(T_celsius, DEB):
    T_kelvin = T_celsius + 273.15
    return np.exp(DEB["T_A"]/DEB["T_ref"] - DEB["T_A"]/T_kelvin)


# ---------------- WEIGHT FUNCTION ----------------
def weight(E, V):
    return V + (E * w_E / (d_V * mu_E))


# ---------------- FEEDING TABLE ----------------
def feeding_rate(weight_g):
    if weight_g < 20:
        return 0.08
    elif weight_g < 100:
        return 0.05
    elif weight_g < 400:
        return 0.03
    else:
        return 0.015


# ---------------- FEEDING FREQUENCY EFFECT ----------------
def feeding_frequency_effect(freq):
    return 1 + 0.03 * (freq - 1)


# ---------------- STRESS FUNCTIONS ----------------
def oxygen_stress(O2):
    if O2 < 3:
        return 0.5
    elif O2 > 6:
        return 1.0
    else:
        return 0.5 + (O2 - 3) / 6


def disease_stress(disease):
    return max(1 - 0.5 * disease, 0.6)


def water_stress(TAN, NO2):
    stress = 1.0
    if TAN > 1:
        stress *= np.exp(-0.4 * (TAN - 1))
    if NO2 > 0.5:
        stress *= np.exp(-0.5 * (NO2 - 0.5))
    return max(stress, 0.6)


def density_stress(density):
    return np.exp(-0.02 * density)


# ---------------- CARRYING CAPACITY ----------------
def carrying_capacity(O2, pond_vol):
    max_biomass = (O2 * pond_vol * 1000) / 2.5
    return max(max_biomass, 1.0)


# ---------------- MORTALITY ----------------
def mortality_rate(O2, disease, TAN, density):
    mu = (
        0.0005
        + 0.04 * disease
        + max(0, 4 - O2) * 0.02
        + max(0, TAN - 1) * 0.08
        + 0.0015 * density
    )
    return min(mu, 0.3)


# ---------------- OXYGEN DYNAMICS ----------------
def update_oxygen(O2_current, biomass, aerator_power, feed_g):
    respiration = 0.0012 * biomass + 0.0004 * feed_g
    aeration = 0.35 * aerator_power
    O2_next = O2_current + aeration - respiration
    return max(O2_next, 2.5)


def night_oxygen_drop(O2):
    return max(O2 - 0.4, 2.5)


def automatic_aerator(O2):
    if O2 < 4:
        return 1.0
    elif O2 < 5:
        return 0.7
    else:
        return 0.3


# ---------------- TAN DYNAMICS ----------------
def update_TAN(TAN_current, feed_g, n_fish, pond_vol, water_exchange):
    TAN_next = TAN_current + (feed_g * N_CONTENT_FEED * n_fish / 1000) / pond_vol
    TAN_next *= (1 - water_exchange / 100)
    return max(TAN_next, 0.0)


# ==========================================================
# STABLE DEB STEP (NO STRUCTURAL COLLAPSE)
# ==========================================================
def deb_step(E, V, DEB, feed_g, T, stress_env, density_factor):

    TC = temp_corr(T, DEB)

    # Assimilation
    feed_energy = feed_g * FEED_ENERGY * ASSIM_EFF
    pA_max = DEB["p_Am"] * V**(2/3) * TC

    pA = min(feed_energy, pA_max) * stress_env

    # Maintenance
    pS = DEB["p_M"] * V * TC

    # Mobilization
    pC = (E/V) * (
        DEB["E_G"] * DEB["v"] * TC * V**(2/3) + pS
    ) / (DEB["kap"] * E/V + DEB["E_G"])

    # Structural growth
    dV = (DEB["kap"] * pC - pS) / DEB["E_G"]

    # Prevent structural shrinkage in grow-out
    if dV < -0.001:
        dV = -0.001

    dE = pA - pC

    E_new = max(E + dE, 1e-8)
    V_new = max(V + dV, 1e-6)

    return E_new, V_new
# ==========================================================
# SECTION 3 — FARMER SIMULATION ENGINE
# ==========================================================

def run_farmer_simulation():

    # ---------- INITIAL STATE ----------
    structure_fraction = 0.7
    V = init_w * structure_fraction
    E = (init_w - V) * mu_E / w_E

    survival = 1.0
    O2_current = O2_init
    TAN_current = TAN_init

    biomass_initial = n_fish * init_w / 1000
    feed_cum = 0.0

    # ---------- STORAGE ----------
    t_series = []
    W_series = []
    bio_series = []
    surv_series = []
    O2_series = []
    TAN_series = []
    feed_series = []

    # ---------- DAILY LOOP ----------
    for day in range(days):

        current_weight = weight(E, V)
        current_fish = n_fish * survival
        biomass = current_fish * current_weight / 1000
        density = biomass / pond_vol

        # -------- FEEDING --------
        if feeding_mode == "Manual (Total Feed per Day)":
            feed_total_today = manual_total_feed
            feed_g = (feed_total_today * 1000) / max(current_fish, 1)
        else:
            feed_percent = feeding_rate(current_weight)
            feed_percent *= feeding_frequency_effect(feeding_frequency)
            feed_g = current_weight * feed_percent

        feed_g *= (1 - FEED_WASTE_BASE)

        # -------- STRESS --------
        stress_env = (
            oxygen_stress(O2_current)
            * disease_stress(disease)
            * water_stress(TAN_current, NO2)
        )

        density_factor = density_stress(density)

        # -------- DEB STEP --------
        E, V = deb_step(E, V, DEB, feed_g, T_mean,
                        stress_env, density_factor)

        current_weight = weight(E, V)
        biomass = current_fish * current_weight / 1000

        # -------- OXYGEN --------
        if aerator_mode == "Manual":
            aerator_power = aerator_power_manual
        else:
            aerator_power = automatic_aerator(O2_current)

        O2_current = update_oxygen(O2_current, biomass,
                                   aerator_power, feed_g)

        O2_current = night_oxygen_drop(O2_current)

        # -------- CARRYING CAPACITY --------
        K = carrying_capacity(O2_current, pond_vol)
        if biomass > K:
            survival *= 0.99

        # -------- TAN --------
        TAN_current = update_TAN(
            TAN_current,
            feed_g,
            n_fish,
            pond_vol,
            water_exchange
        )

        # -------- MORTALITY --------
        mu = mortality_rate(O2_current, disease,
                            TAN_current, density)
        survival *= np.exp(-mu)

        # -------- FEED ACCOUNTING --------
        if feeding_mode == "Manual (Total Feed per Day)":
            feed_cum += manual_total_feed
        else:
            feed_cum += feed_g * current_fish / 1000

        # -------- STORE --------
        t_series.append(day)
        W_series.append(current_weight)
        bio_series.append(biomass)
        surv_series.append(survival)
        O2_series.append(O2_current)
        TAN_series.append(TAN_current)
        feed_series.append(feed_cum)

    # ---------- CONVERT ----------
    t = np.array(t_series)
    W = np.array(W_series)
    bio = np.array(bio_series)
    surv = np.array(surv_series)
    O2_arr = np.array(O2_series)
    TAN_arr = np.array(TAN_series)
    feed_cum_arr = np.array(feed_series)

    # ---------- CYCLE FCR (CORRECTED) ----------
    final_biomass = bio[-1]
    biomass_gain = final_biomass - biomass_initial

    if biomass_gain > 0:
        FCR_cycle = feed_cum_arr[-1] / biomass_gain
    else:
        FCR_cycle = np.nan

    # Clamp unrealistic FCR
    if FCR_cycle > FCR_MAX_REALISTIC:
        FCR_cycle = FCR_MAX_REALISTIC

    # ---------- FARMER LOGISTIC REFERENCE ----------
    # K derived from oxygen-based carrying capacity
    K_farmer = carrying_capacity(O2_arr[-1], pond_vol)
    K_farmer_weight = (K_farmer * 1000) / max(n_fish, 1)

    r = 0.02
    W_logistic = K_farmer_weight / (
        1 + ((K_farmer_weight - W[0]) / W[0])
        * np.exp(-r * t)
    )

    return {
        "t": t,
        "W": W,
        "bio": bio,
        "surv": surv,
        "O2": O2_arr,
        "TAN": TAN_arr,
        "feed": feed_cum_arr,
        "FCR": FCR_cycle,
        "W_logistic": W_logistic
    }


# ----------------------------------------------------------
# RUN FARMER MODE
# ----------------------------------------------------------
if mode == "Farmer Mode":
    farmer_results = run_farmer_simulation()

# ==========================================================
# SECTION 4 — RESEARCH SIMULATION ENGINE
# ==========================================================

def run_research_simulation(DEB_params,
                            T_input,
                            feed_var_percent=0):

    structure_fraction = 0.7
    V = init_w * structure_fraction
    E = (init_w - V) * mu_E / w_E

    survival = 1.0
    O2_current = O2_init
    TAN_current = TAN_init

    biomass_initial = n_fish * init_w / 1000
    feed_cum = 0.0

    # ---------- STORAGE ----------
    t_series = []
    W_series = []
    bio_series = []
    surv_series = []
    O2_series = []
    TAN_series = []
    feed_series = []

    for day in range(days):

        current_weight = weight(E, V)
        current_fish = n_fish * survival
        biomass = current_fish * current_weight / 1000
        density = biomass / pond_vol

        # -------- FEEDING --------
        if feeding_mode == "Manual (Total Feed per Day)":
            feed_today = manual_total_feed * (
                1 + np.random.uniform(-feed_var_percent/100,
                                      feed_var_percent/100)
            )
            feed_g = (feed_today * 1000) / max(current_fish, 1)
        else:
            feed_percent = feeding_rate(current_weight)
            feed_percent *= feeding_frequency_effect(feeding_frequency)
            feed_g = current_weight * feed_percent

        feed_g *= (1 - FEED_WASTE_BASE)

        # -------- STRESS --------
        stress_env = (
            oxygen_stress(O2_current)
            * disease_stress(disease)
            * water_stress(TAN_current, NO2)
        )

        density_factor = density_stress(density)

        # -------- DEB STEP --------
        E, V = deb_step(E, V, DEB_params, feed_g,
                        T_input, stress_env, density_factor)

        current_weight = weight(E, V)
        biomass = current_fish * current_weight / 1000

        # -------- OXYGEN --------
        if aerator_mode == "Manual":
            aerator_power = aerator_power_manual
        else:
            aerator_power = automatic_aerator(O2_current)

        O2_current = update_oxygen(O2_current, biomass,
                                   aerator_power, feed_g)

        O2_current = night_oxygen_drop(O2_current)

        # -------- CARRYING CAPACITY --------
        K = carrying_capacity(O2_current, pond_vol)
        if biomass > K:
            survival *= 0.99

        # -------- TAN --------
        TAN_current = update_TAN(
            TAN_current,
            feed_g,
            n_fish,
            pond_vol,
            water_exchange
        )

        # -------- MORTALITY --------
        mu = mortality_rate(O2_current, disease,
                            TAN_current, density)
        survival *= np.exp(-mu)

        # -------- FEED ACCOUNTING --------
        if feeding_mode == "Manual (Total Feed per Day)":
            feed_cum += feed_today
        else:
            feed_cum += feed_g * current_fish / 1000

        # -------- STORE --------
        t_series.append(day)
        W_series.append(current_weight)
        bio_series.append(biomass)
        surv_series.append(survival)
        O2_series.append(O2_current)
        TAN_series.append(TAN_current)
        feed_series.append(feed_cum)

    # ---------- CONVERT ----------
    t = np.array(t_series)
    W = np.array(W_series)
    bio = np.array(bio_series)
    surv = np.array(surv_series)
    O2_arr = np.array(O2_series)
    TAN_arr = np.array(TAN_series)
    feed_cum_arr = np.array(feed_series)

    # ---------- CYCLE FCR ----------
    final_biomass = bio[-1]
    biomass_gain = final_biomass - biomass_initial

    if biomass_gain > 0:
        FCR_cycle = feed_cum_arr[-1] / biomass_gain
    else:
        FCR_cycle = np.nan

    # ---------- NITROGEN BALANCE ----------
    N_input_total = feed_cum_arr[-1] * N_CONTENT_FEED
    N_retained = biomass_gain * N_RETENTION * 0.16
    N_waste = max(N_input_total - N_retained, 0)

    # ---------- RESEARCH LOGISTIC REFERENCE ----------
    if logistic_K_research is not None:
        K_weight = logistic_K_research
    else:
        K_weight = 800.0  # fallback

    r = 0.02
    W_logistic = K_weight / (
        1 + ((K_weight - W[0]) / W[0])
        * np.exp(-r * t)
    )

    return {
        "t": t,
        "W": W,
        "bio": bio,
        "surv": surv,
        "O2": O2_arr,
        "TAN": TAN_arr,
        "feed": feed_cum_arr,
        "FCR": FCR_cycle,
        "N_input": N_input_total,
        "N_retained": N_retained,
        "N_waste": N_waste,
        "W_logistic": W_logistic
    }


# ----------------------------------------------------------
# RUN RESEARCH MODE
# ----------------------------------------------------------
if mode == "Research Mode":
    research_results = run_research_simulation(DEB, T_mean)
# ==========================================================
# SECTION 5 — FARMER OUTPUT DASHBOARD
# ==========================================================

if mode == "Farmer Mode":

    st.header("Farmer Production Dashboard")

    results = farmer_results

    t = results["t"]
    W = results["W"]
    bio = results["bio"]
    surv = results["surv"]
    O2_arr = results["O2"]
    TAN_arr = results["TAN"]
    feed_arr = results["feed"]
    FCR_cycle = results["FCR"]
    W_logistic = results["W_logistic"]

    final_weight = W[-1]
    final_biomass = bio[-1]
    survival_percent = surv[-1] * 100

    # ---------- ECONOMICS ----------
    revenue = final_biomass * price
    feed_cost_total = feed_arr[-1] * feed_cost
    aeration_cost_total = aerator_cost_per_day * days
    profit = revenue - feed_cost_total - aeration_cost_total

    # ---------- METRICS ----------
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Final Weight (g)", f"{final_weight:.1f}")
    col2.metric("Final Biomass (kg)", f"{final_biomass:.1f}")
    col3.metric("Survival (%)", f"{survival_percent:.1f}")
    col4.metric("Cycle FCR", f"{FCR_cycle:.2f}" if not np.isnan(FCR_cycle) else "N/A")

    st.metric("Estimated Profit ($)", f"{profit:.2f}")

    # ---------- FCR WARNING ----------
    if not np.isnan(FCR_cycle):
        if FCR_cycle > 3:
            st.warning("⚠ High FCR — check feeding strategy.")
        elif FCR_cycle < 1:
            st.warning("⚠ FCR unusually low — verify feed input.")

    # ---------- GROWTH PLOT ----------
    st.subheader("Growth Curve")

    fig1, ax1 = plt.subplots()
    ax1.plot(t, W, label="DEB Growth")
    ax1.plot(t, W_logistic, linestyle="--",
             label="Oxygen-Based Logistic")
    ax1.set_xlabel("Days")
    ax1.set_ylabel("Weight (g)")
    ax1.legend()
    st.pyplot(fig1)

    # ---------- BIOMASS ----------
    st.subheader("Biomass")

    fig2, ax2 = plt.subplots()
    ax2.plot(t, bio)
    ax2.set_xlabel("Days")
    ax2.set_ylabel("Biomass (kg)")
    st.pyplot(fig2)

    # ---------- OXYGEN ----------
    st.subheader("Oxygen (DO)")

    fig3, ax3 = plt.subplots()
    ax3.plot(t, O2_arr)
    ax3.set_xlabel("Days")
    ax3.set_ylabel("DO (mg/L)")
    st.pyplot(fig3)

    if np.any(O2_arr < 3):
        st.warning("⚠ Oxygen dropped below safe level!")

    # ---------- TAN ----------
    st.subheader("TAN")

    fig4, ax4 = plt.subplots()
    ax4.plot(t, TAN_arr)
    ax4.set_xlabel("Days")
    ax4.set_ylabel("TAN (mg/L)")
    st.pyplot(fig4)

    if np.any(TAN_arr > 1.5):
        st.warning("⚠ TAN exceeded safe threshold!")

    # ---------- FEED USAGE ----------
    st.subheader("Cumulative Feed Used (kg)")

    fig5, ax5 = plt.subplots()
    ax5.plot(t, feed_arr)
    ax5.set_xlabel("Days")
    ax5.set_ylabel("Feed (kg)")
    st.pyplot(fig5)

    # ---------- EXPORT ----------
    st.subheader("Download Farm Report")

    farm_df = pd.DataFrame({
        "Day": t,
        "Weight_g": W,
        "Biomass_kg": bio,
        "Survival": surv,
        "O2": O2_arr,
        "TAN": TAN_arr,
        "Feed_kg": feed_arr
    })

    csv_bytes = farm_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Farm Production CSV",
        data=csv_bytes,
        file_name=f"{farm_id}_{pond_id}_farmer_report.csv",
        mime="text/csv"
    )
# ==========================================================
# SECTION 6 — RESEARCH OUTPUT DASHBOARD
# ==========================================================

if mode == "Research Mode":

    st.header("Research Analysis Dashboard")

    results = research_results

    t = results["t"]
    W = results["W"]
    bio = results["bio"]
    surv = results["surv"]
    O2_arr = results["O2"]
    TAN_arr = results["TAN"]
    feed_arr = results["feed"]
    FCR_cycle = results["FCR"]
    W_logistic = results["W_logistic"]

    final_weight = W[-1]
    final_biomass = bio[-1]

    # ---------- METRICS ----------
    col1, col2, col3 = st.columns(3)

    col1.metric("Final Weight (g)", f"{final_weight:.2f}")
    col2.metric("Final Biomass (kg)", f"{final_biomass:.2f}")
    col3.metric("Cycle FCR", f"{FCR_cycle:.2f}" if not np.isnan(FCR_cycle) else "N/A")

    # Flag unrealistic FCR
    if not np.isnan(FCR_cycle):
        if FCR_cycle > FCR_MAX_REALISTIC:
            st.error("FCR exceeds realistic biological threshold.")
        elif FCR_cycle < FCR_MIN_REALISTIC:
            st.warning("FCR below expected biological range.")

    # ---------- GROWTH COMPARISON ----------
    st.subheader("DEB vs Logistic Growth")

    fig1, ax1 = plt.subplots()
    ax1.plot(t, W, label="DEB Growth")
    ax1.plot(t, W_logistic, linestyle="--",
             label="User Logistic Reference")
    ax1.set_xlabel("Days")
    ax1.set_ylabel("Weight (g)")
    ax1.legend()
    st.pyplot(fig1)

    # ---------- BIOMASS ----------
    st.subheader("Biomass")

    fig2, ax2 = plt.subplots()
    ax2.plot(t, bio)
    ax2.set_xlabel("Days")
    ax2.set_ylabel("Biomass (kg)")
    st.pyplot(fig2)

    # ---------- NITROGEN ----------
    st.subheader("Nitrogen Balance")

    st.write(f"Total Nitrogen Input (kg): {results['N_input']:.3f}")
    st.write(f"Nitrogen Retained (kg): {results['N_retained']:.3f}")
    st.write(f"Nitrogen Waste (kg): {results['N_waste']:.3f}")

    # ======================================================
    # MONTE CARLO (OPTION B)
    # ======================================================
    if run_monte_carlo:

        st.subheader("Monte Carlo Analysis")

        final_weights = []
        final_FCRs = []
        growth_curves = []

        for i in range(mc_runs):

            DEB_mc = DEB.copy()

            # Biological variation
            var_factor = parameter_variation / 100
            DEB_mc["p_Am"] *= np.random.uniform(1 - var_factor,
                                                1 + var_factor)
            DEB_mc["p_M"] *= np.random.uniform(1 - var_factor,
                                               1 + var_factor)
            DEB_mc["v"] *= np.random.uniform(1 - var_factor,
                                             1 + var_factor)

            # Environmental variation
            T_mc = T_mean * np.random.uniform(
                1 - environment_variation/100,
                1 + environment_variation/100
            )

            mc_results = run_research_simulation(
                DEB_mc,
                T_mc,
                feed_var_percent=environment_variation
            )

            final_weights.append(mc_results["W"][-1])
            final_FCRs.append(mc_results["FCR"])
            growth_curves.append(mc_results["W"])

        final_weights = np.array(final_weights)
        final_FCRs = np.array(final_FCRs)
        growth_curves = np.array(growth_curves)

        # ---------- WEIGHT DISTRIBUTION ----------
        st.subheader("Final Weight Distribution")

        fig_mc1, ax_mc1 = plt.subplots()
        ax_mc1.hist(final_weights, bins=20)
        ax_mc1.set_xlabel("Final Weight (g)")
        ax_mc1.set_ylabel("Frequency")
        st.pyplot(fig_mc1)

        # ---------- FCR DISTRIBUTION ----------
        st.subheader("Final FCR Distribution")

        fig_mc2, ax_mc2 = plt.subplots()
        ax_mc2.hist(final_FCRs[~np.isnan(final_FCRs)], bins=20)
        ax_mc2.set_xlabel("Final FCR")
        ax_mc2.set_ylabel("Frequency")
        st.pyplot(fig_mc2)

        # ---------- GROWTH ENVELOPE ----------
        st.subheader("Growth Envelope")

        mean_curve = np.mean(growth_curves, axis=0)
        min_curve = np.min(growth_curves, axis=0)
        max_curve = np.max(growth_curves, axis=0)

        fig_mc3, ax_mc3 = plt.subplots()
        ax_mc3.plot(t, mean_curve, label="Mean Growth")
        ax_mc3.fill_between(t, min_curve, max_curve, alpha=0.3)
        ax_mc3.set_xlabel("Days")
        ax_mc3.set_ylabel("Weight (g)")
        ax_mc3.legend()
        st.pyplot(fig_mc3)

        st.write(f"Mean Final Weight: {np.mean(final_weights):.2f} g")
        st.write(f"Std Final Weight: {np.std(final_weights):.2f} g")
        st.write(f"Mean Final FCR: {np.nanmean(final_FCRs):.2f}")
# ==========================================================
# SECTION 7 — CALIBRATION & LEARNING
# ==========================================================

st.header("Model Calibration & Learning")

# ---------- OBSERVED INPUTS ----------
obs_weight = st.number_input("Observed Final Weight (g)", 0.0, 5000.0)
obs_fcr = st.number_input("Observed Final FCR", 0.0, 10.0)

# Select correct results depending on mode
if mode == "Farmer Mode":
    final_weight_model = farmer_results["W"][-1]
    final_fcr_model = farmer_results["FCR"]
else:
    final_weight_model = research_results["W"][-1]
    final_fcr_model = research_results["FCR"]

# Load history
history = farm_memory.get(MEM_KEY, {}).get("history", [])

# ---------- STORE CALIBRATION ----------
if st.button("Store Calibration"):

    if obs_weight > 0 and obs_fcr > 0:

        weight_error = abs(final_weight_model - obs_weight) / max(obs_weight, 1e-6)
        fcr_error = abs(final_fcr_model - obs_fcr) / max(obs_fcr, 1e-6)

        total_error = weight_error + fcr_error

        history.append({
            "date": str(date.today()),
            "weight_error": float(weight_error),
            "fcr_error": float(fcr_error),
            "total_error": float(total_error)
        })

        farm_memory[MEM_KEY] = {
            "params": DEB,
            "history": history[-30:]  # keep last 30 cycles
        }

        with open(MEMORY_FILE, "w") as f:
            json.dump(farm_memory, f, indent=2)

        st.success("Calibration cycle stored.")
    else:
        st.warning("Enter observed weight and FCR before storing.")

# ---------- CONFIDENCE SCORE ----------
if history:

    total_errors = [
        h["total_error"]
        for h in history
        if "total_error" in h
    ]

    if total_errors:
        recent_errors = total_errors[-5:]
        mean_error = np.mean(recent_errors)

        # Skill improves as error decreases
        skill = np.exp(-3 * mean_error)

        # Experience improves with cycles
        experience = np.tanh(len(total_errors) / 5)

        confidence = np.clip(0.3 + 0.7 * skill * experience, 0.3, 0.95)
    else:
        confidence = 0.6
else:
    confidence = 0.6

st.metric("Model Confidence Score", f"{confidence:.2f}")

# ---------- LEARNING TREND ----------
if history and len(history) > 1:

    total_errors = [
        h["total_error"]
        for h in history
        if "total_error" in h
    ]

    if len(total_errors) > 1:

        cycles = list(range(1, len(total_errors) + 1))

        fig_learn, ax_learn = plt.subplots()
        ax_learn.plot(cycles, total_errors, marker="o")
        ax_learn.set_xlabel("Calibration Cycle")
        ax_learn.set_ylabel("Total Error")
        st.pyplot(fig_learn)

        if total_errors[-1] < total_errors[0]:
            st.success("Model accuracy improving over cycles.")
        else:
            st.info("Model performance stable. Continue calibration.")

# ---------- DIAGNOSTICS ----------
st.subheader("Model Diagnostics")

if mode == "Research Mode":
    if np.any(research_results["W"] < 0):
        st.error("Numerical instability detected in growth.")

if mode == "Farmer Mode":
    if np.any(farmer_results["W"] < 0):
        st.error("Numerical instability detected in growth.")

if final_fcr_model > FCR_MAX_REALISTIC:
    st.error("FCR outside biological realistic range.")

if final_fcr_model < FCR_MIN_REALISTIC:
    st.warning("FCR below biological expectation.")
# ==========================================================
# SECTION 8 — TRACEABILITY & EXPORT
# ==========================================================

st.header("Traceability & Export")

# ---------------- TRACEABILITY INPUT ----------------
with st.expander("Generate Traceability QR"):

    farm_name = st.text_input("Farm Name")
    location = st.text_input("Location")
    species = st.text_input("Species", "Tilapia")
    harvest_date = st.date_input("Harvest Date")

    if st.button("Generate QR Code"):

        trace_id = str(uuid.uuid4())

        trace_payload = {
            "farm_id": farm_id,
            "pond_id": pond_id,
            "farm_name": farm_name,
            "location": location,
            "species": species,
            "harvest_date": str(harvest_date),
            "mode": mode,
            "confidence_score": float(confidence),
            "generated_on": str(date.today())
        }

        trace_db[trace_id] = trace_payload

        with open(TRACE_FILE, "w") as f:
            json.dump(trace_db, f, indent=2)

        qr = qrcode.make(trace_id)
        buf = BytesIO()
        qr.save(buf, format="PNG")

        st.image(buf.getvalue(), caption="Scan for Traceability")

# ---------------- EXPORT DATASET ----------------
st.subheader("Export Simulation Data")

if mode == "Farmer Mode":
    export_data = pd.DataFrame({
        "Day": farmer_results["t"],
        "Weight_g": farmer_results["W"],
        "Biomass_kg": farmer_results["bio"],
        "Survival": farmer_results["surv"],
        "O2": farmer_results["O2"],
        "TAN": farmer_results["TAN"],
        "Feed_kg": farmer_results["feed"]
    })
else:
    export_data = pd.DataFrame({
        "Day": research_results["t"],
        "Weight_g": research_results["W"],
        "Biomass_kg": research_results["bio"],
        "Survival": research_results["surv"],
        "O2": research_results["O2"],
        "TAN": research_results["TAN"],
        "Feed_kg": research_results["feed"]
    })

csv_bytes = export_data.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Full Simulation CSV",
    data=csv_bytes,
    file_name=f"{farm_id}_{pond_id}_{mode.replace(' ', '_')}.csv",
    mime="text/csv"
)

# ---------------- SESSION SUMMARY ----------------
st.subheader("Session Summary")

if mode == "Farmer Mode":
    st.write(f"Final Weight: {farmer_results['W'][-1]:.2f} g")
    st.write(f"Final Biomass: {farmer_results['bio'][-1]:.2f} kg")
    st.write(f"Cycle FCR: {farmer_results['FCR']:.2f}")
else:
    st.write(f"Final Weight: {research_results['W'][-1]:.2f} g")
    st.write(f"Final Biomass: {research_results['bio'][-1]:.2f} kg")
    st.write(f"Cycle FCR: {research_results['FCR']:.2f}")

st.write(f"Model Confidence: {confidence:.2f}")

st.success("Twin-Lapia Digital Twin v3 — System Ready.")
