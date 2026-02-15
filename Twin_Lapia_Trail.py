# ================== IMPORTS ==================
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd
import json, os
from datetime import date
import qrcode
from io import BytesIO
import uuid

# ================== FILES ==================
MEMORY_FILE = "farm_memory.json"
TRACE_FILE = "traceability.json"

farm_memory = json.load(open(MEMORY_FILE)) if os.path.exists(MEMORY_FILE) else {}
trace_db = json.load(open(TRACE_FILE)) if os.path.exists(TRACE_FILE) else {}

# ================== CONSTANTS ==================
FEED_ENERGY = 18000        # J/g feed
mu_E = 5.5e5               # J/mol reserve
w_E  = 23.9                # g/mol reserve
d_V  = 1.0                 # g/cm3 structure

N_CONTENT_FEED = 0.08      # kg N / kg feed
N_RETENTION = 0.35         # fraction retained in biomass

# ================== PARAMETERS ==================
BASE_DEB = {
     "p_Am": 350.0,      # FIXED (realistic for tilapia)
    "v": 0.04,
    "kap": 0.85,
    "p_M": 15.0,
    "E_G": 4200.0,
    "E_Hp": 1.0e5,
    "k_J": 0.002,
    "T_ref": 293.15,
    "T_A": 9000.0
    }

# ================== SIDEBAR IDENTIFIERS ==================
st.sidebar.header("Farm / Pond Identification")
farm_id = st.sidebar.text_input("Farm ID", "default_farm")
pond_id = st.sidebar.text_input("Pond ID", "pond_1")
MEM_KEY = f"{farm_id}::{pond_id}"

DEB = BASE_DEB.copy()
if MEM_KEY in farm_memory:
    DEB.update(farm_memory[MEM_KEY]["params"])
    
# ================== LOAD MEMORY ==================
if os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE, "r") as f:
        farm_memory = json.load(f)
else:
    farm_memory = {}

if MEM_KEY in farm_memory:
    DEB.update(farm_memory[MEM_KEY]["params"])
    
# ================== UI ==================
st.title("Twin-Lapia Simulation")

T = st.sidebar.slider("Temperature (°C)", 20, 34, 28)
O2 = st.sidebar.slider("DO (mg/L)", 2.0, 8.0, 6.0)
disease = st.sidebar.slider("Disease Prevalence", 0.0, 1.0, 0.0)
TAN = st.sidebar.slider("TAN (mg/L)", 0.0, 5.0, 0.5)
NO2 = st.sidebar.slider("NO2 (mg/L)", 0.0, 2.0, 0.2)

feed_g = st.sidebar.number_input("Feed per fish per day (g)", 0.0, 150.0, 2.0)
days = st.sidebar.slider("Projection Days", 30, 240, 120)
init_w = st.sidebar.number_input("Initial Weight (g)", 1.0, 2000.0, 50.0)
n_fish = st.sidebar.number_input("Number of Fish", 1, 100000, 1000)
pond_vol = st.sidebar.number_input("Pond Volume (m³)", 1.0, 10000.0, 100.0)

st.sidebar.header("Save Results")
s_path = st.sidebar.text_input(
    "Results file path (CSV)",
    "results/twin_lapia_results.csv"
)
    
# ================== ENVIRONMENTAL STRESS ==================
def oxygen_stress(O2):
    if O2 < 3: return 0.3
    if O2 > 6: return 1.0
    return (O2 - 3) / 3

def disease_stress(dis):
    return max(1 - 0.6 * dis, 0.4)

def water_stress(TAN, NO2):
    s = 1.0
    if TAN > 1: s *= np.exp(-0.5 * (TAN - 1))
    if NO2 > 0.5: s *= np.exp(-0.7 * (NO2 - 0.5))
    return max(s, 0.4)

# ================== TEMPERATURE CORRECTION ==================
def temp_corr(T):
    return np.exp(DEB["T_A"]/DEB["T_ref"] - DEB["T_A"]/T)

def feed_to_f(feed_g, V, T, O2, dis, TAN, NO2):
    TC = temp_corr(T)
    stress = oxygen_stress(O2)*disease_stress(dis)*water_stress(TAN, NO2)
    pA_max = DEB["p_Am"] * V**(2/3) * TC * stress
    return np.clip(feed_g*FEED_ENERGY/(pA_max+1e-8),0,1)
# ================== TRUE DEB WEIGHT ==================
def weight(E, V, E_R):
    return V + (E + E_R) * w_E / (d_V * mu_E)
# ================== DEB MODEL ==================
def deb_ode(t, EVHR, T, feed_g,O2, dis, TAN, NO2):
    E, V, E_H, E_R = EVHR

    TC = temp_corr(T + 273.15)
    v_T  = DEB["v"] * TC
    p_MT = DEB["p_M"] * TC
    k_JT = DEB["k_J"] * TC

    f = feed_to_f(feed_g, V, T, O2, dis, TAN, NO2)

    pA = f * DEB["p_Am"] * V**(2/3) * TC
    pS = p_MT * V

    pC = (E/V) * (
        DEB["E_G"] * v_T * V**(2/3) + pS
    ) / (DEB["kap"] * E/V + DEB["E_G"])

    pJ = k_JT * E_H

    dE = pA - pC
    dV = (DEB["kap"] * pC - pS) / DEB["E_G"]

    if E_H < DEB["E_Hp"]:
        dEH = (1 - DEB["kap"]) * pC - pJ
        dER = 0.0
    else:
        dEH = 0.0
        dER = (1 - DEB["kap"]) * pC - pJ

    return [dE, dV, dEH, dER]

# ================== HELPERS ==================
def mortality(V,O2,dis,TAN):
    return min(0.001+0.05*dis+max(0,4-O2)*0.02+max(0,TAN-1)*0.1,0.2)

# ================== INITIAL STATE ==================
V0 = init_w
E0 = 0.3 * DEB["p_Am"] * V0**(2/3)
EVHR0 = [E0, V0, 0.0, 0.0]

# ================== SOLVE ==================
t = np.arange(days + 1)
stress = oxygen_stress(O2) * disease_stress(disease) * water_stress(TAN, NO2)

sol = solve_ivp(
    lambda t, y: deb_ode(t, y, T, feed_g, O2, disease, TAN, NO2),
    [0, days],
    EVHR0,
    t_eval=t
)

E, V, EH, ER = sol.y
W = weight(E, V, ER)

# ================== SURVIVAL & BIOMASS ==================
surv = np.zeros_like(t, dtype=float)
bio  = np.zeros_like(t, dtype=float)

surv[0] = 1.0
bio[0]  = n_fish * W[0] / 1000

base_mu = 0.002  # realistic daily mortality

for i in range(1, len(t)):

    density = (n_fish * surv[i-1] * W[i-1] / 1000) / pond_vol

    # Soft oxygen stress (bounded)
    O2_eff = max(3.0, O2 - 0.01 * density)

    stress = mortality(O2_eff, disease, TAN, NO2)
    stress = min(stress, 2.0)   # 🔒 cap stress

    mu = base_mu * (1 + stress)

    # Incremental survival (THIS is the key fix)
    surv[i] = surv[i-1] * np.exp(-mu)

    bio[i] = n_fish * surv[i] * W[i] / 1000

# ================== FEED, FCR ==================
TC = temp_corr(T + 273.15)
pA = feed_to_f(feed_g, V, T, O2, disease, TAN, NO2) * DEB["p_Am"] * V**(2/3) * TC
feed_intake = pA / FEED_ENERGY
feed_eaten_cum = np.cumsum(feed_intake) * n_fish / 1000
weight_gain = bio - bio[0] + 1e-8
FCR_series = feed_eaten_cum / weight_gain

# ================== WASTE & NITROGEN ==================
feed_offered = feed_g * n_fish * t / 1000
waste_feed = np.maximum(feed_offered - feed_eaten_cum, 0)
N_input = feed_eaten_cum * N_CONTENT_FEED
N_retained = (bio - bio[0]) * N_RETENTION * 0.16
N_waste = N_input - N_retained

# ================== ECONOMICS ==================
price, feed_cost = 2.5, 1.2
profit = bio * price - feed_eaten_cum * feed_cost
best_day = int(np.argmax(profit))

# ================== ORIGINAL GRAPHS ==================
st.subheader("Individual Growth")
fig1,ax1 = plt.subplots()
ax1.plot(t,W)
ax1.set_xlabel("Days")
ax1.set_ylabel("Weight (g)")
st.pyplot(fig1)

st.subheader("Survival Curve")
fig2,ax2 = plt.subplots()
ax2.plot(t,surv)
ax2.set_xlabel("Days")
ax2.set_ylabel("Survival Probability")
st.pyplot(fig2)

st.subheader("Production Yield")
fig3,ax3 = plt.subplots()
ax3.plot(t,bio)
ax3.set_xlabel("Days")
ax3.set_ylabel("Biomass (kg)")
st.pyplot(fig3)


# ================== OUTPUT ==================
st.metric("Final Weight (g)", f"{W[-1]:.1f}")
st.metric("Harvest Biomass (kg)", f"{bio[best_day]:.1f}")
st.metric("Optimal Harvest Day", best_day)

biomass_gain = max(bio[best_day] - bio[0], 1e-6)
FCR_best = feed_eaten_cum[best_day] / biomass_gain

st.metric("FCR", f"{FCR_best:.2f}")


# ================== FAO VALIDATION ==================
FAO_days = np.array([0, 30, 60, 90, 120])
FAO_weights = np.array([init_w, 120, 350, 600, 900])

fig, ax = plt.subplots()
ax.plot(t, W, label="Model")
ax.plot(FAO_days, FAO_weights, "o--", label="FAO")
ax.set_xlabel("Days")
ax.set_ylabel("Weight (g)")
ax.legend()
st.pyplot(fig)
# ================== AUTO FEED OPTIMIZATION ==================
st.subheader(" Automatic Feed Optimization")
if st.button("Optimize Feed"):
    feed_range = np.linspace(0.5, 8, 20)
    FCRs = []
    for fg in feed_range:
        sol = solve_ivp(
            lambda t, y: deb_ode(t, y, T, fg, O2, disease, TAN, NO2),
            [0, days],
            EVHR0,
            t_eval=t
        )
        E_, V_, _, ER_ = sol.y
        W_ = weight(E_, V_, ER_)
        bio_ = n_fish * W_ / 1000
        pA_ = feed_to_f(fg, V_, T, O2, disease, TAN, NO2) * DEB["p_Am"] * V_**(2/3) * TC
        feed_ = np.cumsum(pA_ / FEED_ENERGY) * n_fish / 1000
        FCRs.append(feed_[-1] / (bio_[-1] - bio_[0] + 1e-8))

    best = feed_range[np.argmin(FCRs)]
    st.success(f"Optimal Feed ≈ {best:.2f} g/fish/day")
# ================== CALIBRATION ==================
st.subheader("Farm Calibration")
obs_w = st.number_input("Observed Avg Weight (g)", 0.0, 5000.0)
obs_fcr = st.number_input("Observed FCR", 0.5, 5.0)

prev = farm_memory.get(MEM_KEY, {})
if obs_w > 0 and obs_fcr > 0:
    ew = obs_w / (W[best_day] + 1e-8)
    ef = obs_fcr / (FCR_series[best_day] + 1e-8)
    lr = min(0.3, 0.1 + 0.02 * len(prev.get("history", [])))
    new_pAm = DEB["p_Am"] * np.exp(lr * (ew - 1))
    new_pM  = DEB["p_M"]  * np.exp(lr * (ef - 1))
    err = abs(ew - 1) + abs(ef - 1)

    if prev and err > prev.get("last_error", np.inf):
        st.warning("Rollback applied")
    else:
        DEB["p_Am"], DEB["p_M"] = new_pAm, new_pM
        hist = prev.get("history", [])
        hist.append({"date": str(date.today()), "ew": ew, "ef": ef})
        farm_memory[MEM_KEY] = {
            "params": DEB,
            "history": hist[-30:],
            "last_error": err
        }
        json.dump(farm_memory, open(MEMORY_FILE, "w"), indent=2)
        st.success("Calibration stored")
#================== CONFIDENCE SCORE (REAL FARM CALIBRATION) ==================
if MEM_KEY in farm_memory and farm_memory[MEM_KEY]["history"]:

    hist = farm_memory[MEM_KEY]["history"]
    n = len(hist)

    recent = hist[-5:]
    err_terms = []

    for h in recent:

        if "sim_weight" in h and "obs_weight" in h:
            w_err = abs(h["sim_weight"] - h["obs_weight"]) / max(h["obs_weight"], 1e-6)
        else:
            continue

        if "sim_fcr" in h and "obs_fcr" in h:
            f_err = abs(h["sim_fcr"] - h["obs_fcr"]) / max(h["obs_fcr"], 1e-6)
        else:
            continue

        err_terms.append(w_err + f_err)

    if err_terms:
        err = np.mean(err_terms)
        skill = np.exp(-2.5 * err)
    else:
        skill = 0.5

    depth = np.tanh(n / 6)

    confidence = np.clip(
        0.3 + 0.7 * skill * depth,
        0.3,
        0.85
    )

else:
    confidence = 0.6

st.metric("Model Confidence", f"{confidence:.2f}")
st.write("Learning cycles:", len(farm_memory.get(MEM_KEY, {}).get("history", [])))



# ================== MULTI-CYCLE LEARNING VISUALIZATION ==================
# ================== MULTI-CYCLE LEARNING VISUALIZATION (ROBUST) ==================
if MEM_KEY in farm_memory and farm_memory[MEM_KEY].get("history"):

    st.subheader("Multi-Cycle Learning Trends")

    hist = farm_memory[MEM_KEY]["history"]

    weight_err = []
    fcr_err    = []
    total_err  = []
    cycles     = []

    for i, h in enumerate(hist):
        if "weight_error" in h and "fcr_error" in h:
            weight_err.append(h["weight_error"])
            fcr_err.append(h["fcr_error"])
            total_err.append(h.get("total_error",
                                   h["weight_error"] + h["fcr_error"]))
            cycles.append(i + 1)

    if not cycles:
        st.info("No calibration error history available yet.")
    else:
        fig4, ax4 = plt.subplots()

        ax4.plot(cycles, weight_err, marker="o", label="Weight Error")
        ax4.plot(cycles, fcr_err, marker="o", label="FCR Error")
        ax4.plot(cycles, total_err, linestyle="--", label="Total Error")

        ax4.set_xlabel("Calibration Cycle")
        ax4.set_ylabel("Normalized Error (↓ is better)")
        ax4.set_title("Per-Pond Learning from Calibration")
        ax4.legend()
        ax4.grid(alpha=0.3)

        st.pyplot(fig4)

1
    
# ================== SAVE RESULTS ==================
results = pd.DataFrame({
    "Day":t,
    "Weight_g":W,
    "Survival":surv,
    "Biomass_kg":bio,
    "FCR":FCR_series,
    "Profit_$":profit
})
results_path = "results/output.csv"  # or .txt, .json, etc.

os.makedirs(os.path.dirname(results_path), exist_ok=True) \
    if os.path.dirname(results_path) else None

results.to_csv(results_path, index=False)
os.makedirs(os.path.dirname(results_path),exist_ok=True) if os.path.dirname(results_path) else None
results.to_csv(results_path,index=False)

st.info("The model is developed based on DEB theory, It learns, calibrates, and grows with usage")

# ================== LOGGING ==================
st.subheader("Water Quality Log")
log = pd.DataFrame({
    "T": [T], "O2":[O2], "TAN":[TAN], "NO2":[NO2], "Disease":[disease]
})
st.dataframe(log)

# ================== SAVE OPTION ==================
st.subheader("Save Results")
csv_bytes = results.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Simulation Results (CSV)",
    data=csv_bytes,
    file_name=f"{farm_id}_{pond_id}_results.csv",
    mime="text/csv"
)

# ================== TRACEABILITY UI (ADDED) ==================
st.subheader(" Product Traceability & QR Code")

with st.expander("Enter Farm Traceability Details"):
    farm_name_t = st.text_input("Farm Name")
    farmer_name_t = st.text_input("Farmer Name")
    location_t = st.text_input("Location")
    species_t = st.text_input("Species", "Tilapia")
    harvest_date_t = st.date_input("Harvest Date")
    production_system_t = st.selectbox(
        "Production System",
        ["Pond", "Cage", "RAS", "Biofloc"]
    )

    if st.button("Generate Traceability QR Code"):
        trace_id = str(uuid.uuid4())

        trace_db[trace_id] = {
            "farm_name": farm_name_t,
            "farmer_name": farmer_name_t,
            "location": location_t,
            "species": species_t,
            "harvest_date": str(harvest_date_t),
            "production_system": production_system_t,
            "farm_id": farm_id,
            "pond_id": pond_id,
            "confidence_score": float(confidence),
            "generated_on": str(date.today())
        }

        with open(TRACE_FILE, "w") as f:
            json.dump(trace_db, f, indent=2)

        public_url = f"?trace_id={trace_id}"

        qr = qrcode.make(public_url)
        buf = BytesIO()
        qr.save(buf, format="PNG")

        st.success("QR Code Generated")
        st.image(buf.getvalue(), caption="Scan to View Farm Details")
        st.code(public_url)
