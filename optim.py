import pandas as pd
import pulp
import numpy as np
import matplotlib.pyplot as plt

# 1. Load Data
df = pd.read_csv("caseA_smart_home_30min_winter_optional.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d-%m-%Y %H:%M')
N = len(df)

# 2. Extract and Convert Parameters (kW to kWh for 30-min intervals)
pv = df['pv_kw'] * 0.5
base = df['base_load_kw'] * 0.5
heat = df['elec_heat_demand_kw'] * 0.5
tariff = df['import_tariff_gbp_per_kwh']
export_price = df['export_price_gbp_per_kwh']

# 3. Initialize Optimization Problem
prob = pulp.LpProblem("Home_Energy_Optimization", pulp.LpMinimize)

# 4. Define Variables
g_imp = pulp.LpVariable.dicts("grid_import", range(N), lowBound=0)
g_exp = pulp.LpVariable.dicts("grid_export", range(N), lowBound=0)

# Constraint: Max charge/discharge of 1.25kWh per 30 minutes
p_ch = pulp.LpVariable.dicts("battery_charge", range(N), lowBound=0, upBound=1.25)
p_dis = pulp.LpVariable.dicts("battery_discharge", range(N), lowBound=0, upBound=1.25)

# Constraint: Battery capacity remains between 0 and 5 kWh
soc = pulp.LpVariable.dicts("state_of_charge", range(N), lowBound=0, upBound=5.0)

# 5. Objective Function
prob += pulp.lpSum([g_imp[t] * tariff[t] - g_exp[t] * export_price[t] for t in range(N)])

# 6. Define Constraints
for t in range(N):
    # Energy Balance Constraint
    prob += (g_imp[t] + pv[t] + p_dis[t] == base[t] + heat[t] + g_exp[t] + p_ch[t], f"Balance_{t}")
    
    # State of Charge Constraint (0.95 Efficiency)
    if t == 0:
        # Constraint: Initial charge is 2.5 kWh
        prob += (soc[t] == 2.5 + p_ch[t] * 0.95 - p_dis[t] / 0.95, f"SOC_{t}")
    else:
        prob += (soc[t] == soc[t-1] + p_ch[t] * 0.95 - p_dis[t] / 0.95, f"SOC_{t}")

# Constraint: Final SOC must exceed or equal initial SOC
prob += (soc[N-1] >= 2.5, "Final_SOC_Requirement")

# 7. Solve
prob.solve()

# Extract Results to DataFrame
df['soc'] = [soc[t].varValue for t in range(N)]
df['g_imp'] = [g_imp[t].varValue for t in range(N)]
df['g_exp'] = [g_exp[t].varValue for t in range(N)]
df['p_ch'] = [p_ch[t].varValue for t in range(N)]
df['p_dis'] = [p_dis[t].varValue for t in range(N)]

# ==========================================
# 8. GENERATE PLOTS
# ==========================================

# Plot 1: Battery SOC Time Series
plt.figure(figsize=(12, 4))
plt.plot(df['timestamp'], df['soc'], label='State of Charge', color='purple')
plt.axhline(y=2.5, color='r', linestyle='--', label='Initial SOC Constraint (2.5 kWh)')
plt.title('Battery State of Charge [Optimisation]')
plt.ylabel('SOC (kWh)')
plt.xlabel('Time')
plt.legend()
plt.tight_layout()
plt.show()

# Plot 2: Total Energy Sources and Loads
total_base = np.sum(base)
total_heat = np.sum(heat)
total_solar = np.sum(pv)
total_g_imp = np.sum(df['g_imp'])
total_b_dis = np.sum(df['p_dis'])
total_b_ch = np.sum(df['p_ch'])
total_g_exp = np.sum(df['g_exp'])

categories = ['Sources:\nGrid Import', 'Sources:\nSolar', 'Sources:\nBatt Discharge', 
              'Loads:\nBase+Heat', 'Loads:\nBatt Charge', 'Loads:\nGrid Export']
values = [total_g_imp, total_solar, total_b_dis, total_base+total_heat, total_b_ch, total_g_exp]
colors = ['blue', 'gold', 'purple', 'red', 'orange', 'green']

plt.figure(figsize=(10, 5))
bars = plt.bar(categories, values, color=colors)
plt.title('Total Energy Sources and Loads (kWh) [Optimisation]')
plt.ylabel('Energy (kWh)')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 5, f"{yval:.1f}", ha='center', va='bottom')
plt.tight_layout()
plt.show()

# Plot 3: Financial Breakdown
total_import_cost = np.sum(df['g_imp'] * tariff)
total_export_revenue = np.sum(df['g_exp'] * export_price)

plt.figure(figsize=(6, 5))
bars = plt.bar(['Total Import Cost', 'Total Export Revenue'], 
               [total_import_cost, total_export_revenue], color=['red', 'green'])
plt.title('Financial Breakdown: Import Cost vs. Export Revenue [Optimisation]')
plt.ylabel('Amount (£)')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"£{yval:.2f}", ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.show()

print(f"Final SOC: {df['soc'].iloc[-1]:.2f} kWh")
print(f"Total Import Cost: £{total_import_cost:.2f}")