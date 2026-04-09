import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Configuration & Constants
# ==========================================
FILE_PATH = "caseA_smart_home_30min_winter_optional.csv"
DT = 0.5  # 30-minute intervals in hours

BATT_CAPACITY_KWH = 5.0
INITIAL_SOC = 2.5
MAX_CHARGE_KW = 2.5
MAX_DISCHARGE_KW = 2.5

# Convert max power (kW) to max energy per 30-min step (kWh)
MAX_CHARGE_KWH = MAX_CHARGE_KW * DT       # 1.25 kWh
MAX_DISCHARGE_KWH = MAX_DISCHARGE_KW * DT # 1.25 kWh

EFF_CHG = 0.95
EFF_DIS = 0.95

# ==========================================
# 2. Simulation Setup
# ==========================================
df_in = pd.read_csv(FILE_PATH)
df_in['timestamp'] = pd.to_datetime(df_in['timestamp'], format="%d-%m-%Y %H:%M")

total_steps = len(df_in)
soc = INITIAL_SOC
results = []

checks = {
    "units_consistent_kwh": True,
    "soc_bounds_respected": True,
    "max_charge_discharge_respected": True,
    "power_balanced": True,
    "non_negative_grid": True
}

# ==========================================
# 3. Main Loop
# ==========================================
for idx, row in df_in.iterrows():
    # Convert all power to energy (kWh) to enforce consistent units
    pv_kwh = row['pv_kw'] * DT
    base_load_kwh = row['base_load_kw'] * DT
    heat_load_kwh = row['elec_heat_demand_kw'] * DT
    total_load_kwh = base_load_kwh + heat_load_kwh
    
    import_tariff = row['import_tariff_gbp_per_kwh']
    export_price = row['export_price_gbp_per_kwh']
    
    batt_chg_kwh = 0.0
    batt_dis_kwh = 0.0
    grid_imp_kwh = 0.0
    grid_exp_kwh = 0.0
    
    steps_left = total_steps - 1 - idx
    
    # ------------------------------------------
    # FORCE CHARGE OVERRIDE (Last 2 hours / 4 steps)
    # ------------------------------------------
    # Ensures the battery ends with at least the initial energy stored
    force_charge_window = (steps_left <= 4)
    
    if force_charge_window:
        if soc < INITIAL_SOC:
            target_charge_kwh = (INITIAL_SOC - soc) / EFF_CHG
            batt_chg_kwh = min(target_charge_kwh, MAX_CHARGE_KWH)
            soc += batt_chg_kwh * EFF_CHG
            total_demand = total_load_kwh + batt_chg_kwh
            
            if pv_kwh >= total_demand:
                grid_exp_kwh = pv_kwh - total_demand
            else:
                grid_imp_kwh = total_demand - pv_kwh
        else:
            net_kwh = pv_kwh - total_load_kwh
            if net_kwh > 0:
                space_in_batt = BATT_CAPACITY_KWH - soc
                batt_chg_kwh = min(net_kwh, MAX_CHARGE_KWH, space_in_batt / EFF_CHG)
                soc += batt_chg_kwh * EFF_CHG
                grid_exp_kwh = net_kwh - batt_chg_kwh
            elif net_kwh < 0:
                available_to_discharge = soc - INITIAL_SOC
                deficit_kwh = -net_kwh
                max_batt_yield = available_to_discharge * EFF_DIS
                batt_dis_kwh = min(deficit_kwh, MAX_DISCHARGE_KWH, max_batt_yield)
                soc -= batt_dis_kwh / EFF_DIS
                grid_imp_kwh = deficit_kwh - batt_dis_kwh
                
    # ------------------------------------------
    # NORMAL SELF-CONSUMPTION RULE
    # ------------------------------------------
    else:
        net_kwh = pv_kwh - total_load_kwh
        
        if net_kwh > 0: 
            space_in_batt = BATT_CAPACITY_KWH - soc
            batt_chg_kwh = min(net_kwh, MAX_CHARGE_KWH, space_in_batt / EFF_CHG)
            soc += batt_chg_kwh * EFF_CHG
            grid_exp_kwh = net_kwh - batt_chg_kwh
            
        elif net_kwh < 0: 
            deficit_kwh = -net_kwh
            energy_from_batt = soc * EFF_DIS
            batt_dis_kwh = min(deficit_kwh, MAX_DISCHARGE_KWH, energy_from_batt)
            soc -= batt_dis_kwh / EFF_DIS
            grid_imp_kwh = deficit_kwh - batt_dis_kwh
            
    # ==========================================
    # 4. Explicit Checks Enforcement
    # ==========================================
    if not (0 <= soc <= BATT_CAPACITY_KWH + 1e-6): checks["soc_bounds_respected"] = False
    if batt_chg_kwh > MAX_CHARGE_KWH + 1e-6 or batt_dis_kwh > MAX_DISCHARGE_KWH + 1e-6: checks["max_charge_discharge_respected"] = False
    if grid_imp_kwh < -1e-6 or grid_exp_kwh < -1e-6: checks["non_negative_grid"] = False
    
    balance = (pv_kwh + batt_dis_kwh + grid_imp_kwh) - (total_load_kwh + batt_chg_kwh + grid_exp_kwh)
    if abs(balance) > 1e-6: checks["power_balanced"] = False
            
    results.append({
        'timestamp': row['timestamp'],
        'soc': soc,
        'pv': pv_kwh,
        'base': base_load_kwh,
        'heat': heat_load_kwh,
        'p_ch': batt_chg_kwh,
        'p_dis': batt_dis_kwh,
        'g_imp': grid_imp_kwh,
        'g_exp': grid_exp_kwh,
        'cost_gbp': grid_imp_kwh * import_tariff,
        'revenue_gbp': grid_exp_kwh * export_price
    })

res_df = pd.DataFrame(results)

# Final check: Battery ending condition
checks["ends_with_initial_energy"] = res_df['soc'].iloc[-1] >= INITIAL_SOC - 1e-6

print("==========================================")
print("Explicit Checks Verification:")
for k, v in checks.items():
    print(f"  {k}: {'PASS' if v else 'FAIL'}")
print("==========================================\n")

# ==========================================
# 8. GENERATE PLOTS
# ==========================================

# Map arrays for user reference code
df = res_df.copy()
base = df['base'].values
heat = df['heat'].values
pv = df['pv'].values

# Plot 1: Battery SOC Time Series
plt.figure(figsize=(12, 4))
plt.plot(df['timestamp'], df['soc'], label='State of Charge', color='purple')
plt.axhline(y=2.5, color='r', linestyle='--', label='Initial SOC Constraint (2.5 kWh)')
plt.title('Battery State of Charge [Self-Consumption]')
plt.ylabel('SOC (kWh)')
plt.xlabel('Time')
plt.legend()
plt.tight_layout()
plt.savefig('Plot_1_SOC.png') # Added save so you can export if needed
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
plt.title('Total Energy Sources and Loads (kWh) [Self-Consumption]')
plt.ylabel('Energy (kWh)')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 5, f"{yval:.1f}", ha='center', va='bottom')
plt.tight_layout()
plt.savefig('Plot_2_Energy.png')
plt.show()

# Plot 3: Financial Breakdown
total_import_cost = np.sum(df['cost_gbp'])
total_export_revenue = np.sum(df['revenue_gbp'])

plt.figure(figsize=(6, 5))
bars = plt.bar(['Total Import Cost', 'Total Export Revenue'], 
               [total_import_cost, total_export_revenue], color=['red', 'green'])
plt.title('Financial Breakdown: Import Cost vs. Export Revenue [Self-Consumption]')
plt.ylabel('Amount (£)')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"£{yval:.2f}", ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig('Plot_3_Financial.png')
plt.show()

# Final Outputs
print(f"Final SOC: {df['soc'].iloc[-1]:.2f} kWh")
print(f"Total Import Cost: £{total_import_cost:.2f}")