
import torch

print(f'PyTorch  : {torch.__version__}')
print(f'CUDA     : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU      : {torch.cuda.get_device_name(0)}')
    print(f'CUDA ver : {torch.version.cuda}')
else:
    print('No GPU — go to Runtime > Change runtime type > T4 GPU')

## Cell 2 — Install PyTorch Geometric

import torch, subprocess, sys

torch_ver = torch.__version__.split('+')[0]
cuda_tag  = 'cu' + torch.version.cuda.replace('.', '') if torch.cuda.is_available() else 'cpu'
wheel_url = f'https://data.pyg.org/whl/torch-{torch_ver}+{cuda_tag}.html'

print(f'Installing PyG for torch {torch_ver} + {cuda_tag}')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'torch_geometric'], check=True)

for pkg in ['pyg_lib', 'torch_scatter', 'torch_sparse', 'torch_cluster', 'torch_spline_conv']:
    r = subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '-q', pkg, '-f', wheel_url],
        capture_output=True, text=True
    )
    print(f'  {pkg:<28} {"OK" if r.returncode == 0 else "skipped"}')

import torch_geometric
print(f'\ntorch_geometric : {torch_geometric.__version__}')

## Cell 3 — Install Flask, redis-py, and other utilities



import flask, redis as redis_lib
print(f'Flask   : {flask.__version__}')
print(f'redis   : {redis_lib.__version__}')
print('All server libraries ready.')


from google.colab import files

uploaded = files.upload()

csv_filename = list(uploaded.keys())[0]
print(f'Uploaded: {csv_filename}')



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# ── Load ─────────────────────────────────────────────────────────
df_raw = pd.read_csv(csv_filename)

print('='*55)
print('DATASET OVERVIEW')
print('='*55)
print(f'Shape            : {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns')
print(f'Columns          : {list(df_raw.columns)}')
print()

print('── Data types ──')
print(df_raw.dtypes.to_string())
print()

print('── First 5 rows ──')
print(df_raw.head().to_string())
print()

# ── Missing values ────────────────────────────────────────────────
print('── Missing values ──')
missing = df_raw.isnull().sum()
print(missing.to_string())
print(f'Total missing: {missing.sum()}')
print()

# ── Duplicates ────────────────────────────────────────────────────
dupes = df_raw.duplicated().sum()
print(f'── Duplicate rows: {dupes} ──')
print()

# ── Numeric summary ───────────────────────────────────────────────
num_cols = ['CarCount','BikeCount','BusCount','TruckCount','Total']
print('── Numeric summary ──')
print(df_raw[num_cols].describe().round(2).to_string())
print()

# ── Temporal structure ────────────────────────────────────────────
print('── Temporal structure ──')
print(f'Unique dates     : {df_raw["Date"].nunique()} days (dates 1–31)')
print(f'Unique times     : {df_raw["Time"].nunique()} slots (96 × 15min = 24h/day)')
print(f'Unique days      : {sorted(df_raw["Day of the week"].unique())}')
print(f'Rows per day     : {df_raw.groupby("Date").size().unique()} (perfectly balanced)')
print()

# ── Class distribution ────────────────────────────────────────────
print('── Traffic Situation distribution ──')
dist = df_raw['Traffic Situation'].value_counts()
for k, v in dist.items():
    print(f'  {k:<8}: {v:>5}  ({100*v/len(df_raw):.1f}%)')
print()

# ── Integrity check: Total == sum of counts ───────────────────────
computed_total = df_raw[['CarCount','BikeCount','BusCount','TruckCount']].sum(axis=1)
mismatch = (computed_total != df_raw['Total']).sum()
print(f'── Total integrity check: {mismatch} mismatches (0 = perfect) ──')
print()

# ── Outlier detection (IQR method) ───────────────────────────────
print('── Outlier detection (IQR method) ──')
for col in num_cols:
    Q1 = df_raw[col].quantile(0.25)
    Q3 = df_raw[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df_raw[col] < Q1 - 1.5*IQR) | (df_raw[col] > Q3 + 1.5*IQR)).sum()
    print(f'  {col:<12}: {outliers:>3} outliers  (Q1={Q1:.0f}  Q3={Q3:.0f}  IQR={IQR:.0f})')
print()

# ── Autocorrelation (key for temporal modelling) ──────────────────
totals = df_raw['Total'].values
mean_t = totals.mean()
var_t  = ((totals - mean_t)**2).mean()
for lag in [1, 4, 8, 16, 96]:
    acf = ((totals[:-lag] - mean_t) * (totals[lag:] - mean_t)).mean() / var_t
    label = {1:'15 min', 4:'1 hour', 8:'2 hours', 16:'4 hours', 96:'24 hours (next day)'}[lag]
    print(f'  Autocorrelation lag {lag:>2} ({label:<20}): {acf:.3f}')

print()
print('EDA complete. No missing values. No duplicates. Dataset is clean and ready.')



import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

fig = plt.figure(figsize=(16, 12))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

# Panel 1 — Average traffic by hour of day
ax1 = fig.add_subplot(gs[0, 0])
df_raw['datetime_str'] = df_raw['Time']
df_raw['hour'] = pd.to_datetime(df_raw['Time'], format='%I:%M:%S %p').dt.hour
hourly_avg = df_raw.groupby('hour')['Total'].mean()
ax1.plot(hourly_avg.index, hourly_avg.values, linewidth=2.5, color='#185FA5', marker='o', markersize=4)
ax1.fill_between(hourly_avg.index, hourly_avg.values, alpha=0.15, color='#185FA5')
ax1.set_title('Average total traffic by hour of day', fontsize=12, fontweight='500', pad=10)
ax1.set_xlabel('Hour'); ax1.set_ylabel('Avg total vehicles')
ax1.set_xticks(range(0, 24, 2))
ax1.grid(True, alpha=0.3)

# Panel 2 — Vehicle type composition
ax2 = fig.add_subplot(gs[0, 1])
vehicle_means = df_raw[['CarCount','BikeCount','BusCount','TruckCount']].mean()
colors = ['#534AB7', '#1D9E75', '#D85A30', '#BA7517']
bars = ax2.bar(vehicle_means.index, vehicle_means.values, color=colors, width=0.6)
for bar, val in zip(bars, vehicle_means.values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val:.1f}', ha='center', va='bottom', fontsize=10)
ax2.set_title('Mean vehicle count by type', fontsize=12, fontweight='500', pad=10)
ax2.set_ylabel('Mean count per 15-min slot')
ax2.tick_params(axis='x', rotation=15)
ax2.grid(True, alpha=0.3, axis='y')

# Panel 3 — Traffic Situation distribution
ax3 = fig.add_subplot(gs[1, 0])
sit_order  = ['low', 'normal', 'high', 'heavy']
sit_counts = df_raw['Traffic Situation'].value_counts().reindex(sit_order)
sit_colors = ['#1D9E75', '#185FA5', '#BA7517', '#D85A30']
wedges, texts, autotexts = ax3.pie(
    sit_counts.values,
    labels=[f'{s}\n({v})' for s, v in zip(sit_order, sit_counts.values)],
    colors=sit_colors,
    autopct='%1.1f%%',
    startangle=90,
    pctdistance=0.75
)
ax3.set_title('Traffic situation distribution (2,976 obs)', fontsize=12, fontweight='500', pad=10)

# Panel 4 — Day-of-week heatmap
ax4 = fig.add_subplot(gs[1, 1])
day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
df_raw['hour_slot'] = df_raw['hour'] // 3  # 8 slots of 3h each
slot_labels = ['0-3h','3-6h','6-9h','9-12h','12-15h','15-18h','18-21h','21-24h']
heatmap_data = df_raw.groupby(['Day of the week','hour_slot'])['Total'].mean().unstack()
heatmap_data = heatmap_data.reindex(day_order)
im = ax4.imshow(heatmap_data.values, aspect='auto', cmap='YlOrRd')
ax4.set_xticks(range(8)); ax4.set_xticklabels(slot_labels, rotation=25, ha='right', fontsize=9)
ax4.set_yticks(range(7)); ax4.set_yticklabels(day_order, fontsize=9)
ax4.set_title('Avg traffic by day × 3h block (heatmap)', fontsize=12, fontweight='500', pad=10)
plt.colorbar(im, ax=ax4, shrink=0.8, label='Avg vehicles')

fig.suptitle('Traffic.csv — Exploratory Data Analysis', fontsize=14, fontweight='500', y=1.01)
plt.savefig('eda_charts.png', dpi=150, bbox_inches='tight')
plt.show()
print('EDA charts saved to eda_charts.png')




import pandas as pd
import numpy as np
import torch
import math
# ── Step 1: Load and sort chronologically ────────────────────────
df = pd.read_csv(csv_filename)
print(f'Raw shape: {df.shape}')
df['datetime_parsed'] = pd.to_datetime(df['Time'], format='%I:%M:%S %p')
df['hour_float'] = df['datetime_parsed'].dt.hour + df['datetime_parsed'].dt.minute / 60.0
day_to_int = {'Monday':0,'Tuesday':1,'Wednesday':2,'Thursday':3,
               'Friday':4,'Saturday':5,'Sunday':6}
df['day_int'] = df['Day of the week'].map(day_to_int)
df['slot_idx'] = (df['Date'].astype(int) - 1) * 96 + (df['hour_float'] * 4).astype(int)
df = df.sort_values('slot_idx').reset_index(drop=True)
print(f'Sorted shape: {df.shape}  |  slot_idx range: {df.slot_idx.min()}–{df.slot_idx.max()}')
# ── Step 2: Cyclical time encoding ───────────────────────────────
df['hour_sin'] = np.sin(2 * np.pi * df['hour_float'] / 24.0)
df['hour_cos'] = np.cos(2 * np.pi * df['hour_float'] / 24.0)
df['day_sin']  = np.sin(2 * np.pi * df['day_int']   / 7.0)
df['day_cos']  = np.cos(2 * np.pi * df['day_int']   / 7.0)
# ── Step 3: Encode Traffic Situation as ordinal severity ─────────
sit_to_ord = {'low':0, 'normal':1, 'high':2, 'heavy':3}
df['traffic_severity'] = df['Traffic Situation'].map(sit_to_ord)
# ── Step 4: Z-score normalise count features ─────────────────────
count_cols = ['CarCount', 'BikeCount', 'BusCount', 'TruckCount', 'Total']
stats = {}
for col in count_cols:
    mu, sigma    = df[col].mean(), df[col].std()
    stats[col]   = (mu, sigma)
    df[col+'_z'] = (df[col] - mu) / sigma
    print(f'  {col:<12}: mean={mu:.2f}  std={sigma:.2f}  → normalised to {col}_z')
# ── Step 5: Create 8 city zones ──────────────────────────────────
print()
print('Building 8 city zones...')
T = len(df)  # 2976 timesteps
zone_demand_raw = {
    0: 0.55*df['CarCount'].values   + 0.15*df['BikeCount'].values + 0.15*df['BusCount'].values  + 0.15*df['TruckCount'].values,
    1: 0.15*df['CarCount'].values   + 0.55*df['BikeCount'].values + 0.15*df['BusCount'].values  + 0.15*df['TruckCount'].values,
    2: 0.15*df['CarCount'].values   + 0.15*df['BikeCount'].values + 0.55*df['BusCount'].values  + 0.15*df['TruckCount'].values,
    3: 0.15*df['CarCount'].values   + 0.15*df['BikeCount'].values + 0.15*df['BusCount'].values  + 0.55*df['TruckCount'].values,
}
for z in range(4):
    base = zone_demand_raw[z]
    zone_demand_raw[z + 4] = np.concatenate([[base[0]], base[:-1]])
zone_demand_norm = {}
zone_stats = {}
for z, signal in zone_demand_raw.items():
    mu, sigma = signal.mean(), signal.std()
    zone_stats[z] = (mu, sigma)
    zone_demand_norm[z] = (signal - mu) / sigma
supply_raw = 3 - df['traffic_severity'].values
supply_norm = (supply_raw - supply_raw.mean()) / supply_raw.std()
sev_norm = (df['traffic_severity'].values - df['traffic_severity'].mean()) / df['traffic_severity'].std()
# ── Step 6: ENGINEERED FEATURES (NEW) ────────────────────────────
# These give the model pre-computed trend information:
#   [7]  rolling_mean_4  : mean demand over last 4 steps (1 hour) — smoothed trend
#   [8]  rolling_std_4   : volatility over last 4 steps — is demand stable or erratic?
#   [9]  momentum        : demand[t] - demand[t-4] — is demand rising or falling?
#   [10] diff_1          : demand[t] - demand[t-1] — instant rate of change
ROLL_WINDOW = 4  # 4 × 15min = 1 hour rolling window
zone_rolling_mean = {}
zone_rolling_std  = {}
zone_momentum     = {}
zone_diff1        = {}
for z in range(8):
    signal = zone_demand_norm[z]
    # Rolling mean (1-hour)
    rm = pd.Series(signal).rolling(ROLL_WINDOW, min_periods=1).mean().values
    zone_rolling_mean[z] = (rm - rm.mean()) / (rm.std() + 1e-8)
    # Rolling std (1-hour volatility)
    rs = pd.Series(signal).rolling(ROLL_WINDOW, min_periods=1).std().fillna(0).values
    zone_rolling_std[z] = (rs - rs.mean()) / (rs.std() + 1e-8)
    # Momentum (4-step difference)
    mom = np.zeros_like(signal)
    mom[ROLL_WINDOW:] = signal[ROLL_WINDOW:] - signal[:-ROLL_WINDOW]
    zone_momentum[z] = (mom - mom.mean()) / (mom.std() + 1e-8)
    # Instant diff
    d1 = np.zeros_like(signal)
    d1[1:] = signal[1:] - signal[:-1]
    zone_diff1[z] = (d1 - d1.mean()) / (d1.std() + 1e-8)
print('Engineered features: rolling_mean_4, rolling_std_4, momentum_4, diff_1')
# ── Step 7: Build feature tensor (11 features now) ───────────────
NUM_ZONES     = 8
NODE_FEATURES = 11        # ← CHANGED from 7
TOTAL_STEPS   = T
telemetry = torch.zeros(NUM_ZONES, TOTAL_STEPS, NODE_FEATURES)
for z in range(NUM_ZONES):
    telemetry[z, :, 0]  = torch.tensor(zone_demand_norm[z],   dtype=torch.float)
    telemetry[z, :, 1]  = torch.tensor(supply_norm,           dtype=torch.float)
    telemetry[z, :, 2]  = torch.tensor(df['hour_sin'].values, dtype=torch.float)
    telemetry[z, :, 3]  = torch.tensor(df['hour_cos'].values, dtype=torch.float)
    telemetry[z, :, 4]  = torch.tensor(df['day_sin'].values,  dtype=torch.float)
    telemetry[z, :, 5]  = torch.tensor(df['day_cos'].values,  dtype=torch.float)
    telemetry[z, :, 6]  = torch.tensor(sev_norm,              dtype=torch.float)
    telemetry[z, :, 7]  = torch.tensor(zone_rolling_mean[z],  dtype=torch.float)  # NEW
    telemetry[z, :, 8]  = torch.tensor(zone_rolling_std[z],   dtype=torch.float)  # NEW
    telemetry[z, :, 9]  = torch.tensor(zone_momentum[z],      dtype=torch.float)  # NEW
    telemetry[z, :, 10] = torch.tensor(zone_diff1[z],         dtype=torch.float)  # NEW
print(f'Telemetry tensor shape: {list(telemetry.shape)}')
print(f'  → (zones={NUM_ZONES}, timesteps={TOTAL_STEPS}, features={NODE_FEATURES})')
# ── Step 8: Build zone graph ─────────────────────────────────────
BASE_LAT, BASE_LON = 12.9716, 77.5946
ZONE_COORDS = [
    (BASE_LAT + 0.000, BASE_LON + 0.000),
    (BASE_LAT + 0.002, BASE_LON + 0.001),
    (BASE_LAT + 0.004, BASE_LON + 0.000),
    (BASE_LAT + 0.006, BASE_LON - 0.001),
    (BASE_LAT + 0.001, BASE_LON + 0.002),
    (BASE_LAT + 0.003, BASE_LON + 0.002),
    (BASE_LAT + 0.005, BASE_LON + 0.001),
    (BASE_LAT + 0.007, BASE_LON + 0.000),
]
def build_zone_graph(zone_coords, radius_km=1.0):
    threshold = radius_km / 111.0
    src, dst, w = [], [], []
    for i, (la1, lo1) in enumerate(zone_coords):
        for j, (la2, lo2) in enumerate(zone_coords):
            if i == j: continue
            d = math.sqrt((la1-la2)**2 + (lo1-lo2)**2)
            if d < threshold:
                src.append(i); dst.append(j); w.append(1.0 / (d + 1e-9))
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    weights    = torch.tensor(w, dtype=torch.float).unsqueeze(1)
    if weights.numel() > 0:
        weights = weights / weights.max()
    return edge_index, weights
edge_index, edge_attr = build_zone_graph(ZONE_COORDS, radius_km=1.0)
print(f'Zone graph     : {NUM_ZONES} zones, {edge_index.shape[1]} directed edges')
ZONE_NAMES = ['car-corridor','bike-corridor','bus-corridor','truck-corridor',
               'lag-car','lag-bike','lag-bus','lag-truck']
zone_meta = [
    {'zone_id': f'Z{i:02d}', 'name': ZONE_NAMES[i],
     'lat': ZONE_COORDS[i][0], 'lon': ZONE_COORDS[i][1]}
    for i in range(NUM_ZONES)
]
print()
print('Preprocessing complete.')
print()
print('── Feature vector layout (per zone per timestep) ──')
feat_names = ['demand_z','supply_z','hour_sin','hour_cos','day_sin','day_cos',
              'traffic_sev_z','rolling_mean','rolling_std','momentum','diff_1']
for i, name in enumerate(feat_names):
    t = telemetry[0, :, i]
    print(f'  [{i:>2}] {name:<16}: min={t.min():.3f}  max={t.max():.3f}  mean={t.mean():.3f}')


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import time
class SpatialEncoder(nn.Module):
    """Two-layer GAT — unchanged from v1."""
    def __init__(self, in_channels, hidden_dim, heads=4, dropout=0.2):
        super().__init__()
        self.gat1  = GATConv(in_channels, hidden_dim // heads,
                              heads=heads, dropout=dropout, concat=True)
        self.gat2  = GATConv(hidden_dim, hidden_dim,
                              heads=1, dropout=dropout, concat=False)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.drop  = nn.Dropout(dropout)
    def forward(self, x, edge_index, edge_attr=None):
        h = self.norm1(F.elu(self.gat1(x, edge_index)))
        h = self.drop(h)
        h = self.norm2(F.elu(self.gat2(h, edge_index)))
        return h
class TemporalAttention(nn.Module):
    """
    NEW: Attention over GRU hidden states.
    Instead of just using h[-1] (the last GRU output), this module
    learns WHICH past timesteps matter most for the prediction.
    Example: for predicting 5PM rush-hour demand, the model might
    attend to yesterday's 5PM (high weight) and ignore 3AM (low weight).
    Uses scaled dot-product attention with a learnable query vector.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.query   = nn.Parameter(torch.randn(hidden_dim))  # learnable query
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.scale   = hidden_dim ** 0.5
    def forward(self, gru_outputs):
        """
        gru_outputs: (num_zones, T, hidden_dim) — all GRU hidden states
        returns:     (num_zones, hidden_dim)    — attention-weighted summary
        """
        keys   = self.key_proj(gru_outputs)                      # (zones, T, hidden)
        scores = (keys * self.query).sum(dim=-1) / self.scale    # (zones, T)
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)        # (zones, T, 1)
        context = (gru_outputs * weights).sum(dim=1)             # (zones, hidden)
        return context
class TemporalEncoder(nn.Module):
    """
    UPGRADED: GRU + Temporal Attention.
    Before: returned only h_n[-1] — lost information from earlier timesteps.
    Now:    returns attention-weighted combination of ALL hidden states.
    """
    def __init__(self, in_channels, hidden_dim, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru  = nn.GRU(in_channels, hidden_dim, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.attention = TemporalAttention(hidden_dim)  # NEW
        self.norm = nn.LayerNorm(hidden_dim)
    def forward(self, x_seq):
        all_hidden, _ = self.gru(x_seq)        # all_hidden: (zones, T, hidden)
        attended = self.attention(all_hidden)    # attended: (zones, hidden)  — NEW
        return self.norm(attended)
class PolarisDemandGNN(nn.Module):
    """
    Spatio-Temporal GNN v2 — with residual skip + temporal attention.
    """
    def __init__(self, node_features=11, hidden_dim=128,
                 gat_heads=4, gru_layers=2, horizon=6, dropout=0.2):
        super().__init__()
        self.horizon      = horizon
        self.spatial_enc  = SpatialEncoder(node_features, hidden_dim, gat_heads, dropout)
        self.temporal_enc = TemporalEncoder(node_features, hidden_dim, gru_layers, dropout)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self.skip = nn.Linear(hidden_dim * 2, hidden_dim)  # residual skip
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout * 0.5),  # light dropout before output
            nn.Linear(hidden_dim // 2, horizon),
        )
    def forward(self, x_seq, edge_index, edge_attr=None):
        x_now        = x_seq[:, -1, :]
        spatial_emb  = self.spatial_enc(x_now, edge_index, edge_attr)
        temporal_emb = self.temporal_enc(x_seq)
        combined     = torch.cat([spatial_emb, temporal_emb], dim=-1)
        fused        = self.fusion(combined) + self.skip(combined)
        return self.head(fused)
class DemandDataset(torch.utils.data.Dataset):
    """Sliding-window dataset — unchanged."""
    def __init__(self, telemetry, window=24, horizon=6, demand_idx=0):
        self.tel     = telemetry
        self.window  = window
        self.horizon = horizon
        self.didx    = demand_idx
        self.length  = telemetry.shape[1] - window - horizon + 1
    def __len__(self): return self.length
    def __getitem__(self, idx):
        x = self.tel[:, idx : idx + self.window, :]
        y = self.tel[:, idx + self.window : idx + self.window + self.horizon, self.didx]
        return x, y
def train_one_epoch(model, loader, edge_index, edge_attr, optimiser, device):
    """Combined Huber + MSE loss for balanced convergence."""
    model.train()
    total = 0.0
    for x_b, y_b in loader:
        x_b = x_b.to(device); y_b = y_b.to(device)
        ei  = edge_index.to(device)
        ea  = edge_attr.to(device) if edge_attr is not None else None
        optimiser.zero_grad()
        preds = torch.stack([model(x_b[i], ei, ea) for i in range(x_b.size(0))])
        # Combined loss: Huber (robust near zero) + MSE (penalises large errors more)
        loss = 0.6 * F.huber_loss(preds, y_b, delta=1.0) + 0.4 * F.mse_loss(preds, y_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        total += loss.item()
    return total / len(loader)
@torch.no_grad()
def evaluate(model, loader, edge_index, edge_attr, device):
    """Full metrics: MAE, RMSE, fixed MAPE, R², direction accuracy, Acc@0.5σ."""
    model.eval()
    ps, ts = [], []
    for x_b, y_b in loader:
        ei = edge_index.to(device)
        ea = edge_attr.to(device) if edge_attr is not None else None
        for i in range(x_b.size(0)):
            ps.append(model(x_b[i].to(device), ei, ea).cpu())
            ts.append(y_b[i])
    p = torch.stack(ps); t = torch.stack(ts)
    mae  = F.l1_loss(p, t).item()
    rmse = torch.sqrt(F.mse_loss(p, t)).item()
    # Fixed MAPE: filter near-zero
    mask = torch.abs(t) > 0.1
    mape = (torch.abs(p[mask] - t[mask]) / torch.abs(t[mask])).mean().item() * 100 if mask.sum() > 0 else 0.0
    # R²
    ss_res = ((t - p) ** 2).sum().item()
    ss_tot = ((t - t.mean()) ** 2).sum().item()
    r2 = 1.0 - ss_res / (ss_tot + 1e-8)
    # Direction accuracy
    if p.shape[0] > 1:
        p_diff = p[1:, :, 0] - p[:-1, :, 0]
        t_diff = t[1:, :, 0] - t[:-1, :, 0]
        dir_acc = ((p_diff * t_diff) > 0).float().mean().item() * 100
    else:
        dir_acc = 0.0
    # Threshold accuracy
    acc_05 = (torch.abs(p - t) < 0.5).float().mean().item() * 100
    return {
        'mae': mae, 'rmse': rmse, 'mape': mape,
        'r2': r2, 'dir_acc': dir_acc, 'acc_0.5std': acc_05,
    }
class PolarisInsightSidecar:
    """Inference wrapper — unchanged."""
    def __init__(self, model, edge_index, edge_attr=None, zone_meta=None, device='cpu'):
        self.device     = torch.device(device)
        self.model      = model.to(self.device).eval()
        self.edge_index = edge_index.to(self.device)
        self.edge_attr  = edge_attr.to(self.device) if edge_attr is not None else None
        self.zone_meta  = zone_meta or []
    @torch.no_grad()
    def forecast(self, x_seq):
        return self.model(x_seq.to(self.device), self.edge_index, self.edge_attr)
    def rider_payload(self, x_seq, top_n=8):
        pred    = self.forecast(x_seq).cpu()
        avg     = pred.mean(dim=1)
        max_avg = avg.max().item()
        cutoff  = max_avg * 0.20
        ranked  = avg.argsort(descending=True).tolist()
        zones   = []
        for idx in ranked:
            a = avg[idx].item()
            if a < cutoff or len(zones) >= top_n: break
            meta = self.zone_meta[idx] if idx < len(self.zone_meta) else {}
            heat = 'high' if a >= max_avg*0.70 else 'medium' if a >= max_avg*0.40 else 'low'
            zones.append({
                'zone_id':          meta.get('zone_id', f'Z{idx:02d}'),
                'name':             meta.get('name', ''),
                'lat':              meta.get('lat', 0.0),
                'lon':              meta.get('lon', 0.0),
                'predicted_demand': [round(v, 3) for v in pred[idx].tolist()],
                'avg_demand':       round(a, 3),
                'heat_level':       heat,
            })
        return {'generated_at': int(time.time()), 'horizon_minutes': 90, 'zones': zones}
    def heatmap_payload(self, x_seq):
        pred = self.forecast(x_seq).cpu()
        avg  = pred.mean(dim=1)
        data = {}
        for idx in range(avg.shape[0]):
            meta = self.zone_meta[idx] if idx < len(self.zone_meta) else {}
            data[meta.get('zone_id', f'Z{idx:02d}')] = round(avg[idx].item(), 3)
        return {'layer': 'predicted_demand', 'generated_at': int(time.time()), 'data': data}
print('All model classes defined (v2 — with temporal attention + 11 features).')



import torch
T_WINDOW = 24   # ← CHANGED from 12: 24 × 15min = 6 hours of history
HORIZON  = 6    # predict next 90 minutes
BATCH    = 32
dataset = DemandDataset(telemetry, window=T_WINDOW, horizon=HORIZON, demand_idx=0)
n      = len(dataset)
n_train = int(0.70 * n)
n_val   = int(0.15 * n)
n_test  = n - n_train - n_val
train_ds = torch.utils.data.Subset(dataset, range(0,             n_train))
val_ds   = torch.utils.data.Subset(dataset, range(n_train,       n_train + n_val))
test_ds  = torch.utils.data.Subset(dataset, range(n_train+n_val, n))
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=BATCH, shuffle=False)
test_loader  = torch.utils.data.DataLoader(test_ds,  batch_size=BATCH, shuffle=False)
print(f'Total windows    : {n}')
print(f'Train windows    : {n_train}  (~{n_train*15/60:.0f}h = {n_train/96:.1f} days)')
print(f'Val windows      : {n_val}   (~{n_val*15/60:.0f}h = {n_val/96:.1f} days)')
print(f'Test windows     : {n_test}   (~{n_test*15/60:.0f}h = {n_test/96:.1f} days)')
print(f'Window size      : {T_WINDOW} steps = {T_WINDOW*15/60:.0f} hours of history')
print()
xb, yb = next(iter(train_loader))
print(f'Batch x shape    : {list(xb.shape)}  (batch × zones × window × features)')
print(f'Batch y shape    : {list(yb.shape)}  (batch × zones × horizon)')



import torch
EPOCHS     = 80
LR         = 3e-3
HIDDEN_DIM = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Training on: {device}')
torch.manual_seed(42)
model = PolarisDemandGNN(
    node_features = NODE_FEATURES,   # 11 now
    hidden_dim    = HIDDEN_DIM,
    gat_heads     = 4,
    gru_layers    = 2,
    horizon       = HORIZON,
    dropout       = 0.2,
).to(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Trainable parameters: {total_params:,}')
optimiser = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=5e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimiser, T_0=10, T_mult=2, eta_min=1e-6
)
print(f'Features         : {NODE_FEATURES} (7 base + 4 engineered)')
print(f'Window           : {T_WINDOW} steps ({T_WINDOW*15/60:.0f}h)')
print(f'Temporal encoder : GRU + Attention (attends over all {T_WINDOW} timesteps)')
print('Model, optimiser, scheduler ready.')





import os, copy
PATIENCE     = 15
best_val_mae = float('inf')
best_state   = None
no_improve   = 0
history      = {
    'train_loss': [], 'val_mae': [], 'val_rmse': [], 'val_mape': [],
    'val_r2': [], 'val_dir_acc': [], 'val_acc05': [], 'lr': [],
}
print(f'{"Ep":>4}  {"Train":>8}  {"ValMAE":>8}  {"RMSE":>7}  '
      f'{"MAPE%":>7}  {"R²":>7}  {"Dir%":>6}  {"Acc%":>6}  {"LR":>10}')
print('-' * 78)
for epoch in range(1, EPOCHS + 1):
    train_loss = train_one_epoch(
        model, train_loader, edge_index, edge_attr, optimiser, device
    )
    val_m = evaluate(model, val_loader, edge_index, edge_attr, device)
    scheduler.step()    # ← steps LR schedule once per epoch
    cur_lr = optimiser.param_groups[0]['lr']
    history['train_loss'].append(train_loss)
    history['val_mae'].append(val_m['mae'])
    history['val_rmse'].append(val_m['rmse'])
    history['val_mape'].append(val_m['mape'])
    history['val_r2'].append(val_m['r2'])
    history['val_dir_acc'].append(val_m['dir_acc'])
    history['val_acc05'].append(val_m['acc_0.5std'])
    history['lr'].append(cur_lr)
    print(f"{epoch:>4}  {train_loss:>8.4f}  {val_m['mae']:>8.4f}  {val_m['rmse']:>7.4f}  "
          f"{val_m['mape']:>6.1f}%  {val_m['r2']:>7.4f}  "
          f"{val_m['dir_acc']:>5.1f}%  {val_m['acc_0.5std']:>5.1f}%  {cur_lr:>10.2e}")
    if val_m['mae'] < best_val_mae:
        best_val_mae = val_m['mae']
        best_state   = copy.deepcopy(model.state_dict())
        no_improve   = 0
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f'\nEarly stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)')
            break
model.load_state_dict(best_state)
os.makedirs('checkpoints', exist_ok=True)
torch.save(best_state, 'checkpoints/demand_gnn_best.pt')
print(f'\nBest Val MAE : {best_val_mae:.4f}')
print('Checkpoint saved → checkpoints/demand_gnn_best.pt')



import matplotlib.pyplot as plt
epochs_ran = len(history['train_loss'])
ep_range   = range(1, epochs_ran + 1)
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
# Panel 1 — Loss curves
axes[0,0].plot(ep_range, history['train_loss'], label='Train (Huber+MSE)', linewidth=2, color='#185FA5')
axes[0,0].plot(ep_range, history['val_mae'],    label='Val MAE',           linewidth=2, color='#D85A30', linestyle='--')
axes[0,0].set_xlabel('Epoch'); axes[0,0].set_ylabel('Loss')
axes[0,0].set_title('Training Loss vs Validation MAE', fontweight='500')
axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)
# Panel 2 — R²
axes[0,1].plot(ep_range, history['val_r2'], linewidth=2, color='#534AB7')
axes[0,1].axhline(y=0, color='gray', linestyle=':', alpha=0.5, label='R²=0 (predicts mean)')
axes[0,1].set_xlabel('Epoch'); axes[0,1].set_ylabel('R²')
axes[0,1].set_title('Validation R²', fontweight='500')
axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)
# Panel 3 — Accuracy metrics
axes[1,0].plot(ep_range, history['val_dir_acc'], label='Direction Acc%', linewidth=2, color='#1D9E75')
axes[1,0].plot(ep_range, history['val_acc05'],   label='Acc@0.5σ%',     linewidth=2, color='#BA7517', linestyle='--')
axes[1,0].axhline(y=50, color='gray', linestyle=':', alpha=0.5, label='50% baseline')
axes[1,0].set_xlabel('Epoch'); axes[1,0].set_ylabel('%')
axes[1,0].set_title('Accuracy metrics', fontweight='500')
axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3)
# Panel 4 — LR schedule
axes[1,1].plot(ep_range, history['lr'], linewidth=2, color='#D85A30')
axes[1,1].set_xlabel('Epoch'); axes[1,1].set_ylabel('LR')
axes[1,1].set_title('LR Schedule (CosineAnnealingWarmRestarts)', fontweight='500')
axes[1,1].set_yscale('log'); axes[1,1].grid(True, alpha=0.3)
plt.suptitle('Polaris GNN v2 — Training curves', fontsize=14, fontweight='500')
plt.tight_layout()
plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print('Training curves saved to training_curves.png')






import torch
import torch.nn.functional as F
model.eval()
ei = edge_index.to(device)
ea = edge_attr.to(device) if edge_attr is not None else None
all_preds, all_targets = [], []
with torch.no_grad():
    for x_b, y_b in test_loader:
        for i in range(x_b.size(0)):
            p = model(x_b[i].to(device), ei, ea).cpu()
            all_preds.append(p)
            all_targets.append(y_b[i])
preds   = torch.stack(all_preds)
targets = torch.stack(all_targets)
# Overall metrics
mae_overall  = F.l1_loss(preds, targets).item()
rmse_overall = torch.sqrt(F.mse_loss(preds, targets)).item()
# Fixed MAPE
mask = torch.abs(targets) > 0.1
mape_overall = (torch.abs(preds[mask] - targets[mask]) / torch.abs(targets[mask])).mean().item() * 100 if mask.sum() > 0 else 0.0
# R²
ss_res = ((targets - preds) ** 2).sum().item()
ss_tot = ((targets - targets.mean()) ** 2).sum().item()
r2_overall = 1.0 - ss_res / (ss_tot + 1e-8)
# Direction accuracy
p_diff = preds[1:, :, 0] - preds[:-1, :, 0]
t_diff = targets[1:, :, 0] - targets[:-1, :, 0]
dir_acc = ((p_diff * t_diff) > 0).float().mean().item() * 100
# Threshold accuracy
acc_05 = (torch.abs(preds - targets) < 0.5).float().mean().item() * 100
print('='*60)
print('TEST SET RESULTS (held-out unseen days) — v2 model')
print('='*60)
print(f'  MAE  (normalised)       : {mae_overall:.4f}')
print(f'  RMSE (normalised)       : {rmse_overall:.4f}')
print(f'  MAPE (filtered >0.1)    : {mape_overall:.2f}%')
print(f'  R² score                : {r2_overall:.4f}')
print(f'  Direction accuracy      : {dir_acc:.1f}%')
print(f'  Accuracy (within 0.5σ)  : {acc_05:.1f}%')
print()
# Per-zone breakdown
print('── Per-zone metrics (test set) ──')
for z in range(NUM_ZONES):
    z_mae  = F.l1_loss(preds[:, z, :], targets[:, z, :]).item()
    z_rmse = torch.sqrt(F.mse_loss(preds[:, z, :], targets[:, z, :])).item()
    z_ss_res = ((targets[:, z, :] - preds[:, z, :]) ** 2).sum().item()
    z_ss_tot = ((targets[:, z, :] - targets[:, z, :].mean()) ** 2).sum().item()
    z_r2 = 1.0 - z_ss_res / (z_ss_tot + 1e-8)
    print(f'  Z{z:02d} ({ZONE_NAMES[z]:<18}): MAE={z_mae:.4f}  RMSE={z_rmse:.4f}  R²={z_r2:.4f}')
print()
print('── Per-timestep MAE (degradation over forecast horizon) ──')
for h in range(HORIZON):
    h_mae = F.l1_loss(preds[:, :, h], targets[:, :, h]).item()
    print(f'  t+{(h+1)*15:>3}min: MAE={h_mae:.4f}')


import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np

# Find best and worst zone by MAE on test
zone_maes = [F.l1_loss(preds[:,z,:], targets[:,z,:]).item() for z in range(NUM_ZONES)]
best_zone  = int(np.argmin(zone_maes))
worst_zone = int(np.argmax(zone_maes))

fig, axes = plt.subplots(2, 1, figsize=(16, 8))

for ax_idx, (z, label) in enumerate([(best_zone, 'best'), (worst_zone, 'worst')]):
    ax = axes[ax_idx]
    # Show first 96 test windows (1 day)
    n_show = min(96, preds.shape[0])
    p_vals = preds[:n_show,   z, 0].numpy()   # t+1 step prediction
    t_vals = targets[:n_show, z, 0].numpy()   # actual
    x_axis = np.arange(n_show) * 15  # minutes

    ax.plot(x_axis, t_vals, label='Actual',    linewidth=2,   color='#185FA5')
    ax.plot(x_axis, p_vals, label='Predicted', linewidth=1.5, color='#D85A30', linestyle='--')
    ax.fill_between(x_axis, t_vals, p_vals, alpha=0.1, color='#D85A30')
    ax.set_title(
        f'Zone Z{z:02d} ({ZONE_NAMES[z]}) — {label} zone  |  '
        f'MAE={zone_maes[z]:.4f} (normalised)',
        fontweight='500'
    )
    ax.set_xlabel('Time (minutes into test period)')
    ax.set_ylabel('Normalised demand')
    ax.legend(); ax.grid(True, alpha=0.3)

plt.suptitle('Predicted vs Actual demand — t+15min horizon (test period)',
             fontsize=13, fontweight='500')
plt.tight_layout()
plt.savefig('prediction_vs_actual.png', dpi=150, bbox_inches='tight')
plt.show()
print('Prediction chart saved to prediction_vs_actual.png')