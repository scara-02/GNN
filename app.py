"""
Traffic Demand Prediction — Gradio App for Hugging Face Spaces
Spatio-Temporal GNN (GAT + GRU with Temporal Attention)
"""
import gradio as gr
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import copy
import tempfile

from model import (
    TrafficDemandGNN, build_zone_graph, preprocess_csv, run_inference,
    NUM_ZONES, NODE_FEATURES, T_WINDOW, HORIZON, ZONE_NAMES, ZONE_META, ZONE_COORDS,
)

# ── Global state ──────────────────────────────────────────────
MODEL_STATE = {
    'model': None,
    'telemetry': None,
    'edge_index': None,
    'edge_attr': None,
    'df': None,
    'trained': False,
    'history': None,
    'zone_stats': None,
}


# ══════════════════════════════════════════════════════════════
# TAB 1: UPLOAD & EDA
# ══════════════════════════════════════════════════════════════

def process_upload(file):
    """Process uploaded Traffic.csv and return EDA results."""
    if file is None:
        return "⚠️ Please upload a Traffic.csv file.", None, None

    try:
        csv_path = file.name if hasattr(file, 'name') else file
        telemetry, df, zone_stats, col_stats = preprocess_csv(csv_path)
        edge_index, edge_attr = build_zone_graph(radius_km=1.0)

        MODEL_STATE['telemetry']  = telemetry
        MODEL_STATE['edge_index'] = edge_index
        MODEL_STATE['edge_attr']  = edge_attr
        MODEL_STATE['df']         = df
        MODEL_STATE['zone_stats'] = zone_stats

        # EDA summary
        summary = f"""## ✅ Dataset Loaded Successfully

| Metric | Value |
|--------|-------|
| **Rows** | {df.shape[0]:,} |
| **Columns** | {df.shape[1]} |
| **Days** | {df['Date'].nunique()} |
| **Time slots/day** | {df.groupby('Date').size().iloc[0]} (15-min intervals) |
| **Missing values** | {df.isnull().sum().sum()} |
| **Duplicate rows** | {df.duplicated().sum()} |

### Vehicle Statistics
| Type | Mean | Std | Min | Max |
|------|------|-----|-----|-----|
| Cars | {df['CarCount'].mean():.1f} | {df['CarCount'].std():.1f} | {df['CarCount'].min()} | {df['CarCount'].max()} |
| Bikes | {df['BikeCount'].mean():.1f} | {df['BikeCount'].std():.1f} | {df['BikeCount'].min()} | {df['BikeCount'].max()} |
| Buses | {df['BusCount'].mean():.1f} | {df['BusCount'].std():.1f} | {df['BusCount'].min()} | {df['BusCount'].max()} |
| Trucks | {df['TruckCount'].mean():.1f} | {df['TruckCount'].std():.1f} | {df['TruckCount'].min()} | {df['TruckCount'].max()} |

### Traffic Situation Distribution
"""
        dist = df['Traffic Situation'].value_counts()
        for k, v in dist.items():
            summary += f"- **{k}**: {v} ({100*v/len(df):.1f}%)\n"

        summary += f"\n### Telemetry Tensor: `{list(telemetry.shape)}` (zones × timesteps × features)"
        summary += f"\n### Zone Graph: {NUM_ZONES} zones, {edge_index.shape[1]} directed edges"

        # EDA charts
        eda_fig = create_eda_charts(df)
        hourly_fig = create_hourly_chart(df)

        return summary, eda_fig, hourly_fig

    except Exception as e:
        return f"❌ Error processing file: {str(e)}", None, None


def create_eda_charts(df):
    """Create EDA visualisation with Plotly."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Vehicle Type Composition', 'Traffic Situation Distribution'),
    )

    # Vehicle composition
    means = df[['CarCount', 'BikeCount', 'BusCount', 'TruckCount']].mean()
    colors = ['#534AB7', '#1D9E75', '#D85A30', '#BA7517']
    fig.add_trace(go.Bar(
        x=means.index, y=means.values,
        marker_color=colors, text=[f'{v:.1f}' for v in means.values],
        textposition='auto', name='Mean Count'
    ), row=1, col=1)

    # Traffic situation
    dist = df['Traffic Situation'].value_counts()
    sit_colors = ['#1D9E75', '#185FA5', '#BA7517', '#D85A30']
    fig.add_trace(go.Pie(
        labels=dist.index, values=dist.values,
        marker_colors=sit_colors, hole=0.4, name='Situations'
    ), row=1, col=2)

    fig.update_layout(
        height=400, showlegend=False,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(30,30,40,0.8)',
    )
    return fig


def create_hourly_chart(df):
    """Traffic by hour of day."""
    hour = pd.to_datetime(df['Time'], format='%I:%M:%S %p').dt.hour
    hourly = df.groupby(hour)['Total'].mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hourly.index, y=hourly.values,
        mode='lines+markers', line=dict(color='#185FA5', width=3),
        fill='tozeroy', fillcolor='rgba(24,95,165,0.15)',
        name='Avg Traffic'
    ))
    fig.update_layout(
        title='Average Total Traffic by Hour of Day',
        xaxis_title='Hour', yaxis_title='Avg Total Vehicles',
        height=400, template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(30,30,40,0.8)',
    )
    return fig


# ══════════════════════════════════════════════════════════════
# TAB 2: TRAIN MODEL
# ══════════════════════════════════════════════════════════════

def train_model(epochs, lr, hidden_dim, progress=gr.Progress()):
    """Train the GNN model and return metrics."""
    if MODEL_STATE['telemetry'] is None:
        return "⚠️ Please upload data first (Tab 1).", None, None

    telemetry  = MODEL_STATE['telemetry']
    edge_index = MODEL_STATE['edge_index']
    edge_attr  = MODEL_STATE['edge_attr']

    epochs     = int(epochs)
    lr         = float(lr)
    hidden_dim = int(hidden_dim)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build dataset
    from torch.utils.data import Dataset, DataLoader, Subset

    class DemandDataset(Dataset):
        def __init__(self, tel, window=T_WINDOW, horizon=HORIZON):
            self.tel = tel; self.window = window; self.horizon = horizon
            self.length = tel.shape[1] - window - horizon + 1
        def __len__(self): return self.length
        def __getitem__(self, idx):
            x = self.tel[:, idx:idx+self.window, :]
            y = self.tel[:, idx+self.window:idx+self.window+self.horizon, 0]
            return x, y

    dataset = DemandDataset(telemetry)
    n = len(dataset)
    n_train = int(0.70 * n)
    n_val   = int(0.15 * n)

    train_ds = Subset(dataset, range(0, n_train))
    val_ds   = Subset(dataset, range(n_train, n_train + n_val))
    test_ds  = Subset(dataset, range(n_train + n_val, n))

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False)

    # Model
    torch.manual_seed(42)
    model = TrafficDemandGNN(
        node_features=NODE_FEATURES, hidden_dim=hidden_dim,
        gat_heads=4, gru_layers=2, horizon=HORIZON, dropout=0.2,
    ).to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimiser, T_0=10, T_mult=2, eta_min=1e-6
    )

    ei = edge_index.to(device)
    ea = edge_attr.to(device)

    history = {'train': [], 'val_mae': [], 'val_rmse': [], 'r2': [], 'dir_acc': [], 'acc05': [], 'lr': []}
    best_mae = float('inf')
    best_state = None
    patience = 15
    no_improve = 0
    log_lines = []

    for epoch in progress.tqdm(range(1, epochs + 1), desc="Training"):
        # Train
        model.train()
        total_loss = 0.0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimiser.zero_grad()
            preds = torch.stack([model(x_b[i], ei, ea) for i in range(x_b.size(0))])
            loss = 0.6 * F.huber_loss(preds, y_b, delta=1.0) + 0.4 * F.mse_loss(preds, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            total_loss += loss.item()
        train_loss = total_loss / len(train_loader)

        # Evaluate
        model.eval()
        ps, ts = [], []
        with torch.no_grad():
            for x_b, y_b in val_loader:
                for i in range(x_b.size(0)):
                    ps.append(model(x_b[i].to(device), ei, ea).cpu())
                    ts.append(y_b[i])
        p = torch.stack(ps); t = torch.stack(ts)

        val_mae  = F.l1_loss(p, t).item()
        val_rmse = torch.sqrt(F.mse_loss(p, t)).item()
        ss_res = ((t - p)**2).sum().item()
        ss_tot = ((t - t.mean())**2).sum().item()
        r2 = 1.0 - ss_res / (ss_tot + 1e-8)

        if p.shape[0] > 1:
            pd_ = p[1:,:,0] - p[:-1,:,0]; td_ = t[1:,:,0] - t[:-1,:,0]
            dir_acc = ((pd_ * td_) > 0).float().mean().item() * 100
        else:
            dir_acc = 0.0
        acc05 = (torch.abs(p - t) < 0.5).float().mean().item() * 100

        scheduler.step()
        cur_lr = optimiser.param_groups[0]['lr']

        history['train'].append(train_loss)
        history['val_mae'].append(val_mae)
        history['val_rmse'].append(val_rmse)
        history['r2'].append(r2)
        history['dir_acc'].append(dir_acc)
        history['acc05'].append(acc05)
        history['lr'].append(cur_lr)

        log_lines.append(
            f"Ep {epoch:>3} | Train {train_loss:.4f} | Val MAE {val_mae:.4f} | "
            f"R² {r2:.4f} | Dir {dir_acc:.1f}% | Acc {acc05:.1f}% | LR {cur_lr:.2e}"
        )

        if val_mae < best_mae:
            best_mae = val_mae
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                log_lines.append(f"\n⏹ Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    MODEL_STATE['model']   = model
    MODEL_STATE['trained'] = True
    MODEL_STATE['history'] = history

    # Test evaluation
    model.eval()
    ps, ts = [], []
    with torch.no_grad():
        for x_b, y_b in test_loader:
            for i in range(x_b.size(0)):
                ps.append(model(x_b[i].to(device), ei, ea).cpu())
                ts.append(y_b[i])
    p = torch.stack(ps); t = torch.stack(ts)
    test_mae  = F.l1_loss(p, t).item()
    test_rmse = torch.sqrt(F.mse_loss(p, t)).item()
    ss_res = ((t - p)**2).sum().item()
    ss_tot = ((t - t.mean())**2).sum().item()
    test_r2 = 1.0 - ss_res / (ss_tot + 1e-8)
    test_acc = (torch.abs(p - t) < 0.5).float().mean().item() * 100

    result = f"""## ✅ Training Complete

### Test Set Results (Held-out Unseen Days)
| Metric | Value |
|--------|-------|
| **MAE** | {test_mae:.4f} |
| **RMSE** | {test_rmse:.4f} |
| **R²** | {test_r2:.4f} |
| **Acc@0.5σ** | {test_acc:.1f}% |
| **Best Val MAE** | {best_mae:.4f} |
| **Epochs** | {len(history['train'])} |
| **Parameters** | {sum(p_.numel() for p_ in model.parameters() if p_.requires_grad):,} |

### Per-Zone MAE
"""
    for z in range(NUM_ZONES):
        z_mae = F.l1_loss(p[:, z, :], t[:, z, :]).item()
        result += f"- **Z{z:02d}** ({ZONE_NAMES[z]}): {z_mae:.4f}\n"

    # Training curves
    curves_fig = create_training_curves(history)
    log_text = "\n".join(log_lines[-20:])  # last 20 lines

    return result, curves_fig, log_text


def create_training_curves(history):
    """Plot training curves using Plotly."""
    epochs = list(range(1, len(history['train']) + 1))

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Loss', 'R² Score', 'Accuracy Metrics', 'Learning Rate'),
    )

    # Loss
    fig.add_trace(go.Scatter(x=epochs, y=history['train'], name='Train', line=dict(color='#185FA5', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=history['val_mae'], name='Val MAE', line=dict(color='#D85A30', width=2, dash='dash')), row=1, col=1)

    # R²
    fig.add_trace(go.Scatter(x=epochs, y=history['r2'], name='R²', line=dict(color='#534AB7', width=2)), row=1, col=2)
    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=1, col=2)

    # Accuracy
    fig.add_trace(go.Scatter(x=epochs, y=history['dir_acc'], name='Direction%', line=dict(color='#1D9E75', width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=history['acc05'], name='Acc@0.5σ%', line=dict(color='#BA7517', width=2, dash='dash')), row=2, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)

    # LR
    fig.add_trace(go.Scatter(x=epochs, y=history['lr'], name='LR', line=dict(color='#D85A30', width=2)), row=2, col=2)

    fig.update_layout(
        height=600, template='plotly_dark', showlegend=True,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(30,30,40,0.8)',
    )
    fig.update_yaxes(type="log", row=2, col=2)
    return fig


# ══════════════════════════════════════════════════════════════
# TAB 3: PREDICT & VISUALISE
# ══════════════════════════════════════════════════════════════

def predict_and_visualise(time_offset):
    """Run prediction and create visualisations."""
    if not MODEL_STATE['trained']:
        return "⚠️ Please train the model first (Tab 2).", None, None

    model      = MODEL_STATE['model']
    telemetry  = MODEL_STATE['telemetry']
    edge_index = MODEL_STATE['edge_index']
    edge_attr  = MODEL_STATE['edge_attr']

    time_offset = int(time_offset)
    max_idx = telemetry.shape[1] - T_WINDOW - HORIZON
    start_idx = max(0, min(time_offset, max_idx))

    y_pred, y_true = run_inference(model, telemetry, edge_index, edge_attr, start_idx)

    # Zone heatmap
    avg_demand = y_pred.mean(axis=1)
    heatmap_fig = create_zone_heatmap(avg_demand)

    # Prediction vs actual
    pred_fig = create_prediction_chart(y_pred, y_true)

    # Summary
    mae = np.abs(y_pred - y_true).mean()
    r2_num = ((y_true - y_pred)**2).sum()
    r2_den = ((y_true - y_true.mean())**2).sum()
    r2 = 1.0 - r2_num / (r2_den + 1e-8)

    summary = f"""## 🔮 Forecast Results

**Window**: timestep {start_idx} → {start_idx + T_WINDOW} (input) → t+{HORIZON*15}min (forecast)

| Metric | Value |
|--------|-------|
| **MAE** | {mae:.4f} |
| **R²** | {r2:.4f} |

### Zone-level Predictions (avg over 90min)
| Zone | Name | Predicted Demand | Heat Level |
|------|------|:----------------:|:----------:|
"""
    max_d = avg_demand.max()
    for z in range(NUM_ZONES):
        d = avg_demand[z]
        heat = '🔴 High' if d >= max_d * 0.7 else '🟡 Medium' if d >= max_d * 0.4 else '🟢 Low'
        summary += f"| Z{z:02d} | {ZONE_NAMES[z]} | {d:.3f} | {heat} |\n"

    return summary, heatmap_fig, pred_fig


def create_zone_heatmap(avg_demand):
    """Create a zone heatmap using Plotly scatter_mapbox."""
    lats = [c[0] for c in ZONE_COORDS]
    lons = [c[1] for c in ZONE_COORDS]

    # Normalise demand for color
    d_min, d_max = avg_demand.min(), avg_demand.max()
    d_norm = (avg_demand - d_min) / (d_max - d_min + 1e-8)

    fig = go.Figure()

    # Zone markers
    fig.add_trace(go.Scattermapbox(
        lat=lats, lon=lons,
        mode='markers+text',
        marker=dict(
            size=20 + d_norm * 30,
            color=avg_demand,
            colorscale='YlOrRd',
            showscale=True,
            colorbar=dict(title='Demand'),
        ),
        text=[f"Z{i:02d} {ZONE_NAMES[i]}<br>Demand: {avg_demand[i]:.3f}" for i in range(NUM_ZONES)],
        textposition='top center',
        textfont=dict(size=10, color='white'),
        hoverinfo='text',
    ))

    fig.update_layout(
        mapbox=dict(
            style='carto-darkmatter',
            center=dict(lat=np.mean(lats), lon=np.mean(lons)),
            zoom=14,
        ),
        height=500,
        margin=dict(l=0, r=0, t=30, b=0),
        title='Zone Demand Heatmap (predicted)',
    )
    return fig


def create_prediction_chart(y_pred, y_true):
    """Plot predicted vs actual for all zones."""
    fig = make_subplots(
        rows=2, cols=4,
        subplot_titles=[f'Z{z:02d} {ZONE_NAMES[z]}' for z in range(NUM_ZONES)],
        vertical_spacing=0.12,
    )

    time_axis = [(h+1)*15 for h in range(HORIZON)]
    colors_actual = '#185FA5'
    colors_pred   = '#D85A30'

    for z in range(NUM_ZONES):
        row = z // 4 + 1
        col = z % 4 + 1
        fig.add_trace(go.Scatter(
            x=time_axis, y=y_true[z], name='Actual' if z == 0 else None,
            line=dict(color=colors_actual, width=2), showlegend=(z == 0),
        ), row=row, col=col)
        fig.add_trace(go.Scatter(
            x=time_axis, y=y_pred[z], name='Predicted' if z == 0 else None,
            line=dict(color=colors_pred, width=2, dash='dash'), showlegend=(z == 0),
        ), row=row, col=col)

    fig.update_layout(
        height=450, template='plotly_dark',
        title='Predicted vs Actual Demand (6 × 15min steps)',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(30,30,40,0.8)',
    )
    return fig


# ══════════════════════════════════════════════════════════════
# GRADIO APP
# ══════════════════════════════════════════════════════════════

DESCRIPTION = """
# 🚦 Traffic Demand Prediction — Spatio-Temporal GNN

**Architecture:** GAT spatial encoder + GRU temporal encoder with attention  
**Data:** Real junction traffic — 31 days, 15-min intervals, 2,976 observations  
**Task:** Predict traffic demand 90 minutes ahead across 8 city zones

### How to use:
1. **Upload** your `Traffic.csv` in the EDA tab
2. **Train** the model (or adjust hyperparameters)
3. **Predict** and explore zone-level demand forecasts
"""

with gr.Blocks(
    title="Traffic Demand GNN",
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="orange",
        neutral_hue="slate",
    ),
    css="""
    .gradio-container { max-width: 1200px !important; }
    .tab-nav button { font-size: 16px !important; padding: 12px 24px !important; }
    """
) as app:

    gr.Markdown(DESCRIPTION)

    with gr.Tabs():
        # ── TAB 1: EDA ──────────────────────────────────────
        with gr.Tab("📊 Upload & EDA", id=0):
            with gr.Row():
                file_input = gr.File(label="Upload Traffic.csv", file_types=[".csv"])
                upload_btn = gr.Button("Process Data", variant="primary", size="lg")

            eda_summary = gr.Markdown(label="Dataset Summary")

            with gr.Row():
                eda_chart = gr.Plot(label="Data Overview")
                hourly_chart = gr.Plot(label="Hourly Pattern")

            upload_btn.click(
                fn=process_upload,
                inputs=[file_input],
                outputs=[eda_summary, eda_chart, hourly_chart],
            )

        # ── TAB 2: TRAIN ────────────────────────────────────
        with gr.Tab("🧠 Train Model", id=1):
            with gr.Row():
                epochs_slider = gr.Slider(10, 100, value=80, step=10, label="Epochs")
                lr_slider     = gr.Slider(0.0001, 0.01, value=0.003, step=0.0005, label="Learning Rate")
                hidden_slider = gr.Slider(64, 256, value=128, step=32, label="Hidden Dimensions")

            train_btn = gr.Button("🚀 Start Training", variant="primary", size="lg")

            train_result = gr.Markdown(label="Results")
            train_curves = gr.Plot(label="Training Curves")
            train_log    = gr.Textbox(label="Training Log (last 20 epochs)", lines=10)

            train_btn.click(
                fn=train_model,
                inputs=[epochs_slider, lr_slider, hidden_slider],
                outputs=[train_result, train_curves, train_log],
            )

        # ── TAB 3: PREDICT ──────────────────────────────────
        with gr.Tab("🔮 Predict", id=2):
            with gr.Row():
                offset_slider = gr.Slider(
                    0, 2900, value=2900, step=96,
                    label="Time Offset (slide to pick different days — 96 = 1 day)"
                )
                predict_btn = gr.Button("Generate Forecast", variant="primary", size="lg")

            pred_summary = gr.Markdown(label="Forecast Summary")

            with gr.Row():
                heatmap_plot = gr.Plot(label="Zone Demand Heatmap")
                pred_plot    = gr.Plot(label="Predicted vs Actual")

            predict_btn.click(
                fn=predict_and_visualise,
                inputs=[offset_slider],
                outputs=[pred_summary, heatmap_plot, pred_plot],
            )

        # ── TAB 4: ABOUT ────────────────────────────────────
        with gr.Tab("ℹ️ About", id=3):
            gr.Markdown("""
## Model Architecture

```
Input: (8 zones × 24 timesteps × 11 features)
  ├── Latest snapshot → SpatialEncoder (GAT ×2, 4 heads) → 128-dim
  └── Full history    → TemporalEncoder (GRU ×2 + Attention) → 128-dim
         concat [256-dim] → Fusion + Residual Skip → 128-dim
         → Output Head → (8 zones × 6 steps) demand forecast
```

### Features (11 per zone per timestep)
| # | Feature | Description |
|---|---------|-------------|
| 0 | demand_z | Z-scored zone demand |
| 1 | supply_z | Inverse traffic severity proxy |
| 2-3 | hour_sin/cos | Cyclical hour encoding |
| 4-5 | day_sin/cos | Cyclical day-of-week encoding |
| 6 | traffic_sev | Normalised severity ordinal |
| 7 | rolling_mean | 1-hour rolling mean of demand |
| 8 | rolling_std | 1-hour rolling std (volatility) |
| 9 | momentum | 1-hour demand momentum |
| 10 | diff_1 | Instant rate of change |

### 8 City Zones
- **Z00–Z03**: Vehicle-class corridors (car, bike, bus, truck)
- **Z04–Z07**: Time-lagged junctions (15-min propagation delay)

### Training
- **Loss**: 60% Huber + 40% MSE (balanced robustness)
- **Optimizer**: Adam (weight_decay=5e-5)
- **Scheduler**: CosineAnnealingWarmRestarts (T₀=10, T_mult=2)
- **Early stopping**: patience=15 on validation MAE
""")


if __name__ == "__main__":
    app.launch()
