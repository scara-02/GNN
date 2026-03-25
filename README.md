---
title: Traffic Demand Prediction — Spatio-Temporal GNN
emoji: 🚦
colorFrom: blue
colorTo: orange
sdk: gradio
sdk_version: 5.23.0
app_file: app.py
pinned: true
license: mit
---

# 🚦 Traffic Demand Prediction — Spatio-Temporal GNN

Predict traffic demand **90 minutes ahead** across 8 city zones using a Graph Neural Network with temporal attention.

## Architecture

```
Input: (8 zones × 24 timesteps × 11 features)
  ├── Latest snapshot → SpatialEncoder (GAT ×2, 4 heads)  → 128-dim
  └── Full history    → TemporalEncoder (GRU ×2 + Attn)   → 128-dim
         concat [256] → Fusion MLP + Residual Skip → 128-dim
         → Output Head → 6-step demand forecast per zone
```

## Features

- **11 engineered features** per zone per timestep (cyclical time, rolling stats, momentum)
- **Graph Attention** learns which neighbouring zones influence each other
- **Temporal Attention** focuses on the most relevant past timesteps
- **Interactive Gradio UI** with EDA, training, prediction, and zone heatmap

## Dataset

Real junction traffic data — 31 days, 15-minute intervals, 2,976 observations.  
Upload your own `Traffic.csv` with columns: `Date`, `Time`, `Day of the week`, `CarCount`, `BikeCount`, `BusCount`, `TruckCount`, `Total`, `Traffic Situation`.

## How to use

1. 📊 **Upload** your `Traffic.csv`
2. 🧠 **Train** the model (adjust epochs, LR, hidden dim)
3. 🔮 **Predict** and explore zone-level demand forecasts with interactive maps
