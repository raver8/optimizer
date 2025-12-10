import gradio as gr
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
import io
from PIL import Image
import numpy as np

# --- 1. CONFIGURATION & DATA LOADING ---
DATASET_ID = "netop/5G-Network-Energy-Consumption"

def load_and_prep_data():
    """
    Connects to Hugging Face Hub, streams the energy dataset,
    and formats it for PyTorch training.
    """
    print(f"🔌 Connecting to Data Center Repository: {DATASET_ID}...")
    
    # stream=True allows us to use massive datasets without downloading them fully
    ds_stream = load_dataset(DATASET_ID, split='train', streaming=True)
    
    # We will extract 1000 data points to train our model quickly
    data = []
    print("   Extracting samples (Load vs Energy)...")
    for i, sample in enumerate(ds_stream):
        if i >= 1000: break
        # The dataset has 'Load' (0-1) and 'Energy' (Wh)
        # We filter for valid data points
        if sample['Load'] is not None and sample['Energy'] is not None:
            data.append([sample['Load'], sample['Energy']])
            
    df = pd.DataFrame(data, columns=['Load', 'Energy'])
    
    # Normalize: Load is 0-1. Energy might be high, so we scale it for stability if needed.
    # For this demo, we keep raw values to show real Wh numbers.
    return df

# --- 2. DEFINE PYTORCH MODEL ---
class EnergyPredictor(nn.Module):
    def __init__(self):
        super(EnergyPredictor, self).__init__()
        # Simple Linear Regression: Power = weight * Load + bias
        # In real data centers, this might be a non-linear curve (Polynomial), 
        # but a linear approximation works for this demo.
        self.linear = nn.Linear(1, 1) 

    def forward(self, x):
        return self.linear(x)

# --- 3. TRAIN THE MODEL (ON STARTUP) ---
df = load_and_prep_data()

# Convert Pandas to PyTorch Tensors
X_train = torch.tensor(df['Load'].values, dtype=torch.float32).unsqueeze(1) # Input: Load
y_train = torch.tensor(df['Energy'].values, dtype=torch.float32).unsqueeze(1) # Target: Energy

model = EnergyPredictor()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

print("🚀 Training PyTorch Energy Model...")
# Quick training loop
epochs = 500
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
print("   Training Complete.")

# --- 4. PREDICTION & VISUALIZATION FUNCTION ---
def predict_power(server_load_pct):
    """
    Takes user input (Server Load %), uses PyTorch to predict power,
    and generates a graph comparing it to the real data center data.
    """
    # Convert input (0-100) to model scale (0-1)
    load_normalized = server_load_pct / 100.0
    input_tensor = torch.tensor([[load_normalized]], dtype=torch.float32)
    
    # Inference
    model.eval()
    with torch.no_grad():
        predicted_energy = model(input_tensor).item()
    
    # --- Visualization ---
    plt.figure(figsize=(10, 5))
    
    # Plot real data points (Background)
    plt.scatter(df['Load']*100, df['Energy'], color='lightgray', s=10, label='Historical Data (Repository)')
    
    # Plot the Model's "Line of Best Fit"
    x_range = torch.linspace(0, 1, 100).unsqueeze(1)
    with torch.no_grad():
        y_range = model(x_range)
    plt.plot(x_range.numpy()*100, y_range.numpy(), color='blue', linewidth=2, label='PyTorch Model Trend')
    
    # Plot the specific user prediction
    plt.scatter([server_load_pct], [predicted_energy], color='red', s=150, zorder=5, label='Your Prediction')
    
    plt.title(f"Data Center Energy Consumption (Node #1042)")
    plt.xlabel("Server Load (%)")
    plt.ylabel("Energy Consumption (Wh)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Save plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    
    return f"{predicted_energy:.2f} Wh", img

# --- 5. UI INTERFACE ---
with gr.Blocks() as demo:
    gr.Markdown("# ⚡ GreenCompute: Data Center Energy Manager")
    gr.Markdown(f"### Utilizing Dataset: `{DATASET_ID}`")
    gr.Markdown("This app pulls real infrastructure power data from the Hub, trains a **PyTorch** regression model, and predicts power usage based on server load.")
    
    with gr.Row():
        with gr.Column():
            slider = gr.Slider(0, 100, value=50, label="Projected Server Load (%)")
            btn = gr.Button("Analyze Power Impact", variant="primary")
        
        with gr.Column():
            result_text = gr.Textbox(label="Predicted Power Consumption")
    
    plot_output = gr.Image(label="Load vs. Energy Curve")
    
    btn.click(predict_power, inputs=slider, outputs=[result_text, plot_output])

demo.launch()     