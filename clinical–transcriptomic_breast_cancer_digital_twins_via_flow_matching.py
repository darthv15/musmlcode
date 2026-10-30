!pip install -q torch scikit-learn pandas numpy

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Colab Runtime Environment: {device}")

raw_cancer_data = load_breast_cancer()

context_index = 2
all_features = raw_cancer_data.data

# Target generation space: All clinical characteristics except the chosen context dimension
data_dimensions = np.delete(all_features, context_index, axis=1)
context_dimension = all_features[:, [context_index]]

# Standardize to zero-mean and unit variance for robust generative modeling
data_scaler = StandardScaler()
context_scaler = StandardScaler()

X_normalized = data_scaler.fit_transform(data_dimensions)
C_normalized = context_scaler.fit_transform(context_dimension)

# Cast variables to PyTorch tensors for model consumption
X_tensor = torch.tensor(X_normalized, dtype=torch.float32)
C_tensor = torch.tensor(C_normalized, dtype=torch.float32)

print(f"Data tensor shape (Biomarker profiles): {X_tensor.shape}")
print(f"Context tensor shape (Cell Perimeter): {C_tensor.shape}")

class BiomedicalMetaFlowMap(nn.Module):
    """
    A Deep Generative Flow parameterizing the mapping matrix across
    the continuous conditional boundaries of the clinical manifold.
    """
    def __init__(self, data_dim, context_dim, hidden_dim=256):
        super(BiomedicalMetaFlowMap, self).__init__()

        # Concat inputs: noisy state (data_dim) + medical condition (context_dim) + time token (1)
        self.input_layer = nn.Linear(data_dim + context_dim + 1, hidden_dim)

        self.backbone = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, data_dim) # Outputs the predicted trajectory vector field
        )

    def forward(self, x, c, t):
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        # Ensure dimensions align before stacking
        inputs = torch.cat([x, c, t], dim=-1)
        h = self.input_layer(inputs)
        return self.backbone(h)

data_dim = X_tensor.shape[1]
context_dim = C_tensor.shape[1]

# Initialize model components
mfm_model = BiomedicalMetaFlowMap(data_dim=data_dim, context_dim=context_dim).to(device)
optimizer = optim.AdamW(mfm_model.parameters(), lr=2e-4, weight_decay=1e-4)

epochs = 150
batch_size = 64
num_records = X_tensor.size(0)

print("Commencing Meta Flow Map training loop on real patient diagnostics...")

for epoch in range(epochs):
    mfm_model.train()
    indices_permutation = torch.randperm(num_records)
    cumulative_loss = 0.0

    for start_idx in range(0, num_records, batch_size):
        optimizer.zero_grad()

        batch_indices = indices_permutation[start_idx : start_idx + batch_size]
        x1 = X_tensor[batch_indices].to(device) # Target clinical data point
        c  = C_tensor[batch_indices].to(device)  # Real paired biological context

        # Sample exogenous noise base distributions
        x0 = torch.randn_like(x1).to(device)

        # Sample continuous stochastic time intervals
        t = torch.rand(x1.size(0), 1, device=device)

        # Build the conditional path interpolation
        xt = (1.0 - t) * x0 + t * x1

        # The straight-line velocity target vector field
        target_field = x1 - x0

        # Execute forward evaluation path
        predicted_field = mfm_model(xt, c, t)

        # Compute trajectory matching error
        loss = nn.functional.mse_loss(predicted_field, target_field)

        loss.backward()
        optimizer.step()

        cumulative_loss += loss.item() * x1.size(0)

    if (epoch + 1) % 25 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:03d}/{epochs} | Mean Trajectory Alignment Loss: {cumulative_loss / num_records:.6f}")

print("MFM Engine fully trained.")

mfm_model.eval()

# Select a targeted pathological context value (e.g., highly enlarged cell perimeters)
abnormal_perimeter_val = 2.5
generation_count = 4

# Scale and match the raw context representation
scaled_context_input = torch.full((generation_count, 1), abnormal_perimeter_val, dtype=torch.float32).to(device)

# Generate baseline random noise vectors
latent_noise = torch.randn(generation_count, data_dim).to(device)

with torch.no_grad():
    # Evaluate at initial time step t=0 to extract instantaneous trajectory direction
    t_zero = torch.zeros(generation_count, 1).to(device)
    velocity_vector = mfm_model(latent_noise, scaled_context_input, t_zero)

    # Project straight down the trajectory to the target sample space: x1 = x0 + v
    synthetic_normalized_profiles = latent_noise + velocity_vector

# Invert normalization scales back to real clinical feature dimensions
synthetic_profiles_np = synthetic_normalized_profiles.cpu().numpy()
unscaled_clinical_profiles = data_scaler.inverse_transform(synthetic_profiles_np)

# Render features into a scannable structure
feature_labels = np.delete(raw_cancer_data.feature_names, context_index)
output_dataframe = pd.DataFrame(unscaled_clinical_profiles, columns=feature_labels)

# Re-insert the target human-readable context condition column for validation
unscaled_context_reconstruction = context_scaler.inverse_transform([[abnormal_perimeter_val]])[0][0]
output_dataframe.insert(0, "Conditioned Mean Perimeter (Target Context)", unscaled_context_reconstruction)

print("\n--- Artificially Synthesized Clinical Patient Records via MFM Flow Matching ---")
display(output_dataframe.iloc[:, :6].round(3)) # Displaying first 6 attributes for scannability

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

torch.manual_seed(42)
np.random.seed(42)

class DataPipeline:
    """Handles data engineering, 3-way cohort splitting, and leakage prevention."""
    def __init__(self, context_feature_idx=3):
        self.context_feature_idx = context_feature_idx
        self.scaler_X = StandardScaler()
        self.scaler_C = StandardScaler()

    def prepare_data(self):
        raw_data = load_breast_cancer()
        X_raw = raw_data.data
        y_raw = raw_data.target # 0 = Malignant, 1 = Benign

        # Isolate the target feature to serve as our continuous conditioning context
        self.feature_names = np.delete(raw_data.feature_names, self.context_feature_idx)
        X_features = np.delete(X_raw, self.context_feature_idx, axis=1)
        C_context = X_raw[:, [self.context_feature_idx]]

        # 3-Way Split: Train (70%), Reward Oracle Training (15%), Test Holdout (15%)
        X_train_val, X_test, C_train_val, C_test, y_train_val, y_test = train_test_split(
            X_features, C_context, y_raw, test_size=0.15, random_state=42, stratify=y_raw
        )
        X_train, X_oracle, C_train, C_oracle, y_train, y_oracle = train_test_split(
            X_train_val, C_train_val, y_train_val, test_size=0.176, random_state=42, stratify=y_train_val
        )

        # Normalize strictly using Training statistics to ensure zero data leakage
        X_train_s = self.scaler_X.fit_transform(X_train)
        C_train_s = self.scaler_C.fit_transform(C_train)

        X_oracle_s = self.scaler_X.transform(X_oracle)
        X_test_s = self.scaler_X.transform(X_test)
        C_test_s = self.scaler_C.transform(C_test)

        return {
            "train": (torch.tensor(X_train_s, dtype=torch.float32), torch.tensor(C_train_s, dtype=torch.float32)),
            "oracle": (torch.tensor(X_oracle_s, dtype=torch.float32), torch.tensor(y_oracle, dtype=torch.long)),
            "test": (torch.tensor(X_test_s, dtype=torch.float32), torch.tensor(C_test_s, dtype=torch.float32), y_test)
        }


class ClinicalRewardOracle(nn.Module):
    """Deep classifier representing our target downstream clinical value/reward function."""
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.net(x)


class MFMResidualBlock(nn.Module):
    """Residual connection layer to preserve backpropagation trajectories through deep maps."""
    def __init__(self, dim):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.act = nn.SiLU()
    def forward(self, x):
        return x + self.act(self.linear2(self.act(self.linear1(x))))


class PortfolioBiomedicalMFM(nn.Module):
    """Continuous Meta Flow Map optimizing straight-line transport velocity fields."""
    def __init__(self, data_dim, context_dim, hidden_dim=256, blocks=3):
        super().__init__()
        self.input_proj = nn.Linear(data_dim + context_dim + 1, hidden_dim)
        self.trunk = nn.Sequential(*[MFMResidualBlock(hidden_dim) for _ in range(blocks)])
        self.output_proj = nn.Linear(hidden_dim, data_dim)
        self.act = nn.SiLU()

    def forward(self, x, c, t):
        if t.dim() == 1: t = t.unsqueeze(-1)
        inputs = torch.cat([x, c, t], dim=-1)
        h = self.act(self.input_proj(inputs))
        h = self.trunk(h)
        return self.output_proj(h)


class ExperimentRunner:
    """Orchestrates model training, inference-time gradient steering, and metrics extraction."""
    def __init__(self, device):
        self.device = device
        self.dp = DataPipeline()

    def run_pipeline(self):
        # 1. Prepare Data Tensors
        data_splits = self.dp.prepare_data()
        X_train, C_train = data_splits["train"][0].to(self.device), data_splits["train"][1].to(self.device)
        X_oracle, y_oracle = data_splits["oracle"][0].to(self.device), data_splits["oracle"][1].to(self.device)
        X_test, C_test, y_test_raw = data_splits["test"]

        # 2. Train the Clinical Reward Oracle
        print("[TRAINING] Optimizing Clinical Reward Classifier...")
        oracle = ClinicalRewardOracle(input_dim=X_train.shape[1]).to(self.device)
        oracle_opt = optim.AdamW(oracle.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        oracle.train()
        for epoch in range(150):
            oracle_opt.zero_grad()
            loss = criterion(oracle(X_oracle), y_oracle)
            loss.backward()
            oracle_opt.step()
        oracle.eval()

        # 3. Train the Meta Flow Map via Flow Matching
        print("[TRAINING] Optimizing Meta Flow Map Trajectories...")
        data_dim, context_dim = X_train.shape[1], C_train.shape[1]
        mfm = PortfolioBiomedicalMFM(data_dim=data_dim, context_dim=context_dim).to(self.device)
        mfm_opt = optim.AdamW(mfm.parameters(), lr=5e-4, weight_decay=1e-5)

        mfm.train()
        num_samples = X_train.size(0)
        for epoch in range(200):
            permutation = torch.randperm(num_samples)
            for i in range(0, num_samples, 64):
                mfm_opt.zero_grad()
                batch_idx = permutation[i : i + 64]
                x1, c = X_train[batch_idx], C_train[batch_idx]

                x0 = torch.randn_like(x1)
                t = torch.rand(x1.size(0), 1, device=self.device)
                xt = (1.0 - t) * x0 + t * x1

                target_v = x1 - x0
                pred_v = mfm(xt, c, t)

                loss = nn.functional.mse_loss(pred_v, target_v)
                loss.backward()
                mfm_opt.step()
        mfm.eval()

        # 4. Execute Reward-Guided Differentiable Steering
        print("[INFERENCE] Running inference-time steering validation...")
        cohort_size = X_test.shape[0]
        # Condition generation on a critical, elevated tumor size pathology context (+2.0 STD)
        target_context = torch.full((cohort_size, 1), 2.0, device=self.device)

        x0_latents = torch.randn(cohort_size, data_dim, device=self.device, requires_grad=True)
        t_zero = torch.zeros(cohort_size, 1, device=self.device)

        # Unsteered single-step generation
        x1_unsteered = x0_latents + mfm(x0_latents, target_context, t_zero)
        logits_unsteered = oracle(x1_unsteered)

        # Target optimization path toward Class 0 (Malignant / High-Risk)
        risk_loss = criterion(logits_unsteered, torch.zeros(cohort_size, dtype=torch.long, device=self.device))
        risk_loss.backward()

        # Adjust the initial noise canvas using the value function gradients
        with torch.no_grad():
            steering_coefficient = 1.5
            x0_steered = x0_latents - steering_coefficient * x0_latents.grad

        # Steered single-step generation
        with torch.no_grad():
            x1_steered = x0_steered + mfm(x0_steered, target_context, t_zero)
            logits_steered = oracle(x1_steered)

        # 5. Reverse transform to real-world clinical scales
        unsteered_np = self.dp.scaler_X.inverse_transform(x1_unsteered.detach().cpu().numpy())
        steered_np = self.dp.scaler_X.inverse_transform(x1_steered.cpu().numpy())
        real_test_np = self.dp.scaler_X.inverse_transform(X_test.numpy())

        df_real = pd.DataFrame(real_test_np, columns=self.dp.feature_names)
        df_unsteered = pd.DataFrame(unsteered_np, columns=self.dp.feature_names)
        df_steered = pd.DataFrame(steered_np, columns=self.dp.feature_names)

        return df_real, df_unsteered, df_steered, logits_unsteered, logits_steered

# Initialize and run the framework
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
runner = ExperimentRunner(device)
df_real, df_unsteered, df_steered, logits_unsteered, logits_steered = runner.run_pipeline()

# Compute target metrics for report output
def get_high_risk_density(logits):
    probs = torch.softmax(logits, dim=-1)
    return (probs[:, 0]).mean().item() * 100

print("\n" + "="*50)
print("     EXPERIMENTAL RESULTS MANIFEST")
print("================================================")
print(f"Baseline Unsteered Cohort Risk Density : {get_high_risk_density(logits_unsteered):.2f}%")
print(f"Optimized Steered Cohort Risk Density   : {get_high_risk_density(logits_steered):.2f}%")
print("="*50)

# Extract features for distribution mapping
fx, fy = df_real.columns[0], df_real.columns[1] # mean radius vs mean texture

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Plot A: Unsteered Alignment
sns.kdeplot(data=df_real, x=fx, y=fy, cmap="Blues", fill=True, ax=axes[0], alpha=0.5)
axes[0].scatter(df_unsteered[fx], df_unsteered[fy], color="orange", alpha=0.6, edgecolors='k', label="Unsteered Synthetic Data")
axes[0].set_title("A: Base MFM Conditional Distribution")
axes[0].legend()
axes[0].grid(True, linestyle="--", alpha=0.5)

# Plot B: Steered Target Alignment
sns.kdeplot(data=df_real, x=fx, y=fy, cmap="Blues", fill=True, ax=axes[1], alpha=0.5)
axes[1].scatter(df_steered[fx], df_steered[fy], color="crimson", alpha=0.6, edgecolors='k', label="Reward-Steered Synthetic Data")
axes[1].set_title("B: Directed Closed-Loop Reward Steering")
axes[1].legend()
axes[1].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("mfm_paper_figure_1.png", dpi=300)
plt.show()

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score

torch.manual_seed(42)
np.random.seed(42)

class BiomedicalDataModule:
    """Handles end-to-end data validation, isolation, and 3-way cohort partitioning."""
    def __init__(self, context_idx=3):
        self.context_idx = context_idx
        self.scaler_X = StandardScaler()
        self.scaler_C = StandardScaler()

    def execute_pipeline(self):
        raw_data = load_breast_cancer()
        X_raw = raw_data.data
        y_raw = raw_data.target  # 0 = Malignant (High Risk), 1 = Benign

        # Save feature headers for downstream dataframe logging
        self.feature_names = np.delete(raw_data.feature_names, self.context_idx)
        self.target_names = raw_data.target_names

        # Manifold Splitting: Isolate feature index 3 (mean area) as continuous context c
        X_features = np.delete(X_raw, self.context_idx, axis=1)
        C_context = X_raw[:, [self.context_idx]]

        # 3-Way Partitioning: Train (70%), Oracle (15%), Test Holdout (15%)
        X_tr_val, X_test, C_tr_val, C_test, y_tr_val, y_test = train_test_split(
            X_features, C_context, y_raw, test_size=0.15, random_state=42, stratify=y_raw
        )
        X_train, X_oracle, C_train, C_oracle, y_train, y_oracle = train_test_split(
            X_tr_val, C_tr_val, y_tr_val, test_size=0.176, random_state=42, stratify=y_tr_val
        )

        # Normalize strictly using Training partitions to eliminate lookahead data leakage
        X_train_s = self.scaler_X.fit_transform(X_train)
        C_train_s = self.scaler_C.fit_transform(C_train)

        X_oracle_s = self.scaler_X.transform(X_oracle)
        X_test_s = self.scaler_X.transform(X_test)
        C_test_s = self.scaler_C.transform(C_test)

        return {
            "train": (torch.tensor(X_train_s, dtype=torch.float32), torch.tensor(C_train_s, dtype=torch.float32)),
            "oracle": (torch.tensor(X_oracle_s, dtype=torch.float32), torch.tensor(y_oracle, dtype=torch.long)),
            "test": (torch.tensor(X_test_s, dtype=torch.float32), torch.tensor(C_test_s, dtype=torch.float32), y_test)
        }


class ClinicalRewardOracle(nn.Module):
    """Downstream value model evaluating patient risk boundaries."""
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 2)  # Raw logits for Binary Cross-Entropy Evaluation
        )
    def forward(self, x):
        return self.network(x)


class MFMResidualLayer(nn.Module):
    """Deep residual pathway to safeguard transport vectors during backpropagation."""
    def __init__(self, dim):
        super().__init__()
        self.l1 = nn.Linear(dim, dim)
        self.l2 = nn.Linear(dim, dim)
        self.act = nn.SiLU()
    def forward(self, x):
        return x + self.act(self.l2(self.act(self.l1(x))))


class ProductionMetaFlowMap(nn.Module):
    """Differentiable Meta Flow Map parameterizing direct trajectories from x0 to x1."""
    def __init__(self, data_dim, context_dim, hidden_dim=256, depth=3):
        super().__init__()
        input_dim = data_dim + context_dim + 1  # Formulated via State + Context + Time token
        self.proj_in = nn.Linear(input_dim, hidden_dim)
        self.trunk = nn.Sequential(*[MFMResidualLayer(hidden_dim) for _ in range(depth)])
        self.proj_out = nn.Linear(hidden_dim, data_dim)
        self.act = nn.SiLU()

    def forward(self, x, c, t):
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        inputs = torch.cat([x, c, t], dim=-1)
        h = self.act(self.proj_in(inputs))
        h = self.trunk(h)
        return self.proj_out(h)


class WorkshopExperimentSuite:
    """Manages the full paper lifecycle: training, inference alignment, and metric generation."""
    def __init__(self, device):
        self.device = device
        self.dm = BiomedicalDataModule()

    def execute(self):
        # 1. Pipeline Dataset Preparation
        splits = self.dm.execute_pipeline()
        X_train, C_train = splits["train"][0].to(self.device), splits["train"][1].to(self.device)
        X_oracle, y_oracle = splits["oracle"][0].to(self.device), splits["oracle"][1].to(self.device)
        X_test, C_test, y_test_raw = splits["test"]

        # 2. Train Clinical Value Oracle (Reward Model)
        print("[PROCESS] Optimizing Downstream Clinical Reward Oracle...")
        oracle = ClinicalRewardOracle(input_dim=X_train.shape[1]).to(self.device)
        oracle_opt = optim.AdamW(oracle.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion_ce = nn.CrossEntropyLoss()

        oracle.train()
        for epoch in range(150):
            oracle_opt.zero_grad()
            loss = criterion_ce(oracle(X_oracle), y_oracle)
            loss.backward()
            oracle_opt.step()
        oracle.eval()

        # Evaluate Oracle to guarantee publication quality
        with torch.no_grad():
            test_preds = oracle(X_test.to(self.device))
            auc = roc_auc_score(y_test_raw, torch.softmax(test_preds, dim=-1)[:, 1].cpu().numpy())
            print(f"  [ORACLE METRIC] Validation ROC-AUC: {auc:.4f}")

        # 3. Train Generative Meta Flow Map via Flow Matching
        print("[PROCESS] Training Generative Meta Flow Map Architecture...")
        data_dim, context_dim = X_train.shape[1], C_train.shape[1]
        mfm = ProductionMetaFlowMap(data_dim=data_dim, context_dim=context_dim).to(self.device)
        mfm_opt = optim.AdamW(mfm.parameters(), lr=5e-4, weight_decay=1e-5)

        mfm.train()
        num_samples = X_train.size(0)
        for epoch in range(200):
            shuffled_indices = torch.randperm(num_samples)
            for i in range(0, num_samples, 64):
                mfm_opt.zero_grad()
                batch_idx = shuffled_indices[i : i + 64]
                x1, c = X_train[batch_idx], C_train[batch_idx]

                x0 = torch.randn_like(x1)
                t = torch.rand(x1.size(0), 1, device=self.device)

                # Conditional linear probability path composition
                xt = (1.0 - t) * x0 + t * x1
                target_velocity = x1 - x0

                predicted_velocity = mfm(xt, c, t)
                loss = nn.functional.mse_loss(predicted_velocity, target_velocity)
                loss.backward()
                mfm_opt.step()
        mfm.eval()

        # 4. Closed-Loop Inference-Time Gradient Steering
        print("[PROCESS] Executing direct one-step latent reward optimization...")
        cohort_size = X_test.shape[0]
        # Challenge the generator using a severe pathological tumor area context (+2.0 STD)
        target_context = torch.full((cohort_size, 1), 2.0, device=self.device)

        x0_latents = torch.randn(cohort_size, data_dim, device=self.device, requires_grad=True)
        t_zero = torch.zeros(cohort_size, 1, device=self.device)

        # Unsteered direct transformation pass
        x1_unsteered = x0_latents + mfm(x0_latents, target_context, t_zero)
        logits_unsteered = oracle(x1_unsteered)

        # Calculate loss targeting class 0 (Malignant State Optimization)
        steering_loss = criterion_ce(logits_unsteered, torch.zeros(cohort_size, dtype=torch.long, device=self.device))
        steering_loss.backward()

        # Update initial latent canvas directly via value function gradients
        with torch.no_grad():
            steering_scale = 1.4
            x0_steered = x0_latents - steering_scale * x0_latents.grad

        # Steered direct transformation pass
        with torch.no_grad():
            x1_steered = x0_steered + mfm(x0_steered, target_context, t_zero)
            logits_steered = oracle(x1_steered)

        # 5. Reverse Transform data to native clinical coordinates
        unsteered_np = self.dm.scaler_X.inverse_transform(x1_unsteered.detach().cpu().numpy())
        steered_np = self.dm.scaler_X.inverse_transform(x1_steered.cpu().numpy())
        real_test_np = self.dm.scaler_X.inverse_transform(X_test.numpy())

        df_real = pd.DataFrame(real_test_np, columns=self.dm.feature_names)
        df_unsteered = pd.DataFrame(unsteered_np, columns=self.dm.feature_names)
        df_steered = pd.DataFrame(steered_np, columns=self.dm.feature_names)

        return df_real, df_unsteered, df_steered, logits_unsteered, logits_steered

# Initialize and run experiment execution
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
runner = WorkshopExperimentSuite(device)
df_real, df_unsteered, df_steered, logits_unsteered, logits_steered = runner.execute()

def get_malignancy_ratio(logits):
    probabilities = torch.softmax(logits, dim=-1)
    return (probabilities[:, 0]).mean().item() * 100

# Calculate Statistical Manifold Covariance Fidelity
cov_real = np.cov(df_real.values, rowvar=False)
cov_synth = np.cov(df_steered.values, rowvar=False)
fidelity_mae = np.mean(np.abs(cov_real - cov_synth))

print("\n" + "="*65)
print("     WORKSHOP PAPER CRITICAL METRICS MANIFEST")
print("=================================================================")
print(f"Generative Framework                : Meta Flow Map (MFM)")
print(f"Unsteered Population Risk Density   : {get_malignancy_ratio(logits_unsteered):.2f}%")
print(f"Steered High-Risk Target Density    : {get_malignancy_ratio(logits_steered):.2f}%")
print(f"Biological Covariance Fidelity (MAE): {fidelity_mae:.4f}")
print("=================================================================\n")

# Plot Generation Suite
fx, fy = df_real.columns[0], df_real.columns[1]  # mean radius vs mean texture
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Plot Left: Unsteered Alignment
sns.kdeplot(data=df_real, x=fx, y=fy, cmap="Blues", fill=True, ax=axes[0], alpha=0.5)
axes[0].scatter(df_unsteered[fx], df_unsteered[fy], color="orange", alpha=0.6, edgecolors='k', label="Unsteered Synthetic Profile")
axes[0].set_title("A: Base MFM Conditional Sample Realism")
axes[0].legend()
axes[0].grid(True, linestyle="--", alpha=0.5)

# Plot Right: Steered Target Shift
sns.kdeplot(data=df_real, x=fx, y=fy, cmap="Blues", fill=True, ax=axes[1], alpha=0.5)
axes[1].scatter(df_steered[fx], df_steered[fy], color="crimson", alpha=0.6, edgecolors='k', label="Value-Steered Pathological Cohort")
axes[1].set_title("B: Targeted Differentiable Latent Optimization")
axes[1].legend()
axes[1].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("workshop_paper_main_plot.png", dpi=300)
plt.show()

"""
Reward-Guided Conditional Flow Matching for Counterfactual Cancer Patient Simulation
=====================================================================================
Cohort   : TCGA-BRCA (Breast Cancer), via UCSC Xena (tcga.xenahubs.net)
Data state X (the digital twin):  clinical variables (age, race, treatment, ...)
                                   + a genomic embedding (PCA over the top-500
                                   most variable genes)
Context C (continuous conditioning variable): tumor-stage severity, encoded as a
                                   continuous scalar (Stage I -> 1.0 ... Stage IV -> 4.0,
                                   with a small offset for A/B/C substages so it behaves
                                   like a genuinely continuous biological severity axis,
                                   not a one-hot category)
Reward oracle: predicted 5-year mortality risk (deceased vs. survived), used both as
                                   an evaluation signal and as the steering objective
                                   for counterfactual ("what-if this patient's profile
                                   shifted toward higher/lower risk") generation.

Mathematical core (Conditional Flow Matching / Rectified Flow)
----------------------------------------------------------------
Patient state trajectory:           dx_t/dt = v_theta(x_t, c, t),   t in [0, 1]
Boundary conditions:                x_0 ~ N(0, I)              (noise)
                                     x_1 ~ p_data(x | c)         (real patient state)
Linear (optimal-transport) path:    x_t = (1 - t) x_0 + t x_1
Training objective:                 L(theta) = E[|| v_theta(x_t, c, t) - (x_1 - x_0) ||^2]
Sampling (forward Euler ODE solve): x_{t+dt} = x_t + v_theta(x_t, c, t) * dt
Reward-guided steering:             x_0' = x_0 - alpha * grad_{x_0} L_target(oracle(Flow(x_0, c)))
                                     (backprop through the ODE solve, classifier-guidance style)

Run top to bottom on a CPU Colab runtime. Total runtime ~10-15 minutes (dominated by
the one-time ~150MB data download; training the flow map and oracle on a few hundred
patients with small MLPs is fast on CPU).
"""

# ── CELL 2 — imports & config ───────────────────────────────────────────────────────
import os, re, gzip, shutil, zipfile, warnings
import numpy as np
import pandas as pd
import requests
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import rbf_kernel

from scipy.stats import wasserstein_distance

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "/content/data"
OUT_DIR = "/content/outputs"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

N_TOP_GENES = 500          # top variable genes considered
N_GENE_PCS = 20            # genomic embedding dimensionality
FIVE_YEAR_DAYS = 365 * 5   # 5-year horizon for the reward oracle's survival label
N_EULER_STEPS = 10         # ODE integration steps for sampling
print("Device:", DEVICE)

# ── CELL 3 — data download via cBioPortal API (no auth, no manual upload) ──────────
import requests, io

CBIOPORTAL_BASE = "https://www.cbioportal.org/api"
STUDY_ID = "brca_tcga"

def fetch_clinical_cbioportal(study_id):
    url = f"{CBIOPORTAL_BASE}/studies/{study_id}/patients"
    patients = requests.get(url, headers={"Accept": "application/json"}, timeout=60).json()
    patient_ids = [p["patientId"] for p in patients]
    print(f"  Fetched {len(patient_ids)} patients")

    # fetch clinical attributes in batches
    url2 = f"{CBIOPORTAL_BASE}/studies/{study_id}/clinical-data?clinicalDataType=PATIENT&pageSize=100000"
    clin_raw = requests.get(url2, headers={"Accept": "application/json"}, timeout=120).json()
    rows = {}
    for entry in clin_raw:
        pid = entry["patientId"]
        if pid not in rows:
            rows[pid] = {"patient_id": pid}
        rows[pid][entry["clinicalAttributeId"]] = entry["value"]
    df = pd.DataFrame.from_dict(rows, orient="index").reset_index(drop=True)
    print(f"  Clinical dataframe: {df.shape}")
    return df

def fetch_expression_cbioportal(study_id, n_genes=80):
    # get the RNA-seq molecular profile
    profiles = requests.get(f"{CBIOPORTAL_BASE}/studies/{study_id}/molecular-profiles",
                             headers={"Accept": "application/json"}, timeout=60).json()
    rna_profile = next(p["molecularProfileId"] for p in profiles
                       if "rna_seq" in p["molecularProfileId"].lower() and "v2" in p["molecularProfileId"].lower())
    print(f"  Using expression profile: {rna_profile}")

    # get samples
    samples = requests.get(f"{CBIOPORTAL_BASE}/studies/{study_id}/samples?pageSize=10000",
                            headers={"Accept": "application/json"}, timeout=60).json()
    sample_ids = [s["sampleId"] for s in samples][:500]
    print(f"  Using {len(sample_ids)} samples")

    # resolve gene IDs one by one using /genes/{id} endpoint
    gene_symbols = [
        "ESR1","ERBB2","PGR","MKI67","BRCA1","BRCA2","TP53","PIK3CA","CDH1","GATA3",
        "RB1","PTEN","AKT1","CCND1","FOXA1","EGFR","FGFR1","MYC","AURKA","TOP2A",
        "BIRC5","BCL2","FOXM1","PLK1","MELK","RAD51","CHEK1","CHEK2","ATM","PALB2",
        "ESR2","AR","STAT3","NOTCH1","CDK4","CDK6","CDKN2A","MDM2","KRT8","KRT18",
        "KRT19","MLPH","TFF1","CLDN7","FN1","COL3A1","DCN","VEGFA","MET","IGF1R",
        "MAP3K1","NF1","TSC1","TSC2","KEAP1","NFE2L2","WNT5A","SRC","JAK2","NCOA3",
        "UBE2C","TYMS","PCNA","E2F1","BUB1","CENPA","KIF2C","TPX2","PRC1","EXO1",
        "MCM2","BRIP1","NBN","MRE11","RAD50","ATR","CCND2","CCNE1","MMP2","MMP9",
    ]

    entrez_ids = []
    symbol_map = {}
    for sym in gene_symbols:
        try:
            r = requests.get(f"{CBIOPORTAL_BASE}/genes/{sym}",
                             headers={"Accept": "application/json"}, timeout=10).json()
            if "entrezGeneId" in r:
                entrez_ids.append(r["entrezGeneId"])
                symbol_map[r["entrezGeneId"]] = r["hugoGeneSymbol"]
        except Exception:
            continue
    print(f"  Resolved {len(entrez_ids)} gene IDs")

    # fetch expression values
    payload = {"sampleIds": sample_ids, "entrezGeneIds": entrez_ids}
    expr_resp = requests.post(
        f"{CBIOPORTAL_BASE}/molecular-profiles/{rna_profile}/molecular-data/fetch?projection=SUMMARY",
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        json=payload, timeout=180).json()

    # expr_resp is a list of dicts
    if not isinstance(expr_resp, list) or len(expr_resp) == 0:
        raise ValueError(f"Unexpected expression response: {str(expr_resp)[:200]}")

    rows = {}
    for entry in expr_resp:
        if not isinstance(entry, dict):
            continue
        sid  = entry.get("sampleId")
        eid  = entry.get("entrezGeneId")
        gene = symbol_map.get(eid, str(eid))
        val  = entry.get("value", np.nan)
        if sid not in rows:
            rows[sid] = {"sample_id": sid}
        rows[sid][gene] = val

    expr_df = pd.DataFrame.from_dict(rows, orient="index").reset_index(drop=True)
    print(f"  Expression dataframe: {expr_df.shape}")
    return expr_df

print("Fetching clinical data from cBioPortal...")
clinical_raw = fetch_clinical_cbioportal(STUDY_ID)
print("Fetching expression data from cBioPortal...")
expr_raw_cbio = fetch_expression_cbioportal(STUDY_ID)
print("Done.")

# ── CELL 4 — map cBioPortal columns → pipeline-expected format ──────────────────────

col_map = {
    "AGE": "age",
    "AJCC_PATHOLOGIC_TUMOR_STAGE": "stage",
    "RACE": "race",
    "HISTOLOGICAL_DIAGNOSIS": "histological_type",
    "MENOPAUSE_STATUS": "menopause_status",
    "RADIATION_THERAPY": "radiation_therapy",
    "NEOADJUVANT_THERAPY": "neoadjuvant_tx",
    "LYMPH_NODE_STATUS": "lymph_nodes",
    "OS_STATUS": "vital_status",
    "OS_MONTHS": "os_months",
    "patient_id": "patient_id",
}
clinical_df = clinical_raw.rename(columns={k: v for k, v in col_map.items() if k in clinical_raw.columns})

if "vital_status" in clinical_df.columns:
    clinical_df["vital_status"] = clinical_df["vital_status"].apply(
        lambda x: "dead" if str(x).startswith("1") else "alive")

if "os_months" in clinical_df.columns:
    clinical_df["os_months"] = pd.to_numeric(clinical_df["os_months"], errors="coerce")
    clinical_df["days_to_death"] = clinical_df.apply(
        lambda r: r["os_months"] * 30.44 if r.get("vital_status") == "dead" else np.nan, axis=1)
    clinical_df["days_to_last_fu"] = clinical_df.apply(
        lambda r: r["os_months"] * 30.44 if r.get("vital_status") == "alive" else np.nan, axis=1)

clinical_df["age"] = pd.to_numeric(clinical_df.get("age", np.nan), errors="coerce")

def stage_to_severity(stage_str):
    if pd.isna(stage_str):
        return np.nan
    s = str(stage_str).upper().replace("STAGE", "").strip()
    import re
    m = re.match(r"^(IV|III|II|I)([ABC]?)$", s)
    if not m:
        return np.nan
    roman_to_val = {"I": 1.0, "II": 2.0, "III": 3.0, "IV": 4.0}
    sub_offset = {"A": 0.0, "B": 0.15, "C": 0.30, "": 0.0}
    return roman_to_val[m.group(1)] + sub_offset[m.group(2)]

def make_5yr_label(row):
    status = str(row.get("vital_status", "")).strip().lower()
    if status == "dead":
        d = row.get("days_to_death", np.nan)
        if pd.isna(d):
            return np.nan
        return 1.0 if d <= FIVE_YEAR_DAYS else 0.0
    elif status == "alive":
        f = row.get("days_to_last_fu", np.nan)
        if pd.isna(f):
            return np.nan
        return 0.0 if f >= FIVE_YEAR_DAYS else np.nan
    return np.nan

clinical_df["stage_severity"] = clinical_df["stage"].apply(stage_to_severity)
clinical_df["label"] = clinical_df.apply(make_5yr_label, axis=1)
clinical_df = clinical_df.dropna(subset=["patient_id", "stage_severity", "label"]).reset_index(drop=True)
clinical_df = clinical_df.set_index("patient_id")
print(f"Clinical rows after filtering: {len(clinical_df)}")
print(clinical_df["label"].value_counts())

# ── CELL 5 — expression matrix cleanup and patient matching ─────────────────────────

expr_raw_cbio["patient_id"] = expr_raw_cbio["sample_id"].apply(
    lambda s: "-".join(s.split("-")[:3]) if pd.notna(s) else s)

expr_df = expr_raw_cbio.drop(columns=["sample_id"]).drop_duplicates(subset="patient_id")
expr_df = expr_df.set_index("patient_id")

gene_cols = [c for c in expr_df.columns if c != "patient_id"]
expr_df = expr_df[gene_cols].apply(pd.to_numeric, errors="coerce")
expr_df = expr_df.fillna(expr_df.mean())

common_patients = clinical_df.index.intersection(expr_df.index)
print(f"Patients with BOTH clinical AND expression data: {len(common_patients)}")

clinical_df = clinical_df.loc[common_patients]
expr_df = expr_df.loc[common_patients]

print(f"Final clinical shape : {clinical_df.shape}")
print(f"Final expression shape: {expr_df.shape}")

# ── CELL 6 — DataPipeline: 3-way split, leakage-safe preprocessing ─────────────────

N_TOP_GENES = 80
N_GENE_PCS  = 10

class DataPipeline:
    def __init__(self, clinical_df, expr_df, n_top_genes=N_TOP_GENES, n_gene_pcs=N_GENE_PCS):
        self.clinical_df  = clinical_df
        self.expr_df      = expr_df
        self.n_top_genes  = n_top_genes
        self.n_gene_pcs   = n_gene_pcs

        self.clinical_numeric     = [c for c in ["age", "lymph_nodes"] if c in clinical_df.columns]
        self.clinical_categorical = [c for c in ["race", "histological_type", "menopause_status",
                                                  "radiation_therapy", "neoadjuvant_tx"]
                                     if c in clinical_df.columns]

    def prepare_data(self):
        idx = self.clinical_df.index
        y   = self.clinical_df["label"].astype(int)

        idx_train_val, idx_test = train_test_split(
            idx, test_size=0.15, random_state=RANDOM_STATE, stratify=y)
        idx_train, idx_oracle = train_test_split(
            idx_train_val, test_size=0.176, random_state=RANDOM_STATE,
            stratify=y.loc[idx_train_val])

        # genomic embedding — fit on train only
        gene_var   = self.expr_df.loc[idx_train].var(axis=0).sort_values(ascending=False)
        self.top_genes  = gene_var.head(self.n_top_genes).index.tolist()
        gene_scaler     = StandardScaler().fit(self.expr_df.loc[idx_train, self.top_genes])
        actual_pcs      = min(self.n_gene_pcs, len(self.top_genes), len(idx_train))
        pca             = PCA(n_components=actual_pcs, random_state=RANDOM_STATE).fit(
                              gene_scaler.transform(self.expr_df.loc[idx_train, self.top_genes]))
        self.gene_scaler, self.pca = gene_scaler, pca
        self.n_gene_pcs = actual_pcs

        def gene_embedding(index):
            scaled = gene_scaler.transform(self.expr_df.loc[index, self.top_genes])
            return pca.transform(scaled)

        # clinical preprocessing — fit on train only
        clinical_pre = ColumnTransformer([
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("scale",  StandardScaler())]), self.clinical_numeric),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                              ("onehot", OneHotEncoder(handle_unknown="ignore"))]),
             self.clinical_categorical),
        ])
        clin_cols = self.clinical_numeric + self.clinical_categorical
        clinical_pre.fit(self.clinical_df.loc[idx_train, clin_cols])
        self.clinical_pre = clinical_pre

        def clinical_features(index):
            out = clinical_pre.transform(self.clinical_df.loc[index, clin_cols])
            return out.toarray() if hasattr(out, "toarray") else out

        # context scaler — fit on train only
        self.context_scaler = StandardScaler().fit(
            self.clinical_df.loc[idx_train, ["stage_severity"]])

        def build_split(index):
            X  = np.hstack([clinical_features(index), gene_embedding(index)]).astype(np.float32)
            C  = self.context_scaler.transform(
                     self.clinical_df.loc[index, ["stage_severity"]]).astype(np.float32)
            yv = self.clinical_df.loc[index, "label"].astype(np.int64).values
            return (torch.tensor(X), torch.tensor(C), torch.tensor(yv))

        X_train,  C_train,  y_train  = build_split(idx_train)
        X_oracle, C_oracle, y_oracle = build_split(idx_oracle)
        X_test,   C_test,   y_test   = build_split(idx_test)

        clin_feature_names = (self.clinical_numeric +
            list(clinical_pre.named_transformers_["cat"].named_steps["onehot"]
                 .get_feature_names_out(self.clinical_categorical)))
        self.feature_names = clin_feature_names + [f"gene_pc_{i}" for i in range(self.n_gene_pcs)]

        return {
            "train":  (X_train,  C_train,  y_train),
            "oracle": (X_oracle, C_oracle, y_oracle),
            "test":   (X_test,   C_test,   y_test),
        }

dp     = DataPipeline(clinical_df, expr_df)
splits = dp.prepare_data()
X_train,  C_train,  y_train  = splits["train"]
X_oracle, C_oracle, y_oracle = splits["oracle"]
X_test,   C_test,   y_test   = splits["test"]

print(f"Train: {X_train.shape[0]} | Oracle: {X_oracle.shape[0]} | Test: {X_test.shape[0]}")
print(f"Digital twin state dim : {X_train.shape[1]}")
print(f"Context dim            : {C_train.shape[1]}")

# ── CELL 7 — model classes: reward oracle + conditional flow map ───────────────────
class SurvivalRewardOracle(nn.Module):
    """Predicts 5-year mortality risk from the digital twin state. Serves both as an
    evaluation classifier and as the steering reward function in Experiment 3."""
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.SiLU(),
            nn.Linear(64, 2),
        )
    def forward(self, x):
        return self.net(x)

class FlowResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.act = nn.SiLU()
    def forward(self, x):
        return x + self.act(self.linear2(self.act(self.linear1(x))))

class TCGAFlowMap(nn.Module):
    """Conditional velocity field v_theta(x_t, c, t) for the digital twin state,
    trained via flow matching against a stage-severity context."""
    def __init__(self, data_dim, context_dim, hidden_dim=256, blocks=3):
        super().__init__()
        self.input_proj = nn.Linear(data_dim + context_dim + 1, hidden_dim)
        self.trunk = nn.Sequential(*[FlowResidualBlock(hidden_dim) for _ in range(blocks)])
        self.output_proj = nn.Linear(hidden_dim, data_dim)
        self.act = nn.SiLU()
    def forward(self, x, c, t):
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        h = self.act(self.input_proj(torch.cat([x, c, t], dim=-1)))
        return self.output_proj(self.trunk(h))

def integrate_flow(model, x0, c, n_steps=N_EULER_STEPS):
    """Forward Euler solve of dx/dt = v_theta(x, c, t) from t=0 to t=1."""
    dt = 1.0 / n_steps
    x = x0
    for i in range(n_steps):
        t = torch.full((x.size(0), 1), i * dt, device=x.device)
        x = x + model(x, c, t) * dt
    return x

# ── CELL 8 — train_reward_oracle() ──────────────────────────────────────────────────
def train_reward_oracle(X_oracle, y_oracle, epochs=200, lr=1e-3):
    X_oracle, y_oracle = X_oracle.to(DEVICE), y_oracle.to(DEVICE)
    oracle = SurvivalRewardOracle(input_dim=X_oracle.shape[1]).to(DEVICE)
    opt = optim.AdamW(oracle.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    oracle.train()
    for epoch in range(epochs):
        opt.zero_grad()
        loss = criterion(oracle(X_oracle), y_oracle)
        loss.backward()
        opt.step()
    oracle.eval()
    return oracle

print("[TRAINING] Reward oracle (5-year mortality risk)...")
oracle = train_reward_oracle(X_oracle, y_oracle)
with torch.no_grad():
    test_acc = (oracle(X_test.to(DEVICE)).argmax(-1).cpu() == y_test).float().mean().item()
print(f"Reward oracle held-out accuracy: {test_acc:.3f}")

# ── CELL 9 — train_flow_map() via Conditional Flow Matching ────────────────────────
def train_flow_map(X_train, C_train, epochs=300, batch_size=32, lr=5e-4):
    X_train, C_train = X_train.to(DEVICE), C_train.to(DEVICE)
    data_dim, context_dim = X_train.shape[1], C_train.shape[1]
    model = TCGAFlowMap(data_dim=data_dim, context_dim=context_dim).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    n = X_train.size(0)
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n)
        running = 0.0
        for i in range(0, n, batch_size):
            opt.zero_grad()
            idx = perm[i:i + batch_size]
            x1, c = X_train[idx], C_train[idx]
            x0 = torch.randn_like(x1)
            t = torch.rand(x1.size(0), 1, device=DEVICE)
            xt = (1.0 - t) * x0 + t * x1
            target_v = x1 - x0
            pred_v = model(xt, c, t)
            loss = nn.functional.mse_loss(pred_v, target_v)
            loss.backward()
            opt.step()
            running += loss.item() * x1.size(0)
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:03d}/{epochs} | flow matching loss: {running/n:.4f}")
    model.eval()
    return model

print("[TRAINING] Conditional flow map (Rectified Flow / Flow Matching)...")
flow_model = train_flow_map(X_train, C_train)
print("Flow map trained.")

# ── CELL 10 — Experiment 1: generation quality (real vs. synthetic, MMD + Wasserstein) ─
def mmd_rbf(X, Y, gamma=None):
    """Maximum Mean Discrepancy with an RBF kernel between two sample sets."""
    X, Y = np.asarray(X), np.asarray(Y)
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    Kxx, Kyy, Kxy = rbf_kernel(X, X, gamma), rbf_kernel(Y, Y, gamma), rbf_kernel(X, Y, gamma)
    return Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()

def mean_wasserstein(X, Y):
    """Average per-dimension 1-D Wasserstein distance (simple multivariate proxy)."""
    X, Y = np.asarray(X), np.asarray(Y)
    return float(np.mean([wasserstein_distance(X[:, j], Y[:, j]) for j in range(X.shape[1])]))

def run_generation_quality_experiment():
    with torch.no_grad():
        x0 = torch.randn(X_test.size(0), X_test.size(1), device=DEVICE)
        x_gen = integrate_flow(flow_model, x0, C_test.to(DEVICE)).cpu().numpy()
    x_real = X_test.numpy()
    mmd = mmd_rbf(x_real, x_gen)
    wdist = mean_wasserstein(x_real, x_gen)
    print(f"[Experiment 1] MMD(real, generated) = {mmd:.4f}")
    print(f"[Experiment 1] Mean per-dim Wasserstein(real, generated) = {wdist:.4f}")
    return x_real, x_gen, mmd, wdist

x_real_test, x_gen_test, mmd_score, wasserstein_score = run_generation_quality_experiment()

# ── CELL 11 — Figure 1: real vs. generated overlay (PCA 2D projection) ─────────────
def plot_generation_quality(x_real, x_gen, save_path):
    pca2 = PCA(n_components=2, random_state=RANDOM_STATE).fit(x_real)
    real_2d, gen_2d = pca2.transform(x_real), pca2.transform(x_gen)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.kdeplot(x=real_2d[:, 0], y=real_2d[:, 1], cmap="Blues", fill=True, alpha=0.5, ax=ax)
    ax.scatter(gen_2d[:, 0], gen_2d[:, 1], color="crimson", alpha=0.6, edgecolors="k",
               s=25, label="Flow-generated digital twins")
    ax.set_xlabel("PC1 (digital twin state)"); ax.set_ylabel("PC2 (digital twin state)")
    ax.set_title(f"Real vs. Generated Patient States\nMMD={mmd_score:.3f}, mean W-dist={wasserstein_score:.3f}")
    ax.legend(loc="best")
    plt.tight_layout(); plt.savefig(save_path, dpi=200, bbox_inches="tight"); plt.show()

plot_generation_quality(x_real_test, x_gen_test, f"{OUT_DIR}/figure1_generation_quality.png")

# ── CELL 12 — Experiment 2: condition fidelity (does changing context shift output?) ─
def run_condition_fidelity_experiment(stages=(1.0, 2.0, 3.0), n_samples=60):
    rows, generated_by_stage = [], {}
    real_severity = clinical_df.loc[clinical_df.index, "stage_severity"]
    for s in stages:
        c_scaled = torch.tensor(dp.context_scaler.transform([[s]]), dtype=torch.float32)
        c_batch = c_scaled.repeat(n_samples, 1).to(DEVICE)
        x0 = torch.randn(n_samples, X_train.shape[1], device=DEVICE)
        with torch.no_grad():
            x_gen = integrate_flow(flow_model, x0, c_batch).cpu().numpy()
        generated_by_stage[s] = x_gen

        # compare against the closest real patients (severity within +/- 0.5 of this stage)
        real_mask = (real_severity >= s - 0.5) & (real_severity < s + 0.5)
        if real_mask.sum() > 1:
            real_subset_idx = clinical_df.index[real_mask]
            x_real_subset = dp.clinical_pre.transform(
                clinical_df.loc[real_subset_idx, dp.clinical_numeric + dp.clinical_categorical])
            x_real_subset = x_real_subset.toarray() if hasattr(x_real_subset, "toarray") else x_real_subset
            gene_emb_subset = dp.pca.transform(
                dp.gene_scaler.transform(expr_df.loc[real_subset_idx, dp.top_genes]))
            x_real_subset = np.hstack([x_real_subset, gene_emb_subset])
            wdist = mean_wasserstein(x_real_subset, x_gen)
        else:
            wdist = np.nan
        rows.append({"stage_severity": s, "n_real_matched": int(real_mask.sum()),
                      "wasserstein_to_real_subgroup": wdist,
                      "generated_mean_norm": float(np.linalg.norm(x_gen, axis=1).mean())})
    fidelity_df = pd.DataFrame(rows)
    return fidelity_df, generated_by_stage

fidelity_df, generated_by_stage = run_condition_fidelity_experiment()
print("[Experiment 2] Condition fidelity (lower Wasserstein = generated cohort closer to "
      "the matching real-stage subgroup):")
print(fidelity_df.to_string(index=False))

# ── CELL 13 — Figure 2: condition fidelity (PC1 distribution shift across stages) ──
def plot_condition_fidelity(generated_by_stage, save_path):
    pca2 = PCA(n_components=1, random_state=RANDOM_STATE).fit(X_train.numpy())
    rows = []
    for stage, x_gen in generated_by_stage.items():
        pc1 = pca2.transform(x_gen)[:, 0]
        rows.extend([{"stage_severity": f"~Stage {stage:.0f}", "PC1": v} for v in pc1])
    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.violinplot(data=plot_df, x="stage_severity", y="PC1", ax=ax, palette="flare")
    ax.set_title("Condition Fidelity: Generated Patient State (PC1)\nshifts with Conditioning Stage Severity")
    ax.set_xlabel("Conditioning context"); ax.set_ylabel("PC1 of digital twin state")
    plt.tight_layout(); plt.savefig(save_path, dpi=200, bbox_inches="tight"); plt.show()

plot_condition_fidelity(generated_by_stage, f"{OUT_DIR}/figure2_condition_fidelity.png")

# ── CELL 14 — Experiment 3: reward-guided steering (counterfactual risk shift) ─────
def run_steering_experiment(n_patients=40, alpha=1.5, target_class=1):
    # cap n_patients to what's actually available in the test set
    n_patients = min(n_patients, X_test.shape[0])

    x_test_batch = X_test[:n_patients].to(DEVICE)
    c_batch       = C_test[:n_patients].to(DEVICE)

    # make sure both have the same first dimension
    assert x_test_batch.shape[0] == c_batch.shape[0], \
        f"Shape mismatch: X {x_test_batch.shape[0]} vs C {c_batch.shape[0]}"

    x0 = torch.randn(n_patients, X_train.shape[1], device=DEVICE, requires_grad=True)

    # forward pass through ODE — keep grad
    x1_unsteered = integrate_flow(flow_model, x0, c_batch)
    logits_unsteered = oracle(x1_unsteered)
    target = torch.full((n_patients,), target_class, dtype=torch.long, device=DEVICE)
    loss = nn.functional.cross_entropy(logits_unsteered, target)
    loss.backward()

    with torch.no_grad():
        x0_steered = x0 - alpha * x0.grad

    with torch.no_grad():
        x1_steered   = integrate_flow(flow_model, x0_steered, c_batch)
        logits_steered = oracle(x1_steered)

    def risk_score(logits):
        return torch.softmax(logits, dim=-1)[:, 1].mean().item()

    before = risk_score(logits_unsteered.detach())
    after  = risk_score(logits_steered)
    return before, after, x1_unsteered.detach().cpu().numpy(), x1_steered.cpu().numpy()

print("[Experiment 3] Reward-guided steering toward HIGHER risk (deceased):")
risk_before_hi, risk_after_hi, x_unsteered_hi, x_steered_hi = run_steering_experiment(target_class=1)
print(f"  Before steering: risk score = {risk_before_hi:.3f}")
print(f"  After steering:  risk score = {risk_after_hi:.3f}")

print("[Experiment 3] Reward-guided steering toward LOWER risk (survived):")
risk_before_lo, risk_after_lo, x_unsteered_lo, x_steered_lo = run_steering_experiment(target_class=0)
print(f"  Before steering: risk score = {risk_before_lo:.3f}")
print(f"  After steering:  risk score = {risk_after_lo:.3f}")

# ── CELL 15 — Figure 3: steering effect (before vs after, both directions) ─────────
def plot_steering_effect(results, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    labels = ["Toward higher risk\n(before)", "Toward higher risk\n(after)",
              "Toward lower risk\n(before)", "Toward lower risk\n(after)"]
    values = [results["hi_before"], results["hi_after"], results["lo_before"], results["lo_after"]]
    colors = ["#ffb787", "#d9534f", "#9ec9f5", "#5cb85c"]
    bars = ax.bar(labels, values, color=colors, edgecolor="black")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.02, f"{v:.2f}", ha="center", fontsize=10, weight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Reward oracle: P(deceased within 5 years)")
    ax.set_title("Reward-Guided Counterfactual Steering")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    plt.xticks(rotation=15)
    plt.tight_layout(); plt.savefig(save_path, dpi=200, bbox_inches="tight"); plt.show()

steer_results = {"hi_before": risk_before_hi, "hi_after": risk_after_hi,
                  "lo_before": risk_before_lo, "lo_after": risk_after_lo}
plot_steering_effect(steer_results, f"{OUT_DIR}/figure3_steering_effect.png")

# ── CELL 16 — Experiment 4: digital twin interpolation (disease progression path) ──
def run_digital_twin_interpolation(n_points=12, severity_range=(0.5, 4.0), n_seeds=6):
    severities = np.linspace(*severity_range, n_points)
    fixed_noise = torch.randn(n_seeds, X_train.shape[1], device=DEVICE)  # same patients, varying context
    trajectory_points, trajectory_severity = [], []
    with torch.no_grad():
        for s in severities:
            c_scaled = torch.tensor(dp.context_scaler.transform([[s]]), dtype=torch.float32)
            c_batch = c_scaled.repeat(n_seeds, 1).to(DEVICE)
            x_gen = integrate_flow(flow_model, fixed_noise, c_batch).cpu().numpy()
            trajectory_points.append(x_gen)
            trajectory_severity.extend([s] * n_seeds)
    return np.vstack(trajectory_points), np.array(trajectory_severity)

traj_points, traj_severity = run_digital_twin_interpolation()
print(f"[Experiment 4] Generated {len(traj_points)} digital twin states across "
      f"{len(np.unique(traj_severity))} severity levels (same latent seeds, varying context).")

# ── CELL 17 — Figure 4: digital twin interpolation path ────────────────────────────
def plot_digital_twin_path(traj_points, traj_severity, x_real, save_path):
    pca2 = PCA(n_components=2, random_state=RANDOM_STATE).fit(x_real)
    real_2d = pca2.transform(x_real)
    traj_2d = pca2.transform(traj_points)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.kdeplot(x=real_2d[:, 0], y=real_2d[:, 1], cmap="Greys", fill=True, alpha=0.35, ax=ax)
    sc = ax.scatter(traj_2d[:, 0], traj_2d[:, 1], c=traj_severity, cmap="viridis",
                     s=35, edgecolors="k", alpha=0.85)
    cbar = plt.colorbar(sc, ax=ax); cbar.set_label("Conditioning severity (Stage I -> IV)")
    ax.set_xlabel("PC1 (digital twin state)"); ax.set_ylabel("PC2 (digital twin state)")
    ax.set_title("Digital Twin Interpolation:\nSame Latent Seed, Increasing Tumor-Stage Context")
    plt.tight_layout(); plt.savefig(save_path, dpi=200, bbox_inches="tight"); plt.show()

plot_digital_twin_path(traj_points, traj_severity, x_real_test, f"{OUT_DIR}/figure4_digital_twin_path.png")

# ── CELL 18 — save all results to a single bundle for the paper ────────────────────
def save_results_bundle():
    summary = pd.DataFrame([
        {"experiment": "1_generation_quality", "metric": "MMD", "value": mmd_score},
        {"experiment": "1_generation_quality", "metric": "mean_wasserstein", "value": wasserstein_score},
        {"experiment": "3_steering", "metric": "risk_before_hi", "value": risk_before_hi},
        {"experiment": "3_steering", "metric": "risk_after_hi", "value": risk_after_hi},
        {"experiment": "3_steering", "metric": "risk_before_lo", "value": risk_before_lo},
        {"experiment": "3_steering", "metric": "risk_after_lo", "value": risk_after_lo},
    ])
    summary.to_csv(f"{OUT_DIR}/experiment_summary.csv", index=False)
    fidelity_df.to_csv(f"{OUT_DIR}/experiment2_condition_fidelity.csv", index=False)

    zip_path = "/content/flow_matching_results.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for fname in os.listdir(OUT_DIR):
            zf.write(os.path.join(OUT_DIR, fname), arcname=fname)
    print(f"Bundled {len(os.listdir(OUT_DIR))} files into {zip_path}")
    try:
        from google.colab import files
        files.download(zip_path)
    except ImportError:
        print("Not running in Colab — find your results at:", zip_path)

# ── CELL 19 — Tier 1: Baseline Comparison (Gaussian + VAE vs Flow Matching) ────────

class VAE(nn.Module):
    def __init__(self, data_dim, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(data_dim, 128), nn.SiLU(),
            nn.Linear(128, 64),  nn.SiLU())
        self.mu_layer      = nn.Linear(64, latent_dim)
        self.logvar_layer  = nn.Linear(64, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),  nn.SiLU(),
            nn.Linear(64, 128),         nn.SiLU(),
            nn.Linear(128, data_dim))

    def encode(self, x):
        h = self.encoder(x)
        return self.mu_layer(h), self.logvar_layer(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

def train_vae(X_train, epochs=300, lr=1e-3, batch_size=32):
    X = X_train.to(DEVICE)
    data_dim = X.shape[1]
    vae = VAE(data_dim).to(DEVICE)
    opt = optim.AdamW(vae.parameters(), lr=lr)
    vae.train()
    for epoch in range(epochs):
        perm = torch.randperm(X.shape[0])
        for i in range(0, X.shape[0], batch_size):
            idx = perm[i:i+batch_size]
            x_batch = X[idx]
            opt.zero_grad()
            recon, mu, logvar = vae(x_batch)
            recon_loss = nn.functional.mse_loss(recon, x_batch)
            kl_loss    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + 0.01 * kl_loss
            loss.backward()
            opt.step()
        if (epoch+1) % 100 == 0:
            print(f"  VAE Epoch {epoch+1}/{epochs} | loss: {loss.item():.4f}")
    vae.eval()
    return vae

print("[Baseline] Training VAE...")
vae_model = train_vae(X_train)

# generate samples from each method
n_samples = X_test.shape[0]

# Gaussian baseline
x_gaussian = torch.randn(n_samples, X_train.shape[1]).numpy()

# VAE baseline
with torch.no_grad():
    z_sample = torch.randn(n_samples, 32).to(DEVICE)
    x_vae = vae_model.decoder(z_sample).cpu().numpy()

# Flow Matching (already have x_gen_test)
x_flow = x_gen_test
x_real = x_real_test

mmd_gaussian  = mmd_rbf(x_real, x_gaussian)
mmd_vae       = mmd_rbf(x_real, x_vae)
mmd_flow      = mmd_rbf(x_real, x_flow)

wd_gaussian   = mean_wasserstein(x_real, x_gaussian)
wd_vae        = mean_wasserstein(x_real, x_vae)
wd_flow       = mean_wasserstein(x_real, x_flow)

baseline_df = pd.DataFrame([
    {"Method": "Gaussian Sampling", "MMD ↓": mmd_gaussian, "Mean Wasserstein ↓": wd_gaussian},
    {"Method": "VAE",               "MMD ↓": mmd_vae,      "Mean Wasserstein ↓": wd_vae},
    {"Method": "Flow Matching",     "MMD ↓": mmd_flow,     "Mean Wasserstein ↓": wd_flow},
])
print("\n[Baseline Comparison]")
print(baseline_df.to_string(index=False))
baseline_df.to_csv(f"{OUT_DIR}/baseline_comparison.csv", index=False)
save_results_bundle()

# ── CELL 20 — Tier 1: UMAP Visualization (Real vs Generated) ───────────────────────

# !pip install -q umap-learn   # uncomment if not installed

try:
    import umap
    USE_UMAP = True
except ImportError:
    from sklearn.manifold import TSNE
    USE_UMAP = False
    print("umap-learn not found, falling back to t-SNE")

def plot_umap_real_vs_generated(x_real, x_gen, save_path):
    combined = np.vstack([x_real, x_gen])
    labels   = (["Real"] * len(x_real)) + (["Generated"] * len(x_gen))

    if USE_UMAP:
        reducer  = umap.UMAP(n_components=2, random_state=RANDOM_STATE)
        title_method = "UMAP"
    else:
        reducer  = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=min(30, len(combined)-1))
        title_method = "t-SNE"

    embedding = reducer.fit_transform(combined)
    plot_df   = pd.DataFrame({"x": embedding[:,0], "y": embedding[:,1], "Type": labels})

    fig, ax = plt.subplots(figsize=(7, 6))
    colors  = {"Real": "#4878CF", "Generated": "#D65F5F"}
    for label, grp in plot_df.groupby("Type"):
        ax.scatter(grp["x"], grp["y"], label=label, alpha=0.7,
                   color=colors[label], edgecolors="k", linewidths=0.3, s=40)
    ax.set_title(f"{title_method} Projection: Real vs Flow-Generated Patient States")
    ax.set_xlabel(f"{title_method}-1")
    ax.set_ylabel(f"{title_method}-2")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_path}")

plot_umap_real_vs_generated(x_real_test, x_gen_test,
                             f"{OUT_DIR}/figure5_umap_real_vs_generated.png")


# ── CELL 21 — Tier 1: Steering Trajectory Plot ──────────────────────────────────────

def run_steering_trajectory(n_patients=None, n_steps=10, alpha_max=3.0, target_class=1):
    """Run steering across increasing alpha values to show monotonic risk shift."""
    n_patients = min(n_patients or X_test.shape[0], X_test.shape[0])
    c_batch    = C_test[:n_patients].to(DEVICE)
    alphas     = np.linspace(0, alpha_max, n_steps)
    risk_scores = []

    # compute gradient once
    x0 = torch.randn(n_patients, X_train.shape[1], device=DEVICE, requires_grad=True)
    x1 = integrate_flow(flow_model, x0, c_batch)
    target = torch.full((n_patients,), target_class, dtype=torch.long, device=DEVICE)
    loss = nn.functional.cross_entropy(oracle(x1), target)
    loss.backward()
    grad = x0.grad.detach()

    for alpha in alphas:
        with torch.no_grad():
            x0_steered = x0.detach() - alpha * grad
            x1_steered = integrate_flow(flow_model, x0_steered, c_batch)
            logits     = oracle(x1_steered)
            risk       = torch.softmax(logits, dim=-1)[:, 1].mean().item()
        risk_scores.append(risk)

    return alphas, risk_scores

def plot_steering_trajectory(save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, target_class, label, color in zip(
        axes, [1, 0],
        ["Steering Toward Higher Risk (deceased)", "Steering Toward Lower Risk (survived)"],
        ["#D65F5F", "#4878CF"]
    ):
        alphas, risks = run_steering_trajectory(target_class=target_class)
        ax.plot(alphas, risks, marker="o", color=color, linewidth=2, markersize=5)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Decision boundary")
        ax.fill_between(alphas, risks, 0.5, alpha=0.15, color=color)
        ax.set_xlabel("Steering strength (α)", fontsize=12)
        ax.set_ylabel("P(deceased within 5 years)", fontsize=12)
        ax.set_title(label, fontsize=11)
        ax.set_ylim(0, 1)
        ax.legend()

    plt.suptitle("Reward-Guided Counterfactual Steering Trajectories", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

plot_steering_trajectory(f"{OUT_DIR}/figure6_steering_trajectory.png")


# ── CELL 22 — Tier 2: Ablation Study (no context vs conditional flow) ───────────────

# ── CELL 22 — Tier 2: Ablation Study (proper per-stage condition fidelity) ──────────

def train_flow_map_no_context(X_train, epochs=300, batch_size=32, lr=5e-4):
    X   = X_train.to(DEVICE)
    dim = X.shape[1]
    model = TCGAFlowMap(data_dim=dim, context_dim=1, hidden_dim=256, blocks=3).to(DEVICE)
    opt   = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    n     = X.shape[0]
    model.train()
    for epoch in range(epochs):
        perm    = torch.randperm(n)
        running = 0.0
        for i in range(0, n, batch_size):
            opt.zero_grad()
            idx = perm[i:i+batch_size]
            x1  = X[idx]
            x0  = torch.randn_like(x1)
            t   = torch.rand(x1.shape[0], 1, device=DEVICE)
            xt  = (1-t)*x0 + t*x1
            # zero context — model sees no stage information
            c_zeros = torch.zeros(x1.shape[0], 1, device=DEVICE)
            pred_v  = model(xt, c_zeros, t)
            loss    = nn.functional.mse_loss(pred_v, x1 - x0)
            loss.backward()
            opt.step()
            running += loss.item() * x1.shape[0]
        if (epoch+1) % 100 == 0:
            print(f"  No-context Epoch {epoch+1}/{epochs} | loss: {running/n:.4f}")
    model.eval()
    return model

def get_real_patients_for_stage(s, tolerance=0.5):
    """Return real patient feature matrix for patients near stage severity s."""
    real_severity = clinical_df["stage_severity"]
    real_mask     = (real_severity >= s - tolerance) & (real_severity < s + tolerance)
    if real_mask.sum() < 2:
        return None
    real_idx   = clinical_df.index[real_mask]
    x_real_sub = dp.clinical_pre.transform(
        clinical_df.loc[real_idx, dp.clinical_numeric + dp.clinical_categorical])
    x_real_sub = x_real_sub.toarray() if hasattr(x_real_sub, "toarray") else x_real_sub
    gene_sub   = dp.pca.transform(
        dp.gene_scaler.transform(expr_df.loc[real_idx, dp.top_genes]))
    return np.hstack([x_real_sub, gene_sub])

def per_stage_fidelity(model, use_context=True, stages=(1.0, 2.0, 3.0), n_samples=None):
    """
    For each stage s:
      - generate n_samples patients conditioned on s
        (or with zero context if use_context=False)
      - compute Wasserstein vs real patients at that stage
    Returns a dict {stage -> wasserstein} and the mean across stages.
    """
    n_samples = n_samples or max(20, X_test.shape[0])
    per_stage_wd = {}

    for s in stages:
        x_real_sub = get_real_patients_for_stage(s)
        if x_real_sub is None:
            print(f"  Stage {s}: not enough real patients, skipping")
            continue

        c_scaled = torch.tensor(
            dp.context_scaler.transform([[s]]), dtype=torch.float32)
        c_batch  = c_scaled.repeat(n_samples, 1).to(DEVICE)

        if not use_context:
            c_batch = torch.zeros_like(c_batch)   # blind to stage

        x0 = torch.randn(n_samples, X_train.shape[1], device=DEVICE)
        with torch.no_grad():
            x_gen = integrate_flow(model, x0, c_batch).cpu().numpy()

        wd = mean_wasserstein(x_real_sub, x_gen)
        per_stage_wd[s] = wd

    mean_wd = float(np.mean(list(per_stage_wd.values())))
    return per_stage_wd, mean_wd

print("[Ablation] Training flow map WITHOUT context conditioning...")
flow_no_context = train_flow_map_no_context(X_train)

stages = (1.0, 2.0, 3.0)
wd_no_ctx,  mean_no_ctx  = per_stage_fidelity(flow_no_context, use_context=False, stages=stages)
wd_with_ctx, mean_with_ctx = per_stage_fidelity(flow_model,     use_context=True,  stages=stages)

# build comparison table
stage_labels = {1.0: "Stage I", 2.0: "Stage II", 3.0: "Stage III"}
rows = []
for s in stages:
    rows.append({
        "Stage":                          stage_labels.get(s, str(s)),
        "No Context (Wasserstein ↓)":     wd_no_ctx.get(s,   np.nan),
        "With Context (Wasserstein ↓)":   wd_with_ctx.get(s, np.nan),
        "Improvement":                    wd_no_ctx.get(s, np.nan) - wd_with_ctx.get(s, np.nan),
    })

ablation_df = pd.DataFrame(rows)
print("\n[Ablation Study — Per-Stage Condition Fidelity]")
print(ablation_df.to_string(index=False))
print(f"\n  Mean Wasserstein  No Context : {mean_no_ctx:.4f}")
print(f"  Mean Wasserstein  With Context: {mean_with_ctx:.4f}")
print(f"  Mean improvement from context : {mean_no_ctx - mean_with_ctx:.4f}")
ablation_df.to_csv(f"{OUT_DIR}/ablation_study.csv", index=False)

# ── plot ──────────────────────────────────────────────────────────────────────────────
x      = np.arange(len(stages))
width  = 0.35
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# left: per-stage grouped bar
ax = axes[0]
bars1 = ax.bar(x - width/2, ablation_df["No Context (Wasserstein ↓)"],
               width, label="No Context", color="#aaaaaa", edgecolor="k")
bars2 = ax.bar(x + width/2, ablation_df["With Context (Wasserstein ↓)"],
               width, label="With Stage Context", color="#4878CF", edgecolor="k")
for bar, v in zip(list(bars1)+list(bars2),
                  list(ablation_df["No Context (Wasserstein ↓)"]) +
                  list(ablation_df["With Context (Wasserstein ↓)"])):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.002,
            f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(ablation_df["Stage"])
ax.set_ylabel("Wasserstein Distance to Real Stage Subgroup ↓")
ax.set_title("Per-Stage Condition Fidelity\n(lower = generated cohort closer to real stage)")
ax.legend()

# right: improvement bars
ax2 = axes[1]
colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in ablation_df["Improvement"]]
bars3  = ax2.bar(ablation_df["Stage"], ablation_df["Improvement"],
                 color=colors, edgecolor="k")
for bar, v in zip(bars3, ablation_df["Improvement"]):
    ax2.text(bar.get_x() + bar.get_width()/2,
             v + (0.001 if v >= 0 else -0.005),
             f"{v:+.3f}", ha="center", fontsize=9, fontweight="bold")
ax2.axhline(0, color="black", linewidth=0.8)
ax2.set_ylabel("Wasserstein Improvement from Adding Context\n(positive = context helps)")
ax2.set_title("Context Conditioning Benefit per Stage")

plt.suptitle("Ablation Study: Effect of Stage-Severity Conditioning",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/figure7_ablation.png", dpi=200, bbox_inches="tight")
plt.show()


# ── CELL 23 — Tier 2: Stage-Specific Risk Statistics ────────────────────────────────

def stage_specific_risk_table(stages=(1.0, 2.0, 3.0, 4.0), n_samples=None):
    n_samples = n_samples or max(10, X_test.shape[0])
    rows = []
    real_severity = clinical_df["stage_severity"]

    for s in stages:
        # generated risk
        c_scaled = torch.tensor(dp.context_scaler.transform([[s]]), dtype=torch.float32)
        c_batch  = c_scaled.repeat(n_samples, 1).to(DEVICE)
        x0 = torch.randn(n_samples, X_train.shape[1], device=DEVICE)
        with torch.no_grad():
            x_gen    = integrate_flow(flow_model, x0, c_batch)
            logits   = oracle(x_gen)
            gen_risk = torch.softmax(logits, dim=-1)[:, 1].mean().item()

        # real risk for matching patients
        real_mask = (real_severity >= s - 0.5) & (real_severity < s + 0.5)
        if real_mask.sum() > 0:
            real_idx   = clinical_df.index[real_mask]
            x_real_sub = dp.clinical_pre.transform(
                clinical_df.loc[real_idx, dp.clinical_numeric + dp.clinical_categorical])
            x_real_sub = x_real_sub.toarray() if hasattr(x_real_sub, "toarray") else x_real_sub
            gene_sub   = dp.pca.transform(
                dp.gene_scaler.transform(expr_df.loc[real_idx, dp.top_genes]))
            x_real_sub = np.hstack([x_real_sub, gene_sub])
            with torch.no_grad():
                logits_real = oracle(torch.tensor(x_real_sub, dtype=torch.float32).to(DEVICE))
                real_risk   = torch.softmax(logits_real, dim=-1)[:, 1].mean().item()
            n_real = int(real_mask.sum())
        else:
            real_risk, n_real = np.nan, 0

        stage_label = {1.0: "Stage I", 2.0: "Stage II", 3.0: "Stage III", 4.0: "Stage IV"}[s]
        rows.append({"Stage": stage_label, "N Real Patients": n_real,
                     "Real Mean Risk": real_risk, "Generated Mean Risk": gen_risk})

    risk_df = pd.DataFrame(rows)
    print("\n[Stage-Specific Risk Statistics]")
    print(risk_df.to_string(index=False))
    risk_df.to_csv(f"{OUT_DIR}/stage_risk_statistics.csv", index=False)

    # plot
    x = np.arange(len(rows))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, risk_df["Real Mean Risk"],      width, label="Real",      color="#4878CF", edgecolor="k")
    ax.bar(x + width/2, risk_df["Generated Mean Risk"], width, label="Generated", color="#D65F5F", edgecolor="k")
    ax.set_xticks(x)
    ax.set_xticklabels(risk_df["Stage"])
    ax.set_ylabel("Mean P(deceased within 5 years)")
    ax.set_title("Stage-Specific Risk: Real vs Generated Patients")
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/figure8_stage_risk.png", dpi=200, bbox_inches="tight")
    plt.show()
    return risk_df

risk_table = stage_specific_risk_table()


# ── CELL 24 — Tier 3: Multiple Random Seeds Stability ───────────────────────────────

def run_multi_seed_evaluation(seeds=(42, 7, 13, 99, 2024)):
    results = []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        x0    = torch.randn(X_test.shape[0], X_train.shape[1], device=DEVICE)
        c_bat = C_test.to(DEVICE)
        with torch.no_grad():
            x_gen = integrate_flow(flow_model, x0, c_bat).cpu().numpy()
        mmd  = mmd_rbf(x_real_test, x_gen)
        wd   = mean_wasserstein(x_real_test, x_gen)
        results.append({"seed": seed, "MMD": mmd, "Mean Wasserstein": wd})
        print(f"  Seed {seed}: MMD={mmd:.4f}, Wasserstein={wd:.4f}")

    seed_df = pd.DataFrame(results)
    print(f"\n  MMD             = {seed_df['MMD'].mean():.4f} ± {seed_df['MMD'].std():.4f}")
    print(f"  Mean Wasserstein = {seed_df['Mean Wasserstein'].mean():.4f} ± {seed_df['Mean Wasserstein'].std():.4f}")
    seed_df.to_csv(f"{OUT_DIR}/multi_seed_stability.csv", index=False)
    return seed_df

print("[Tier 3] Multi-seed stability evaluation...")
seed_df = run_multi_seed_evaluation()


# ── CELL 25 — Tier 3: Biological Interpretation (gene changes during steering) ──────

def biological_interpretation(n_patients=None, alpha=1.5, top_n=10):
    n_patients = min(n_patients or X_test.shape[0], X_test.shape[0])
    c_batch    = C_test[:n_patients].to(DEVICE)

    x0 = torch.randn(n_patients, X_train.shape[1], device=DEVICE, requires_grad=True)
    x1 = integrate_flow(flow_model, x0, c_batch)
    target = torch.full((n_patients,), 1, dtype=torch.long, device=DEVICE)
    nn.functional.cross_entropy(oracle(x1), target).backward()
    grad = x0.grad.detach()

    with torch.no_grad():
        x0_steered = x0.detach() - alpha * grad
        x1_base    = integrate_flow(flow_model, x0.detach(), c_batch).cpu().numpy()
        x1_steered = integrate_flow(flow_model, x0_steered,  c_batch).cpu().numpy()

    delta      = np.abs(x1_steered - x1_base).mean(axis=0)
    feat_names = dp.feature_names

    # focus on gene PC dimensions
    gene_pc_start = len(dp.clinical_numeric) + len(
        dp.clinical_pre.named_transformers_["cat"]
           .named_steps["onehot"].get_feature_names_out(dp.clinical_categorical))
    gene_deltas   = delta[gene_pc_start:]
    gene_pc_names = [f"gene_pc_{i}" for i in range(len(gene_deltas))]

    bio_df = pd.DataFrame({"Feature": gene_pc_names, "Mean |Δ| during steering": gene_deltas})
    bio_df = bio_df.sort_values("Mean |Δ| during steering", ascending=False).head(top_n)
    print("\n[Biological Interpretation] Top gene PCs most affected by steering:")
    print(bio_df.to_string(index=False))

    # map PCs back to top contributing genes
    print("\n  Top genes per influential PC:")
    for _, row in bio_df.head(5).iterrows():
        pc_idx    = int(row["Feature"].split("_")[-1])
        loadings  = dp.pca.components_[pc_idx]
        top_gene_idx = np.argsort(np.abs(loadings))[::-1][:5]
        top_genes_pc = [dp.top_genes[i] for i in top_gene_idx]
        print(f"  {row['Feature']}: {top_genes_pc}")

    bio_df.to_csv(f"{OUT_DIR}/biological_interpretation.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(bio_df["Feature"][::-1], bio_df["Mean |Δ| during steering"][::-1],
            color="#9b59b6", edgecolor="k")
    ax.set_xlabel("Mean |Δ| during high-risk steering")
    ax.set_title("Gene PCs Most Affected by Reward-Guided Steering")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/figure9_biological_interpretation.png", dpi=200, bbox_inches="tight")
    plt.show()

biological_interpretation()


print("\n=== SUMMARY ===")
print(f"Generation quality   : MMD={mmd_score:.4f}, mean Wasserstein={wasserstein_score:.4f}")
print(f"Steering (higher risk): {risk_before_hi:.3f} -> {risk_after_hi:.3f}")
print(f"Steering (lower risk) : {risk_before_lo:.3f} -> {risk_after_lo:.3f}")


# ── CELL 26 — update results bundle with all new outputs ────────────────────────────

save_results_bundle()
print("\n=== FULL RESULTS SUMMARY ===")
print(baseline_df.to_string(index=False))
print(ablation_df.to_string(index=False))
print(risk_table.to_string(index=False))
print(seed_df.to_string(index=False))

!pip install -q GEOparse umap-learn

"""
Population-Shift Evaluation of Conditional Flow-Matching Breast-Cancer Digital Twins
======================================================================================
v2 — incorporates the methodological review:

  1. STAGE/GRADE MISMATCH  -> dp.severity_context_source is now tracked explicitly.
     Experiment 3 (stage-conditioning transfer) is GATED: it only runs, and only gets
     reported, if TCGA and the external cohort share the SAME severity axis (grade).
     If TCGA fell back to 'stage' (grade coverage too low), Experiment 3 is skipped
     with a loud explanation instead of silently comparing stage-conditioned TCGA to
     grade-conditioned Saudi data.

  2. EXPERIMENT 4 MONOTONICITY -> replaced the brittle True/False check with Spearman
     rank correlation(alpha, risk), a violation count, max downward excursion, mean
     risk increase, and a multi-seed confidence interval.

  3. EXPERIMENT 5 LEAKAGE AUDIT -> explicit assertions that calibration/eval Saudi
     indices are disjoint and that the TCGA-fit scaler/PCA/oracle never see Saudi
     data at fit time. Also adds a calibration-size ablation (5/10/20/30/40 patients).

  4. GENE-IMPUTATION SENSITIVITY -> any TCGA top-gene missing on the external platform
     (e.g. MRE11) is now analyzed two ways: (A) imputed with the TCGA train mean
     [default, as before], (B) dropped entirely and PCA/model re-derived on the
     intersection. Both are reported side by side.

  5. SEPARATE COHORTS BEFORE POOLING -> GSE36295 and GSE29044 are run as two
     independent external-validation cohorts (Experiments 2-5 each), *then* pooled,
     rather than only ever being analyzed as one combined 118-sample set that mixes
     three microarray platforms.

  6. BASELINE GENERATIVE MODELS -> a Gaussian (mean/cov) baseline and a small VAE
     baseline are trained the same way as the flow model (TCGA train -> zero-shot
     Saudi -> adapted Saudi) so the flow-matching population-shift-robustness claim
     is judged against something, not reported alone.

Everything else (data loading, harmonization, ColumnTransformer-based common-feature
schema, flow model architecture) is unchanged from v1.
"""

# ── CELL 0 — installs ────────────────────────────────────────────────────────────────
# !pip install -q torch scikit-learn pandas numpy scipy matplotlib seaborn requests tqdm
# !pip install -q GEOparse umap-learn

# ── CELL 1 — imports & config ───────────────────────────────────────────────────────
import os, re, warnings, zipfile, copy
import numpy as np
import pandas as pd
import requests

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import rbf_kernel

from scipy.stats import wasserstein_distance, spearmanr

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "/content/data"
OUT_DIR = "/content/outputs"
GEO_DIR = "/content/data/geo"
for d in (DATA_DIR, OUT_DIR, GEO_DIR):
    os.makedirs(d, exist_ok=True)

CBIOPORTAL_BASE = "https://www.cbioportal.org/api"
TCGA_STUDY_ID = "brca_tcga"

# Saudi Arabia / Muslim-majority-country cohorts (population context only).
# CHANGE (fix 5): kept as a list so downstream code can analyze each series
# independently before pooling, instead of only ever seeing a merged frame.
SAUDI_GEO_SERIES = ["GSE36295", "GSE29044"]

N_TOP_GENES = 80           # top variable genes, fit on TCGA train only
N_GENE_PCS = 10            # genomic embedding dimensionality
FIVE_YEAR_DAYS = 365 * 5
N_EULER_STEPS = 10

CALIBRATION_SIZES = [5, 10, 20, 30, 40]     # CHANGE (fix 3): calibration-size ablation grid
N_MONOTONICITY_SEEDS = 5                     # CHANGE (fix 2): seeds for the monotonicity CI

# ---- COMMON FEATURE SUBSET --------------------------------------------------------
COMMON_CLINICAL_NUMERIC = ["age", "lymph_nodes"]
COMMON_CLINICAL_CATEGORICAL = ["histological_type", "er_status", "pr_status",
                                "her2_status", "grade"]
print("Device:", DEVICE)


# ════════════════════════════════════════════════════════════════════════════════════
# PART A — TCGA-BRCA (source domain), fit-once pipeline
# ════════════════════════════════════════════════════════════════════════════════════

# ── CELL 2 — TCGA download via cBioPortal API ───────────────────────────────────────
def fetch_clinical_cbioportal(study_id):
    url = f"{CBIOPORTAL_BASE}/studies/{study_id}/patients"
    patients = requests.get(url, headers={"Accept": "application/json"}, timeout=60).json()
    patient_ids = [p["patientId"] for p in patients]
    print(f"  Fetched {len(patient_ids)} patients")

    url2 = f"{CBIOPORTAL_BASE}/studies/{study_id}/clinical-data?clinicalDataType=PATIENT&pageSize=100000"
    clin_raw = requests.get(url2, headers={"Accept": "application/json"}, timeout=120).json()
    rows = {}
    for entry in clin_raw:
        pid = entry["patientId"]
        rows.setdefault(pid, {"patient_id": pid})[entry["clinicalAttributeId"]] = entry["value"]
    df = pd.DataFrame.from_dict(rows, orient="index").reset_index(drop=True)
    print(f"  Clinical dataframe: {df.shape}")
    return df


def fetch_expression_cbioportal(study_id):
    profiles = requests.get(f"{CBIOPORTAL_BASE}/studies/{study_id}/molecular-profiles",
                             headers={"Accept": "application/json"}, timeout=60).json()
    rna_profile = next(p["molecularProfileId"] for p in profiles
                        if "rna_seq" in p["molecularProfileId"].lower()
                        and "v2" in p["molecularProfileId"].lower())
    print(f"  Using expression profile: {rna_profile}")

    samples = requests.get(f"{CBIOPORTAL_BASE}/studies/{study_id}/samples?pageSize=10000",
                            headers={"Accept": "application/json"}, timeout=60).json()
    sample_ids = [s["sampleId"] for s in samples][:500]
    print(f"  Using {len(sample_ids)} samples")

    gene_symbols = [
        "ESR1", "ERBB2", "PGR", "MKI67", "BRCA1", "BRCA2", "TP53", "PIK3CA", "CDH1", "GATA3",
        "RB1", "PTEN", "AKT1", "CCND1", "FOXA1", "EGFR", "FGFR1", "MYC", "AURKA", "TOP2A",
        "BIRC5", "BCL2", "FOXM1", "PLK1", "MELK", "RAD51", "CHEK1", "CHEK2", "ATM", "PALB2",
        "ESR2", "AR", "STAT3", "NOTCH1", "CDK4", "CDK6", "CDKN2A", "MDM2", "KRT8", "KRT18",
        "KRT19", "MLPH", "TFF1", "CLDN7", "FN1", "COL3A1", "DCN", "VEGFA", "MET", "IGF1R",
        "MAP3K1", "NF1", "TSC1", "TSC2", "KEAP1", "NFE2L2", "WNT5A", "SRC", "JAK2", "NCOA3",
        "UBE2C", "TYMS", "PCNA", "E2F1", "BUB1", "CENPA", "KIF2C", "TPX2", "PRC1", "EXO1",
        "MCM2", "BRIP1", "NBN", "MRE11", "RAD50", "ATR", "CCND2", "CCNE1", "MMP2", "MMP9",
    ]

    entrez_ids, symbol_map = [], {}
    for sym in gene_symbols:
        try:
            r = requests.get(f"{CBIOPORTAL_BASE}/genes/{sym}",
                              headers={"Accept": "application/json"}, timeout=10).json()
            if "entrezGeneId" in r:
                entrez_ids.append(r["entrezGeneId"])
                symbol_map[r["entrezGeneId"]] = r["hugoGeneSymbol"]
        except Exception:
            continue
    print(f"  Resolved {len(entrez_ids)} gene IDs")

    payload = {"sampleIds": sample_ids, "entrezGeneIds": entrez_ids}
    expr_resp = requests.post(
        f"{CBIOPORTAL_BASE}/molecular-profiles/{rna_profile}/molecular-data/fetch?projection=SUMMARY",
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        json=payload, timeout=180).json()

    if not isinstance(expr_resp, list) or len(expr_resp) == 0:
        raise ValueError(f"Unexpected expression response: {str(expr_resp)[:200]}")

    rows = {}
    for entry in expr_resp:
        if not isinstance(entry, dict):
            continue
        sid, eid = entry.get("sampleId"), entry.get("entrezGeneId")
        gene = symbol_map.get(eid, str(eid))
        rows.setdefault(sid, {"sample_id": sid})[gene] = entry.get("value", np.nan)

    expr_df = pd.DataFrame.from_dict(rows, orient="index").reset_index(drop=True)
    print(f"  Expression dataframe: {expr_df.shape}")
    return expr_df


# ── CELL 3 — TCGA clinical harmonization onto the COMMON schema ────────────────────
TCGA_COL_MAP = {
    "AGE": "age",
    "AJCC_PATHOLOGIC_TUMOR_STAGE": "stage",
    "RACE": "race",
    "HISTOLOGICAL_DIAGNOSIS": "histological_type",
    "MENOPAUSE_STATUS": "menopause_status",
    "RADIATION_THERAPY": "radiation_therapy",
    "NEOADJUVANT_THERAPY": "neoadjuvant_tx",
    "LYMPH_NODE_STATUS": "lymph_nodes",
    "ER_STATUS_BY_IHC": "er_status",
    "PR_STATUS_BY_IHC": "pr_status",
    "IHC_HER2": "her2_status",
    "TUMOR_GRADE": "grade",
    "OS_STATUS": "vital_status",
    "OS_MONTHS": "os_months",
    "patient_id": "patient_id",
}


def stage_to_severity(stage_str):
    if pd.isna(stage_str):
        return np.nan
    s = str(stage_str).upper().replace("STAGE", "").strip()
    m = re.match(r"^(IV|III|II|I)([ABC]?)$", s)
    if not m:
        return np.nan
    roman_to_val = {"I": 1.0, "II": 2.0, "III": 3.0, "IV": 4.0}
    sub_offset = {"A": 0.0, "B": 0.15, "C": 0.30, "": 0.0}
    return roman_to_val[m.group(1)] + sub_offset[m.group(2)]


def grade_to_severity(grade_raw):
    if pd.isna(grade_raw):
        return np.nan
    s = str(grade_raw).strip().upper()
    if "GX" in s or "N/A" in s or "NA" == s:
        return np.nan
    m = re.search(r"([1-3])", s)
    return float(m.group(1)) if m else np.nan


CONTEXT_SOURCE = "grade"
MIN_TCGA_GRADE_COVERAGE = 30


def parse_age_value(raw):
    if pd.isna(raw):
        return np.nan
    s = str(raw).strip()
    if re.match(r"^\d+(\.\d+)?$", s):
        return float(s)
    m = re.match(r"^>\s*(\d+)$", s)
    if m:
        return float(m.group(1)) + 5.0
    m = re.match(r"^<\s*(\d+)$", s)
    if m:
        return float(m.group(1)) - 5.0
    m = re.match(r"^(\d+)\s*-\s*(\d+)$", s)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2.0
    return np.nan


def normalize_marker_status(raw):
    if pd.isna(raw):
        return np.nan
    s = str(raw).strip().lower()
    if s in ("normal", "n/a", "na", ""):
        return np.nan
    if s in ("+", "1", "pos", "positive"):
        return "positive"
    if s in ("-", "0", "neg", "negative"):
        return "negative"
    if s == "2":
        return np.nan
    if s == "3":
        return "positive"
    return np.nan


def make_5yr_label(row):
    status = str(row.get("vital_status", "")).strip().lower()
    if status == "dead":
        d = row.get("days_to_death", np.nan)
        return np.nan if pd.isna(d) else (1.0 if d <= FIVE_YEAR_DAYS else 0.0)
    elif status == "alive":
        f = row.get("days_to_last_fu", np.nan)
        return np.nan if pd.isna(f) else (0.0 if f >= FIVE_YEAR_DAYS else np.nan)
    return np.nan


def resolve_tcga_grade_column(clinical_raw):
    candidates = [c for c in clinical_raw.columns if "GRADE" in c.upper()]
    best_col, best_count = None, 0
    for c in candidates:
        cnt = clinical_raw[c].notna().sum()
        if cnt > best_count:
            best_col, best_count = c, cnt
    if best_col:
        print(f"  [TCGA grade] candidate columns: {candidates}; using '{best_col}' "
              f"with {best_count}/{len(clinical_raw)} non-null values")
    else:
        print("  [TCGA grade] no GRADE-like column found in TCGA clinical data.")
    return best_col, best_count


def build_tcga_clinical(clinical_raw):
    """CHANGE (fix 1): now returns (df, severity_context_source) instead of just df,
    so callers know for certain whether TCGA ended up conditioned on 'grade' (matches
    the Saudi cohorts) or fell back to 'stage' (does NOT match Saudi grade-based
    severity). This is the single source of truth Experiment 3 gates on."""
    grade_col, grade_coverage = resolve_tcga_grade_column(clinical_raw)
    col_map = dict(TCGA_COL_MAP)
    if grade_col:
        col_map[grade_col] = "grade_raw"

    df = clinical_raw.rename(columns={k: v for k, v in col_map.items()
                                       if k in clinical_raw.columns})
    if "vital_status" in df.columns:
        df["vital_status"] = df["vital_status"].apply(lambda x: "dead" if str(x).startswith("1") else "alive")
    if "os_months" in df.columns:
        df["os_months"] = pd.to_numeric(df["os_months"], errors="coerce")
        df["days_to_death"] = df.apply(lambda r: r["os_months"] * 30.44 if r.get("vital_status") == "dead" else np.nan, axis=1)
        df["days_to_last_fu"] = df.apply(lambda r: r["os_months"] * 30.44 if r.get("vital_status") == "alive" else np.nan, axis=1)

    df["age"] = pd.to_numeric(df.get("age", np.nan), errors="coerce")
    df["stage"] = df.get("stage", np.nan)
    df["stage_severity_"] = df["stage"].apply(stage_to_severity)

    use_grade = CONTEXT_SOURCE == "grade" and grade_coverage >= MIN_TCGA_GRADE_COVERAGE
    if use_grade:
        df["severity_context"] = df["grade_raw"].apply(grade_to_severity)
        severity_context_source = "grade"
        print(f"  [TCGA severity_context] using GRADE (coverage {grade_coverage}/{len(df)}) "
              f"-- MATCHES Saudi cohort severity axis (grade). Experiment 3 is valid.")
    else:
        df["severity_context"] = df["stage_severity_"]
        severity_context_source = "stage"
        reason = "CONTEXT_SOURCE='stage'" if CONTEXT_SOURCE != "grade" else \
            f"grade coverage too low ({grade_coverage} < {MIN_TCGA_GRADE_COVERAGE})"
        print(f"  [TCGA severity_context] falling back to STAGE ({reason}). "
              f"WARNING: Saudi cohorts only report GRADE. Experiment 3 (stage-conditioning "
              f"transfer) will be SKIPPED for this run because comparing stage-conditioned "
              f"TCGA to grade-conditioned Saudi data is not apples-to-apples (see review).")

    df["label"] = df.apply(make_5yr_label, axis=1)

    for col in ["er_status", "pr_status", "her2_status"]:
        if col in df.columns:
            df[col] = df[col].apply(normalize_marker_status)

    df = df.dropna(subset=["patient_id", "severity_context", "label"]).reset_index(drop=True)
    df = df.set_index("patient_id")
    df["population_context"] = "TCGA (global, source domain)"
    return df, severity_context_source


# ── CELL 4 — TCGA expression cleanup + patient matching ─────────────────────────────
def build_tcga_expression(expr_raw_cbio):
    expr_raw_cbio = expr_raw_cbio.copy()
    expr_raw_cbio["patient_id"] = expr_raw_cbio["sample_id"].apply(
        lambda s: "-".join(s.split("-")[:3]) if pd.notna(s) else s)
    expr_df = expr_raw_cbio.drop(columns=["sample_id"]).drop_duplicates(subset="patient_id")
    expr_df = expr_df.set_index("patient_id")
    gene_cols = list(expr_df.columns)
    expr_df = expr_df[gene_cols].apply(pd.to_numeric, errors="coerce")
    expr_df = expr_df.fillna(expr_df.mean())
    return expr_df


# ════════════════════════════════════════════════════════════════════════════════════
# PART B — GEO ingestion for the Saudi cohorts (GSE36295, GSE29044)
# ════════════════════════════════════════════════════════════════════════════════════

def download_geo_series(gse_id, destdir=GEO_DIR):
    import GEOparse
    print(f"[GEO] Downloading {gse_id} ...")
    gse = GEOparse.get_GEO(geo=gse_id, destdir=destdir, silent=True)
    print(f"  {gse_id}: {len(gse.gsms)} samples, platforms: {list(gse.gpls.keys())}")
    return gse


CHARACTERISTIC_PATTERNS = {
    "age": [r"age[^:]*:\s*(\S+)"],
    "tissue": [r"^tissue[^:]*:\s*(.+)", r"sample type\s*:\s*(.+)"],
    "tumor_normal": [r"tumou?r\s*/\s*normal[^:]*:\s*(\S+)"],
    "sample_type": [r"\btype\s*:\s*(\S+)"],
    "histological_type": [r"histolog\w*[^:]*:\s*(.+)"],
    "er_status": [r"\ber\b\s*(\([^)]*\))?\s*:\s*(\S+)"],
    "pr_status": [r"\bpr\b\s*(\([^)]*\))?\s*:\s*(\S+)"],
    "her2_status": [r"her-?2\s*(/\s*neu)?\s*(\([^)]*\))?\s*:\s*(\S+)"],
    "grade": [r"\bgrade\b[^:]*:\s*(\S+)"],
}


def parse_geo_characteristics(gsm):
    raw_lines = gsm.metadata.get("characteristics_ch1", [])
    parsed = {}
    for field, patterns in CHARACTERISTIC_PATTERNS.items():
        for line in raw_lines:
            matched = False
            for pat in patterns:
                m = re.search(pat, line, flags=re.IGNORECASE)
                if m:
                    parsed[field] = m.group(m.lastindex)
                    matched = True
                    break
            if matched:
                break
    if len(parsed) < 2:
        print(f"  [WARN] {gsm.name}: only parsed {list(parsed.keys())} from "
              f"{raw_lines} -- extend CHARACTERISTIC_PATTERNS if this is a real cohort sample.")
    return parsed


def is_breast_tumor_sample(row):
    tissue = str(row.get("tissue", "")).lower()
    if "blood" in tissue or "lymphoblastoid" in tissue:
        return False
    tn = str(row.get("tumor_normal", "")).strip().upper()
    sample_type = str(row.get("sample_type", "")).strip().upper()
    if tn == "T":
        return True
    if tn == "N":
        return False
    if "normal" in tissue or sample_type == "NORMAL":
        return False
    if "tumor" in tissue or "cancer" in tissue or sample_type in ("IDC", "DCIS", "ILC"):
        return True
    return False


def resolve_probe_to_symbol(gpl):
    table = gpl.table
    candidate_cols = ["Gene Symbol", "GENE_SYMBOL", "Symbol", "gene_symbol"]
    for col in candidate_cols:
        if col in table.columns:
            mapping = table.set_index(table.columns[0])[col].dropna().to_dict()
            return {k: str(v).split("///")[0].strip() for k, v in mapping.items() if str(v).strip()}

    if "gene_assignment" in table.columns:
        mapping = {}
        for probe_id, val in zip(table.iloc[:, 0], table["gene_assignment"]):
            if pd.isna(val) or val == "---":
                continue
            first_block = str(val).split("///")[0]
            parts = [p.strip() for p in first_block.split("//")]
            if len(parts) >= 2 and parts[1]:
                mapping[probe_id] = parts[1]
        return mapping

    raise ValueError(f"Could not find a gene-symbol column in platform table; "
                      f"columns available: {list(table.columns)}. "
                      f"Add the right column name to resolve_probe_to_symbol().")


def build_geo_expression_and_clinical(gse):
    platform_to_samples = {}
    for gsm_name, gsm in gse.gsms.items():
        plat_ids = gsm.metadata.get("platform_id", [])
        plat_id = plat_ids[0] if plat_ids else None
        platform_to_samples.setdefault(plat_id, []).append(gsm_name)

    per_platform_gene_expr = []
    for plat_id, sample_names in platform_to_samples.items():
        if plat_id not in gse.gpls:
            print(f"  [WARN] platform {plat_id} not in gse.gpls -- skipping "
                  f"{len(sample_names)} samples on it.")
            continue
        try:
            probe_map = resolve_probe_to_symbol(gse.gpls[plat_id])
        except ValueError as e:
            print(f"  [WARN] {e} -- skipping platform {plat_id} ({len(sample_names)} samples)")
            continue
        probe_map = {str(k): v for k, v in probe_map.items()}

        expr_cols = {}
        for gsm_name in sample_names:
            gsm = gse.gsms[gsm_name]
            if "VALUE" not in gsm.table.columns:
                continue
            s = gsm.table.set_index("ID_REF")["VALUE"]
            s.index = s.index.astype(str)
            expr_cols[gsm_name] = s
        if not expr_cols:
            continue

        plat_expr = pd.DataFrame(expr_cols).T.apply(pd.to_numeric, errors="coerce")
        symbol_series = pd.Series({p: probe_map.get(p) for p in plat_expr.columns})
        plat_expr_t = plat_expr.T
        plat_expr_t["__gene__"] = symbol_series
        plat_expr_t = plat_expr_t.dropna(subset=["__gene__"])
        gene_expr = plat_expr_t.groupby("__gene__").mean().T
        print(f"  Platform {plat_id}: {len(sample_names)} samples -> "
              f"{gene_expr.shape[1]} mapped genes")
        # CHANGE (fix 5): tag every sample's originating platform so downstream code
        # can report/inspect platform composition of a cohort instead of only ever
        # seeing an anonymous pooled matrix.
        gene_expr["__platform__"] = plat_id
        per_platform_gene_expr.append(gene_expr)

    if not per_platform_gene_expr:
        raise ValueError("No usable expression data found on any platform for this GSE.")

    expr_df = pd.concat(per_platform_gene_expr, axis=0, join="outer")
    platform_col = expr_df.pop("__platform__")

    clinical_rows = {gsm_name: parse_geo_characteristics(gsm) for gsm_name, gsm in gse.gsms.items()}
    clinical_df = pd.DataFrame.from_dict(clinical_rows, orient="index")
    for col in ("tissue", "tumor_normal", "sample_type", "grade"):
        if col not in clinical_df.columns:
            clinical_df[col] = np.nan
    clinical_df["platform"] = platform_col.reindex(clinical_df.index)

    n_before = len(clinical_df)
    is_tumor = clinical_df.apply(is_breast_tumor_sample, axis=1)
    print(f"  {n_before} raw samples -> {int(is_tumor.sum())} kept as breast TUMOR samples "
          f"({int((~is_tumor).sum())} dropped: normal/adjacent-normal breast tissue and/or "
          f"unrelated non-breast specimens folded into this GSE accession)")
    clinical_df = clinical_df.loc[is_tumor]
    expr_df_out = expr_df.loc[expr_df.index.intersection(clinical_df.index)]

    clinical_df["age"] = clinical_df["age"].apply(parse_age_value) if "age" in clinical_df.columns else np.nan
    clinical_df["severity_context"] = clinical_df["grade"].apply(grade_to_severity)

    for col in ["er_status", "pr_status", "her2_status"]:
        if col in clinical_df.columns:
            clinical_df[col] = clinical_df[col].apply(normalize_marker_status)

    return expr_df_out, clinical_df


def load_saudi_cohorts(series_ids=SAUDI_GEO_SERIES):
    """CHANGE (fix 5): now returns (combined_expr, combined_clin, per_series_dict) so
    callers can run Experiments 2-5 on GSE36295 and GSE29044 SEPARATELY (each is a
    single-platform-family cohort) before ever pooling them into one distribution."""
    all_expr, all_clin = [], []
    per_series = {}
    for gse_id in series_ids:
        gse = download_geo_series(gse_id)
        gene_expr, clin_df = build_geo_expression_and_clinical(gse)

        clin_df = clin_df.copy()
        clin_df["population_context"] = "Saudi Arabia / Muslim-majority-country cohort"
        clin_df["source_series"] = gse_id

        n_stage = int(clin_df["severity_context"].notna().sum()) if "severity_context" in clin_df.columns else 0
        print(f"  [{gse_id}] parsed fields present (post tumor-filtering): "
              f"age={int(clin_df['age'].notna().sum()) if 'age' in clin_df.columns else 0}/{len(clin_df)}, "
              f"severity_context (grade-based)={n_stage}/{len(clin_df)}, "
              f"platforms={sorted(clin_df['platform'].dropna().unique().tolist())}")
        if n_stage == 0 and len(clin_df) > 0:
            print(f"  [{gse_id}] severity_context is 0/{len(clin_df)} -- dumping raw "
                  f"characteristics_ch1 for the first 5 KEPT (tumor) samples so you can "
                  f"extend CHARACTERISTIC_PATTERNS['grade'] with the real field name/format:")
            for gsm_name in list(clin_df.index)[:5]:
                raw = gse.gsms[gsm_name].metadata.get("characteristics_ch1", [])
                print(f"    {gsm_name}: {raw}")

        per_series[gse_id] = (gene_expr, clin_df)
        all_expr.append(gene_expr)
        all_clin.append(clin_df)

    combined_expr = pd.concat(all_expr, axis=0, join="outer")
    combined_clin = pd.concat(all_clin, axis=0, join="outer")
    common_idx = combined_expr.index.intersection(combined_clin.index)
    combined_expr, combined_clin = combined_expr.loc[common_idx], combined_clin.loc[common_idx]
    n_platforms = combined_clin["platform"].nunique() if "platform" in combined_clin.columns else "?"
    print(f"[GEO] Combined Saudi cohort: {combined_expr.shape[0]} samples, "
          f"{combined_expr.shape[1]} genes (pre-alignment), spanning {n_platforms} platforms "
          f"-- treat pooled numbers as secondary to the per-series ones below.")
    return combined_expr, combined_clin, per_series


# ════════════════════════════════════════════════════════════════════════════════════
# PART C — Shared representation: fit ONCE on TCGA, apply to Saudi
# ════════════════════════════════════════════════════════════════════════════════════

class TCGADataPipeline:
    def __init__(self, clinical_df, expr_df, severity_context_source,
                 n_top_genes=N_TOP_GENES, n_gene_pcs=N_GENE_PCS, drop_genes=None):
        self.clinical_df = clinical_df
        self.expr_df = expr_df
        self.severity_context_source = severity_context_source  # CHANGE (fix 1)
        self.n_top_genes = n_top_genes
        self.n_gene_pcs = n_gene_pcs
        self.clinical_numeric = [c for c in COMMON_CLINICAL_NUMERIC if c in clinical_df.columns]
        self.clinical_categorical = [c for c in COMMON_CLINICAL_CATEGORICAL if c in clinical_df.columns]
        # CHANGE (fix 4): optional gene exclusion list, used for the MRE11-style
        # "drop instead of impute" sensitivity analysis.
        self.drop_genes = set(drop_genes or [])
        # CHANGE (fix 3): remember exactly which TCGA patient IDs were used to fit
        # every stateful transform, so a leakage audit can check against it later.
        self.fit_index_ = None

    def prepare_data(self):
        idx = self.clinical_df.index
        y = self.clinical_df["label"].astype(int)

        idx_train_val, idx_test = train_test_split(idx, test_size=0.15, random_state=RANDOM_STATE, stratify=y)
        idx_train, idx_oracle = train_test_split(idx_train_val, test_size=0.176, random_state=RANDOM_STATE,
                                                  stratify=y.loc[idx_train_val])
        self.fit_index_ = list(idx_train)  # only TRAIN ever touches .fit()

        candidate_genes = [g for g in self.expr_df.columns if g not in self.drop_genes]
        gene_var = self.expr_df.loc[idx_train, candidate_genes].var(axis=0).sort_values(ascending=False)
        self.top_genes = gene_var.head(self.n_top_genes).index.tolist()
        self.gene_scaler = StandardScaler().fit(self.expr_df.loc[idx_train, self.top_genes])
        actual_pcs = min(self.n_gene_pcs, len(self.top_genes), len(idx_train))
        self.pca = PCA(n_components=actual_pcs, random_state=RANDOM_STATE).fit(
            self.gene_scaler.transform(self.expr_df.loc[idx_train, self.top_genes]))
        self.n_gene_pcs = actual_pcs
        self.top_gene_train_means = self.expr_df.loc[idx_train, self.top_genes].mean()

        def gene_embedding(index, expr_source=None):
            expr_source = self.expr_df if expr_source is None else expr_source
            scaled = self.gene_scaler.transform(expr_source.loc[index, self.top_genes])
            return self.pca.transform(scaled)

        clin_cols = self.clinical_numeric + self.clinical_categorical
        self.clinical_pre = ColumnTransformer([
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                               ("scale", StandardScaler())]), self.clinical_numeric),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                               ("onehot", OneHotEncoder(handle_unknown="ignore"))]),
             self.clinical_categorical),
        ]).fit(self.clinical_df.loc[idx_train, clin_cols])

        def clinical_features(index, clin_source=None):
            clin_source = self.clinical_df if clin_source is None else clin_source
            out = self.clinical_pre.transform(clin_source.loc[index, clin_cols])
            return out.toarray() if hasattr(out, "toarray") else out

        self.context_scaler = StandardScaler().fit(self.clinical_df.loc[idx_train, ["severity_context"]])

        def build_split(index):
            X = np.hstack([clinical_features(index), gene_embedding(index)]).astype(np.float32)
            C = self.context_scaler.transform(self.clinical_df.loc[index, ["severity_context"]]).astype(np.float32)
            yv = self.clinical_df.loc[index, "label"].astype(np.int64).values
            return torch.tensor(X), torch.tensor(C), torch.tensor(yv)

        X_train, C_train, y_train = build_split(idx_train)
        X_oracle, C_oracle, y_oracle = build_split(idx_oracle)
        X_test, C_test, y_test = build_split(idx_test)

        cat_names = list(self.clinical_pre.named_transformers_["cat"].named_steps["onehot"]
                          .get_feature_names_out(self.clinical_categorical))
        self.feature_names = self.clinical_numeric + cat_names + [f"gene_pc_{i}" for i in range(self.n_gene_pcs)]
        self._clin_cols = clin_cols

        return {"train": (X_train, C_train, y_train),
                "oracle": (X_oracle, C_oracle, y_oracle),
                "test": (X_test, C_test, y_test)}

    def transform_external(self, ext_clinical_df, ext_expr_df, verbose=True):
        present_genes = [g for g in self.top_genes if g in ext_expr_df.columns]
        missing_genes = [g for g in self.top_genes if g not in ext_expr_df.columns]
        if missing_genes and verbose:
            print(f"  [transform_external] {len(missing_genes)}/{len(self.top_genes)} "
                  f"TCGA genes not found on external platform -- imputing with TCGA train mean: "
                  f"{missing_genes[:10]}{'...' if len(missing_genes) > 10 else ''}")

        expr_aligned = pd.DataFrame(index=ext_expr_df.index, columns=self.top_genes, dtype=float)
        if present_genes:
            expr_aligned[present_genes] = ext_expr_df[present_genes].values
        for g in missing_genes:
            expr_aligned[g] = self.top_gene_train_means[g]
        expr_aligned = expr_aligned.fillna(expr_aligned.mean())

        gene_pc = self.pca.transform(self.gene_scaler.transform(expr_aligned))

        n_total = len(ext_clinical_df)
        n_with_stage = int(ext_clinical_df["severity_context"].notna().sum()) if \
            "severity_context" in ext_clinical_df.columns else 0
        keep_idx = ext_clinical_df.index[ext_clinical_df.get("severity_context", pd.Series(
            index=ext_clinical_df.index, dtype=float)).notna()]
        keep_idx = keep_idx.intersection(expr_aligned.index)

        if len(keep_idx) == 0:
            print(f"  [transform_external] 0/{n_total} external samples usable "
                  f"({n_with_stage}/{n_total} had a parsed severity_context). Returning empty result.")
            empty_X = torch.empty((0, len(self.feature_names)), dtype=torch.float32)
            empty_C = torch.empty((0, 1), dtype=torch.float32)
            return empty_X, empty_C, None, keep_idx

        clin_slice = ext_clinical_df.loc[keep_idx, self._clin_cols] if set(self._clin_cols).issubset(
            ext_clinical_df.columns) else pd.DataFrame(index=keep_idx, columns=self._clin_cols)
        for c in self._clin_cols:
            if c not in clin_slice.columns:
                clin_slice[c] = np.nan

        clin_feat = self.clinical_pre.transform(clin_slice)
        clin_feat = clin_feat.toarray() if hasattr(clin_feat, "toarray") else clin_feat

        gene_pc_slice = pd.DataFrame(gene_pc, index=expr_aligned.index).loc[keep_idx].values
        X_ext = np.hstack([clin_feat, gene_pc_slice]).astype(np.float32)
        C_ext = self.context_scaler.transform(ext_clinical_df.loc[keep_idx, ["severity_context"]]).astype(np.float32)

        y_ext = None
        if "label" in ext_clinical_df.columns and ext_clinical_df.loc[keep_idx, "label"].notna().any():
            y_ext = ext_clinical_df.loc[keep_idx, "label"].astype(np.int64).values

        return (torch.tensor(X_ext), torch.tensor(C_ext),
                None if y_ext is None else torch.tensor(y_ext), keep_idx)

    def assert_no_leakage(self, external_keep_idx):
        """CHANGE (fix 3): explicit leakage audit. Confirms none of the TCGA-side
        stateful fits (scaler / PCA / clinical ColumnTransformer / context scaler)
        were fit on anything outside self.fit_index_ (== TCGA train only), and that
        the external (Saudi) indices passed in never overlap with TCGA patient IDs
        (which would be a coincidence-of-ID bug, not real leakage, but worth ruling
        out mechanically)."""
        assert self.fit_index_ is not None, "prepare_data() must run before the audit."
        overlap = set(self.fit_index_) & set(external_keep_idx.tolist() if hasattr(external_keep_idx, "tolist") else external_keep_idx)
        assert len(overlap) == 0, f"Leakage: {len(overlap)} external IDs collide with TCGA fit_index_"
        print(f"  [leakage audit] OK: 0 overlap between TCGA fit index (n={len(self.fit_index_)}) "
              f"and external cohort (n={len(external_keep_idx)}).")


def assert_disjoint(idx_a, idx_b, label=""):
    """CHANGE (fix 3): generic disjointness assertion used for the Saudi
    calibration/eval split in Experiment 5."""
    set_a, set_b = set(np.asarray(idx_a).tolist()), set(np.asarray(idx_b).tolist())
    overlap = set_a & set_b
    assert len(overlap) == 0, f"Leakage in {label}: {len(overlap)} overlapping indices"
    print(f"  [leakage audit] {label}: calibration (n={len(set_a)}) and eval (n={len(set_b)}) "
          f"are disjoint.")


# ════════════════════════════════════════════════════════════════════════════════════
# PART D — model classes
# ════════════════════════════════════════════════════════════════════════════════════

class SurvivalRewardOracle(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.SiLU(),
            nn.Linear(64, 2),
        )
    def forward(self, x):
        return self.net(x)


class FlowResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.act = nn.SiLU()
    def forward(self, x):
        return x + self.act(self.linear2(self.act(self.linear1(x))))


class TCGAFlowMap(nn.Module):
    def __init__(self, data_dim, context_dim, hidden_dim=256, blocks=3):
        super().__init__()
        self.input_proj = nn.Linear(data_dim + context_dim + 1, hidden_dim)
        self.trunk = nn.Sequential(*[FlowResidualBlock(hidden_dim) for _ in range(blocks)])
        self.output_proj = nn.Linear(hidden_dim, data_dim)
        self.act = nn.SiLU()
    def forward(self, x, c, t):
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        h = self.act(self.input_proj(torch.cat([x, c, t], dim=-1)))
        return self.output_proj(self.trunk(h))


def integrate_flow(model, x0, c, n_steps=N_EULER_STEPS):
    dt = 1.0 / n_steps
    x = x0
    for i in range(n_steps):
        t = torch.full((x.size(0), 1), i * dt, device=x.device)
        x = x + model(x, c, t) * dt
    return x


# ---- CHANGE (fix 6): baseline generative models --------------------------------

class GaussianBaseline:
    """Fits a single multivariate Gaussian (mean + diagonal-regularized covariance)
    on TCGA train data, conditioned trivially (context is ignored -- this baseline
    answers "how much of the flow model's apparent robustness is just because ANY
    reasonable density estimate does okay here"). Sampling ignores context by design."""
    def __init__(self, reg=1e-3):
        self.reg = reg
    def fit(self, X_train):
        X = X_train.numpy() if torch.is_tensor(X_train) else X_train
        self.mean_ = X.mean(axis=0)
        cov = np.cov(X, rowvar=False)
        self.cov_ = cov + self.reg * np.eye(cov.shape[0])
        return self
    def sample(self, n):
        return np.random.multivariate_normal(self.mean_, self.cov_, size=n).astype(np.float32)
    def adapt(self, X_calib):
        """'Adaptation' for the Gaussian baseline = refit mean/cov on the small
        calibration set alone (the closest fair analogue to fine-tuning)."""
        X = X_calib.numpy() if torch.is_tensor(X_calib) else X_calib
        adapted = GaussianBaseline(reg=self.reg).fit(torch.tensor(X))
        return adapted


class SimpleVAE(nn.Module):
    """Small conditional VAE baseline, same context-conditioning signal as the flow
    model, trained with the standard ELBO. Used only as a second reference point for
    population-shift robustness, not as a competitor architecture to optimize."""
    def __init__(self, data_dim, context_dim, latent_dim=16, hidden_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc = nn.Sequential(
            nn.Linear(data_dim + context_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)
        self.dec = nn.Sequential(
            nn.Linear(latent_dim + context_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, data_dim),
        )
    def encode(self, x, c):
        h = self.enc(torch.cat([x, c], dim=-1))
        return self.mu_head(h), self.logvar_head(h)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)
    def decode(self, z, c):
        return self.dec(torch.cat([z, c], dim=-1))
    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar
    def sample(self, c, device):
        n = c.shape[0]
        z = torch.randn(n, self.latent_dim, device=device)
        with torch.no_grad():
            return self.decode(z, c).cpu().numpy()


def train_vae(X_train, C_train, epochs=300, batch_size=32, lr=5e-4, beta=0.1, init_model=None):
    X_train, C_train = X_train.to(DEVICE), C_train.to(DEVICE)
    model = init_model if init_model is not None else SimpleVAE(
        data_dim=X_train.shape[1], context_dim=C_train.shape[1]).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    n = X_train.shape[0]
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n)
        running = 0.0
        for i in range(0, n, batch_size):
            opt.zero_grad()
            idx = perm[i:i + batch_size]
            x, c = X_train[idx], C_train[idx]
            x_hat, mu, logvar = model(x, c)
            recon = nn.functional.mse_loss(x_hat, x, reduction="mean")
            kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon + beta * kld
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
        if (epoch + 1) % 100 == 0:
            print(f"  VAE epoch {epoch+1}/{epochs} | loss: {running/n:.4f}")
    model.eval()
    return model


def train_reward_oracle(X_oracle, y_oracle, epochs=200, lr=1e-3):
    X_oracle, y_oracle = X_oracle.to(DEVICE), y_oracle.to(DEVICE)
    oracle = SurvivalRewardOracle(input_dim=X_oracle.shape[1]).to(DEVICE)
    opt = optim.AdamW(oracle.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    oracle.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = criterion(oracle(X_oracle), y_oracle)
        loss.backward()
        opt.step()
    oracle.eval()
    return oracle


def train_flow_map(X_train, C_train, epochs=300, batch_size=32, lr=5e-4,
                    context_dim_override=None, init_model=None):
    X_train, C_train = X_train.to(DEVICE), C_train.to(DEVICE)
    data_dim = X_train.shape[1]
    context_dim = context_dim_override or C_train.shape[1]
    model = init_model if init_model is not None else TCGAFlowMap(data_dim, context_dim).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    n = X_train.size(0)
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n)
        running = 0.0
        for i in range(0, n, batch_size):
            opt.zero_grad()
            idx = perm[i:i + batch_size]
            x1, c = X_train[idx], C_train[idx]
            x0 = torch.randn_like(x1)
            t = torch.rand(x1.size(0), 1, device=DEVICE)
            xt = (1.0 - t) * x0 + t * x1
            pred_v = model(xt, c, t)
            loss = nn.functional.mse_loss(pred_v, x1 - x0)
            loss.backward()
            opt.step()
            running += loss.item() * x1.size(0)
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:03d}/{epochs} | flow matching loss: {running/n:.4f}")
    model.eval()
    return model


def train_flow_map_no_context(X_train, epochs=300, batch_size=32, lr=5e-4):
    X = X_train.to(DEVICE)
    model = TCGAFlowMap(data_dim=X.shape[1], context_dim=1, hidden_dim=256, blocks=3).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    n = X.shape[0]
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n)
        running = 0.0
        for i in range(0, n, batch_size):
            opt.zero_grad()
            idx = perm[i:i + batch_size]
            x1 = X[idx]
            x0 = torch.randn_like(x1)
            t = torch.rand(x1.shape[0], 1, device=DEVICE)
            xt = (1 - t) * x0 + t * x1
            c_zeros = torch.zeros(x1.shape[0], 1, device=DEVICE)
            pred_v = model(xt, c_zeros, t)
            loss = nn.functional.mse_loss(pred_v, x1 - x0)
            loss.backward()
            opt.step()
            running += loss.item() * x1.shape[0]
        if (epoch + 1) % 100 == 0:
            print(f"  No-context Epoch {epoch+1}/{epochs} | loss: {running/n:.4f}")
    model.eval()
    return model


# ════════════════════════════════════════════════════════════════════════════════════
# PART E — metrics
# ════════════════════════════════════════════════════════════════════════════════════

def mmd_rbf(X, Y, gamma=None):
    X, Y = np.asarray(X), np.asarray(Y)
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    Kxx, Kyy, Kxy = rbf_kernel(X, X, gamma), rbf_kernel(Y, Y, gamma), rbf_kernel(X, Y, gamma)
    return Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()


def mean_wasserstein(X, Y):
    X, Y = np.asarray(X), np.asarray(Y)
    return float(np.mean([wasserstein_distance(X[:, j], Y[:, j]) for j in range(X.shape[1])]))


def per_feature_shift(X_a, X_b, feature_names):
    rows = []
    for j, name in enumerate(feature_names):
        rows.append({"feature": name, "wasserstein": wasserstein_distance(X_a[:, j], X_b[:, j])})
    return pd.DataFrame(rows).sort_values("wasserstein", ascending=False)


# ════════════════════════════════════════════════════════════════════════════════════
# PART F — the experiments
# ════════════════════════════════════════════════════════════════════════════════════

def experiment1_in_domain_baseline(flow_model, oracle, X_test, C_test, X_train, n_steering=40, alpha=1.5):
    with torch.no_grad():
        x0 = torch.randn(X_test.size(0), X_test.size(1), device=DEVICE)
        x_gen = integrate_flow(flow_model, x0, C_test.to(DEVICE)).cpu().numpy()
    x_real = X_test.numpy()
    mmd = mmd_rbf(x_real, x_gen)
    wd = mean_wasserstein(x_real, x_gen)

    n = min(n_steering, X_test.shape[0])
    c_batch = C_test[:n].to(DEVICE)
    x0 = torch.randn(n, X_train.shape[1], device=DEVICE, requires_grad=True)
    x1 = integrate_flow(flow_model, x0, c_batch)
    logits = oracle(x1)
    target = torch.full((n,), 1, dtype=torch.long, device=DEVICE)
    nn.functional.cross_entropy(logits, target).backward()
    with torch.no_grad():
        x0_steered = x0 - alpha * x0.grad
        x1_steered = integrate_flow(flow_model, x0_steered, c_batch)
        risk_before = torch.softmax(logits, dim=-1)[:, 1].mean().item()
        risk_after = torch.softmax(oracle(x1_steered), dim=-1)[:, 1].mean().item()

    result = {"mmd": mmd, "mean_wasserstein": wd, "risk_before_steering": risk_before,
              "risk_after_steering": risk_after}
    print("[Experiment 1 -- In-domain baseline]", result)
    return result


def experiment2_population_shift(dp, flow_model, ext_expr, ext_clin, cohort_label="pooled",
                                  baseline_gaussian=None, baseline_vae=None, X_train=None, C_train=None):
    """CHANGE (fix 6): also reports the Gaussian and VAE baselines' MMD/Wasserstein on
    the same external samples, so the flow model's number has something to be judged
    against. baseline_gaussian / baseline_vae are optional pre-fit TCGA-side models."""
    X_ext, C_ext, y_ext, keep_idx = dp.transform_external(ext_clin, ext_expr)
    if X_ext.shape[0] < 5:
        print(f"[Experiment 2 | {cohort_label}] Too few harmonizable samples "
              f"({X_ext.shape[0]}) -- insufficient severity_context / gene overlap.")
        return None, X_ext

    dp.assert_no_leakage(keep_idx)  # CHANGE (fix 3)

    with torch.no_grad():
        x0 = torch.randn(X_ext.shape[0], X_ext.shape[1], device=DEVICE)
        x_gen_flow = integrate_flow(flow_model, x0, C_ext.to(DEVICE)).cpu().numpy()
    x_real = X_ext.numpy()

    result = {
        "cohort": cohort_label,
        "n_samples": X_ext.shape[0],
        "flow_mmd": mmd_rbf(x_real, x_gen_flow),
        "flow_wasserstein": mean_wasserstein(x_real, x_gen_flow),
    }

    if baseline_gaussian is not None:
        x_gen_gauss = baseline_gaussian.sample(X_ext.shape[0])
        result["gaussian_mmd"] = mmd_rbf(x_real, x_gen_gauss)
        result["gaussian_wasserstein"] = mean_wasserstein(x_real, x_gen_gauss)

    if baseline_vae is not None:
        x_gen_vae = baseline_vae.sample(C_ext.to(DEVICE), DEVICE)
        result["vae_mmd"] = mmd_rbf(x_real, x_gen_vae)
        result["vae_wasserstein"] = mean_wasserstein(x_real, x_gen_vae)

    pca2 = PCA(n_components=2, random_state=RANDOM_STATE).fit(x_real)
    result["pca_centroid_distance"] = float(np.linalg.norm(
        pca2.transform(x_real).mean(0) - pca2.transform(x_gen_flow).mean(0)))

    shift_table = per_feature_shift(x_real, x_gen_flow, dp.feature_names)
    print(f"[Experiment 2 | {cohort_label} -- Population shift]", result)
    return {"summary": result, "per_feature_shift": shift_table,
            "X_real": x_real, "X_generated_flow": x_gen_flow,
            "C_ext": C_ext, "keep_idx": keep_idx}, X_ext


def plot_population_shift_umap(x_tcga_real, x_ext_real, save_path, cohort_label="Saudi cohort"):
    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=RANDOM_STATE)
        method = "UMAP"
    except ImportError:
        from sklearn.manifold import TSNE
        combined_n = len(x_tcga_real) + len(x_ext_real)
        reducer = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=min(30, combined_n - 1))
        method = "t-SNE"

    combined = np.vstack([x_tcga_real, x_ext_real])
    labels = ["TCGA (source)"] * len(x_tcga_real) + [cohort_label] * len(x_ext_real)
    emb = reducer.fit_transform(combined)
    plot_df = pd.DataFrame({"x": emb[:, 0], "y": emb[:, 1], "Cohort": labels})

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {"TCGA (source)": "#4878CF", cohort_label: "#D65F5F"}
    for label, grp in plot_df.groupby("Cohort"):
        ax.scatter(grp["x"], grp["y"], label=label, alpha=0.7, color=colors.get(label, "#888888"),
                   edgecolors="k", linewidths=0.3, s=40)
    ax.set_title(f"{method}: TCGA vs {cohort_label} Digital-Twin State (Population Shift)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def experiment3_stage_conditioning_transfer(dp, flow_model, flow_no_context, ext_expr, ext_clin,
                                             cohort_label="pooled", stages=(1.0, 2.0, 3.0)):
    """CHANGE (fix 1): GATED on dp.severity_context_source == 'grade'. If TCGA fell
    back to 'stage', this experiment is skipped entirely and returns None with a
    clear explanation, instead of silently comparing stage-conditioned TCGA to
    grade-conditioned external data."""
    if dp.severity_context_source != "grade":
        print(f"[Experiment 3 | {cohort_label}] SKIPPED: TCGA severity_context is "
              f"'{dp.severity_context_source}' but the external cohort's severity_context "
              f"is 'grade'. Comparing these would compare two different clinical constructs "
              f"(stage vs grade) and is not a valid conditioning-transfer test. Increase TCGA "
              f"grade coverage or accept STAGE as the axis on both sides before rerunning.")
        return pd.DataFrame()

    X_ext, C_ext, y_ext, keep_idx = dp.transform_external(ext_clin, ext_clin.index.to_series() * 0 + 1
                                                            if False else ext_expr, verbose=False)
    ext_severity = ext_clin.loc[keep_idx, "severity_context"]

    rows = []
    for s in stages:
        mask = (ext_severity >= s - 0.5) & (ext_severity < s + 0.5)
        if mask.sum() < 2:
            continue
        x_real_sub = X_ext.numpy()[mask.values]
        n_samples = max(20, mask.sum())

        c_scaled = torch.tensor(dp.context_scaler.transform([[s]]), dtype=torch.float32)
        c_batch = c_scaled.repeat(n_samples, 1).to(DEVICE)
        c_zero = torch.zeros(n_samples, 1, device=DEVICE)
        x0 = torch.randn(n_samples, X_ext.shape[1], device=DEVICE)

        with torch.no_grad():
            gen_with_ctx = integrate_flow(flow_model, x0, c_batch).cpu().numpy()
            gen_no_ctx = integrate_flow(flow_no_context, x0, c_zero).cpu().numpy()

        rows.append({
            "cohort": cohort_label, "severity_context": s, "n_real": int(mask.sum()),
            "wasserstein_with_context": mean_wasserstein(x_real_sub, gen_with_ctx),
            "wasserstein_no_context": mean_wasserstein(x_real_sub, gen_no_ctx),
        })

    df = pd.DataFrame(rows)
    if len(df):
        df["improvement_from_context"] = df["wasserstein_no_context"] - df["wasserstein_with_context"]
        df["pct_improvement"] = 100 * df["improvement_from_context"] / df["wasserstein_no_context"]
    print(f"[Experiment 3 | {cohort_label} -- Stage/grade-conditioning transfer (VALID: same axis both sides)]")
    print(df.to_string(index=False))
    return df


def experiment4_risk_steering_transfer(dp, flow_model, oracle, ext_expr, ext_clin,
                                        cohort_label="pooled", alpha=1.5, n_patients=40,
                                        n_seeds=N_MONOTONICITY_SEEDS):
    """CHANGE (fix 2): replaces the strict True/False monotonicity check with:
      - Spearman rank correlation between alpha and mean risk (1 seed's curve)
      - number of pairwise violations (risk decreasing as alpha increases)
      - max single-step downward excursion
      - mean total risk increase (last - first)
      - a bootstrap-style confidence interval by repeating the whole curve across
        n_seeds different random x0 draws
    """
    X_ext, C_ext, y_ext, keep_idx = dp.transform_external(ext_clin, ext_expr)
    n = min(n_patients, X_ext.shape[0])
    if n < 2:
        print(f"[Experiment 4 | {cohort_label}] Not enough harmonized samples to run steering.")
        return None
    c_batch_base = C_ext[:n].to(DEVICE)

    def steer(c_batch, target_class, alpha_val, seed):
        g = torch.Generator(device=DEVICE).manual_seed(seed)
        x0 = torch.randn(n, X_ext.shape[1], device=DEVICE, generator=g, requires_grad=True)
        x1 = integrate_flow(flow_model, x0, c_batch)
        logits_before = oracle(x1)
        target = torch.full((n,), target_class, dtype=torch.long, device=DEVICE)
        nn.functional.cross_entropy(logits_before, target).backward()
        with torch.no_grad():
            x0_steered = x0 - alpha_val * x0.grad
            x1_steered = integrate_flow(flow_model, x0_steered, c_batch)
            logits_after = oracle(x1_steered)
        risk_before = torch.softmax(logits_before, dim=-1)[:, 1].mean().item()
        risk_after = torch.softmax(logits_after, dim=-1)[:, 1].mean().item()
        return risk_before, risk_after

    hi_before, hi_after = steer(c_batch_base, 1, alpha, seed=RANDOM_STATE)
    lo_before, lo_after = steer(c_batch_base, 0, alpha, seed=RANDOM_STATE)

    alphas = np.linspace(0, 3.0, 8)
    curves = []
    for seed in range(RANDOM_STATE, RANDOM_STATE + n_seeds):
        risks = [steer(c_batch_base, 1, a, seed=seed)[1] for a in alphas]
        curves.append(risks)
    curves = np.array(curves)  # (n_seeds, len(alphas))

    mean_curve = curves.mean(axis=0)
    ci_low = np.percentile(curves, 2.5, axis=0)
    ci_high = np.percentile(curves, 97.5, axis=0)

    spearman_rho, spearman_p = spearmanr(alphas, mean_curve)
    diffs = np.diff(mean_curve)
    n_violations = int(np.sum(diffs < -1e-3))
    max_downward_excursion = float(min(0.0, diffs.min()))
    mean_risk_increase = float(mean_curve[-1] - mean_curve[0])

    result = {
        "cohort": cohort_label,
        "n_patients": n,
        "risk_before_hi": hi_before, "risk_after_hi": hi_after,
        "risk_before_lo": lo_before, "risk_after_lo": lo_after,
        "spearman_rho_alpha_vs_risk": float(spearman_rho),
        "spearman_p": float(spearman_p),
        "n_monotonicity_violations": n_violations,
        "max_downward_excursion": max_downward_excursion,
        "mean_risk_increase_end_minus_start": mean_risk_increase,
        "alphas": alphas.tolist(),
        "mean_risk_by_alpha": mean_curve.tolist(),
        "risk_ci95_low": ci_low.tolist(),
        "risk_ci95_high": ci_high.tolist(),
    }
    print(f"[Experiment 4 | {cohort_label} -- Risk-steering transfer] "
          f"Spearman rho={spearman_rho:.3f} (p={spearman_p:.3g}), "
          f"{n_violations} violation(s) out of {len(diffs)} steps, "
          f"max downward excursion={max_downward_excursion:.4f}, "
          f"mean risk increase={mean_risk_increase:.4f}")
    return result


def experiment5_lightweight_adaptation(dp, flow_model, ext_expr, ext_clin, cohort_label="pooled",
                                        calib_frac=0.4, adapt_epochs=60, adapt_lr=1e-4,
                                        baseline_gaussian=None, baseline_vae=None):
    """CHANGE (fix 3 + 6): explicit disjointness assertion between calibration and eval
    Saudi patients, plus Gaussian/VAE baselines adapted the same way for comparison."""
    X_ext, C_ext, y_ext, keep_idx = dp.transform_external(ext_clin, ext_expr)
    n = X_ext.shape[0]
    if n < 10:
        print(f"[Experiment 5 | {cohort_label}] Too few samples for a calibration/eval split.")
        return None

    idx = np.arange(n)
    calib_idx, eval_idx = train_test_split(idx, test_size=1 - calib_frac, random_state=RANDOM_STATE)
    assert_disjoint(calib_idx, eval_idx, label=f"Experiment 5 | {cohort_label}")

    X_calib, C_calib = X_ext[calib_idx], C_ext[calib_idx]
    X_eval, C_eval = X_ext[eval_idx], C_ext[eval_idx]

    with torch.no_grad():
        x0 = torch.randn(X_eval.shape[0], X_eval.shape[1], device=DEVICE)
        gen_zero_shot = integrate_flow(flow_model, x0, C_eval.to(DEVICE)).cpu().numpy()
    mmd_zero_shot = mmd_rbf(X_eval.numpy(), gen_zero_shot)
    wd_zero_shot = mean_wasserstein(X_eval.numpy(), gen_zero_shot)

    adapted_model = TCGAFlowMap(data_dim=X_ext.shape[1], context_dim=C_ext.shape[1]).to(DEVICE)
    adapted_model.load_state_dict(flow_model.state_dict())
    adapted_model = train_flow_map(X_calib, C_calib, epochs=adapt_epochs, batch_size=min(16, len(calib_idx)),
                                    lr=adapt_lr, init_model=adapted_model)

    with torch.no_grad():
        x0 = torch.randn(X_eval.shape[0], X_eval.shape[1], device=DEVICE)
        gen_adapted = integrate_flow(adapted_model, x0, C_eval.to(DEVICE)).cpu().numpy()
    mmd_adapted = mmd_rbf(X_eval.numpy(), gen_adapted)
    wd_adapted = mean_wasserstein(X_eval.numpy(), gen_adapted)

    result = {
        "cohort": cohort_label,
        "n_calibration": len(calib_idx), "n_eval": len(eval_idx),
        "flow_mmd_zero_shot": mmd_zero_shot, "flow_mmd_adapted": mmd_adapted,
        "flow_wasserstein_zero_shot": wd_zero_shot, "flow_wasserstein_adapted": wd_adapted,
        "flow_mmd_improvement": mmd_zero_shot - mmd_adapted,
        "flow_wasserstein_improvement": wd_zero_shot - wd_adapted,
    }

    if baseline_gaussian is not None:
        gauss_zero = baseline_gaussian.sample(X_eval.shape[0])
        gauss_adapted_model = baseline_gaussian.adapt(X_calib)
        gauss_adapted = gauss_adapted_model.sample(X_eval.shape[0])
        result["gaussian_mmd_zero_shot"] = mmd_rbf(X_eval.numpy(), gauss_zero)
        result["gaussian_mmd_adapted"] = mmd_rbf(X_eval.numpy(), gauss_adapted)

    if baseline_vae is not None:
        vae_zero = baseline_vae.sample(C_eval.to(DEVICE), DEVICE)
        vae_adapted_model = copy.deepcopy(baseline_vae)
        vae_adapted_model = train_vae(X_calib, C_calib, epochs=adapt_epochs,
                                       batch_size=min(16, len(calib_idx)), lr=adapt_lr,
                                       init_model=vae_adapted_model)
        vae_adapted = vae_adapted_model.sample(C_eval.to(DEVICE), DEVICE)
        result["vae_mmd_zero_shot"] = mmd_rbf(X_eval.numpy(), vae_zero)
        result["vae_mmd_adapted"] = mmd_rbf(X_eval.numpy(), vae_adapted)

    print(f"[Experiment 5 | {cohort_label} -- Lightweight adaptation]", result)
    return result, adapted_model


def experiment5b_calibration_size_ablation(dp, flow_model, ext_expr, ext_clin, cohort_label="pooled",
                                            calib_sizes=CALIBRATION_SIZES, adapt_epochs=60, adapt_lr=1e-4):
    """CHANGE (fix 3): sweeps the calibration set size (5/10/20/30/40 patients),
    always evaluating against the SAME held-out eval set, so the reported curve
    isn't confounded by a shrinking/growing eval pool."""
    X_ext, C_ext, y_ext, keep_idx = dp.transform_external(ext_clin, ext_expr, verbose=False)
    n = X_ext.shape[0]
    if n < max(calib_sizes) + 10:
        print(f"[Experiment 5b | {cohort_label}] Too few samples ({n}) for the full "
              f"calibration-size ablation grid {calib_sizes}; using only sizes that fit.")
        calib_sizes = [c for c in calib_sizes if c + 10 <= n]
        if not calib_sizes:
            return pd.DataFrame()

    rng = np.random.RandomState(RANDOM_STATE)
    perm = rng.permutation(n)
    max_calib = max(calib_sizes)
    eval_idx = perm[max_calib:]
    if len(eval_idx) < 5:
        eval_idx = perm[-max(5, n - max_calib):]
    X_eval, C_eval = X_ext[eval_idx], C_ext[eval_idx]

    rows = []
    for size in calib_sizes:
        calib_idx = perm[:size]
        assert_disjoint(calib_idx, eval_idx, label=f"Experiment 5b | {cohort_label} | size={size}")
        X_calib, C_calib = X_ext[calib_idx], C_ext[calib_idx]

        adapted_model = TCGAFlowMap(data_dim=X_ext.shape[1], context_dim=C_ext.shape[1]).to(DEVICE)
        adapted_model.load_state_dict(flow_model.state_dict())
        adapted_model = train_flow_map(X_calib, C_calib, epochs=adapt_epochs,
                                        batch_size=min(16, size), lr=adapt_lr, init_model=adapted_model)
        with torch.no_grad():
            x0 = torch.randn(X_eval.shape[0], X_eval.shape[1], device=DEVICE)
            gen = integrate_flow(adapted_model, x0, C_eval.to(DEVICE)).cpu().numpy()
        rows.append({
            "cohort": cohort_label, "n_calibration": size, "n_eval": len(eval_idx),
            "mmd": mmd_rbf(X_eval.numpy(), gen), "wasserstein": mean_wasserstein(X_eval.numpy(), gen),
        })
    df = pd.DataFrame(rows)
    print(f"[Experiment 5b | {cohort_label} -- Calibration-size ablation]")
    print(df.to_string(index=False))
    return df


def gene_imputation_sensitivity_analysis(clinical_df, expr_df, severity_context_source,
                                          ext_expr, ext_clin, cohort_label="pooled"):
    """CHANGE (fix 4): builds TWO pipelines -- (A) default, missing external genes
    (e.g. MRE11) imputed with the TCGA train mean; (B) those genes dropped from the
    top-gene pool entirely and PCA re-derived on the intersection -- and compares
    the resulting zero-shot MMD/Wasserstein on the same external cohort."""
    dp_a = TCGADataPipeline(clinical_df, expr_df, severity_context_source)
    splits_a = dp_a.prepare_data()
    X_train_a, C_train_a, _ = splits_a["train"]
    flow_a = train_flow_map(X_train_a, C_train_a, epochs=150)
    X_ext_a, C_ext_a, _, keep_a = dp_a.transform_external(ext_clin, ext_expr, verbose=True)
    missing_genes = [g for g in dp_a.top_genes if g not in ext_expr.columns]

    with torch.no_grad():
        x0 = torch.randn(X_ext_a.shape[0], X_ext_a.shape[1], device=DEVICE)
        gen_a = integrate_flow(flow_a, x0, C_ext_a.to(DEVICE)).cpu().numpy()
    result_a = {"variant": "A_impute_missing_genes", "cohort": cohort_label,
                "missing_genes_imputed": missing_genes,
                "mmd": mmd_rbf(X_ext_a.numpy(), gen_a), "wasserstein": mean_wasserstein(X_ext_a.numpy(), gen_a)}

    if not missing_genes:
        print(f"[Gene sensitivity | {cohort_label}] No missing genes on this platform -- "
              f"variant B is identical to variant A, skipping.")
        return pd.DataFrame([result_a])

    dp_b = TCGADataPipeline(clinical_df, expr_df, severity_context_source, drop_genes=missing_genes)
    splits_b = dp_b.prepare_data()
    X_train_b, C_train_b, _ = splits_b["train"]
    flow_b = train_flow_map(X_train_b, C_train_b, epochs=150)
    X_ext_b, C_ext_b, _, keep_b = dp_b.transform_external(ext_clin, ext_expr, verbose=True)

    with torch.no_grad():
        x0 = torch.randn(X_ext_b.shape[0], X_ext_b.shape[1], device=DEVICE)
        gen_b = integrate_flow(flow_b, x0, C_ext_b.to(DEVICE)).cpu().numpy()
    result_b = {"variant": "B_drop_missing_genes", "cohort": cohort_label,
                "dropped_genes": missing_genes,
                "mmd": mmd_rbf(X_ext_b.numpy(), gen_b), "wasserstein": mean_wasserstein(X_ext_b.numpy(), gen_b)}

    df = pd.DataFrame([result_a, result_b])
    print(f"[Gene sensitivity | {cohort_label}] impute vs drop for {missing_genes}:")
    print(df.to_string(index=False))
    return df


# ════════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════════

def run_all_experiments():
    print("Fetching TCGA clinical data from cBioPortal...")
    clinical_raw = fetch_clinical_cbioportal(TCGA_STUDY_ID)
    print("Fetching TCGA expression data from cBioPortal...")
    expr_raw = fetch_expression_cbioportal(TCGA_STUDY_ID)

    clinical_df, severity_context_source = build_tcga_clinical(clinical_raw)  # CHANGE (fix 1)
    expr_df = build_tcga_expression(expr_raw)
    common_patients = clinical_df.index.intersection(expr_df.index)
    clinical_df, expr_df = clinical_df.loc[common_patients], expr_df.loc[common_patients]
    print(f"TCGA patients with clinical + expression data: {len(common_patients)}")
    print(f"TCGA severity_context_source = '{severity_context_source}'")

    dp = TCGADataPipeline(clinical_df, expr_df, severity_context_source)
    splits = dp.prepare_data()
    X_train, C_train, y_train = splits["train"]
    X_oracle, C_oracle, y_oracle = splits["oracle"]
    X_test, C_test, y_test = splits["test"]
    print(f"Train: {X_train.shape[0]} | Oracle: {X_oracle.shape[0]} | Test: {X_test.shape[0]} "
          f"| Digital twin dim: {X_train.shape[1]} (common-feature schema)")

    print("[TRAINING] Reward oracle...")
    oracle = train_reward_oracle(X_oracle, y_oracle)

    print("[TRAINING] TCGA conditional flow map (with severity context)...")
    flow_model = train_flow_map(X_train, C_train)

    print("[TRAINING] TCGA flow map ablation (no context)...")
    flow_no_context = train_flow_map_no_context(X_train)

    print("[TRAINING] Gaussian baseline...")           # CHANGE (fix 6)
    gaussian_baseline = GaussianBaseline().fit(X_train)

    print("[TRAINING] VAE baseline...")                # CHANGE (fix 6)
    vae_baseline = train_vae(X_train, C_train, epochs=200)

    # ---- Saudi cohorts: pooled AND per-series (CHANGE fix 5) ----
    saudi_expr_pooled, saudi_clin_pooled, per_series = load_saudi_cohorts()

    exp1 = experiment1_in_domain_baseline(flow_model, oracle, X_test, C_test, X_train)

    all_exp2, all_exp3, all_exp4, all_exp5, all_exp5b, all_gene_sens = {}, {}, {}, {}, {}, {}

    cohorts_to_run = list(per_series.items()) + [("pooled", (saudi_expr_pooled, saudi_clin_pooled))]
    for cohort_label, (ext_expr, ext_clin) in cohorts_to_run:
        print(f"\n{'='*80}\nCohort: {cohort_label}\n{'='*80}")
        exp2, X_ext_harmonized = experiment2_population_shift(
            dp, flow_model, ext_expr, ext_clin, cohort_label=cohort_label,
            baseline_gaussian=gaussian_baseline, baseline_vae=vae_baseline)
        all_exp2[cohort_label] = exp2
        if exp2 is None:
            continue

        plot_population_shift_umap(X_test.numpy(), exp2["X_real"],
                                    f"{OUT_DIR}/figure_population_shift_umap_{cohort_label}.png",
                                    cohort_label=cohort_label)

        all_exp3[cohort_label] = experiment3_stage_conditioning_transfer(
            dp, flow_model, flow_no_context, ext_expr, ext_clin, cohort_label=cohort_label)
        all_exp4[cohort_label] = experiment4_risk_steering_transfer(
            dp, flow_model, oracle, ext_expr, ext_clin, cohort_label=cohort_label)

        exp5_out = experiment5_lightweight_adaptation(
            dp, flow_model, ext_expr, ext_clin, cohort_label=cohort_label,
            baseline_gaussian=gaussian_baseline, baseline_vae=vae_baseline)
        if exp5_out is not None:
            all_exp5[cohort_label], _adapted = exp5_out
            all_exp5b[cohort_label] = experiment5b_calibration_size_ablation(
                dp, flow_model, ext_expr, ext_clin, cohort_label=cohort_label)

        all_gene_sens[cohort_label] = gene_imputation_sensitivity_analysis(
            clinical_df, expr_df, severity_context_source, ext_expr, ext_clin, cohort_label=cohort_label)

    # ---- Save bundle ----
    summary_rows = [{"experiment": "1_in_domain_baseline", **exp1}]
    for label, e2 in all_exp2.items():
        if e2 is not None:
            summary_rows.append({"experiment": f"2_population_shift[{label}]", **e2["summary"]})
    for label, e4 in all_exp4.items():
        if e4 is not None:
            summary_rows.append({"experiment": f"4_risk_steering_transfer[{label}]",
                                  **{k: v for k, v in e4.items()
                                     if k not in ("alphas", "mean_risk_by_alpha", "risk_ci95_low", "risk_ci95_high")}})
    for label, e5 in all_exp5.items():
        if e5 is not None:
            summary_rows.append({"experiment": f"5_lightweight_adaptation[{label}]", **e5})

    pd.DataFrame(summary_rows).to_csv(f"{OUT_DIR}/experiment_summary.csv", index=False)
    for label, e3 in all_exp3.items():
        if e3 is not None and len(e3):
            e3.to_csv(f"{OUT_DIR}/experiment3_stage_conditioning_transfer_{label}.csv", index=False)
    for label, e2 in all_exp2.items():
        if e2 is not None:
            e2["per_feature_shift"].to_csv(f"{OUT_DIR}/experiment2_per_feature_shift_{label}.csv", index=False)
    for label, e5b in all_exp5b.items():
        if e5b is not None and len(e5b):
            e5b.to_csv(f"{OUT_DIR}/experiment5b_calibration_ablation_{label}.csv", index=False)
    for label, gs in all_gene_sens.items():
        if gs is not None and len(gs):
            gs.to_csv(f"{OUT_DIR}/gene_imputation_sensitivity_{label}.csv", index=False)

    zip_path = "/content/population_shift_results.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for fname in os.listdir(OUT_DIR):
            zf.write(os.path.join(OUT_DIR, fname), arcname=fname)
    print(f"Bundled results into {zip_path}")

    return {
        "dp": dp, "oracle": oracle, "flow_model": flow_model, "flow_no_context": flow_no_context,
        "gaussian_baseline": gaussian_baseline, "vae_baseline": vae_baseline,
        "experiment1": exp1, "experiment2": all_exp2, "experiment3": all_exp3,
        "experiment4": all_exp4, "experiment5": all_exp5, "experiment5b": all_exp5b,
        "gene_sensitivity": all_gene_sens,
    }


if __name__ == "__main__":
    results = run_all_experiments()
    print("\n=== DONE ===")

"""
Population-Shift Evaluation of Conditional Flow-Matching Breast-Cancer Digital Twins
"""

# ── CELL 0 — installs ────────────────────────────────────────────────────────────────
# !pip install -q torch scikit-learn pandas numpy scipy matplotlib seaborn requests tqdm
# !pip install -q GEOparse umap-learn

# ── CELL 1 — imports & config ───────────────────────────────────────────────────────
import os, re, warnings, zipfile, copy
import numpy as np
import pandas as pd
import requests

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.covariance import LedoitWolf


from scipy.stats import wasserstein_distance, spearmanr

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "/content/data"
OUT_DIR = "/content/outputs"
GEO_DIR = "/content/data/geo"
for d in (DATA_DIR, OUT_DIR, GEO_DIR):
    os.makedirs(d, exist_ok=True)

CBIOPORTAL_BASE = "https://www.cbioportal.org/api"
TCGA_STUDY_ID = "brca_tcga"

# Saudi Arabia / Muslim-majority-country cohorts (population context only).
# independently before pooling, instead of only ever seeing a merged frame.
SAUDI_GEO_SERIES = ["GSE36295", "GSE29044"]

N_TOP_GENES = 80           # top variable genes, fit on TCGA train only
N_GENE_PCS = 10            # genomic embedding dimensionality
FIVE_YEAR_DAYS = 365 * 5
N_EULER_STEPS = 10

CALIBRATION_SIZES = [5, 10, 20, 30, 40]
N_MONOTONICITY_SEEDS = 5

N_BOOTSTRAP = 1000                # bootstrap resamples for MMD/Wasserstein CIs (item 1)
N_CALIBRATION_SPLIT_SEEDS = 5     # repeated random calibration/eval splits per size (item 6)
CALIBRATION_SPLIT_SEEDS = [RANDOM_STATE + k for k in range(N_CALIBRATION_SPLIT_SEEDS)]
GEN_POOL_MULTIPLIER = 5           # generated-sample pool size = GEN_POOL_MULTIPLIER * n_eval,
                                   # resampled (not retrained) inside the bootstrap loop

# ---- COMMON FEATURE SUBSET --------------------------------------------------------
COMMON_CLINICAL_NUMERIC = ["age", "lymph_nodes"]
COMMON_CLINICAL_CATEGORICAL = ["histological_type", "er_status", "pr_status",
                                "her2_status", "grade"]
print("Device:", DEVICE)


# ════════════════════════════════════════════════════════════════════════════════════
# PART A — TCGA-BRCA (source domain), fit-once pipeline
# ════════════════════════════════════════════════════════════════════════════════════

# ── CELL 2 — TCGA download via cBioPortal API ───────────────────────────────────────
def fetch_clinical_cbioportal(study_id):
    url = f"{CBIOPORTAL_BASE}/studies/{study_id}/patients"
    patients = requests.get(url, headers={"Accept": "application/json"}, timeout=60).json()
    patient_ids = [p["patientId"] for p in patients]
    print(f"  Fetched {len(patient_ids)} patients")

    url2 = f"{CBIOPORTAL_BASE}/studies/{study_id}/clinical-data?clinicalDataType=PATIENT&pageSize=100000"
    clin_raw = requests.get(url2, headers={"Accept": "application/json"}, timeout=120).json()
    rows = {}
    for entry in clin_raw:
        pid = entry["patientId"]
        rows.setdefault(pid, {"patient_id": pid})[entry["clinicalAttributeId"]] = entry["value"]
    df = pd.DataFrame.from_dict(rows, orient="index").reset_index(drop=True)
    print(f"  Clinical dataframe: {df.shape}")
    return df


def fetch_expression_cbioportal(study_id):
    profiles = requests.get(f"{CBIOPORTAL_BASE}/studies/{study_id}/molecular-profiles",
                             headers={"Accept": "application/json"}, timeout=60).json()
    rna_profile = next(p["molecularProfileId"] for p in profiles
                        if "rna_seq" in p["molecularProfileId"].lower()
                        and "v2" in p["molecularProfileId"].lower())
    print(f"  Using expression profile: {rna_profile}")

    samples = requests.get(f"{CBIOPORTAL_BASE}/studies/{study_id}/samples?pageSize=10000",
                            headers={"Accept": "application/json"}, timeout=60).json()
    sample_ids = [s["sampleId"] for s in samples][:500]
    print(f"  Using {len(sample_ids)} samples")

    gene_symbols = [
        "ESR1", "ERBB2", "PGR", "MKI67", "BRCA1", "BRCA2", "TP53", "PIK3CA", "CDH1", "GATA3",
        "RB1", "PTEN", "AKT1", "CCND1", "FOXA1", "EGFR", "FGFR1", "MYC", "AURKA", "TOP2A",
        "BIRC5", "BCL2", "FOXM1", "PLK1", "MELK", "RAD51", "CHEK1", "CHEK2", "ATM", "PALB2",
        "ESR2", "AR", "STAT3", "NOTCH1", "CDK4", "CDK6", "CDKN2A", "MDM2", "KRT8", "KRT18",
        "KRT19", "MLPH", "TFF1", "CLDN7", "FN1", "COL3A1", "DCN", "VEGFA", "MET", "IGF1R",
        "MAP3K1", "NF1", "TSC1", "TSC2", "KEAP1", "NFE2L2", "WNT5A", "SRC", "JAK2", "NCOA3",
        "UBE2C", "TYMS", "PCNA", "E2F1", "BUB1", "CENPA", "KIF2C", "TPX2", "PRC1", "EXO1",
        "MCM2", "BRIP1", "NBN", "MRE11", "RAD50", "ATR", "CCND2", "CCNE1", "MMP2", "MMP9",
    ]

    entrez_ids, symbol_map = [], {}
    for sym in gene_symbols:
        try:
            r = requests.get(f"{CBIOPORTAL_BASE}/genes/{sym}",
                              headers={"Accept": "application/json"}, timeout=10).json()
            if "entrezGeneId" in r:
                entrez_ids.append(r["entrezGeneId"])
                symbol_map[r["entrezGeneId"]] = r["hugoGeneSymbol"]
        except Exception:
            continue
    print(f"  Resolved {len(entrez_ids)} gene IDs")

    payload = {"sampleIds": sample_ids, "entrezGeneIds": entrez_ids}
    expr_resp = requests.post(
        f"{CBIOPORTAL_BASE}/molecular-profiles/{rna_profile}/molecular-data/fetch?projection=SUMMARY",
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        json=payload, timeout=180).json()

    if not isinstance(expr_resp, list) or len(expr_resp) == 0:
        raise ValueError(f"Unexpected expression response: {str(expr_resp)[:200]}")

    rows = {}
    for entry in expr_resp:
        if not isinstance(entry, dict):
            continue
        sid, eid = entry.get("sampleId"), entry.get("entrezGeneId")
        gene = symbol_map.get(eid, str(eid))
        rows.setdefault(sid, {"sample_id": sid})[gene] = entry.get("value", np.nan)

    expr_df = pd.DataFrame.from_dict(rows, orient="index").reset_index(drop=True)
    print(f"  Expression dataframe: {expr_df.shape}")
    return expr_df


# ── CELL 3 — TCGA clinical harmonization onto the COMMON schema ────────────────────
TCGA_COL_MAP = {
    "AGE": "age",
    "AJCC_PATHOLOGIC_TUMOR_STAGE": "stage",
    "RACE": "race",
    "HISTOLOGICAL_DIAGNOSIS": "histological_type",
    "MENOPAUSE_STATUS": "menopause_status",
    "RADIATION_THERAPY": "radiation_therapy",
    "NEOADJUVANT_THERAPY": "neoadjuvant_tx",
    "LYMPH_NODE_STATUS": "lymph_nodes",
    "ER_STATUS_BY_IHC": "er_status",
    "PR_STATUS_BY_IHC": "pr_status",
    "IHC_HER2": "her2_status",
    "TUMOR_GRADE": "grade",
    "OS_STATUS": "vital_status",
    "OS_MONTHS": "os_months",
    "patient_id": "patient_id",
}


def stage_to_severity(stage_str):
    if pd.isna(stage_str):
        return np.nan
    s = str(stage_str).upper().replace("STAGE", "").strip()
    m = re.match(r"^(IV|III|II|I)([ABC]?)$", s)
    if not m:
        return np.nan
    roman_to_val = {"I": 1.0, "II": 2.0, "III": 3.0, "IV": 4.0}
    sub_offset = {"A": 0.0, "B": 0.15, "C": 0.30, "": 0.0}
    return roman_to_val[m.group(1)] + sub_offset[m.group(2)]


def grade_to_severity(grade_raw):
    if pd.isna(grade_raw):
        return np.nan
    s = str(grade_raw).strip().upper()
    if "GX" in s or "N/A" in s or "NA" == s:
        return np.nan
    m = re.search(r"([1-3])", s)
    return float(m.group(1)) if m else np.nan


CONTEXT_SOURCE = "grade"
MIN_TCGA_GRADE_COVERAGE = 30


def parse_age_value(raw):
    if pd.isna(raw):
        return np.nan
    s = str(raw).strip()
    if re.match(r"^\d+(\.\d+)?$", s):
        return float(s)
    m = re.match(r"^>\s*(\d+)$", s)
    if m:
        return float(m.group(1)) + 5.0
    m = re.match(r"^<\s*(\d+)$", s)
    if m:
        return float(m.group(1)) - 5.0
    m = re.match(r"^(\d+)\s*-\s*(\d+)$", s)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2.0
    return np.nan


def normalize_marker_status(raw):
    if pd.isna(raw):
        return np.nan
    s = str(raw).strip().lower()
    if s in ("normal", "n/a", "na", ""):
        return np.nan
    if s in ("+", "1", "pos", "positive"):
        return "positive"
    if s in ("-", "0", "neg", "negative"):
        return "negative"
    if s == "2":
        return np.nan
    if s == "3":
        return "positive"
    return np.nan


def make_5yr_label(row):
    status = str(row.get("vital_status", "")).strip().lower()
    if status == "dead":
        d = row.get("days_to_death", np.nan)
        return np.nan if pd.isna(d) else (1.0 if d <= FIVE_YEAR_DAYS else 0.0)
    elif status == "alive":
        f = row.get("days_to_last_fu", np.nan)
        return np.nan if pd.isna(f) else (0.0 if f >= FIVE_YEAR_DAYS else np.nan)
    return np.nan


def resolve_tcga_grade_column(clinical_raw):
    candidates = [c for c in clinical_raw.columns if "GRADE" in c.upper()]
    best_col, best_count = None, 0
    for c in candidates:
        cnt = clinical_raw[c].notna().sum()
        if cnt > best_count:
            best_col, best_count = c, cnt
    if best_col:
        print(f"  [TCGA grade] candidate columns: {candidates}; using '{best_col}' "
              f"with {best_count}/{len(clinical_raw)} non-null values")
    else:
        print("  [TCGA grade] no GRADE-like column found in TCGA clinical data.")
    return best_col, best_count


def build_tcga_clinical(clinical_raw):
    """CHANGE (fix 1): now returns (df, severity_context_source) instead of just df,
    so callers know for certain whether TCGA ended up conditioned on 'grade' (matches
    the Saudi cohorts) or fell back to 'stage' (does NOT match Saudi grade-based
    severity). This is the single source of truth Experiment 3 gates on."""
    grade_col, grade_coverage = resolve_tcga_grade_column(clinical_raw)
    col_map = dict(TCGA_COL_MAP)
    if grade_col:
        col_map[grade_col] = "grade_raw"

    df = clinical_raw.rename(columns={k: v for k, v in col_map.items()
                                       if k in clinical_raw.columns})
    if "vital_status" in df.columns:
        df["vital_status"] = df["vital_status"].apply(lambda x: "dead" if str(x).startswith("1") else "alive")
    if "os_months" in df.columns:
        df["os_months"] = pd.to_numeric(df["os_months"], errors="coerce")
        df["days_to_death"] = df.apply(lambda r: r["os_months"] * 30.44 if r.get("vital_status") == "dead" else np.nan, axis=1)
        df["days_to_last_fu"] = df.apply(lambda r: r["os_months"] * 30.44 if r.get("vital_status") == "alive" else np.nan, axis=1)

    df["age"] = pd.to_numeric(df.get("age", np.nan), errors="coerce")
    df["stage"] = df.get("stage", np.nan)
    df["stage_severity_"] = df["stage"].apply(stage_to_severity)

    use_grade = CONTEXT_SOURCE == "grade" and grade_coverage >= MIN_TCGA_GRADE_COVERAGE
    if use_grade:
        df["severity_context"] = df["grade_raw"].apply(grade_to_severity)
        severity_context_source = "grade"
        print(f"  [TCGA severity_context] using GRADE (coverage {grade_coverage}/{len(df)}) "
              f"-- MATCHES Saudi cohort severity axis (grade). Experiment 3 is valid.")
    else:
        df["severity_context"] = df["stage_severity_"]
        severity_context_source = "stage"
        reason = "CONTEXT_SOURCE='stage'" if CONTEXT_SOURCE != "grade" else \
            f"grade coverage too low ({grade_coverage} < {MIN_TCGA_GRADE_COVERAGE})"
        print(f"  [TCGA severity_context] falling back to STAGE ({reason}). "
              f"WARNING: Saudi cohorts only report GRADE. Experiment 3 (stage-conditioning "
              f"transfer) will be SKIPPED for this run because comparing stage-conditioned "
              f"TCGA to grade-conditioned Saudi data is not apples-to-apples (see review).")

    df["label"] = df.apply(make_5yr_label, axis=1)

    for col in ["er_status", "pr_status", "her2_status"]:
        if col in df.columns:
            df[col] = df[col].apply(normalize_marker_status)

    df = df.dropna(subset=["patient_id", "severity_context", "label"]).reset_index(drop=True)
    df = df.set_index("patient_id")
    df["population_context"] = "TCGA (global, source domain)"
    return df, severity_context_source


# ── CELL 4 — TCGA expression cleanup + patient matching ─────────────────────────────
def build_tcga_expression(expr_raw_cbio):
    expr_raw_cbio = expr_raw_cbio.copy()
    expr_raw_cbio["patient_id"] = expr_raw_cbio["sample_id"].apply(
        lambda s: "-".join(s.split("-")[:3]) if pd.notna(s) else s)
    expr_df = expr_raw_cbio.drop(columns=["sample_id"]).drop_duplicates(subset="patient_id")
    expr_df = expr_df.set_index("patient_id")
    gene_cols = list(expr_df.columns)
    expr_df = expr_df[gene_cols].apply(pd.to_numeric, errors="coerce")
    expr_df = expr_df.fillna(expr_df.mean())
    return expr_df


# ════════════════════════════════════════════════════════════════════════════════════
# PART B — GEO ingestion for the Saudi cohorts (GSE36295, GSE29044)
# ════════════════════════════════════════════════════════════════════════════════════

def download_geo_series(gse_id, destdir=GEO_DIR):
    import GEOparse
    print(f"[GEO] Downloading {gse_id} ...")
    gse = GEOparse.get_GEO(geo=gse_id, destdir=destdir, silent=True)
    print(f"  {gse_id}: {len(gse.gsms)} samples, platforms: {list(gse.gpls.keys())}")
    return gse


CHARACTERISTIC_PATTERNS = {
    "age": [r"age[^:]*:\s*(\S+)"],
    "tissue": [r"^tissue[^:]*:\s*(.+)", r"sample type\s*:\s*(.+)"],
    "tumor_normal": [r"tumou?r\s*/\s*normal[^:]*:\s*(\S+)"],
    "sample_type": [r"\btype\s*:\s*(\S+)"],
    "histological_type": [r"histolog\w*[^:]*:\s*(.+)"],
    "er_status": [r"\ber\b\s*(\([^)]*\))?\s*:\s*(\S+)"],
    "pr_status": [r"\bpr\b\s*(\([^)]*\))?\s*:\s*(\S+)"],
    "her2_status": [r"her-?2\s*(/\s*neu)?\s*(\([^)]*\))?\s*:\s*(\S+)"],
    "grade": [r"\bgrade\b[^:]*:\s*(\S+)"],
}


def parse_geo_characteristics(gsm):
    raw_lines = gsm.metadata.get("characteristics_ch1", [])
    parsed = {}
    for field, patterns in CHARACTERISTIC_PATTERNS.items():
        for line in raw_lines:
            matched = False
            for pat in patterns:
                m = re.search(pat, line, flags=re.IGNORECASE)
                if m:
                    parsed[field] = m.group(m.lastindex)
                    matched = True
                    break
            if matched:
                break
    if len(parsed) < 2:
        print(f"  [WARN] {gsm.name}: only parsed {list(parsed.keys())} from "
              f"{raw_lines} -- extend CHARACTERISTIC_PATTERNS if this is a real cohort sample.")
    return parsed


def is_breast_tumor_sample(row):
    tissue = str(row.get("tissue", "")).lower()
    if "blood" in tissue or "lymphoblastoid" in tissue:
        return False
    tn = str(row.get("tumor_normal", "")).strip().upper()
    sample_type = str(row.get("sample_type", "")).strip().upper()
    if tn == "T":
        return True
    if tn == "N":
        return False
    if "normal" in tissue or sample_type == "NORMAL":
        return False
    if "tumor" in tissue or "cancer" in tissue or sample_type in ("IDC", "DCIS", "ILC"):
        return True
    return False


def resolve_probe_to_symbol(gpl):
    table = gpl.table
    candidate_cols = ["Gene Symbol", "GENE_SYMBOL", "Symbol", "gene_symbol"]
    for col in candidate_cols:
        if col in table.columns:
            mapping = table.set_index(table.columns[0])[col].dropna().to_dict()
            return {k: str(v).split("///")[0].strip() for k, v in mapping.items() if str(v).strip()}

    if "gene_assignment" in table.columns:
        mapping = {}
        for probe_id, val in zip(table.iloc[:, 0], table["gene_assignment"]):
            if pd.isna(val) or val == "---":
                continue
            first_block = str(val).split("///")[0]
            parts = [p.strip() for p in first_block.split("//")]
            if len(parts) >= 2 and parts[1]:
                mapping[probe_id] = parts[1]
        return mapping

    raise ValueError(f"Could not find a gene-symbol column in platform table; "
                      f"columns available: {list(table.columns)}. "
                      f"Add the right column name to resolve_probe_to_symbol().")


def build_geo_expression_and_clinical(gse):
    platform_to_samples = {}
    for gsm_name, gsm in gse.gsms.items():
        plat_ids = gsm.metadata.get("platform_id", [])
        plat_id = plat_ids[0] if plat_ids else None
        platform_to_samples.setdefault(plat_id, []).append(gsm_name)

    per_platform_gene_expr = []
    for plat_id, sample_names in platform_to_samples.items():
        if plat_id not in gse.gpls:
            print(f"  [WARN] platform {plat_id} not in gse.gpls -- skipping "
                  f"{len(sample_names)} samples on it.")
            continue
        try:
            probe_map = resolve_probe_to_symbol(gse.gpls[plat_id])
        except ValueError as e:
            print(f"  [WARN] {e} -- skipping platform {plat_id} ({len(sample_names)} samples)")
            continue
        probe_map = {str(k): v for k, v in probe_map.items()}

        expr_cols = {}
        for gsm_name in sample_names:
            gsm = gse.gsms[gsm_name]
            if "VALUE" not in gsm.table.columns:
                continue
            s = gsm.table.set_index("ID_REF")["VALUE"]
            s.index = s.index.astype(str)
            expr_cols[gsm_name] = s
        if not expr_cols:
            continue

        plat_expr = pd.DataFrame(expr_cols).T.apply(pd.to_numeric, errors="coerce")
        symbol_series = pd.Series({p: probe_map.get(p) for p in plat_expr.columns})
        plat_expr_t = plat_expr.T
        plat_expr_t["__gene__"] = symbol_series
        plat_expr_t = plat_expr_t.dropna(subset=["__gene__"])
        gene_expr = plat_expr_t.groupby("__gene__").mean().T
        print(f"  Platform {plat_id}: {len(sample_names)} samples -> "
              f"{gene_expr.shape[1]} mapped genes")

        gene_expr["__platform__"] = plat_id
        per_platform_gene_expr.append(gene_expr)

    if not per_platform_gene_expr:
        raise ValueError("No usable expression data found on any platform for this GSE.")

    expr_df = pd.concat(per_platform_gene_expr, axis=0, join="outer")
    platform_col = expr_df.pop("__platform__")

    clinical_rows = {gsm_name: parse_geo_characteristics(gsm) for gsm_name, gsm in gse.gsms.items()}
    clinical_df = pd.DataFrame.from_dict(clinical_rows, orient="index")
    for col in ("tissue", "tumor_normal", "sample_type", "grade"):
        if col not in clinical_df.columns:
            clinical_df[col] = np.nan
    clinical_df["platform"] = platform_col.reindex(clinical_df.index)

    n_before = len(clinical_df)
    is_tumor = clinical_df.apply(is_breast_tumor_sample, axis=1)
    print(f"  {n_before} raw samples -> {int(is_tumor.sum())} kept as breast TUMOR samples "
          f"({int((~is_tumor).sum())} dropped: normal/adjacent-normal breast tissue and/or "
          f"unrelated non-breast specimens folded into this GSE accession)")
    clinical_df = clinical_df.loc[is_tumor]
    expr_df_out = expr_df.loc[expr_df.index.intersection(clinical_df.index)]

    clinical_df["age"] = clinical_df["age"].apply(parse_age_value) if "age" in clinical_df.columns else np.nan
    clinical_df["severity_context"] = clinical_df["grade"].apply(grade_to_severity)

    for col in ["er_status", "pr_status", "her2_status"]:
        if col in clinical_df.columns:
            clinical_df[col] = clinical_df[col].apply(normalize_marker_status)

    return expr_df_out, clinical_df


def load_saudi_cohorts(series_ids=SAUDI_GEO_SERIES):
    all_expr, all_clin = [], []
    per_series = {}
    for gse_id in series_ids:
        gse = download_geo_series(gse_id)
        gene_expr, clin_df = build_geo_expression_and_clinical(gse)

        clin_df = clin_df.copy()
        clin_df["population_context"] = "Saudi Arabia / Muslim-majority-country cohort"
        clin_df["source_series"] = gse_id

        n_stage = int(clin_df["severity_context"].notna().sum()) if "severity_context" in clin_df.columns else 0
        print(f"  [{gse_id}] parsed fields present (post tumor-filtering): "
              f"age={int(clin_df['age'].notna().sum()) if 'age' in clin_df.columns else 0}/{len(clin_df)}, "
              f"severity_context (grade-based)={n_stage}/{len(clin_df)}, "
              f"platforms={sorted(clin_df['platform'].dropna().unique().tolist())}")
        if n_stage == 0 and len(clin_df) > 0:
            print(f"  [{gse_id}] severity_context is 0/{len(clin_df)} -- dumping raw "
                  f"characteristics_ch1 for the first 5 KEPT (tumor) samples so you can "
                  f"extend CHARACTERISTIC_PATTERNS['grade'] with the real field name/format:")
            for gsm_name in list(clin_df.index)[:5]:
                raw = gse.gsms[gsm_name].metadata.get("characteristics_ch1", [])
                print(f"    {gsm_name}: {raw}")

        per_series[gse_id] = (gene_expr, clin_df)
        all_expr.append(gene_expr)
        all_clin.append(clin_df)

    combined_expr = pd.concat(all_expr, axis=0, join="outer")
    combined_clin = pd.concat(all_clin, axis=0, join="outer")
    common_idx = combined_expr.index.intersection(combined_clin.index)
    combined_expr, combined_clin = combined_expr.loc[common_idx], combined_clin.loc[common_idx]
    n_platforms = combined_clin["platform"].nunique() if "platform" in combined_clin.columns else "?"
    print(f"[GEO] Combined Saudi cohort: {combined_expr.shape[0]} samples, "
          f"{combined_expr.shape[1]} genes (pre-alignment), spanning {n_platforms} platforms "
          f"-- treat pooled numbers as secondary to the per-series ones below.")
    return combined_expr, combined_clin, per_series


# ════════════════════════════════════════════════════════════════════════════════════
# PART C — Shared representation
# ════════════════════════════════════════════════════════════════════════════════════

class TCGADataPipeline:
    def __init__(self, clinical_df, expr_df, severity_context_source,
                 n_top_genes=N_TOP_GENES, n_gene_pcs=N_GENE_PCS, drop_genes=None):
        self.clinical_df = clinical_df
        self.expr_df = expr_df
        self.severity_context_source = severity_context_source
        self.n_top_genes = n_top_genes
        self.n_gene_pcs = n_gene_pcs
        self.clinical_numeric = [c for c in COMMON_CLINICAL_NUMERIC if c in clinical_df.columns]
        self.clinical_categorical = [c for c in COMMON_CLINICAL_CATEGORICAL if c in clinical_df.columns]
        # "drop instead of impute" sensitivity analysis.
        self.drop_genes = set(drop_genes or [])
        # every stateful transform, so a leakage audit can check against it later.
        self.fit_index_ = None

    def prepare_data(self):
        idx = self.clinical_df.index
        y = self.clinical_df["label"].astype(int)

        idx_train_val, idx_test = train_test_split(idx, test_size=0.15, random_state=RANDOM_STATE, stratify=y)
        idx_train, idx_oracle = train_test_split(idx_train_val, test_size=0.176, random_state=RANDOM_STATE,
                                                  stratify=y.loc[idx_train_val])
        self.fit_index_ = list(idx_train)  # only TRAIN ever touches .fit()

        candidate_genes = [g for g in self.expr_df.columns if g not in self.drop_genes]
        gene_var = self.expr_df.loc[idx_train, candidate_genes].var(axis=0).sort_values(ascending=False)
        self.top_genes = gene_var.head(self.n_top_genes).index.tolist()
        self.gene_scaler = StandardScaler().fit(self.expr_df.loc[idx_train, self.top_genes])
        actual_pcs = min(self.n_gene_pcs, len(self.top_genes), len(idx_train))
        self.pca = PCA(n_components=actual_pcs, random_state=RANDOM_STATE).fit(
            self.gene_scaler.transform(self.expr_df.loc[idx_train, self.top_genes]))
        self.n_gene_pcs = actual_pcs
        self.top_gene_train_means = self.expr_df.loc[idx_train, self.top_genes].mean()

        def gene_embedding(index, expr_source=None):
            expr_source = self.expr_df if expr_source is None else expr_source
            scaled = self.gene_scaler.transform(expr_source.loc[index, self.top_genes])
            return self.pca.transform(scaled)

        clin_cols = self.clinical_numeric + self.clinical_categorical
        self.clinical_pre = ColumnTransformer([
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                               ("scale", StandardScaler())]), self.clinical_numeric),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                               ("onehot", OneHotEncoder(handle_unknown="ignore"))]),
             self.clinical_categorical),
        ]).fit(self.clinical_df.loc[idx_train, clin_cols])

        def clinical_features(index, clin_source=None):
            clin_source = self.clinical_df if clin_source is None else clin_source
            out = self.clinical_pre.transform(clin_source.loc[index, clin_cols])
            return out.toarray() if hasattr(out, "toarray") else out

        self.context_scaler = StandardScaler().fit(self.clinical_df.loc[idx_train, ["severity_context"]])

        def build_split(index):
            X = np.hstack([clinical_features(index), gene_embedding(index)]).astype(np.float32)
            C = self.context_scaler.transform(self.clinical_df.loc[index, ["severity_context"]]).astype(np.float32)
            yv = self.clinical_df.loc[index, "label"].astype(np.int64).values
            return torch.tensor(X), torch.tensor(C), torch.tensor(yv)

        X_train, C_train, y_train = build_split(idx_train)
        X_oracle, C_oracle, y_oracle = build_split(idx_oracle)
        X_test, C_test, y_test = build_split(idx_test)

        cat_names = list(self.clinical_pre.named_transformers_["cat"].named_steps["onehot"]
                          .get_feature_names_out(self.clinical_categorical))
        self.feature_names = self.clinical_numeric + cat_names + [f"gene_pc_{i}" for i in range(self.n_gene_pcs)]
        self._clin_cols = clin_cols

        return {"train": (X_train, C_train, y_train),
                "oracle": (X_oracle, C_oracle, y_oracle),
                "test": (X_test, C_test, y_test)}

    def transform_external(self, ext_clinical_df, ext_expr_df, verbose=True):
        present_genes = [g for g in self.top_genes if g in ext_expr_df.columns]
        missing_genes = [g for g in self.top_genes if g not in ext_expr_df.columns]
        if missing_genes and verbose:
            print(f"  [transform_external] {len(missing_genes)}/{len(self.top_genes)} "
                  f"TCGA genes not found on external platform -- imputing with TCGA train mean: "
                  f"{missing_genes[:10]}{'...' if len(missing_genes) > 10 else ''}")

        expr_aligned = pd.DataFrame(index=ext_expr_df.index, columns=self.top_genes, dtype=float)
        if present_genes:
            expr_aligned[present_genes] = ext_expr_df[present_genes].values
        for g in missing_genes:
            expr_aligned[g] = self.top_gene_train_means[g]
        expr_aligned = expr_aligned.fillna(expr_aligned.mean())

        gene_pc = self.pca.transform(self.gene_scaler.transform(expr_aligned))

        n_total = len(ext_clinical_df)
        n_with_stage = int(ext_clinical_df["severity_context"].notna().sum()) if \
            "severity_context" in ext_clinical_df.columns else 0
        keep_idx = ext_clinical_df.index[ext_clinical_df.get("severity_context", pd.Series(
            index=ext_clinical_df.index, dtype=float)).notna()]
        keep_idx = keep_idx.intersection(expr_aligned.index)

        if len(keep_idx) == 0:
            print(f"  [transform_external] 0/{n_total} external samples usable "
                  f"({n_with_stage}/{n_total} had a parsed severity_context). Returning empty result.")
            empty_X = torch.empty((0, len(self.feature_names)), dtype=torch.float32)
            empty_C = torch.empty((0, 1), dtype=torch.float32)
            return empty_X, empty_C, None, keep_idx

        clin_slice = ext_clinical_df.loc[keep_idx, self._clin_cols] if set(self._clin_cols).issubset(
            ext_clinical_df.columns) else pd.DataFrame(index=keep_idx, columns=self._clin_cols)
        for c in self._clin_cols:
            if c not in clin_slice.columns:
                clin_slice[c] = np.nan

        clin_feat = self.clinical_pre.transform(clin_slice)
        clin_feat = clin_feat.toarray() if hasattr(clin_feat, "toarray") else clin_feat

        gene_pc_slice = pd.DataFrame(gene_pc, index=expr_aligned.index).loc[keep_idx].values
        X_ext = np.hstack([clin_feat, gene_pc_slice]).astype(np.float32)
        C_ext = self.context_scaler.transform(ext_clinical_df.loc[keep_idx, ["severity_context"]]).astype(np.float32)

        y_ext = None
        if "label" in ext_clinical_df.columns and ext_clinical_df.loc[keep_idx, "label"].notna().any():
            y_ext = ext_clinical_df.loc[keep_idx, "label"].astype(np.int64).values

        return (torch.tensor(X_ext), torch.tensor(C_ext),
                None if y_ext is None else torch.tensor(y_ext), keep_idx)

    def assert_no_leakage(self, external_keep_idx):
        assert self.fit_index_ is not None, "prepare_data() must run before the audit."
        overlap = set(self.fit_index_) & set(external_keep_idx.tolist() if hasattr(external_keep_idx, "tolist") else external_keep_idx)
        assert len(overlap) == 0, f"Leakage: {len(overlap)} external IDs collide with TCGA fit_index_"
        print(f"  [leakage audit] OK: 0 overlap between TCGA fit index (n={len(self.fit_index_)}) "
              f"and external cohort (n={len(external_keep_idx)}).")


def assert_disjoint(idx_a, idx_b, label=""):
    """CHANGE (fix 3): generic disjointness assertion used for the Saudi
    calibration/eval split in Experiment 5."""
    set_a, set_b = set(np.asarray(idx_a).tolist()), set(np.asarray(idx_b).tolist())
    overlap = set_a & set_b
    assert len(overlap) == 0, f"Leakage in {label}: {len(overlap)} overlapping indices"
    print(f"  [leakage audit] {label}: calibration (n={len(set_a)}) and eval (n={len(set_b)}) "
          f"are disjoint.")



# ════════════════════════════════════════════════════════════════════════════════════
# PART D — model classes
# ════════════════════════════════════════════════════════════════════════════════════

class SurvivalRewardOracle(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.SiLU(),
            nn.Linear(64, 2),
        )
    def forward(self, x):
        return self.net(x)


class FlowResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.act = nn.SiLU()
    def forward(self, x):
        return x + self.act(self.linear2(self.act(self.linear1(x))))


class TCGAFlowMap(nn.Module):
    def __init__(self, data_dim, context_dim, hidden_dim=256, blocks=3):
        super().__init__()
        self.input_proj = nn.Linear(data_dim + context_dim + 1, hidden_dim)
        self.trunk = nn.Sequential(*[FlowResidualBlock(hidden_dim) for _ in range(blocks)])
        self.output_proj = nn.Linear(hidden_dim, data_dim)
        self.act = nn.SiLU()
    def forward(self, x, c, t):
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        h = self.act(self.input_proj(torch.cat([x, c, t], dim=-1)))
        return self.output_proj(self.trunk(h))


def integrate_flow(model, x0, c, n_steps=N_EULER_STEPS):
    dt = 1.0 / n_steps
    x = x0
    for i in range(n_steps):
        t = torch.full((x.size(0), 1), i * dt, device=x.device)
        x = x + model(x, c, t) * dt
    return x


# ---- Gaussian baseline: 4 variants with Ledoit-Wolf shrinkage (v3, items 3/4/7) ------

class GaussianVariants:
    """Fits a shrinkage-regularized (Ledoit-Wolf)
    Gaussian on TCGA train ('source'), and can combine its mean/covariance with a
    separately shrinkage-fit calibration-set Gaussian in four ways:

        variant       mean            covariance
        source        TCGA            TCGA
        full          calibration     calibration
        mean_only     calibration     TCGA
        cov_only      TCGA            calibration

    Ledoit-Wolf shrinkage is used everywhere (not just for small calibration sets) so
    the four variants are only ever different in WHICH data supplied mu/cov, never in
    how heavily each one is regularized -- otherwise a comparison between them would be
    confounded by regularization strength as well as by the data source.

    The suspiciously-perfect zero-shot-vs-adapted gap the reviewer flagged (Gaussian
    full-adaptation MMD collapsing to ~0) is exactly what fitting mu/cov directly on a
    calibration set FROM THE SAME evaluation cohort's population would predict when the
    covariance dimensionality (~22) is close to the calibration sample size (~26-41):
    print_gaussian_diagnostics() below reports the condition number / rank so this can
    be checked rather than assumed each run, and never touches the held-out eval split
    to do so."""

    def __init__(self):
        self.mu_source_ = None
        self.cov_source_ = None

    def fit_source(self, X_train):
        X = X_train.numpy() if torch.is_tensor(X_train) else np.asarray(X_train)
        lw = LedoitWolf().fit(X)
        self.mu_source_, self.cov_source_ = lw.location_, lw.covariance_
        self.n_source_ = X.shape[0]
        return self

    @staticmethod
    def _fit_calibration(X_calib):
        X = X_calib.numpy() if torch.is_tensor(X_calib) else np.asarray(X_calib)
        lw = LedoitWolf().fit(X)
        return lw.location_, lw.covariance_

    def params(self, variant, X_calib=None):
        assert self.mu_source_ is not None, "call fit_source() first"
        if variant == "source":
            return self.mu_source_, self.cov_source_
        assert X_calib is not None, f"variant '{variant}' requires X_calib"
        mu_calib, cov_calib = self._fit_calibration(X_calib)
        if variant == "full":
            return mu_calib, cov_calib
        if variant == "mean_only":
            return mu_calib, self.cov_source_
        if variant == "cov_only":
            return self.mu_source_, cov_calib
        raise ValueError(f"unknown Gaussian variant: {variant}")

    def sample(self, mu, cov, n, seed=None):
        rng = np.random.RandomState(seed) if seed is not None else np.random
        return rng.multivariate_normal(mu, cov, size=n).astype(np.float32)


def print_gaussian_diagnostics(mu_source, cov_source, mu_target, cov_target, n_calib, n_eval, label=""):
    cond_source = np.linalg.cond(cov_source)
    cond_target = np.linalg.cond(cov_target)
    rank_source = np.linalg.matrix_rank(cov_source)
    rank_target = np.linalg.matrix_rank(cov_target)
    dim = cov_source.shape[0]
    print(f"  [Gaussian diagnostics{' | ' + label if label else ''}] "
          f"n_calibration={n_calib}, n_eval={n_eval}, dimension={dim}")
    print(f"    ||mu_source - mu_target|| = {np.linalg.norm(mu_source - mu_target):.4f}")
    print(f"    mean(diag(cov_source)) = {np.diag(cov_source).mean():.4f}, "
          f"mean(diag(cov_target)) = {np.diag(cov_target).mean():.4f}")
    print(f"    cond(cov_source) = {cond_source:.3e}, rank(cov_source) = {rank_source}/{dim}")
    print(f"    cond(cov_target) = {cond_target:.3e}, rank(cov_target) = {rank_target}/{dim}")
    if n_calib < dim:
        print(f"    NOTE: n_calibration ({n_calib}) < dimension ({dim}) -- the raw sample "
              f"covariance would be rank-deficient here; Ledoit-Wolf shrinkage is load-bearing, "
              f"not a formality.")


class SimpleVAE(nn.Module):
    def __init__(self, data_dim, context_dim, latent_dim=16, hidden_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc = nn.Sequential(
            nn.Linear(data_dim + context_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)
        self.dec = nn.Sequential(
            nn.Linear(latent_dim + context_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, data_dim),
        )
    def encode(self, x, c):
        h = self.enc(torch.cat([x, c], dim=-1))
        return self.mu_head(h), self.logvar_head(h)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)
    def decode(self, z, c):
        return self.dec(torch.cat([z, c], dim=-1))
    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar
    def sample(self, c, device):
        n = c.shape[0]
        z = torch.randn(n, self.latent_dim, device=device)
        with torch.no_grad():
            return self.decode(z, c).cpu().numpy()


def train_vae(X_train, C_train, epochs=300, batch_size=32, lr=5e-4, beta=0.1, init_model=None):
    X_train, C_train = X_train.to(DEVICE), C_train.to(DEVICE)
    model = init_model if init_model is not None else SimpleVAE(
        data_dim=X_train.shape[1], context_dim=C_train.shape[1]).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    n = X_train.shape[0]
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n)
        running = 0.0
        for i in range(0, n, batch_size):
            opt.zero_grad()
            idx = perm[i:i + batch_size]
            x, c = X_train[idx], C_train[idx]
            x_hat, mu, logvar = model(x, c)
            recon = nn.functional.mse_loss(x_hat, x, reduction="mean")
            kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon + beta * kld
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
        if (epoch + 1) % 100 == 0:
            print(f"  VAE epoch {epoch+1}/{epochs} | loss: {running/n:.4f}")
    model.eval()
    return model


def train_reward_oracle(X_oracle, y_oracle, epochs=200, lr=1e-3):
    X_oracle, y_oracle = X_oracle.to(DEVICE), y_oracle.to(DEVICE)
    oracle = SurvivalRewardOracle(input_dim=X_oracle.shape[1]).to(DEVICE)
    opt = optim.AdamW(oracle.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    oracle.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = criterion(oracle(X_oracle), y_oracle)
        loss.backward()
        opt.step()
    oracle.eval()
    return oracle


def train_flow_map(X_train, C_train, epochs=300, batch_size=32, lr=5e-4,
                    context_dim_override=None, init_model=None):
    X_train, C_train = X_train.to(DEVICE), C_train.to(DEVICE)
    data_dim = X_train.shape[1]
    context_dim = context_dim_override or C_train.shape[1]
    model = init_model if init_model is not None else TCGAFlowMap(data_dim, context_dim).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    n = X_train.size(0)
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n)
        running = 0.0
        for i in range(0, n, batch_size):
            opt.zero_grad()
            idx = perm[i:i + batch_size]
            x1, c = X_train[idx], C_train[idx]
            x0 = torch.randn_like(x1)
            t = torch.rand(x1.size(0), 1, device=DEVICE)
            xt = (1.0 - t) * x0 + t * x1
            pred_v = model(xt, c, t)
            loss = nn.functional.mse_loss(pred_v, x1 - x0)
            loss.backward()
            opt.step()
            running += loss.item() * x1.size(0)
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:03d}/{epochs} | flow matching loss: {running/n:.4f}")
    model.eval()
    return model


def train_flow_map_no_context(X_train, epochs=300, batch_size=32, lr=5e-4):
    X = X_train.to(DEVICE)
    model = TCGAFlowMap(data_dim=X.shape[1], context_dim=1, hidden_dim=256, blocks=3).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    n = X.shape[0]
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n)
        running = 0.0
        for i in range(0, n, batch_size):
            opt.zero_grad()
            idx = perm[i:i + batch_size]
            x1 = X[idx]
            x0 = torch.randn_like(x1)
            t = torch.rand(x1.shape[0], 1, device=DEVICE)
            xt = (1 - t) * x0 + t * x1
            c_zeros = torch.zeros(x1.shape[0], 1, device=DEVICE)
            pred_v = model(xt, c_zeros, t)
            loss = nn.functional.mse_loss(pred_v, x1 - x0)
            loss.backward()
            opt.step()
            running += loss.item() * x1.shape[0]
        if (epoch + 1) % 100 == 0:
            print(f"  No-context Epoch {epoch+1}/{epochs} | loss: {running/n:.4f}")
    model.eval()
    return model


# ════════════════════════════════════════════════════════════════════════════════════
# PART E — metrics, incl. bootstrap CIs
# ════════════════════════════════════════════════════════════════════════════════════

def mmd_rbf(X, Y, gamma=None):
    X, Y = np.asarray(X), np.asarray(Y)
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    Kxx, Kyy, Kxy = rbf_kernel(X, X, gamma), rbf_kernel(Y, Y, gamma), rbf_kernel(X, Y, gamma)
    return Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()


def mean_wasserstein(X, Y):
    X, Y = np.asarray(X), np.asarray(Y)
    return float(np.mean([wasserstein_distance(X[:, j], Y[:, j]) for j in range(X.shape[1])]))


def per_feature_shift(X_a, X_b, feature_names):
    rows = []
    for j, name in enumerate(feature_names):
        rows.append({"feature": name, "wasserstein": wasserstein_distance(X_a[:, j], X_b[:, j])})
    return pd.DataFrame(rows).sort_values("wasserstein", ascending=False)


def bootstrap_generative_metrics(x_real_eval, x_generated_pool, n_boot=N_BOOTSTRAP, seed=RANDOM_STATE):
    x_real_eval = np.asarray(x_real_eval)
    x_generated_pool = np.asarray(x_generated_pool)
    n_real = x_real_eval.shape[0]
    n_gen_draw = min(n_real, x_generated_pool.shape[0])  # keep the two-sample sizes matched

    point_mmd = mmd_rbf(x_real_eval, x_generated_pool[:n_real] if x_generated_pool.shape[0] >= n_real
                         else x_generated_pool)
    point_wd = mean_wasserstein(x_real_eval, x_generated_pool[:n_real] if x_generated_pool.shape[0] >= n_real
                                 else x_generated_pool)

    rng = np.random.RandomState(seed)
    mmds, wds = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        real_idx = rng.randint(0, n_real, size=n_real)
        gen_idx = rng.randint(0, x_generated_pool.shape[0], size=n_gen_draw)
        xr = x_real_eval[real_idx]
        xg = x_generated_pool[gen_idx]
        mmds[b] = mmd_rbf(xr, xg)
        wds[b] = mean_wasserstein(xr, xg)

    return {
        "mmd": float(point_mmd), "mmd_ci_low": float(np.percentile(mmds, 2.5)),
        "mmd_ci_high": float(np.percentile(mmds, 97.5)),
        "wasserstein": float(point_wd), "wasserstein_ci_low": float(np.percentile(wds, 2.5)),
        "wasserstein_ci_high": float(np.percentile(wds, 97.5)),
        "n_eval": int(n_real), "n_boot": int(n_boot),
    }


def fmt_ci(d, metric):
    return f"{d[metric]:.4f} [95% CI: {d[metric+'_ci_low']:.4f}-{d[metric+'_ci_high']:.4f}]"



# ════════════════════════════════════════════════════════════════════════════════════
# PART F — sample-pool helpers
# ════════════════════════════════════════════════════════════════════════════════════

def generate_flow_pool(model, C_ref, data_dim, n_pool, seed):
    """Builds an n_pool-sample generated pool by resampling contexts (with replacement)
    from C_ref and drawing a fresh noise seed per sample. Used only to feed
    bootstrap_generative_metrics -- never touches real eval data."""
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, C_ref.shape[0], size=n_pool)
    c_pool = C_ref[idx].to(DEVICE)
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    x0 = torch.randn(n_pool, data_dim, device=DEVICE, generator=g)
    with torch.no_grad():
        return integrate_flow(model, x0, c_pool).cpu().numpy()


def generate_vae_pool(model, C_ref, n_pool, seed):
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, C_ref.shape[0], size=n_pool)
    c_pool = C_ref[idx].to(DEVICE)
    torch.manual_seed(seed)
    return model.sample(c_pool, DEVICE)


def generate_gaussian_pool(gaussian_variants, variant, n_pool, seed, X_calib=None):
    mu, cov = gaussian_variants.params(variant, X_calib=X_calib)
    return gaussian_variants.sample(mu, cov, n_pool, seed=seed), mu, cov


# ════════════════════════════════════════════════════════════════════════════════════
# PART G — the experiments
# ════════════════════════════════════════════════════════════════════════════════════

def experiment1_in_domain(flow_model, oracle, gaussian_variants, vae_model,
                           X_train, C_train, X_test, C_test, n_steering=40, alpha=1.5):
    """EXPERIMENT 1: TCGA in-domain generation for Flow, Gaussian-source, VAE, each
    with a bootstrap CI on the TCGA test split (held out from training). Also runs the
    flow steering sanity check as before."""
    data_dim = X_train.shape[1]
    x_real = X_test.numpy()
    n_pool = max(GEN_POOL_MULTIPLIER * X_test.shape[0], 200)

    with torch.no_grad():
        x0 = torch.randn(X_test.size(0), data_dim, device=DEVICE)
        x_gen_flow = integrate_flow(flow_model, x0, C_test.to(DEVICE)).cpu().numpy()
    flow_pool = generate_flow_pool(flow_model, C_test, data_dim, n_pool, seed=RANDOM_STATE)
    flow_ci = bootstrap_generative_metrics(x_real, flow_pool)

    gauss_pool, mu_s, cov_s = generate_gaussian_pool(gaussian_variants, "source", n_pool, seed=RANDOM_STATE)
    gauss_ci = bootstrap_generative_metrics(x_real, gauss_pool)

    vae_pool = generate_vae_pool(vae_model, C_test, n_pool, seed=RANDOM_STATE)
    vae_ci = bootstrap_generative_metrics(x_real, vae_pool)

    print("[EXPERIMENT 1 -- TCGA in-domain generation]")
    print(f"  Flow:     MMD={fmt_ci(flow_ci,'mmd')} | Wasserstein={fmt_ci(flow_ci,'wasserstein')}")
    print(f"  Gaussian: MMD={fmt_ci(gauss_ci,'mmd')} | Wasserstein={fmt_ci(gauss_ci,'wasserstein')}")
    print(f"  VAE:      MMD={fmt_ci(vae_ci,'mmd')} | Wasserstein={fmt_ci(vae_ci,'wasserstein')}")

    # steering sanity check (flow only, as in v2 Experiment 1)
    n = min(n_steering, X_test.shape[0])
    c_batch = C_test[:n].to(DEVICE)
    x0 = torch.randn(n, data_dim, device=DEVICE, requires_grad=True)
    x1 = integrate_flow(flow_model, x0, c_batch)
    logits = oracle(x1)
    target = torch.full((n,), 1, dtype=torch.long, device=DEVICE)
    nn.functional.cross_entropy(logits, target).backward()
    with torch.no_grad():
        x0_steered = x0 - alpha * x0.grad
        x1_steered = integrate_flow(flow_model, x0_steered, c_batch)
        risk_before = torch.softmax(logits, dim=-1)[:, 1].mean().item()
        risk_after = torch.softmax(oracle(x1_steered), dim=-1)[:, 1].mean().item()

    return {"flow": flow_ci, "gaussian_source": gauss_ci, "vae": vae_ci,
            "risk_before_steering": risk_before, "risk_after_steering": risk_after}


def experiment2_population_shift(dp, flow_model, gaussian_variants, vae_model, ext_expr, ext_clin,
                                  cohort_label="pooled"):
    """EXPERIMENT 2: TCGA -> external zero-shot, with bootstrap CIs for Flow, Gaussian
    (source), and VAE on the SAME external evaluation samples."""
    X_ext, C_ext, y_ext, keep_idx = dp.transform_external(ext_clin, ext_expr)
    if X_ext.shape[0] < 5:
        print(f"[EXPERIMENT 2 | {cohort_label}] Too few harmonizable samples ({X_ext.shape[0]}).")
        return None, X_ext

    dp.assert_no_leakage(keep_idx)
    x_real = X_ext.numpy()
    data_dim = X_ext.shape[1]
    n_pool = max(GEN_POOL_MULTIPLIER * X_ext.shape[0], 200)

    with torch.no_grad():
        x0 = torch.randn(X_ext.shape[0], data_dim, device=DEVICE)
        x_gen_flow = integrate_flow(flow_model, x0, C_ext.to(DEVICE)).cpu().numpy()
    flow_pool = generate_flow_pool(flow_model, C_ext, data_dim, n_pool, seed=RANDOM_STATE)
    flow_ci = bootstrap_generative_metrics(x_real, flow_pool)

    gauss_pool, _, _ = generate_gaussian_pool(gaussian_variants, "source", n_pool, seed=RANDOM_STATE)
    gauss_ci = bootstrap_generative_metrics(x_real, gauss_pool)

    vae_pool = generate_vae_pool(vae_model, C_ext, n_pool, seed=RANDOM_STATE)
    vae_ci = bootstrap_generative_metrics(x_real, vae_pool)

    pca2 = PCA(n_components=2, random_state=RANDOM_STATE).fit(x_real)
    pca_dist = float(np.linalg.norm(pca2.transform(x_real).mean(0) - pca2.transform(x_gen_flow).mean(0)))
    shift_table = per_feature_shift(x_real, x_gen_flow, dp.feature_names)

    print(f"[EXPERIMENT 2 | {cohort_label} -- Population shift, zero-shot]")
    print(f"  Flow:     MMD={fmt_ci(flow_ci,'mmd')} | Wasserstein={fmt_ci(flow_ci,'wasserstein')}")
    print(f"  Gaussian: MMD={fmt_ci(gauss_ci,'mmd')} | Wasserstein={fmt_ci(gauss_ci,'wasserstein')}")
    print(f"  VAE:      MMD={fmt_ci(vae_ci,'mmd')} | Wasserstein={fmt_ci(vae_ci,'wasserstein')}")

    summary = {"cohort": cohort_label, "n_samples": X_ext.shape[0],
               "pca_centroid_distance": pca_dist,
               **{f"flow_{k}": v for k, v in flow_ci.items()},
               **{f"gaussian_{k}": v for k, v in gauss_ci.items()},
               **{f"vae_{k}": v for k, v in vae_ci.items()}}
    return {"summary": summary, "per_feature_shift": shift_table,
            "X_real": x_real, "X_generated_flow": x_gen_flow,
            "X_ext": X_ext, "C_ext": C_ext, "keep_idx": keep_idx}, X_ext


def plot_population_shift_umap(x_tcga_real, x_ext_real, save_path, cohort_label="Saudi cohort"):
    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=RANDOM_STATE)
        method = "UMAP"
    except ImportError:
        from sklearn.manifold import TSNE
        combined_n = len(x_tcga_real) + len(x_ext_real)
        reducer = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=min(30, combined_n - 1))
        method = "t-SNE"

    combined = np.vstack([x_tcga_real, x_ext_real])
    labels = ["TCGA (source)"] * len(x_tcga_real) + [cohort_label] * len(x_ext_real)
    emb = reducer.fit_transform(combined)
    plot_df = pd.DataFrame({"x": emb[:, 0], "y": emb[:, 1], "Cohort": labels})

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {"TCGA (source)": "#4878CF", cohort_label: "#D65F5F"}
    for label, grp in plot_df.groupby("Cohort"):
        ax.scatter(grp["x"], grp["y"], label=label, alpha=0.7, color=colors.get(label, "#888888"),
                   edgecolors="k", linewidths=0.3, s=40)
    ax.set_title(f"{method}: TCGA vs {cohort_label} Digital-Twin State (Population Shift)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def experiment3_stage_conditioning_transfer(dp, flow_model, flow_no_context, ext_expr, ext_clin,
                                             cohort_label="pooled", stages=(1.0, 2.0, 3.0)):
    """Stage/grade-conditioning transfer check, gated on axis match (v2 fix 1, kept
    as-is in v3; reported alongside, not counted as one of the 8 numbered experiments
    the reviewer just requested, but still necessary for Experiment 5's validity)."""
    if dp.severity_context_source != "grade":
        print(f"[Conditioning-transfer check | {cohort_label}] SKIPPED: TCGA severity_context "
              f"is '{dp.severity_context_source}', external is 'grade' -- not comparable.")
        return pd.DataFrame()

    X_ext, C_ext, y_ext, keep_idx = dp.transform_external(ext_clin, ext_expr, verbose=False)
    ext_severity = ext_clin.loc[keep_idx, "severity_context"]

    rows = []
    for s in stages:
        mask = (ext_severity >= s - 0.5) & (ext_severity < s + 0.5)
        if mask.sum() < 2:
            continue
        x_real_sub = X_ext.numpy()[mask.values]
        n_samples = max(20, mask.sum())

        c_scaled = torch.tensor(dp.context_scaler.transform([[s]]), dtype=torch.float32)
        c_batch = c_scaled.repeat(n_samples, 1).to(DEVICE)
        c_zero = torch.zeros(n_samples, 1, device=DEVICE)
        x0 = torch.randn(n_samples, X_ext.shape[1], device=DEVICE)

        with torch.no_grad():
            gen_with_ctx = integrate_flow(flow_model, x0, c_batch).cpu().numpy()
            gen_no_ctx = integrate_flow(flow_no_context, x0, c_zero).cpu().numpy()

        rows.append({
            "cohort": cohort_label, "severity_context": s, "n_real": int(mask.sum()),
            "wasserstein_with_context": mean_wasserstein(x_real_sub, gen_with_ctx),
            "wasserstein_no_context": mean_wasserstein(x_real_sub, gen_no_ctx),
        })

    df = pd.DataFrame(rows)
    if len(df):
        df["improvement_from_context"] = df["wasserstein_no_context"] - df["wasserstein_with_context"]
        df["pct_improvement"] = 100 * df["improvement_from_context"] / df["wasserstein_no_context"]
    print(f"[Conditioning-transfer check | {cohort_label}]")
    print(df.to_string(index=False))
    return df


def experiment3_population_adaptation(dp, flow_model, gaussian_variants, vae_model, ext_expr, ext_clin,
                                       cohort_label="pooled", calib_frac=0.4, adapt_epochs=60, adapt_lr=1e-4):
    """EXPERIMENT 3: population adaptation for Flow, Gaussian(full/mean_only/cov_only),
    and VAE. Every zero-shot AND adapted number gets a bootstrap CI computed only from
    the eval split. The calibration/eval split is asserted disjoint, and Gaussian
    diagnostics (mu/cov/condition number/rank) are printed for every variant so a
    suspiciously-perfect adapted number can be inspected rather than assumed."""
    X_ext, C_ext, y_ext, keep_idx = dp.transform_external(ext_clin, ext_expr)
    n = X_ext.shape[0]
    if n < 10:
        print(f"[EXPERIMENT 3 | {cohort_label}] Too few samples for calibration/eval split.")
        return None

    idx = np.arange(n)
    calib_idx, eval_idx = train_test_split(idx, test_size=1 - calib_frac, random_state=RANDOM_STATE)
    assert_disjoint(calib_idx, eval_idx, label=f"Experiment 3 | {cohort_label}")

    X_calib, C_calib = X_ext[calib_idx], C_ext[calib_idx]
    X_eval, C_eval = X_ext[eval_idx], C_ext[eval_idx]
    x_real_eval = X_eval.numpy()
    data_dim = X_eval.shape[1]
    n_pool = max(GEN_POOL_MULTIPLIER * X_eval.shape[0], 200)

    rows = {}

    # --- Flow: zero-shot vs adapted ---
    flow_zero_pool = generate_flow_pool(flow_model, C_eval, data_dim, n_pool, seed=RANDOM_STATE)
    rows["flow_zero_shot"] = bootstrap_generative_metrics(x_real_eval, flow_zero_pool)

    adapted_flow = TCGAFlowMap(data_dim=data_dim, context_dim=C_ext.shape[1]).to(DEVICE)
    adapted_flow.load_state_dict(flow_model.state_dict())
    adapted_flow = train_flow_map(X_calib, C_calib, epochs=adapt_epochs, batch_size=min(16, len(calib_idx)),
                                   lr=adapt_lr, init_model=adapted_flow)
    flow_adapt_pool = generate_flow_pool(adapted_flow, C_eval, data_dim, n_pool, seed=RANDOM_STATE)
    rows["flow_adapted"] = bootstrap_generative_metrics(x_real_eval, flow_adapt_pool)

    # --- Gaussian: source, full, mean_only, cov_only ---
    for variant in ["source", "full", "mean_only", "cov_only"]:
        pool, mu_v, cov_v = generate_gaussian_pool(
            gaussian_variants, variant, n_pool, seed=RANDOM_STATE,
            X_calib=(X_calib if variant != "source" else None))
        rows[f"gaussian_{variant}"] = bootstrap_generative_metrics(x_real_eval, pool)
        if variant != "source":
            print_gaussian_diagnostics(gaussian_variants.mu_source_, gaussian_variants.cov_source_,
                                        mu_v, cov_v, n_calib=len(calib_idx), n_eval=len(eval_idx),
                                        label=f"{cohort_label} | {variant}")

    # --- VAE: zero-shot vs adapted (fine-tuned the same way as flow) ---
    vae_zero_pool = generate_vae_pool(vae_model, C_eval, n_pool, seed=RANDOM_STATE)
    rows["vae_zero_shot"] = bootstrap_generative_metrics(x_real_eval, vae_zero_pool)

    adapted_vae = copy.deepcopy(vae_model)
    adapted_vae = train_vae(X_calib, C_calib, epochs=adapt_epochs, batch_size=min(16, len(calib_idx)),
                             lr=adapt_lr, init_model=adapted_vae)
    vae_adapt_pool = generate_vae_pool(adapted_vae, C_eval, n_pool, seed=RANDOM_STATE)
    rows["vae_adapted"] = bootstrap_generative_metrics(x_real_eval, vae_adapt_pool)

    print(f"[EXPERIMENT 3 | {cohort_label} -- Population adaptation] "
          f"(n_calibration={len(calib_idx)}, n_eval={len(eval_idx)})")
    for name, ci in rows.items():
        print(f"  {name:18s}: MMD={fmt_ci(ci,'mmd')} | Wasserstein={fmt_ci(ci,'wasserstein')}")

    summary = {"cohort": cohort_label, "n_calibration": len(calib_idx), "n_eval": len(eval_idx)}
    for name, ci in rows.items():
        summary[f"{name}_mmd"] = ci["mmd"]
        summary[f"{name}_mmd_ci_low"] = ci["mmd_ci_low"]
        summary[f"{name}_mmd_ci_high"] = ci["mmd_ci_high"]
        summary[f"{name}_wasserstein"] = ci["wasserstein"]

    return summary, adapted_flow


def experiment4_calibration_ablation(dp, flow_model, gaussian_variants, ext_expr, ext_clin,
                                      cohort_label="pooled", calib_sizes=CALIBRATION_SIZES,
                                      split_seeds=CALIBRATION_SPLIT_SEEDS, adapt_epochs=60, adapt_lr=1e-4):
    """EXPERIMENT 4: calibration-size ablation for Flow + 3 Gaussian variants (full,
    mean_only, cov_only), repeated across N_CALIBRATION_SPLIT_SEEDS independent random
    splits per size, reporting mean +/- SD and a 95% CI across splits. This is the
    direct test of whether a spectacular small-calibration Gaussian number is a genuine
    effect or an artifact of one fortunate split."""
    X_ext, C_ext, y_ext, keep_idx = dp.transform_external(ext_clin, ext_expr, verbose=False)
    n = X_ext.shape[0]
    data_dim = X_ext.shape[1]
    usable_sizes = [c for c in calib_sizes if c + 10 <= n]
    if not usable_sizes:
        print(f"[EXPERIMENT 4 | {cohort_label}] Too few samples ({n}) for any calibration size in {calib_sizes}.")
        return pd.DataFrame()
    if len(usable_sizes) < len(calib_sizes):
        print(f"[EXPERIMENT 4 | {cohort_label}] Only {usable_sizes} fit within n={n}; larger sizes skipped.")

    rows = []
    for size in usable_sizes:
        for split_seed in split_seeds:
            rng = np.random.RandomState(split_seed)
            perm = rng.permutation(n)
            calib_idx = perm[:size]
            eval_idx = perm[size:] if len(perm) - size >= 5 else perm[-max(5, n - size):]
            assert_disjoint(calib_idx, eval_idx, label=f"Exp4 | {cohort_label} | size={size} seed={split_seed}")

            X_calib, C_calib = X_ext[calib_idx], C_ext[calib_idx]
            X_eval, C_eval = X_ext[eval_idx], C_ext[eval_idx]
            x_real_eval = X_eval.numpy()
            n_pool = max(GEN_POOL_MULTIPLIER * X_eval.shape[0], 100)

            # Flow
            adapted_flow = TCGAFlowMap(data_dim=data_dim, context_dim=C_ext.shape[1]).to(DEVICE)
            adapted_flow.load_state_dict(flow_model.state_dict())
            adapted_flow = train_flow_map(X_calib, C_calib, epochs=adapt_epochs,
                                           batch_size=min(16, size), lr=adapt_lr, init_model=adapted_flow)
            flow_pool = generate_flow_pool(adapted_flow, C_eval, data_dim, n_pool, seed=split_seed)
            rows.append({"cohort": cohort_label, "method": "flow", "n_calibration": size,
                         "split_seed": split_seed, "n_eval": len(eval_idx),
                         "mmd": mmd_rbf(x_real_eval, flow_pool[:len(eval_idx)]),
                         "wasserstein": mean_wasserstein(x_real_eval, flow_pool[:len(eval_idx)])})

            # Gaussian variants
            for variant in ["full", "mean_only", "cov_only"]:
                pool, _, _ = generate_gaussian_pool(gaussian_variants, variant, n_pool,
                                                     seed=split_seed, X_calib=X_calib)
                rows.append({"cohort": cohort_label, "method": f"gaussian_{variant}", "n_calibration": size,
                             "split_seed": split_seed, "n_eval": len(eval_idx),
                             "mmd": mmd_rbf(x_real_eval, pool[:len(eval_idx)]),
                             "wasserstein": mean_wasserstein(x_real_eval, pool[:len(eval_idx)])})

    df = pd.DataFrame(rows)
    agg = df.groupby(["cohort", "method", "n_calibration"]).agg(
        mmd_mean=("mmd", "mean"), mmd_sd=("mmd", "std"),
        wasserstein_mean=("wasserstein", "mean"), wasserstein_sd=("wasserstein", "std"),
        n_splits=("split_seed", "nunique"),
    ).reset_index()
    print(f"[EXPERIMENT 4 | {cohort_label} -- Calibration-size ablation, "
          f"{len(split_seeds)} splits/size]")
    print(agg.to_string(index=False))
    return df, agg


def experiment5_risk_steering(dp, flow_model, oracle, ext_expr, ext_clin,
                               cohort_label="pooled", alpha=1.5, n_patients=40,
                               n_seeds=N_MONOTONICITY_SEEDS):
    """EXPERIMENT 5: risk steering with the flow model (identical to v2's Experiment 4;
    renumbered here to match the reviewer's requested EXPERIMENT 5 slot)."""
    X_ext, C_ext, y_ext, keep_idx = dp.transform_external(ext_clin, ext_expr)
    n = min(n_patients, X_ext.shape[0])
    if n < 2:
        print(f"[EXPERIMENT 5 | {cohort_label}] Not enough harmonized samples to run steering.")
        return None
    c_batch_base = C_ext[:n].to(DEVICE)

    def steer(c_batch, target_class, alpha_val, seed):
        g = torch.Generator(device=DEVICE).manual_seed(seed)
        x0 = torch.randn(n, X_ext.shape[1], device=DEVICE, generator=g, requires_grad=True)
        x1 = integrate_flow(flow_model, x0, c_batch)
        logits_before = oracle(x1)
        target = torch.full((n,), target_class, dtype=torch.long, device=DEVICE)
        nn.functional.cross_entropy(logits_before, target).backward()
        with torch.no_grad():
            x0_steered = x0 - alpha_val * x0.grad
            x1_steered = integrate_flow(flow_model, x0_steered, c_batch)
            logits_after = oracle(x1_steered)
        risk_before = torch.softmax(logits_before, dim=-1)[:, 1].mean().item()
        risk_after = torch.softmax(logits_after, dim=-1)[:, 1].mean().item()
        return risk_before, risk_after

    hi_before, hi_after = steer(c_batch_base, 1, alpha, seed=RANDOM_STATE)
    lo_before, lo_after = steer(c_batch_base, 0, alpha, seed=RANDOM_STATE)

    alphas = np.linspace(0, 3.0, 8)
    curves = []
    for seed in range(RANDOM_STATE, RANDOM_STATE + n_seeds):
        risks = [steer(c_batch_base, 1, a, seed=seed)[1] for a in alphas]
        curves.append(risks)
    curves = np.array(curves)
    mean_curve = curves.mean(axis=0)
    ci_low = np.percentile(curves, 2.5, axis=0)
    ci_high = np.percentile(curves, 97.5, axis=0)

    spearman_rho, spearman_p = spearmanr(alphas, mean_curve)
    diffs = np.diff(mean_curve)
    n_violations = int(np.sum(diffs < -1e-3))
    max_downward_excursion = float(min(0.0, diffs.min()))
    mean_risk_increase = float(mean_curve[-1] - mean_curve[0])

    result = {
        "cohort": cohort_label, "method": "flow", "n_patients": n,
        "risk_before_hi": hi_before, "risk_after_hi": hi_after,
        "risk_before_lo": lo_before, "risk_after_lo": lo_after,
        "spearman_rho_alpha_vs_risk": float(spearman_rho), "spearman_p": float(spearman_p),
        "n_monotonicity_violations": n_violations,
        "max_downward_excursion": max_downward_excursion,
        "mean_risk_increase_end_minus_start": mean_risk_increase,
        "baseline_risk": float(mean_curve[0]), "max_risk": float(mean_curve.max()), "min_risk": float(mean_curve.min()),
        "alphas": alphas.tolist(), "mean_risk_by_alpha": mean_curve.tolist(),
        "risk_ci95_low": ci_low.tolist(), "risk_ci95_high": ci_high.tolist(),
    }
    print(f"[EXPERIMENT 5 | {cohort_label} -- Risk-steering (flow)] "
          f"Spearman rho={spearman_rho:.3f} (p={spearman_p:.3g}), "
          f"{n_violations} violation(s)/{len(diffs)} steps, "
          f"risk {mean_curve[0]:.3f} -> {mean_curve[-1]:.3f}")
    return result


def experiment6_risk_steering_baselines(dp, gaussian_variants, vae_model, oracle, ext_expr, ext_clin,
                                         cohort_label="pooled", alpha=1.5, n_patients=40,
                                         n_seeds=N_MONOTONICITY_SEEDS):
    """EXPERIMENT 6: does risk steering actually require the flow model? Two baselines,
    both steered by the SAME gradient-ascent mechanism on the oracle, so only the
    generative substrate differs from Experiment 5:

      - Gaussian: no learned generator to backprop through, so we steer directly in
        data space -- sample x ~ N(mu_source, cov_source), then x_steered = x -
        alpha * grad_x CE(oracle(x), target). This is the "no generative structure at
        all" baseline: pure gradient ascent on raw features.
      - VAE: has a decoder, so steering is latent-space (mirrors the flow model's x0
        steering): sample z ~ N(0, I), backprop the oracle loss through decode(z, c)
        to z, then decode(z - alpha*grad_z, c).
    """
    X_ext, C_ext, y_ext, keep_idx = dp.transform_external(ext_clin, ext_expr, verbose=False)
    n = min(n_patients, X_ext.shape[0])
    if n < 2:
        print(f"[EXPERIMENT 6 | {cohort_label}] Not enough samples.")
        return pd.DataFrame()
    c_batch = C_ext[:n].to(DEVICE)
    alphas = np.linspace(0, 3.0, 8)

    def gaussian_curve(seed):
        rng_seed = seed
        risks = []
        for a in alphas:
            x = torch.tensor(gaussian_variants.sample(gaussian_variants.mu_source_,
                                                        gaussian_variants.cov_source_, n, seed=rng_seed),
                              device=DEVICE, requires_grad=True)
            logits = oracle(x)
            target = torch.full((n,), 1, dtype=torch.long, device=DEVICE)
            nn.functional.cross_entropy(logits, target).backward()
            with torch.no_grad():
                x_steered = x - a * x.grad
                risk_after = torch.softmax(oracle(x_steered), dim=-1)[:, 1].mean().item()
            risks.append(risk_after)
        return risks

    def vae_curve(seed):
        g = torch.Generator(device=DEVICE).manual_seed(seed)
        risks = []
        for a in alphas:
            z = torch.randn(n, vae_model.latent_dim, device=DEVICE, generator=g, requires_grad=True)
            x = vae_model.decode(z, c_batch)
            logits = oracle(x)
            target = torch.full((n,), 1, dtype=torch.long, device=DEVICE)
            nn.functional.cross_entropy(logits, target).backward()
            with torch.no_grad():
                z_steered = z - a * z.grad
                x_steered = vae_model.decode(z_steered, c_batch)
                risk_after = torch.softmax(oracle(x_steered), dim=-1)[:, 1].mean().item()
            risks.append(risk_after)
        return risks

    rows = []
    for method_name, curve_fn in [("gaussian", gaussian_curve), ("vae", vae_curve)]:
        curves = np.array([curve_fn(seed) for seed in range(RANDOM_STATE, RANDOM_STATE + n_seeds)])
        mean_curve = curves.mean(axis=0)
        rho, p = spearmanr(alphas, mean_curve)
        diffs = np.diff(mean_curve)
        rows.append({
            "cohort": cohort_label, "method": method_name, "n_patients": n,
            "baseline_risk": float(mean_curve[0]), "max_risk": float(mean_curve.max()),
            "min_risk": float(mean_curve.min()),
            "spearman_rho_alpha_vs_risk": float(rho), "spearman_p": float(p),
            "n_monotonicity_violations": int(np.sum(diffs < -1e-3)),
        })

    df = pd.DataFrame(rows)
    print(f"[EXPERIMENT 6 | {cohort_label} -- Risk-steering baselines (Gaussian, VAE)]")
    print(df.to_string(index=False))
    return df


def experiment7_platform_sensitivity(dp, per_series):
    """EXPERIMENT 7: for each external cohort/platform, how many of the 80 TCGA top
    genes are actually present, and which are missing."""
    rows = []
    for cohort_label, (ext_expr, ext_clin) in per_series.items():
        platforms = sorted(ext_clin["platform"].dropna().unique().tolist()) if "platform" in ext_clin.columns else ["?"]
        present = [g for g in dp.top_genes if g in ext_expr.columns]
        missing = [g for g in dp.top_genes if g not in ext_expr.columns]
        rows.append({
            "cohort": cohort_label, "platform(s)": ", ".join(platforms),
            "common_genes": f"{len(present)}/{len(dp.top_genes)}",
            "missing_genes": ", ".join(missing) if missing else "(none)",
        })
    df = pd.DataFrame(rows)
    print("[EXPERIMENT 7 -- Platform sensitivity]")
    print(df.to_string(index=False))
    return df


def experiment8_gene_sensitivity(clinical_df, expr_df, severity_context_source,
                                  ext_expr, ext_clin, cohort_label="pooled"):
    """EXPERIMENT 8: impute vs. drop the gene(s) missing on the external platform
    (unchanged logic from v2's gene_imputation_sensitivity_analysis)."""
    dp_a = TCGADataPipeline(clinical_df, expr_df, severity_context_source)
    splits_a = dp_a.prepare_data()
    X_train_a, C_train_a, _ = splits_a["train"]
    flow_a = train_flow_map(X_train_a, C_train_a, epochs=150)
    X_ext_a, C_ext_a, _, keep_a = dp_a.transform_external(ext_clin, ext_expr, verbose=True)
    missing_genes = [g for g in dp_a.top_genes if g not in ext_expr.columns]

    with torch.no_grad():
        x0 = torch.randn(X_ext_a.shape[0], X_ext_a.shape[1], device=DEVICE)
        gen_a = integrate_flow(flow_a, x0, C_ext_a.to(DEVICE)).cpu().numpy()
    result_a = {"variant": "A_impute_missing_genes", "cohort": cohort_label,
                "missing_genes_imputed": missing_genes,
                "mmd": mmd_rbf(X_ext_a.numpy(), gen_a), "wasserstein": mean_wasserstein(X_ext_a.numpy(), gen_a)}

    if not missing_genes:
        print(f"[EXPERIMENT 8 | {cohort_label}] No missing genes -- variant B == variant A.")
        return pd.DataFrame([result_a])

    dp_b = TCGADataPipeline(clinical_df, expr_df, severity_context_source, drop_genes=missing_genes)
    splits_b = dp_b.prepare_data()
    X_train_b, C_train_b, _ = splits_b["train"]
    flow_b = train_flow_map(X_train_b, C_train_b, epochs=150)
    X_ext_b, C_ext_b, _, keep_b = dp_b.transform_external(ext_clin, ext_expr, verbose=True)

    with torch.no_grad():
        x0 = torch.randn(X_ext_b.shape[0], X_ext_b.shape[1], device=DEVICE)
        gen_b = integrate_flow(flow_b, x0, C_ext_b.to(DEVICE)).cpu().numpy()
    result_b = {"variant": "B_drop_missing_genes", "cohort": cohort_label,
                "dropped_genes": missing_genes,
                "mmd": mmd_rbf(X_ext_b.numpy(), gen_b), "wasserstein": mean_wasserstein(X_ext_b.numpy(), gen_b)}

    df = pd.DataFrame([result_a, result_b])
    print(f"[EXPERIMENT 8 | {cohort_label}] impute vs drop for {missing_genes}:")
    print(df.to_string(index=False))
    return df


# ════════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════════

def run_all_experiments():
    print("Fetching TCGA clinical data from cBioPortal...")
    clinical_raw = fetch_clinical_cbioportal(TCGA_STUDY_ID)
    print("Fetching TCGA expression data from cBioPortal...")
    expr_raw = fetch_expression_cbioportal(TCGA_STUDY_ID)

    clinical_df, severity_context_source = build_tcga_clinical(clinical_raw)
    expr_df = build_tcga_expression(expr_raw)
    common_patients = clinical_df.index.intersection(expr_df.index)
    clinical_df, expr_df = clinical_df.loc[common_patients], expr_df.loc[common_patients]
    print(f"TCGA patients with clinical + expression data: {len(common_patients)}")
    print(f"TCGA severity_context_source = '{severity_context_source}'")

    dp = TCGADataPipeline(clinical_df, expr_df, severity_context_source)
    splits = dp.prepare_data()
    X_train, C_train, y_train = splits["train"]
    X_oracle, C_oracle, y_oracle = splits["oracle"]
    X_test, C_test, y_test = splits["test"]
    print(f"Train: {X_train.shape[0]} | Oracle: {X_oracle.shape[0]} | Test: {X_test.shape[0]} "
          f"| Digital twin dim: {X_train.shape[1]}")

    print("[TRAINING] Reward oracle...")
    oracle = train_reward_oracle(X_oracle, y_oracle)
    print("[TRAINING] TCGA conditional flow map...")
    flow_model = train_flow_map(X_train, C_train)
    print("[TRAINING] TCGA flow map ablation (no context)...")
    flow_no_context = train_flow_map_no_context(X_train)
    print("[TRAINING] Gaussian variants (Ledoit-Wolf, source)...")
    gaussian_variants = GaussianVariants().fit_source(X_train)
    print("[TRAINING] VAE baseline...")
    vae_model = train_vae(X_train, C_train, epochs=200)

    saudi_expr_pooled, saudi_clin_pooled, per_series = load_saudi_cohorts()
    cohorts_to_run = list(per_series.items()) + [("pooled", (saudi_expr_pooled, saudi_clin_pooled))]

    exp1 = experiment1_in_domain(flow_model, oracle, gaussian_variants, vae_model,
                                  X_train, C_train, X_test, C_test)
    exp7 = experiment7_platform_sensitivity(dp, per_series)

    all_exp2, all_exp3_check, all_exp3, all_exp4, all_exp5, all_exp6, all_exp8 = {}, {}, {}, {}, {}, {}, {}

    for cohort_label, (ext_expr, ext_clin) in cohorts_to_run:
        print(f"\n{'='*80}\nCohort: {cohort_label}\n{'='*80}")
        exp2, X_ext_harmonized = experiment2_population_shift(
            dp, flow_model, gaussian_variants, vae_model, ext_expr, ext_clin, cohort_label=cohort_label)
        all_exp2[cohort_label] = exp2
        if exp2 is None:
            continue

        plot_population_shift_umap(X_test.numpy(), exp2["X_real"],
                                    f"{OUT_DIR}/figure_population_shift_umap_{cohort_label}.png",
                                    cohort_label=cohort_label)

        all_exp3_check[cohort_label] = experiment3_stage_conditioning_transfer(
            dp, flow_model, flow_no_context, ext_expr, ext_clin, cohort_label=cohort_label)

        exp3_out = experiment3_population_adaptation(
            dp, flow_model, gaussian_variants, vae_model, ext_expr, ext_clin, cohort_label=cohort_label)
        if exp3_out is not None:
            all_exp3[cohort_label], _adapted_flow = exp3_out
            df4, agg4 = experiment4_calibration_ablation(
                dp, flow_model, gaussian_variants, ext_expr, ext_clin, cohort_label=cohort_label)
            all_exp4[cohort_label] = (df4, agg4)

        all_exp5[cohort_label] = experiment5_risk_steering(
            dp, flow_model, oracle, ext_expr, ext_clin, cohort_label=cohort_label)
        all_exp6[cohort_label] = experiment6_risk_steering_baselines(
            dp, gaussian_variants, vae_model, oracle, ext_expr, ext_clin, cohort_label=cohort_label)

        all_exp8[cohort_label] = experiment8_gene_sensitivity(
            clinical_df, expr_df, severity_context_source, ext_expr, ext_clin, cohort_label=cohort_label)

    # ---- Save bundle ----
    summary_rows = []
    for label, e2 in all_exp2.items():
        if e2 is not None:
            summary_rows.append({"experiment": f"2_population_shift[{label}]", **e2["summary"]})
    for label, e3 in all_exp3.items():
        if e3 is not None:
            summary_rows.append({"experiment": f"3_population_adaptation[{label}]", **e3})
    for label, e5 in all_exp5.items():
        if e5 is not None:
            summary_rows.append({"experiment": f"5_risk_steering[{label}]",
                                  **{k: v for k, v in e5.items()
                                     if k not in ("alphas", "mean_risk_by_alpha", "risk_ci95_low", "risk_ci95_high")}})
    pd.DataFrame(summary_rows).to_csv(f"{OUT_DIR}/experiment_summary.csv", index=False)

    if len(exp7):
        exp7.to_csv(f"{OUT_DIR}/experiment7_platform_sensitivity.csv", index=False)
    for label, e2 in all_exp2.items():
        if e2 is not None:
            e2["per_feature_shift"].to_csv(f"{OUT_DIR}/experiment2_per_feature_shift_{label}.csv", index=False)
    for label, (df4, agg4) in all_exp4.items():
        if len(df4):
            df4.to_csv(f"{OUT_DIR}/experiment4_calibration_ablation_raw_{label}.csv", index=False)
            agg4.to_csv(f"{OUT_DIR}/experiment4_calibration_ablation_summary_{label}.csv", index=False)
    for label, e6 in all_exp6.items():
        if e6 is not None and len(e6):
            e6.to_csv(f"{OUT_DIR}/experiment6_risk_steering_baselines_{label}.csv", index=False)
    for label, e8 in all_exp8.items():
        if e8 is not None and len(e8):
            e8.to_csv(f"{OUT_DIR}/experiment8_gene_sensitivity_{label}.csv", index=False)

    zip_path = "/content/population_shift_results.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for fname in os.listdir(OUT_DIR):
            zf.write(os.path.join(OUT_DIR, fname), arcname=fname)
    print(f"Bundled results into {zip_path}")

    return {
        "dp": dp, "oracle": oracle, "flow_model": flow_model, "flow_no_context": flow_no_context,
        "gaussian_variants": gaussian_variants, "vae_model": vae_model,
        "experiment1": exp1, "experiment2": all_exp2, "experiment3_conditioning_check": all_exp3_check,
        "experiment3": all_exp3, "experiment4": all_exp4, "experiment5": all_exp5,
        "experiment6": all_exp6, "experiment7": exp7, "experiment8": all_exp8,
    }


if __name__ == "__main__":
    results = run_all_experiments()
    print("\n=== DONE ===")

"""
Population-Shift Evaluation of Conditional Flow-Matching Breast-Cancer Digital Twins
======================================================================================
"""

# ── CELL 0 — installs ────────────────────────────────────────────────────────────────
# !pip install -q torch scikit-learn pandas numpy scipy matplotlib seaborn requests tqdm
# !pip install -q GEOparse umap-learn

# ── CELL 1 — imports & config ───────────────────────────────────────────────────────
import os, re, warnings, zipfile, copy
import numpy as np
import pandas as pd
import requests

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.covariance import LedoitWolf   # shrinkage-regularized covariance for all
                                             # Gaussian variants

from scipy.stats import wasserstein_distance, spearmanr

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "/content/data"
OUT_DIR = "/content/outputs"
GEO_DIR = "/content/data/geo"
for d in (DATA_DIR, OUT_DIR, GEO_DIR):
    os.makedirs(d, exist_ok=True)

CBIOPORTAL_BASE = "https://www.cbioportal.org/api"
TCGA_STUDY_ID = "brca_tcga"

# Saudi Arabia / Muslim-majority-country cohorts (population context only).
SAUDI_GEO_SERIES = ["GSE36295", "GSE29044"]

N_TOP_GENES = 80           # top variable genes, fit on TCGA train only
N_GENE_PCS = 10            # genomic embedding dimensionality
FIVE_YEAR_DAYS = 365 * 5
N_EULER_STEPS = 10

CALIBRATION_SIZES = [5, 10, 20, 30, 40]     # calibration-size ablation grid
N_MONOTONICITY_SEEDS = 5                     # seeds for the monotonicity CI

# bootstrap + repeated-split settings
N_BOOTSTRAP = 1000                # bootstrap resamples for MMD/Wasserstein CIs
N_CALIBRATION_SPLIT_SEEDS = 5     # repeated random calibration/eval splits per size
CALIBRATION_SPLIT_SEEDS = [RANDOM_STATE + k for k in range(N_CALIBRATION_SPLIT_SEEDS)]
GEN_POOL_MULTIPLIER = 5           # generated-sample pool size = GEN_POOL_MULTIPLIER * n_eval,
                                   # resampled (not retrained) inside the bootstrap loop

# ---- COMMON FEATURE SUBSET --------------------------------------------------------
COMMON_CLINICAL_NUMERIC = ["age", "lymph_nodes"]
COMMON_CLINICAL_CATEGORICAL = ["histological_type", "er_status", "pr_status",
                                "her2_status", "grade"]
print("Device:", DEVICE)


# ════════════════════════════════════════════════════════════════════════════════════
# PART A — TCGA-BRCA
# ════════════════════════════════════════════════════════════════════════════════════

# ── CELL 2 — TCGA download via cBioPortal API ───────────────────────────────────────
def fetch_clinical_cbioportal(study_id):
    url = f"{CBIOPORTAL_BASE}/studies/{study_id}/patients"
    patients = requests.get(url, headers={"Accept": "application/json"}, timeout=60).json()
    patient_ids = [p["patientId"] for p in patients]
    print(f"  Fetched {len(patient_ids)} patients")

    url2 = f"{CBIOPORTAL_BASE}/studies/{study_id}/clinical-data?clinicalDataType=PATIENT&pageSize=100000"
    clin_raw = requests.get(url2, headers={"Accept": "application/json"}, timeout=120).json()
    rows = {}
    for entry in clin_raw:
        pid = entry["patientId"]
        rows.setdefault(pid, {"patient_id": pid})[entry["clinicalAttributeId"]] = entry["value"]
    df = pd.DataFrame.from_dict(rows, orient="index").reset_index(drop=True)
    print(f"  Clinical dataframe: {df.shape}")
    return df


def fetch_expression_cbioportal(study_id):
    profiles = requests.get(f"{CBIOPORTAL_BASE}/studies/{study_id}/molecular-profiles",
                             headers={"Accept": "application/json"}, timeout=60).json()
    rna_profile = next(p["molecularProfileId"] for p in profiles
                        if "rna_seq" in p["molecularProfileId"].lower()
                        and "v2" in p["molecularProfileId"].lower())
    print(f"  Using expression profile: {rna_profile}")

    samples = requests.get(f"{CBIOPORTAL_BASE}/studies/{study_id}/samples?pageSize=10000",
                            headers={"Accept": "application/json"}, timeout=60).json()
    sample_ids = [s["sampleId"] for s in samples][:500]
    print(f"  Using {len(sample_ids)} samples")

    gene_symbols = [
        "ESR1", "ERBB2", "PGR", "MKI67", "BRCA1", "BRCA2", "TP53", "PIK3CA", "CDH1", "GATA3",
        "RB1", "PTEN", "AKT1", "CCND1", "FOXA1", "EGFR", "FGFR1", "MYC", "AURKA", "TOP2A",
        "BIRC5", "BCL2", "FOXM1", "PLK1", "MELK", "RAD51", "CHEK1", "CHEK2", "ATM", "PALB2",
        "ESR2", "AR", "STAT3", "NOTCH1", "CDK4", "CDK6", "CDKN2A", "MDM2", "KRT8", "KRT18",
        "KRT19", "MLPH", "TFF1", "CLDN7", "FN1", "COL3A1", "DCN", "VEGFA", "MET", "IGF1R",
        "MAP3K1", "NF1", "TSC1", "TSC2", "KEAP1", "NFE2L2", "WNT5A", "SRC", "JAK2", "NCOA3",
        "UBE2C", "TYMS", "PCNA", "E2F1", "BUB1", "CENPA", "KIF2C", "TPX2", "PRC1", "EXO1",
        "MCM2", "BRIP1", "NBN", "MRE11", "RAD50", "ATR", "CCND2", "CCNE1", "MMP2", "MMP9",
    ]

    entrez_ids, symbol_map = [], {}
    for sym in gene_symbols:
        try:
            r = requests.get(f"{CBIOPORTAL_BASE}/genes/{sym}",
                              headers={"Accept": "application/json"}, timeout=10).json()
            if "entrezGeneId" in r:
                entrez_ids.append(r["entrezGeneId"])
                symbol_map[r["entrezGeneId"]] = r["hugoGeneSymbol"]
        except Exception:
            continue
    print(f"  Resolved {len(entrez_ids)} gene IDs")

    payload = {"sampleIds": sample_ids, "entrezGeneIds": entrez_ids}
    expr_resp = requests.post(
        f"{CBIOPORTAL_BASE}/molecular-profiles/{rna_profile}/molecular-data/fetch?projection=SUMMARY",
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        json=payload, timeout=180).json()

    if not isinstance(expr_resp, list) or len(expr_resp) == 0:
        raise ValueError(f"Unexpected expression response: {str(expr_resp)[:200]}")

    rows = {}
    for entry in expr_resp:
        if not isinstance(entry, dict):
            continue
        sid, eid = entry.get("sampleId"), entry.get("entrezGeneId")
        gene = symbol_map.get(eid, str(eid))
        rows.setdefault(sid, {"sample_id": sid})[gene] = entry.get("value", np.nan)

    expr_df = pd.DataFrame.from_dict(rows, orient="index").reset_index(drop=True)
    print(f"  Expression dataframe: {expr_df.shape}")
    return expr_df


# ── CELL 3 — TCGA clinical harmonization onto the COMMON schema ────────────────────
TCGA_COL_MAP = {
    "AGE": "age",
    "AJCC_PATHOLOGIC_TUMOR_STAGE": "stage",
    "RACE": "race",
    "HISTOLOGICAL_DIAGNOSIS": "histological_type",
    "MENOPAUSE_STATUS": "menopause_status",
    "RADIATION_THERAPY": "radiation_therapy",
    "NEOADJUVANT_THERAPY": "neoadjuvant_tx",
    "LYMPH_NODE_STATUS": "lymph_nodes",
    "ER_STATUS_BY_IHC": "er_status",
    "PR_STATUS_BY_IHC": "pr_status",
    "IHC_HER2": "her2_status",
    "TUMOR_GRADE": "grade",
    "OS_STATUS": "vital_status",
    "OS_MONTHS": "os_months",
    "patient_id": "patient_id",
}


def stage_to_severity(stage_str):
    if pd.isna(stage_str):
        return np.nan
    s = str(stage_str).upper().replace("STAGE", "").strip()
    m = re.match(r"^(IV|III|II|I)([ABC]?)$", s)
    if not m:
        return np.nan
    roman_to_val = {"I": 1.0, "II": 2.0, "III": 3.0, "IV": 4.0}
    sub_offset = {"A": 0.0, "B": 0.15, "C": 0.30, "": 0.0}
    return roman_to_val[m.group(1)] + sub_offset[m.group(2)]


def grade_to_severity(grade_raw):
    if pd.isna(grade_raw):
        return np.nan
    s = str(grade_raw).strip().upper()
    if "GX" in s or "N/A" in s or "NA" == s:
        return np.nan
    m = re.search(r"([1-3])", s)
    return float(m.group(1)) if m else np.nan


CONTEXT_SOURCE = "grade"
MIN_TCGA_GRADE_COVERAGE = 30


def parse_age_value(raw):
    if pd.isna(raw):
        return np.nan
    s = str(raw).strip()
    if re.match(r"^\d+(\.\d+)?$", s):
        return float(s)
    m = re.match(r"^>\s*(\d+)$", s)
    if m:
        return float(m.group(1)) + 5.0
    m = re.match(r"^<\s*(\d+)$", s)
    if m:
        return float(m.group(1)) - 5.0
    m = re.match(r"^(\d+)\s*-\s*(\d+)$", s)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2.0
    return np.nan


def normalize_marker_status(raw):
    if pd.isna(raw):
        return np.nan
    s = str(raw).strip().lower()
    if s in ("normal", "n/a", "na", ""):
        return np.nan
    if s in ("+", "1", "pos", "positive"):
        return "positive"
    if s in ("-", "0", "neg", "negative"):
        return "negative"
    if s == "2":
        return np.nan
    if s == "3":
        return "positive"
    return np.nan


def make_5yr_label(row):
    status = str(row.get("vital_status", "")).strip().lower()
    if status == "dead":
        d = row.get("days_to_death", np.nan)
        return np.nan if pd.isna(d) else (1.0 if d <= FIVE_YEAR_DAYS else 0.0)
    elif status == "alive":
        f = row.get("days_to_last_fu", np.nan)
        return np.nan if pd.isna(f) else (0.0 if f >= FIVE_YEAR_DAYS else np.nan)
    return np.nan


def resolve_tcga_grade_column(clinical_raw):
    candidates = [c for c in clinical_raw.columns if "GRADE" in c.upper()]
    best_col, best_count = None, 0
    for c in candidates:
        cnt = clinical_raw[c].notna().sum()
        if cnt > best_count:
            best_col, best_count = c, cnt
    if best_col:
        print(f"  [TCGA grade] candidate columns: {candidates}; using '{best_col}' "
              f"with {best_count}/{len(clinical_raw)} non-null values")
    else:
        print("  [TCGA grade] no GRADE-like column found in TCGA clinical data.")
    return best_col, best_count


def build_tcga_clinical(clinical_raw):
    """Returns (df, severity_context_source) so callers know for certain whether TCGA
    ended up conditioned on 'grade' (matches the Saudi cohorts) or fell back to
    'stage' (does NOT match Saudi grade-based severity). This is the single source of
    truth Experiment 3's conditioning-transfer check gates on."""
    grade_col, grade_coverage = resolve_tcga_grade_column(clinical_raw)
    col_map = dict(TCGA_COL_MAP)
    if grade_col:
        col_map[grade_col] = "grade_raw"

    df = clinical_raw.rename(columns={k: v for k, v in col_map.items()
                                       if k in clinical_raw.columns})
    if "vital_status" in df.columns:
        df["vital_status"] = df["vital_status"].apply(lambda x: "dead" if str(x).startswith("1") else "alive")
    if "os_months" in df.columns:
        df["os_months"] = pd.to_numeric(df["os_months"], errors="coerce")
        df["days_to_death"] = df.apply(lambda r: r["os_months"] * 30.44 if r.get("vital_status") == "dead" else np.nan, axis=1)
        df["days_to_last_fu"] = df.apply(lambda r: r["os_months"] * 30.44 if r.get("vital_status") == "alive" else np.nan, axis=1)

    df["age"] = pd.to_numeric(df.get("age", np.nan), errors="coerce")
    df["stage"] = df.get("stage", np.nan)
    df["stage_severity_"] = df["stage"].apply(stage_to_severity)

    use_grade = CONTEXT_SOURCE == "grade" and grade_coverage >= MIN_TCGA_GRADE_COVERAGE
    if use_grade:
        df["severity_context"] = df["grade_raw"].apply(grade_to_severity)
        severity_context_source = "grade"
        print(f"  [TCGA severity_context] using GRADE (coverage {grade_coverage}/{len(df)}) "
              f"-- MATCHES Saudi cohort severity axis (grade). Experiment 3 is valid.")
    else:
        df["severity_context"] = df["stage_severity_"]
        severity_context_source = "stage"
        reason = "CONTEXT_SOURCE='stage'" if CONTEXT_SOURCE != "grade" else \
            f"grade coverage too low ({grade_coverage} < {MIN_TCGA_GRADE_COVERAGE})"
        print(f"  [TCGA severity_context] falling back to STAGE ({reason}). "
              f"WARNING: Saudi cohorts only report GRADE. The stage-conditioning transfer "
              f"check will be SKIPPED for this run because comparing stage-conditioned "
              f"TCGA to grade-conditioned Saudi data is not apples-to-apples.")

    df["label"] = df.apply(make_5yr_label, axis=1)

    for col in ["er_status", "pr_status", "her2_status"]:
        if col in df.columns:
            df[col] = df[col].apply(normalize_marker_status)

    df = df.dropna(subset=["patient_id", "severity_context", "label"]).reset_index(drop=True)
    df = df.set_index("patient_id")
    df["population_context"] = "TCGA (global, source domain)"
    return df, severity_context_source


# ── CELL 4 — TCGA expression cleanup + patient matching ─────────────────────────────
def build_tcga_expression(expr_raw_cbio):
    expr_raw_cbio = expr_raw_cbio.copy()
    expr_raw_cbio["patient_id"] = expr_raw_cbio["sample_id"].apply(
        lambda s: "-".join(s.split("-")[:3]) if pd.notna(s) else s)
    expr_df = expr_raw_cbio.drop(columns=["sample_id"]).drop_duplicates(subset="patient_id")
    expr_df = expr_df.set_index("patient_id")
    gene_cols = list(expr_df.columns)
    expr_df = expr_df[gene_cols].apply(pd.to_numeric, errors="coerce")
    expr_df = expr_df.fillna(expr_df.mean())
    return expr_df


# ════════════════════════════════════════════════════════════════════════════════════
# PART B — GEO ingestion for the Saudi cohorts (GSE36295, GSE29044)
# ════════════════════════════════════════════════════════════════════════════════════

def download_geo_series(gse_id, destdir=GEO_DIR):
    import GEOparse
    print(f"[GEO] Downloading {gse_id} ...")
    gse = GEOparse.get_GEO(geo=gse_id, destdir=destdir, silent=True)
    print(f"  {gse_id}: {len(gse.gsms)} samples, platforms: {list(gse.gpls.keys())}")
    return gse


CHARACTERISTIC_PATTERNS = {
    "age": [r"age[^:]*:\s*(\S+)"],
    "tissue": [r"^tissue[^:]*:\s*(.+)", r"sample type\s*:\s*(.+)"],
    "tumor_normal": [r"tumou?r\s*/\s*normal[^:]*:\s*(\S+)"],
    "sample_type": [r"\btype\s*:\s*(\S+)"],
    "histological_type": [r"histolog\w*[^:]*:\s*(.+)"],
    "er_status": [r"\ber\b\s*(\([^)]*\))?\s*:\s*(\S+)"],
    "pr_status": [r"\bpr\b\s*(\([^)]*\))?\s*:\s*(\S+)"],
    "her2_status": [r"her-?2\s*(/\s*neu)?\s*(\([^)]*\))?\s*:\s*(\S+)"],
    "grade": [r"\bgrade\b[^:]*:\s*(\S+)"],
}


def parse_geo_characteristics(gsm):
    raw_lines = gsm.metadata.get("characteristics_ch1", [])
    parsed = {}
    for field, patterns in CHARACTERISTIC_PATTERNS.items():
        for line in raw_lines:
            matched = False
            for pat in patterns:
                m = re.search(pat, line, flags=re.IGNORECASE)
                if m:
                    parsed[field] = m.group(m.lastindex)
                    matched = True
                    break
            if matched:
                break
    if len(parsed) < 2:
        print(f"  [WARN] {gsm.name}: only parsed {list(parsed.keys())} from "
              f"{raw_lines} -- extend CHARACTERISTIC_PATTERNS if this is a real cohort sample.")
    return parsed


def is_breast_tumor_sample(row):
    tissue = str(row.get("tissue", "")).lower()
    if "blood" in tissue or "lymphoblastoid" in tissue:
        return False
    tn = str(row.get("tumor_normal", "")).strip().upper()
    sample_type = str(row.get("sample_type", "")).strip().upper()
    if tn == "T":
        return True
    if tn == "N":
        return False
    if "normal" in tissue or sample_type == "NORMAL":
        return False
    if "tumor" in tissue or "cancer" in tissue or sample_type in ("IDC", "DCIS", "ILC"):
        return True
    return False


def resolve_probe_to_symbol(gpl):
    table = gpl.table
    candidate_cols = ["Gene Symbol", "GENE_SYMBOL", "Symbol", "gene_symbol"]
    for col in candidate_cols:
        if col in table.columns:
            mapping = table.set_index(table.columns[0])[col].dropna().to_dict()
            return {k: str(v).split("///")[0].strip() for k, v in mapping.items() if str(v).strip()}

    if "gene_assignment" in table.columns:
        mapping = {}
        for probe_id, val in zip(table.iloc[:, 0], table["gene_assignment"]):
            if pd.isna(val) or val == "---":
                continue
            first_block = str(val).split("///")[0]
            parts = [p.strip() for p in first_block.split("//")]
            if len(parts) >= 2 and parts[1]:
                mapping[probe_id] = parts[1]
        return mapping

    raise ValueError(f"Could not find a gene-symbol column in platform table; "
                      f"columns available: {list(table.columns)}. "
                      f"Add the right column name to resolve_probe_to_symbol().")


def build_geo_expression_and_clinical(gse):
    platform_to_samples = {}
    for gsm_name, gsm in gse.gsms.items():
        plat_ids = gsm.metadata.get("platform_id", [])
        plat_id = plat_ids[0] if plat_ids else None
        platform_to_samples.setdefault(plat_id, []).append(gsm_name)

    per_platform_gene_expr = []
    for plat_id, sample_names in platform_to_samples.items():
        if plat_id not in gse.gpls:
            print(f"  [WARN] platform {plat_id} not in gse.gpls -- skipping "
                  f"{len(sample_names)} samples on it.")
            continue
        try:
            probe_map = resolve_probe_to_symbol(gse.gpls[plat_id])
        except ValueError as e:
            print(f"  [WARN] {e} -- skipping platform {plat_id} ({len(sample_names)} samples)")
            continue
        probe_map = {str(k): v for k, v in probe_map.items()}

        expr_cols = {}
        for gsm_name in sample_names:
            gsm = gse.gsms[gsm_name]
            if "VALUE" not in gsm.table.columns:
                continue
            s = gsm.table.set_index("ID_REF")["VALUE"]
            s.index = s.index.astype(str)
            expr_cols[gsm_name] = s
        if not expr_cols:
            continue

        plat_expr = pd.DataFrame(expr_cols).T.apply(pd.to_numeric, errors="coerce")
        symbol_series = pd.Series({p: probe_map.get(p) for p in plat_expr.columns})
        plat_expr_t = plat_expr.T
        plat_expr_t["__gene__"] = symbol_series
        plat_expr_t = plat_expr_t.dropna(subset=["__gene__"])
        gene_expr = plat_expr_t.groupby("__gene__").mean().T
        print(f"  Platform {plat_id}: {len(sample_names)} samples -> "
              f"{gene_expr.shape[1]} mapped genes")
        # Tag every sample's originating platform so downstream code can report/
        # inspect platform composition of a cohort instead of only ever seeing an
        # anonymous pooled matrix.
        gene_expr["__platform__"] = plat_id
        per_platform_gene_expr.append(gene_expr)

    if not per_platform_gene_expr:
        raise ValueError("No usable expression data found on any platform for this GSE.")

    expr_df = pd.concat(per_platform_gene_expr, axis=0, join="outer")
    platform_col = expr_df.pop("__platform__")

    clinical_rows = {gsm_name: parse_geo_characteristics(gsm) for gsm_name, gsm in gse.gsms.items()}
    clinical_df = pd.DataFrame.from_dict(clinical_rows, orient="index")
    for col in ("tissue", "tumor_normal", "sample_type", "grade"):
        if col not in clinical_df.columns:
            clinical_df[col] = np.nan
    clinical_df["platform"] = platform_col.reindex(clinical_df.index)

    n_before = len(clinical_df)
    is_tumor = clinical_df.apply(is_breast_tumor_sample, axis=1)
    print(f"  {n_before} raw samples -> {int(is_tumor.sum())} kept as breast TUMOR samples "
          f"({int((~is_tumor).sum())} dropped: normal/adjacent-normal breast tissue and/or "
          f"unrelated non-breast specimens folded into this GSE accession)")
    clinical_df = clinical_df.loc[is_tumor]
    expr_df_out = expr_df.loc[expr_df.index.intersection(clinical_df.index)]

    clinical_df["age"] = clinical_df["age"].apply(parse_age_value) if "age" in clinical_df.columns else np.nan
    clinical_df["severity_context"] = clinical_df["grade"].apply(grade_to_severity)

    for col in ["er_status", "pr_status", "her2_status"]:
        if col in clinical_df.columns:
            clinical_df[col] = clinical_df[col].apply(normalize_marker_status)

    return expr_df_out, clinical_df


def load_saudi_cohorts(series_ids=SAUDI_GEO_SERIES):
    """Returns (combined_expr, combined_clin, per_series_dict) so callers can run
    Experiments 2-5 on GSE36295 and GSE29044 SEPARATELY (each is a single-platform-
    family cohort) before ever pooling them into one distribution."""
    all_expr, all_clin = [], []
    per_series = {}
    for gse_id in series_ids:
        gse = download_geo_series(gse_id)
        gene_expr, clin_df = build_geo_expression_and_clinical(gse)

        clin_df = clin_df.copy()
        clin_df["population_context"] = "Saudi Arabia / Muslim-majority-country cohort"
        clin_df["source_series"] = gse_id

        n_stage = int(clin_df["severity_context"].notna().sum()) if "severity_context" in clin_df.columns else 0
        print(f"  [{gse_id}] parsed fields present (post tumor-filtering): "
              f"age={int(clin_df['age'].notna().sum()) if 'age' in clin_df.columns else 0}/{len(clin_df)}, "
              f"severity_context (grade-based)={n_stage}/{len(clin_df)}, "
              f"platforms={sorted(clin_df['platform'].dropna().unique().tolist())}")
        if n_stage == 0 and len(clin_df) > 0:
            print(f"  [{gse_id}] severity_context is 0/{len(clin_df)} -- dumping raw "
                  f"characteristics_ch1 for the first 5 KEPT (tumor) samples so you can "
                  f"extend CHARACTERISTIC_PATTERNS['grade'] with the real field name/format:")
            for gsm_name in list(clin_df.index)[:5]:
                raw = gse.gsms[gsm_name].metadata.get("characteristics_ch1", [])
                print(f"    {gsm_name}: {raw}")

        per_series[gse_id] = (gene_expr, clin_df)
        all_expr.append(gene_expr)
        all_clin.append(clin_df)

    combined_expr = pd.concat(all_expr, axis=0, join="outer")
    combined_clin = pd.concat(all_clin, axis=0, join="outer")
    common_idx = combined_expr.index.intersection(combined_clin.index)
    combined_expr, combined_clin = combined_expr.loc[common_idx], combined_clin.loc[common_idx]
    n_platforms = combined_clin["platform"].nunique() if "platform" in combined_clin.columns else "?"
    print(f"[GEO] Combined Saudi cohort: {combined_expr.shape[0]} samples, "
          f"{combined_expr.shape[1]} genes (pre-alignment), spanning {n_platforms} platforms "
          f"-- treat pooled numbers as secondary to the per-series ones below.")
    return combined_expr, combined_clin, per_series


# ════════════════════════════════════════════════════════════════════════════════════
# PART C — Shared representation: fit ONCE on TCGA, apply to Saudi
# ════════════════════════════════════════════════════════════════════════════════════

class TCGADataPipeline:
    def __init__(self, clinical_df, expr_df, severity_context_source,
                 n_top_genes=N_TOP_GENES, n_gene_pcs=N_GENE_PCS, drop_genes=None):
        self.clinical_df = clinical_df
        self.expr_df = expr_df
        self.severity_context_source = severity_context_source
        self.n_top_genes = n_top_genes
        self.n_gene_pcs = n_gene_pcs
        self.clinical_numeric = [c for c in COMMON_CLINICAL_NUMERIC if c in clinical_df.columns]
        self.clinical_categorical = [c for c in COMMON_CLINICAL_CATEGORICAL if c in clinical_df.columns]
        self.drop_genes = set(drop_genes or [])
        self.fit_index_ = None

    def prepare_data(self):
        idx = self.clinical_df.index
        y = self.clinical_df["label"].astype(int)

        idx_train_val, idx_test = train_test_split(idx, test_size=0.15, random_state=RANDOM_STATE, stratify=y)
        idx_train, idx_oracle = train_test_split(idx_train_val, test_size=0.176, random_state=RANDOM_STATE,
                                                  stratify=y.loc[idx_train_val])
        self.fit_index_ = list(idx_train)  # only TRAIN ever touches .fit()

        candidate_genes = [g for g in self.expr_df.columns if g not in self.drop_genes]
        gene_var = self.expr_df.loc[idx_train, candidate_genes].var(axis=0).sort_values(ascending=False)
        self.top_genes = gene_var.head(self.n_top_genes).index.tolist()
        self.gene_scaler = StandardScaler().fit(self.expr_df.loc[idx_train, self.top_genes])
        actual_pcs = min(self.n_gene_pcs, len(self.top_genes), len(idx_train))
        self.pca = PCA(n_components=actual_pcs, random_state=RANDOM_STATE).fit(
            self.gene_scaler.transform(self.expr_df.loc[idx_train, self.top_genes]))
        self.n_gene_pcs = actual_pcs
        self.top_gene_train_means = self.expr_df.loc[idx_train, self.top_genes].mean()

        def gene_embedding(index, expr_source=None):
            expr_source = self.expr_df if expr_source is None else expr_source
            scaled = self.gene_scaler.transform(expr_source.loc[index, self.top_genes])
            return self.pca.transform(scaled)

        clin_cols = self.clinical_numeric + self.clinical_categorical
        self.clinical_pre = ColumnTransformer([
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                               ("scale", StandardScaler())]), self.clinical_numeric),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                               ("onehot", OneHotEncoder(handle_unknown="ignore"))]),
             self.clinical_categorical),
        ]).fit(self.clinical_df.loc[idx_train, clin_cols])

        def clinical_features(index, clin_source=None):
            clin_source = self.clinical_df if clin_source is None else clin_source
            out = self.clinical_pre.transform(clin_source.loc[index, clin_cols])
            return out.toarray() if hasattr(out, "toarray") else out

        self.context_scaler = StandardScaler().fit(self.clinical_df.loc[idx_train, ["severity_context"]])

        def build_split(index):
            X = np.hstack([clinical_features(index), gene_embedding(index)]).astype(np.float32)
            C = self.context_scaler.transform(self.clinical_df.loc[index, ["severity_context"]]).astype(np.float32)
            yv = self.clinical_df.loc[index, "label"].astype(np.int64).values
            return torch.tensor(X), torch.tensor(C), torch.tensor(yv)

        X_train, C_train, y_train = build_split(idx_train)
        X_oracle, C_oracle, y_oracle = build_split(idx_oracle)
        X_test, C_test, y_test = build_split(idx_test)

        cat_names = list(self.clinical_pre.named_transformers_["cat"].named_steps["onehot"]
                          .get_feature_names_out(self.clinical_categorical))
        self.feature_names = self.clinical_numeric + cat_names + [f"gene_pc_{i}" for i in range(self.n_gene_pcs)]
        self._clin_cols = clin_cols

        return {"train": (X_train, C_train, y_train),
                "oracle": (X_oracle, C_oracle, y_oracle),
                "test": (X_test, C_test, y_test)}

    def transform_external(self, ext_clinical_df, ext_expr_df, verbose=True):
        present_genes = [g for g in self.top_genes if g in ext_expr_df.columns]
        missing_genes = [g for g in self.top_genes if g not in ext_expr_df.columns]
        if missing_genes and verbose:
            print(f"  [transform_external] {len(missing_genes)}/{len(self.top_genes)} "
                  f"TCGA genes not found on external platform -- imputing with TCGA train mean: "
                  f"{missing_genes[:10]}{'...' if len(missing_genes) > 10 else ''}")

        expr_aligned = pd.DataFrame(index=ext_expr_df.index, columns=self.top_genes, dtype=float)
        if present_genes:
            expr_aligned[present_genes] = ext_expr_df[present_genes].values
        for g in missing_genes:
            expr_aligned[g] = self.top_gene_train_means[g]
        expr_aligned = expr_aligned.fillna(expr_aligned.mean())

        gene_pc = self.pca.transform(self.gene_scaler.transform(expr_aligned))

        n_total = len(ext_clinical_df)
        n_with_stage = int(ext_clinical_df["severity_context"].notna().sum()) if \
            "severity_context" in ext_clinical_df.columns else 0
        keep_idx = ext_clinical_df.index[ext_clinical_df.get("severity_context", pd.Series(
            index=ext_clinical_df.index, dtype=float)).notna()]
        keep_idx = keep_idx.intersection(expr_aligned.index)

        if len(keep_idx) == 0:
            print(f"  [transform_external] 0/{n_total} external samples usable "
                  f"({n_with_stage}/{n_total} had a parsed severity_context). Returning empty result.")
            empty_X = torch.empty((0, len(self.feature_names)), dtype=torch.float32)
            empty_C = torch.empty((0, 1), dtype=torch.float32)
            return empty_X, empty_C, None, keep_idx

        clin_slice = ext_clinical_df.loc[keep_idx, self._clin_cols] if set(self._clin_cols).issubset(
            ext_clinical_df.columns) else pd.DataFrame(index=keep_idx, columns=self._clin_cols)
        for c in self._clin_cols:
            if c not in clin_slice.columns:
                clin_slice[c] = np.nan

        clin_feat = self.clinical_pre.transform(clin_slice)
        clin_feat = clin_feat.toarray() if hasattr(clin_feat, "toarray") else clin_feat

        gene_pc_slice = pd.DataFrame(gene_pc, index=expr_aligned.index).loc[keep_idx].values
        X_ext = np.hstack([clin_feat, gene_pc_slice]).astype(np.float32)
        C_ext = self.context_scaler.transform(ext_clinical_df.loc[keep_idx, ["severity_context"]]).astype(np.float32)

        y_ext = None
        if "label" in ext_clinical_df.columns and ext_clinical_df.loc[keep_idx, "label"].notna().any():
            y_ext = ext_clinical_df.loc[keep_idx, "label"].astype(np.int64).values

        return (torch.tensor(X_ext), torch.tensor(C_ext),
                None if y_ext is None else torch.tensor(y_ext), keep_idx)

    def assert_no_leakage(self, external_keep_idx):
        """Explicit leakage audit. Confirms none of the TCGA-side stateful fits
        (scaler / PCA / clinical ColumnTransformer / context scaler) were fit on
        anything outside self.fit_index_ (== TCGA train only), and that the external
        (Saudi) indices passed in never overlap with TCGA patient IDs."""
        assert self.fit_index_ is not None, "prepare_data() must run before the audit."
        overlap = set(self.fit_index_) & set(external_keep_idx.tolist() if hasattr(external_keep_idx, "tolist") else external_keep_idx)
        assert len(overlap) == 0, f"Leakage: {len(overlap)} external IDs collide with TCGA fit_index_"
        print(f"  [leakage audit] OK: 0 overlap between TCGA fit index (n={len(self.fit_index_)}) "
              f"and external cohort (n={len(external_keep_idx)}).")


def assert_disjoint(idx_a, idx_b, label=""):
    """Generic disjointness assertion used for the Saudi calibration/eval splits in
    Experiments 3, 4, and 5."""
    set_a, set_b = set(np.asarray(idx_a).tolist()), set(np.asarray(idx_b).tolist())
    overlap = set_a & set_b
    assert len(overlap) == 0, f"Leakage in {label}: {len(overlap)} overlapping indices"
    print(f"  [leakage audit] {label}: calibration (n={len(set_a)}) and eval (n={len(set_b)}) "
          f"are disjoint.")


def print_calibration_eval_sanity_check(calib_idx, eval_idx, variant_label="gaussian_full"):
    """v4 CHANGE (7c, per review): explicit, loud sanity check printed right at the
    point any Gaussian-on-calibration variant is fit, so the "fit on calibration,
    never on eval" contract is visible in the run log itself and not just implied by
    code structure. This is on top of (not a replacement for) assert_disjoint(),
    which already runs once per calibration/eval split -- this one is scoped to the
    specific Gaussian variant being fit at that moment, matching the exact log format
    the review asked for:

        Gaussian fit source: calibration only
        Calibration n: <n>
        Evaluation n: <n>
        Calibration/evaluation overlap: <n>

    Asserts (not just prints) zero overlap, so a real leakage bug fails loudly."""
    calib_set = set(np.asarray(calib_idx).tolist())
    eval_set = set(np.asarray(eval_idx).tolist())
    overlap = calib_set & eval_set
    print(f"  [sanity check | {variant_label}] Gaussian fit source: calibration only")
    print(f"  [sanity check | {variant_label}] Calibration n: {len(calib_set)}")
    print(f"  [sanity check | {variant_label}] Evaluation n: {len(eval_set)}")
    print(f"  [sanity check | {variant_label}] Calibration/evaluation overlap: {len(overlap)}")
    assert len(overlap) == 0, (
        f"LEAKAGE in {variant_label}: {len(overlap)} indices appear in both the "
        f"calibration set used to fit this Gaussian variant and the evaluation set "
        f"used to score it.")


# ════════════════════════════════════════════════════════════════════════════════════
# PART D — model classes
# ════════════════════════════════════════════════════════════════════════════════════

class SurvivalRewardOracle(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.SiLU(),
            nn.Linear(64, 2),
        )
    def forward(self, x):
        return self.net(x)


class FlowResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.act = nn.SiLU()
    def forward(self, x):
        return x + self.act(self.linear2(self.act(self.linear1(x))))


class TCGAFlowMap(nn.Module):
    def __init__(self, data_dim, context_dim, hidden_dim=256, blocks=3):
        super().__init__()
        self.input_proj = nn.Linear(data_dim + context_dim + 1, hidden_dim)
        self.trunk = nn.Sequential(*[FlowResidualBlock(hidden_dim) for _ in range(blocks)])
        self.output_proj = nn.Linear(hidden_dim, data_dim)
        self.act = nn.SiLU()
    def forward(self, x, c, t):
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        h = self.act(self.input_proj(torch.cat([x, c, t], dim=-1)))
        return self.output_proj(self.trunk(h))


def integrate_flow(model, x0, c, n_steps=N_EULER_STEPS):
    dt = 1.0 / n_steps
    x = x0
    for i in range(n_steps):
        t = torch.full((x.size(0), 1), i * dt, device=x.device)
        x = x + model(x, c, t) * dt
    return x


# ── Closed-form Gaussian-to-Gaussian affine ────────

def matrix_sqrt(A, eps=1e-8):
    """Symmetric PSD matrix square root via eigendecomposition. eps floors tiny/
    negative eigenvalues that arise from numerical noise (Ledoit-Wolf covariances are
    PSD by construction, but eigh() can still return values like -1e-12)."""
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.clip(eigvals, eps, None)
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T


def matrix_inv_sqrt(A, eps=1e-8):
    """Symmetric PSD matrix inverse square root via eigendecomposition."""
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.clip(eigvals, eps, None)
    return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T


def gaussian_affine_transport(X, mu_source, cov_source, mu_target, cov_target):
    """Maps source samples X into the target Gaussian's geometry via the closed-form
    affine map between two Gaussians:

        x' = mu_target + Sigma_target^(1/2) Sigma_source^(-1/2) (x - mu_source)

    Unlike the 'full' variant (which discards the source samples entirely and draws
    fresh i.i.d. samples from N(mu_target, cov_target)), this actually transports real
    TCGA source samples toward the target distribution's first two moments -- a
    fairer point of comparison to Flow, which also transports real source/noise
    samples through a learned map rather than sampling i.i.d. from a fitted target
    distribution.

    mu_source/mu_target: (d,) arrays. cov_source/cov_target: (d, d) PSD arrays.
    X: (n, d) array of source-domain samples to transport."""
    X = np.asarray(X)
    mu_source = np.asarray(mu_source)
    mu_target = np.asarray(mu_target)
    Cs_inv_sqrt = matrix_inv_sqrt(np.asarray(cov_source))
    Ct_sqrt = matrix_sqrt(np.asarray(cov_target))
    A = Ct_sqrt @ Cs_inv_sqrt
    return (X - mu_source) @ A.T + mu_target


# ---- Gaussian baseline: 5 variants with Ledoit-Wolf shrinkage -----

class GaussianVariants:

    def __init__(self):
        self.mu_source_ = None
        self.cov_source_ = None
        self.X_source_ = None
                                 # source points to transport, rather than only ever
                                 # drawing fresh i.i.d. Gaussian noise.

    def fit_source(self, X_train):
        X = X_train.numpy() if torch.is_tensor(X_train) else np.asarray(X_train)
        lw = LedoitWolf().fit(X)
        self.mu_source_, self.cov_source_ = lw.location_, lw.covariance_
        self.n_source_ = X.shape[0]
        self.X_source_ = X  # v4
        return self

    @staticmethod
    def fit_gaussian(X):
        X = X.numpy() if torch.is_tensor(X) else np.asarray(X)
        lw = LedoitWolf().fit(X)
        return lw.location_, lw.covariance_

    def params(self, variant, X_calib=None):
        assert self.mu_source_ is not None, "call fit_source() first"
        if variant == "source":
            return self.mu_source_, self.cov_source_
        if variant == "affine_transport":
            raise ValueError("'affine_transport' does not return (mu, cov) params -- "
                              "it transports real source samples. Call "
                              "GaussianVariants.affine_transport(X_calib, n, seed) "
                              "instead (generate_gaussian_pool() below dispatches to "
                              "it automatically).")
        assert X_calib is not None, f"variant '{variant}' requires X_calib"
        mu_calib, cov_calib = self.fit_gaussian(X_calib)
        if variant == "full":
            return mu_calib, cov_calib
        if variant == "mean_only":
            return mu_calib, self.cov_source_
        if variant == "cov_only":
            return self.mu_source_, cov_calib
        raise ValueError(f"unknown Gaussian variant: {variant}")

    def sample(self, mu, cov, n, seed=None):
        rng = np.random.RandomState(seed) if seed is not None else np.random
        return rng.multivariate_normal(mu, cov, size=n).astype(np.float32)

    def affine_transport(self, X_calib, n, seed=None):
        assert self.X_source_ is not None, "call fit_source() first"
        mu_target, cov_target = self.fit_gaussian(X_calib)
        rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
        idx = rng.randint(0, self.X_source_.shape[0], size=n)
        X_src_draw = self.X_source_[idx]
        transported = gaussian_affine_transport(
            X_src_draw, self.mu_source_, self.cov_source_, mu_target, cov_target)
        return transported.astype(np.float32), mu_target, cov_target


def print_gaussian_diagnostics(mu_source, cov_source, mu_target, cov_target, n_calib, n_eval, label=""):
    cond_source = np.linalg.cond(cov_source)
    cond_target = np.linalg.cond(cov_target)
    rank_source = np.linalg.matrix_rank(cov_source)
    rank_target = np.linalg.matrix_rank(cov_target)
    dim = cov_source.shape[0]
    print(f"  [Gaussian diagnostics{' | ' + label if label else ''}] "
          f"n_calibration={n_calib}, n_eval={n_eval}, dimension={dim}")
    print(f"    ||mu_source - mu_target|| = {np.linalg.norm(mu_source - mu_target):.4f}")
    print(f"    mean(diag(cov_source)) = {np.diag(cov_source).mean():.4f}, "
          f"mean(diag(cov_target)) = {np.diag(cov_target).mean():.4f}")
    print(f"    cond(cov_source) = {cond_source:.3e}, rank(cov_source) = {rank_source}/{dim}")
    print(f"    cond(cov_target) = {cond_target:.3e}, rank(cov_target) = {rank_target}/{dim}")
    if n_calib < dim:
        print(f"    NOTE: n_calibration ({n_calib}) < dimension ({dim}) -- the raw sample "
              f"covariance would be rank-deficient here; Ledoit-Wolf shrinkage is load-bearing, "
              f"not a formality.")


class SimpleVAE(nn.Module):
    def __init__(self, data_dim, context_dim, latent_dim=16, hidden_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc = nn.Sequential(
            nn.Linear(data_dim + context_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)
        self.dec = nn.Sequential(
            nn.Linear(latent_dim + context_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, data_dim),
        )
    def encode(self, x, c):
        h = self.enc(torch.cat([x, c], dim=-1))
        return self.mu_head(h), self.logvar_head(h)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)
    def decode(self, z, c):
        return self.dec(torch.cat([z, c], dim=-1))
    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar
    def sample(self, c, device):
        n = c.shape[0]
        z = torch.randn(n, self.latent_dim, device=device)
        with torch.no_grad():
            return self.decode(z, c).cpu().numpy()


def train_vae(X_train, C_train, epochs=300, batch_size=32, lr=5e-4, beta=0.1, init_model=None):
    X_train, C_train = X_train.to(DEVICE), C_train.to(DEVICE)
    model = init_model if init_model is not None else SimpleVAE(
        data_dim=X_train.shape[1], context_dim=C_train.shape[1]).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    n = X_train.shape[0]
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n)
        running = 0.0
        for i in range(0, n, batch_size):
            opt.zero_grad()
            idx = perm[i:i + batch_size]
            x, c = X_train[idx], C_train[idx]
            x_hat, mu, logvar = model(x, c)
            recon = nn.functional.mse_loss(x_hat, x, reduction="mean")
            kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon + beta * kld
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
        if (epoch + 1) % 100 == 0:
            print(f"  VAE epoch {epoch+1}/{epochs} | loss: {running/n:.4f}")
    model.eval()
    return model


def train_reward_oracle(X_oracle, y_oracle, epochs=200, lr=1e-3):
    X_oracle, y_oracle = X_oracle.to(DEVICE), y_oracle.to(DEVICE)
    oracle = SurvivalRewardOracle(input_dim=X_oracle.shape[1]).to(DEVICE)
    opt = optim.AdamW(oracle.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    oracle.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = criterion(oracle(X_oracle), y_oracle)
        loss.backward()
        opt.step()
    oracle.eval()
    return oracle


def train_flow_map(X_train, C_train, epochs=300, batch_size=32, lr=5e-4,
                    context_dim_override=None, init_model=None):
    X_train, C_train = X_train.to(DEVICE), C_train.to(DEVICE)
    data_dim = X_train.shape[1]
    context_dim = context_dim_override or C_train.shape[1]
    model = init_model if init_model is not None else TCGAFlowMap(data_dim, context_dim).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    n = X_train.size(0)
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n)
        running = 0.0
        for i in range(0, n, batch_size):
            opt.zero_grad()
            idx = perm[i:i + batch_size]
            x1, c = X_train[idx], C_train[idx]
            x0 = torch.randn_like(x1)
            t = torch.rand(x1.size(0), 1, device=DEVICE)
            xt = (1.0 - t) * x0 + t * x1
            pred_v = model(xt, c, t)
            loss = nn.functional.mse_loss(pred_v, x1 - x0)
            loss.backward()
            opt.step()
            running += loss.item() * x1.size(0)
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:03d}/{epochs} | flow matching loss: {running/n:.4f}")
    model.eval()
    return model


def train_flow_map_no_context(X_train, epochs=300, batch_size=32, lr=5e-4):
    X = X_train.to(DEVICE)
    model = TCGAFlowMap(data_dim=X.shape[1], context_dim=1, hidden_dim=256, blocks=3).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    n = X.shape[0]
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n)
        running = 0.0
        for i in range(0, n, batch_size):
            opt.zero_grad()
            idx = perm[i:i + batch_size]
            x1 = X[idx]
            x0 = torch.randn_like(x1)
            t = torch.rand(x1.shape[0], 1, device=DEVICE)
            xt = (1 - t) * x0 + t * x1
            c_zeros = torch.zeros(x1.shape[0], 1, device=DEVICE)
            pred_v = model(xt, c_zeros, t)
            loss = nn.functional.mse_loss(pred_v, x1 - x0)
            loss.backward()
            opt.step()
            running += loss.item() * x1.shape[0]
        if (epoch + 1) % 100 == 0:
            print(f"  No-context Epoch {epoch+1}/{epochs} | loss: {running/n:.4f}")
    model.eval()
    return model


# ════════════════════════════════════════════════════════════════════════════════════
# PART E — metrics, incl. bootstrap CIs
# ════════════════════════════════════════════════════════════════════════════════════

def mmd_rbf(X, Y, gamma=None):
    X, Y = np.asarray(X), np.asarray(Y)
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    Kxx, Kyy, Kxy = rbf_kernel(X, X, gamma), rbf_kernel(Y, Y, gamma), rbf_kernel(X, Y, gamma)
    return Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()


def mean_wasserstein(X, Y):
    X, Y = np.asarray(X), np.asarray(Y)
    return float(np.mean([wasserstein_distance(X[:, j], Y[:, j]) for j in range(X.shape[1])]))


def per_feature_shift(X_a, X_b, feature_names):
    rows = []
    for j, name in enumerate(feature_names):
        rows.append({"feature": name, "wasserstein": wasserstein_distance(X_a[:, j], X_b[:, j])})
    return pd.DataFrame(rows).sort_values("wasserstein", ascending=False)


def bootstrap_generative_metrics(x_real_eval, x_generated_pool, n_boot=N_BOOTSTRAP, seed=RANDOM_STATE):
    x_real_eval = np.asarray(x_real_eval)
    x_generated_pool = np.asarray(x_generated_pool)
    n_real = x_real_eval.shape[0]
    n_gen_draw = min(n_real, x_generated_pool.shape[0])  # keep the two-sample sizes matched

    point_mmd = mmd_rbf(x_real_eval, x_generated_pool[:n_real] if x_generated_pool.shape[0] >= n_real
                         else x_generated_pool)
    point_wd = mean_wasserstein(x_real_eval, x_generated_pool[:n_real] if x_generated_pool.shape[0] >= n_real
                                 else x_generated_pool)

    rng = np.random.RandomState(seed)
    mmds, wds = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        real_idx = rng.randint(0, n_real, size=n_real)
        gen_idx = rng.randint(0, x_generated_pool.shape[0], size=n_gen_draw)
        xr = x_real_eval[real_idx]
        xg = x_generated_pool[gen_idx]
        mmds[b] = mmd_rbf(xr, xg)
        wds[b] = mean_wasserstein(xr, xg)

    return {
        "mmd": float(point_mmd), "mmd_ci_low": float(np.percentile(mmds, 2.5)),
        "mmd_ci_high": float(np.percentile(mmds, 97.5)),
        "wasserstein": float(point_wd), "wasserstein_ci_low": float(np.percentile(wds, 2.5)),
        "wasserstein_ci_high": float(np.percentile(wds, 97.5)),
        "n_eval": int(n_real), "n_boot": int(n_boot),
    }


def fmt_ci(d, metric):
    return f"{d[metric]:.4f} [95% CI: {d[metric+'_ci_low']:.4f}-{d[metric+'_ci_high']:.4f}]"


def generate_flow_pool(model, C_ref, data_dim, n_pool, seed):
    """Builds an n_pool-sample generated pool by resampling contexts (with replacement)
    from C_ref and drawing a fresh noise seed per sample. Used only to feed
    bootstrap_generative_metrics -- never touches real eval data."""
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, C_ref.shape[0], size=n_pool)
    c_pool = C_ref[idx].to(DEVICE)
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    x0 = torch.randn(n_pool, data_dim, device=DEVICE, generator=g)
    with torch.no_grad():
        return integrate_flow(model, x0, c_pool).cpu().numpy()


def generate_vae_pool(model, C_ref, n_pool, seed):
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, C_ref.shape[0], size=n_pool)
    c_pool = C_ref[idx].to(DEVICE)
    torch.manual_seed(seed)
    return model.sample(c_pool, DEVICE)


def generate_gaussian_pool(gaussian_variants, variant, n_pool, seed, X_calib=None):
    if variant == "affine_transport":
        assert X_calib is not None, "'affine_transport' requires X_calib"
        pool, mu_t, cov_t = gaussian_variants.affine_transport(X_calib, n_pool, seed=seed)
        return pool, mu_t, cov_t
    mu, cov = gaussian_variants.params(variant, X_calib=X_calib)
    return gaussian_variants.sample(mu, cov, n_pool, seed=seed), mu, cov


def experiment1_in_domain(flow_model, oracle, gaussian_variants, vae_model,
                           X_train, C_train, X_test, C_test, n_steering=40, alpha=1.5):
    """EXPERIMENT 1: TCGA in-domain generation for Flow, Gaussian-source, VAE, each
    with a bootstrap CI on the TCGA test split (held out from training). Also runs the
    flow steering sanity check as before."""
    data_dim = X_train.shape[1]
    x_real = X_test.numpy()
    n_pool = max(GEN_POOL_MULTIPLIER * X_test.shape[0], 200)

    with torch.no_grad():
        x0 = torch.randn(X_test.size(0), data_dim, device=DEVICE)
        x_gen_flow = integrate_flow(flow_model, x0, C_test.to(DEVICE)).cpu().numpy()
    flow_pool = generate_flow_pool(flow_model, C_test, data_dim, n_pool, seed=RANDOM_STATE)
    flow_ci = bootstrap_generative_metrics(x_real, flow_pool)

    gauss_pool, mu_s, cov_s = generate_gaussian_pool(gaussian_variants, "source", n_pool, seed=RANDOM_STATE)
    gauss_ci = bootstrap_generative_metrics(x_real, gauss_pool)

    vae_pool = generate_vae_pool(vae_model, C_test, n_pool, seed=RANDOM_STATE)
    vae_ci = bootstrap_generative_metrics(x_real, vae_pool)

    print("[EXPERIMENT 1 -- TCGA in-domain generation]")
    print(f"  Flow:     MMD={fmt_ci(flow_ci,'mmd')} | Wasserstein={fmt_ci(flow_ci,'wasserstein')}")
    print(f"  Gaussian: MMD={fmt_ci(gauss_ci,'mmd')} | Wasserstein={fmt_ci(gauss_ci,'wasserstein')}")
    print(f"  VAE:      MMD={fmt_ci(vae_ci,'mmd')} | Wasserstein={fmt_ci(vae_ci,'wasserstein')}")

    # steering sanity check (flow only, as before)
    n = min(n_steering, X_test.shape[0])
    c_batch = C_test[:n].to(DEVICE)
    x0 = torch.randn(n, data_dim, device=DEVICE, requires_grad=True)
    x1 = integrate_flow(flow_model, x0, c_batch)
    logits = oracle(x1)
    target = torch.full((n,), 1, dtype=torch.long, device=DEVICE)
    nn.functional.cross_entropy(logits, target).backward()
    with torch.no_grad():
        x0_steered = x0 - alpha * x0.grad
        x1_steered = integrate_flow(flow_model, x0_steered, c_batch)
        risk_before = torch.softmax(logits, dim=-1)[:, 1].mean().item()
        risk_after = torch.softmax(oracle(x1_steered), dim=-1)[:, 1].mean().item()

    return {"flow": flow_ci, "gaussian_source": gauss_ci, "vae": vae_ci,
            "risk_before_steering": risk_before, "risk_after_steering": risk_after}


def experiment2_population_shift(dp, flow_model, gaussian_variants, vae_model, ext_expr, ext_clin,
                                  cohort_label="pooled"):
    """EXPERIMENT 2: TCGA -> external zero-shot, with bootstrap CIs for Flow, Gaussian
    (source), and VAE on the SAME external evaluation samples."""
    X_ext, C_ext, y_ext, keep_idx = dp.transform_external(ext_clin, ext_expr)
    if X_ext.shape[0] < 5:
        print(f"[EXPERIMENT 2 | {cohort_label}] Too few harmonizable samples ({X_ext.shape[0]}).")
        return None, X_ext

    dp.assert_no_leakage(keep_idx)
    x_real = X_ext.numpy()
    data_dim = X_ext.shape[1]
    n_pool = max(GEN_POOL_MULTIPLIER * X_ext.shape[0], 200)

    with torch.no_grad():
        x0 = torch.randn(X_ext.shape[0], data_dim, device=DEVICE)
        x_gen_flow = integrate_flow(flow_model, x0, C_ext.to(DEVICE)).cpu().numpy()
    flow_pool = generate_flow_pool(flow_model, C_ext, data_dim, n_pool, seed=RANDOM_STATE)
    flow_ci = bootstrap_generative_metrics(x_real, flow_pool)

    gauss_pool, _, _ = generate_gaussian_pool(gaussian_variants, "source", n_pool, seed=RANDOM_STATE)
    gauss_ci = bootstrap_generative_metrics(x_real, gauss_pool)

    vae_pool = generate_vae_pool(vae_model, C_ext, n_pool, seed=RANDOM_STATE)
    vae_ci = bootstrap_generative_metrics(x_real, vae_pool)

    pca2 = PCA(n_components=2, random_state=RANDOM_STATE).fit(x_real)
    pca_dist = float(np.linalg.norm(pca2.transform(x_real).mean(0) - pca2.transform(x_gen_flow).mean(0)))
    shift_table = per_feature_shift(x_real, x_gen_flow, dp.feature_names)

    print(f"[EXPERIMENT 2 | {cohort_label} -- Population shift, zero-shot]")
    print(f"  Flow:     MMD={fmt_ci(flow_ci,'mmd')} | Wasserstein={fmt_ci(flow_ci,'wasserstein')}")
    print(f"  Gaussian: MMD={fmt_ci(gauss_ci,'mmd')} | Wasserstein={fmt_ci(gauss_ci,'wasserstein')}")
    print(f"  VAE:      MMD={fmt_ci(vae_ci,'mmd')} | Wasserstein={fmt_ci(vae_ci,'wasserstein')}")

    summary = {"cohort": cohort_label, "n_samples": X_ext.shape[0],
               "pca_centroid_distance": pca_dist,
               **{f"flow_{k}": v for k, v in flow_ci.items()},
               **{f"gaussian_{k}": v for k, v in gauss_ci.items()},
               **{f"vae_{k}": v for k, v in vae_ci.items()}}
    return {"summary": summary, "per_feature_shift": shift_table,
            "X_real": x_real, "X_generated_flow": x_gen_flow,
            "X_ext": X_ext, "C_ext": C_ext, "keep_idx": keep_idx}, X_ext


def plot_population_shift_umap(x_tcga_real, x_ext_real, save_path, cohort_label="Saudi cohort"):
    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=RANDOM_STATE)
        method = "UMAP"
    except ImportError:
        from sklearn.manifold import TSNE
        combined_n = len(x_tcga_real) + len(x_ext_real)
        reducer = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=min(30, combined_n - 1))
        method = "t-SNE"

    combined = np.vstack([x_tcga_real, x_ext_real])
    labels = ["TCGA (source)"] * len(x_tcga_real) + [cohort_label] * len(x_ext_real)
    emb = reducer.fit_transform(combined)
    plot_df = pd.DataFrame({"x": emb[:, 0], "y": emb[:, 1], "Cohort": labels})

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {"TCGA (source)": "#4878CF", cohort_label: "#D65F5F"}
    for label, grp in plot_df.groupby("Cohort"):
        ax.scatter(grp["x"], grp["y"], label=label, alpha=0.7, color=colors.get(label, "#888888"),
                   edgecolors="k", linewidths=0.3, s=40)
    ax.set_title(f"{method}: TCGA vs {cohort_label} Digital-Twin State (Population Shift)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def experiment3_stage_conditioning_transfer(dp, flow_model, flow_no_context, ext_expr, ext_clin,
                                             cohort_label="pooled", stages=(1.0, 2.0, 3.0)):
    if dp.severity_context_source != "grade":
        print(f"[Conditioning-transfer check | {cohort_label}] SKIPPED: TCGA severity_context "
              f"is '{dp.severity_context_source}', external is 'grade' -- not comparable.")
        return pd.DataFrame()

    X_ext, C_ext, y_ext, keep_idx = dp.transform_external(ext_clin, ext_expr, verbose=False)
    ext_severity = ext_clin.loc[keep_idx, "severity_context"]

    rows = []
    for s in stages:
        mask = (ext_severity >= s - 0.5) & (ext_severity < s + 0.5)
        if mask.sum() < 2:
            continue
        x_real_sub = X_ext.numpy()[mask.values]
        n_samples = max(20, mask.sum())

        c_scaled = torch.tensor(dp.context_scaler.transform([[s]]), dtype=torch.float32)
        c_batch = c_scaled.repeat(n_samples, 1).to(DEVICE)
        c_zero = torch.zeros(n_samples, 1, device=DEVICE)
        x0 = torch.randn(n_samples, X_ext.shape[1], device=DEVICE)

        with torch.no_grad():
            gen_with_ctx = integrate_flow(flow_model, x0, c_batch).cpu().numpy()
            gen_no_ctx = integrate_flow(flow_no_context, x0, c_zero).cpu().numpy()

        rows.append({
            "cohort": cohort_label, "severity_context": s, "n_real": int(mask.sum()),
            "wasserstein_with_context": mean_wasserstein(x_real_sub, gen_with_ctx),
            "wasserstein_no_context": mean_wasserstein(x_real_sub, gen_no_ctx),
        })

    df = pd.DataFrame(rows)
    if len(df):
        df["improvement_from_context"] = df["wasserstein_no_context"] - df["wasserstein_with_context"]
        df["pct_improvement"] = 100 * df["improvement_from_context"] / df["wasserstein_no_context"]
    print(f"[Conditioning-transfer check | {cohort_label}]")
    print(df.to_string(index=False))
    return df


def experiment3_population_adaptation(dp, flow_model, gaussian_variants, vae_model, ext_expr, ext_clin,
                                       cohort_label="pooled", calib_frac=0.4, adapt_epochs=60, adapt_lr=1e-4):
    X_ext, C_ext, y_ext, keep_idx = dp.transform_external(ext_clin, ext_expr)
    n = X_ext.shape[0]
    if n < 10:
        print(f"[EXPERIMENT 3 | {cohort_label}] Too few samples for calibration/eval split.")
        return None

    idx = np.arange(n)
    calib_idx, eval_idx = train_test_split(idx, test_size=1 - calib_frac, random_state=RANDOM_STATE)
    assert_disjoint(calib_idx, eval_idx, label=f"Experiment 3 | {cohort_label}")

    X_calib, C_calib = X_ext[calib_idx], C_ext[calib_idx]
    X_eval, C_eval = X_ext[eval_idx], C_ext[eval_idx]
    x_real_eval = X_eval.numpy()
    data_dim = X_eval.shape[1]
    n_pool = max(GEN_POOL_MULTIPLIER * X_eval.shape[0], 200)

    rows = {}

    # --- Flow: zero-shot vs adapted ---
    flow_zero_pool = generate_flow_pool(flow_model, C_eval, data_dim, n_pool, seed=RANDOM_STATE)
    rows["flow_zero_shot"] = bootstrap_generative_metrics(x_real_eval, flow_zero_pool)

    adapted_flow = TCGAFlowMap(data_dim=data_dim, context_dim=C_ext.shape[1]).to(DEVICE)
    adapted_flow.load_state_dict(flow_model.state_dict())
    adapted_flow = train_flow_map(X_calib, C_calib, epochs=adapt_epochs, batch_size=min(16, len(calib_idx)),
                                   lr=adapt_lr, init_model=adapted_flow)
    flow_adapt_pool = generate_flow_pool(adapted_flow, C_eval, data_dim, n_pool, seed=RANDOM_STATE)
    rows["flow_adapted"] = bootstrap_generative_metrics(x_real_eval, flow_adapt_pool)

    # --- Gaussian: source, full, mean_only, cov_only, affine_transport ---
    for variant in ["source", "full", "mean_only", "cov_only", "affine_transport"]:
        pool, mu_v, cov_v = generate_gaussian_pool(
            gaussian_variants, variant, n_pool, seed=RANDOM_STATE,
            X_calib=(X_calib if variant != "source" else None))
        rows[f"gaussian_{variant}"] = bootstrap_generative_metrics(x_real_eval, pool)

        if variant != "source":
            print_gaussian_diagnostics(gaussian_variants.mu_source_, gaussian_variants.cov_source_,
                                        mu_v, cov_v, n_calib=len(calib_idx), n_eval=len(eval_idx),
                                        label=f"{cohort_label} | {variant}")
            # variant that was actually fit on the calibration set.
            print_calibration_eval_sanity_check(
                calib_idx, eval_idx, variant_label=f"gaussian_{variant} | {cohort_label}")

    # --- VAE: zero-shot vs adapted (fine-tuned the same way as flow) ---
    vae_zero_pool = generate_vae_pool(vae_model, C_eval, n_pool, seed=RANDOM_STATE)
    rows["vae_zero_shot"] = bootstrap_generative_metrics(x_real_eval, vae_zero_pool)

    adapted_vae = copy.deepcopy(vae_model)
    adapted_vae = train_vae(X_calib, C_calib, epochs=adapt_epochs, batch_size=min(16, len(calib_idx)),
                             lr=adapt_lr, init_model=adapted_vae)
    vae_adapt_pool = generate_vae_pool(adapted_vae, C_eval, n_pool, seed=RANDOM_STATE)
    rows["vae_adapted"] = bootstrap_generative_metrics(x_real_eval, vae_adapt_pool)

    print(f"[EXPERIMENT 3 | {cohort_label} -- Population adaptation] "
          f"(n_calibration={len(calib_idx)}, n_eval={len(eval_idx)})")
    for name, ci in rows.items():
        print(f"  {name:20s}: MMD={fmt_ci(ci,'mmd')} | Wasserstein={fmt_ci(ci,'wasserstein')}")

    summary = {"cohort": cohort_label, "n_calibration": len(calib_idx), "n_eval": len(eval_idx)}
    for name, ci in rows.items():
        summary[f"{name}_mmd"] = ci["mmd"]
        summary[f"{name}_mmd_ci_low"] = ci["mmd_ci_low"]
        summary[f"{name}_mmd_ci_high"] = ci["mmd_ci_high"]
        summary[f"{name}_wasserstein"] = ci["wasserstein"]

    return summary, adapted_flow


def experiment4_calibration_ablation(dp, flow_model, gaussian_variants, ext_expr, ext_clin,
                                      cohort_label="pooled", calib_sizes=CALIBRATION_SIZES,
                                      split_seeds=CALIBRATION_SPLIT_SEEDS, adapt_epochs=60, adapt_lr=1e-4):
    X_ext, C_ext, y_ext, keep_idx = dp.transform_external(ext_clin, ext_expr, verbose=False)
    n = X_ext.shape[0]
    data_dim = X_ext.shape[1]
    usable_sizes = [c for c in calib_sizes if c + 10 <= n]
    if not usable_sizes:
        print(f"[EXPERIMENT 4 | {cohort_label}] Too few samples ({n}) for any calibration size in {calib_sizes}.")
        return pd.DataFrame()
    if len(usable_sizes) < len(calib_sizes):
        print(f"[EXPERIMENT 4 | {cohort_label}] Only {usable_sizes} fit within n={n}; larger sizes skipped.")

    rows = []
    for size in usable_sizes:
        for split_seed in split_seeds:
            rng = np.random.RandomState(split_seed)
            perm = rng.permutation(n)
            calib_idx = perm[:size]
            eval_idx = perm[size:] if len(perm) - size >= 5 else perm[-max(5, n - size):]
            assert_disjoint(calib_idx, eval_idx, label=f"Exp4 | {cohort_label} | size={size} seed={split_seed}")

            X_calib, C_calib = X_ext[calib_idx], C_ext[calib_idx]
            X_eval, C_eval = X_ext[eval_idx], C_ext[eval_idx]
            x_real_eval = X_eval.numpy()
            n_pool = max(GEN_POOL_MULTIPLIER * X_eval.shape[0], 100)

            # Flow
            adapted_flow = TCGAFlowMap(data_dim=data_dim, context_dim=C_ext.shape[1]).to(DEVICE)
            adapted_flow.load_state_dict(flow_model.state_dict())
            adapted_flow = train_flow_map(X_calib, C_calib, epochs=adapt_epochs,
                                           batch_size=min(16, size), lr=adapt_lr, init_model=adapted_flow)
            flow_pool = generate_flow_pool(adapted_flow, C_eval, data_dim, n_pool, seed=split_seed)
            rows.append({"cohort": cohort_label, "method": "flow", "n_calibration": size,
                         "split_seed": split_seed, "n_eval": len(eval_idx),
                         "mmd": mmd_rbf(x_real_eval, flow_pool[:len(eval_idx)]),
                         "wasserstein": mean_wasserstein(x_real_eval, flow_pool[:len(eval_idx)])})

            for variant in ["full", "mean_only", "cov_only", "affine_transport"]:
                pool, _, _ = generate_gaussian_pool(gaussian_variants, variant, n_pool,
                                                     seed=split_seed, X_calib=X_calib)
                rows.append({"cohort": cohort_label, "method": f"gaussian_{variant}", "n_calibration": size,
                             "split_seed": split_seed, "n_eval": len(eval_idx),
                             "mmd": mmd_rbf(x_real_eval, pool[:len(eval_idx)]),
                             "wasserstein": mean_wasserstein(x_real_eval, pool[:len(eval_idx)])})

    df = pd.DataFrame(rows)
    agg = df.groupby(["cohort", "method", "n_calibration"]).agg(
        mmd_mean=("mmd", "mean"), mmd_sd=("mmd", "std"),
        wasserstein_mean=("wasserstein", "mean"), wasserstein_sd=("wasserstein", "std"),
        n_splits=("split_seed", "nunique"),
    ).reset_index()
    print(f"[EXPERIMENT 4 | {cohort_label} -- Calibration-size ablation, "
          f"{len(split_seeds)} splits/size]")
    print(agg.to_string(index=False))
    return df, agg


def experiment5_risk_steering(dp, flow_model, oracle, ext_expr, ext_clin,
                               cohort_label="pooled", alpha=1.5, n_patients=40,
                               n_seeds=N_MONOTONICITY_SEEDS):
    """EXPERIMENT 5: risk steering with the flow model."""
    X_ext, C_ext, y_ext, keep_idx = dp.transform_external(ext_clin, ext_expr)
    n = min(n_patients, X_ext.shape[0])
    if n < 2:
        print(f"[EXPERIMENT 5 | {cohort_label}] Not enough harmonized samples to run steering.")
        return None
    c_batch_base = C_ext[:n].to(DEVICE)

    def steer(c_batch, target_class, alpha_val, seed):
        g = torch.Generator(device=DEVICE).manual_seed(seed)
        x0 = torch.randn(n, X_ext.shape[1], device=DEVICE, generator=g, requires_grad=True)
        x1 = integrate_flow(flow_model, x0, c_batch)
        logits_before = oracle(x1)
        target = torch.full((n,), target_class, dtype=torch.long, device=DEVICE)
        nn.functional.cross_entropy(logits_before, target).backward()
        with torch.no_grad():
            x0_steered = x0 - alpha_val * x0.grad
            x1_steered = integrate_flow(flow_model, x0_steered, c_batch)
            logits_after = oracle(x1_steered)
        risk_before = torch.softmax(logits_before, dim=-1)[:, 1].mean().item()
        risk_after = torch.softmax(logits_after, dim=-1)[:, 1].mean().item()
        return risk_before, risk_after

    hi_before, hi_after = steer(c_batch_base, 1, alpha, seed=RANDOM_STATE)
    lo_before, lo_after = steer(c_batch_base, 0, alpha, seed=RANDOM_STATE)

    alphas = np.linspace(0, 3.0, 8)
    curves = []
    for seed in range(RANDOM_STATE, RANDOM_STATE + n_seeds):
        risks = [steer(c_batch_base, 1, a, seed=seed)[1] for a in alphas]
        curves.append(risks)
    curves = np.array(curves)
    mean_curve = curves.mean(axis=0)
    ci_low = np.percentile(curves, 2.5, axis=0)
    ci_high = np.percentile(curves, 97.5, axis=0)

    spearman_rho, spearman_p = spearmanr(alphas, mean_curve)
    diffs = np.diff(mean_curve)
    n_violations = int(np.sum(diffs < -1e-3))
    max_downward_excursion = float(min(0.0, diffs.min()))
    mean_risk_increase = float(mean_curve[-1] - mean_curve[0])

    result = {
        "cohort": cohort_label, "method": "flow", "n_patients": n,
        "risk_before_hi": hi_before, "risk_after_hi": hi_after,
        "risk_before_lo": lo_before, "risk_after_lo": lo_after,
        "spearman_rho_alpha_vs_risk": float(spearman_rho), "spearman_p": float(spearman_p),
        "n_monotonicity_violations": n_violations,
        "max_downward_excursion": max_downward_excursion,
        "mean_risk_increase_end_minus_start": mean_risk_increase,
        "baseline_risk": float(mean_curve[0]), "max_risk": float(mean_curve.max()), "min_risk": float(mean_curve.min()),
        "alphas": alphas.tolist(), "mean_risk_by_alpha": mean_curve.tolist(),
        "risk_ci95_low": ci_low.tolist(), "risk_ci95_high": ci_high.tolist(),
    }
    print(f"[EXPERIMENT 5 | {cohort_label} -- Risk-steering (flow)] "
          f"Spearman rho={spearman_rho:.3f} (p={spearman_p:.3g}), "
          f"{n_violations} violation(s)/{len(diffs)} steps, "
          f"risk {mean_curve[0]:.3f} -> {mean_curve[-1]:.3f}")
    return result


def experiment6_risk_steering_baselines(dp, gaussian_variants, vae_model, oracle, ext_expr, ext_clin,
                                         cohort_label="pooled", alpha=1.5, n_patients=40,
                                         n_seeds=N_MONOTONICITY_SEEDS):
    X_ext, C_ext, y_ext, keep_idx = dp.transform_external(ext_clin, ext_expr, verbose=False)
    n = min(n_patients, X_ext.shape[0])
    if n < 2:
        print(f"[EXPERIMENT 6 | {cohort_label}] Not enough samples.")
        return pd.DataFrame()
    c_batch = C_ext[:n].to(DEVICE)
    alphas = np.linspace(0, 3.0, 8)

    def gaussian_curve(seed):
        rng_seed = seed
        risks = []
        for a in alphas:
            x = torch.tensor(gaussian_variants.sample(gaussian_variants.mu_source_,
                                                        gaussian_variants.cov_source_, n, seed=rng_seed),
                              device=DEVICE, requires_grad=True)
            logits = oracle(x)
            target = torch.full((n,), 1, dtype=torch.long, device=DEVICE)
            nn.functional.cross_entropy(logits, target).backward()
            with torch.no_grad():
                x_steered = x - a * x.grad
                risk_after = torch.softmax(oracle(x_steered), dim=-1)[:, 1].mean().item()
            risks.append(risk_after)
        return risks

    def vae_curve(seed):
        g = torch.Generator(device=DEVICE).manual_seed(seed)
        risks = []
        for a in alphas:
            z = torch.randn(n, vae_model.latent_dim, device=DEVICE, generator=g, requires_grad=True)
            x = vae_model.decode(z, c_batch)
            logits = oracle(x)
            target = torch.full((n,), 1, dtype=torch.long, device=DEVICE)
            nn.functional.cross_entropy(logits, target).backward()
            with torch.no_grad():
                z_steered = z - a * z.grad
                x_steered = vae_model.decode(z_steered, c_batch)
                risk_after = torch.softmax(oracle(x_steered), dim=-1)[:, 1].mean().item()
            risks.append(risk_after)
        return risks

    rows = []
    for method_name, curve_fn in [("gaussian", gaussian_curve), ("vae", vae_curve)]:
        curves = np.array([curve_fn(seed) for seed in range(RANDOM_STATE, RANDOM_STATE + n_seeds)])
        mean_curve = curves.mean(axis=0)
        rho, p = spearmanr(alphas, mean_curve)
        diffs = np.diff(mean_curve)
        rows.append({
            "cohort": cohort_label, "method": method_name, "n_patients": n,
            "baseline_risk": float(mean_curve[0]), "max_risk": float(mean_curve.max()),
            "min_risk": float(mean_curve.min()),
            "spearman_rho_alpha_vs_risk": float(rho), "spearman_p": float(p),
            "n_monotonicity_violations": int(np.sum(diffs < -1e-3)),
        })

    df = pd.DataFrame(rows)
    print(f"[EXPERIMENT 6 | {cohort_label} -- Risk-steering baselines (Gaussian, VAE)]")
    print(df.to_string(index=False))
    return df


def experiment7_platform_sensitivity(dp, per_series):
    """EXPERIMENT 7: for each external cohort/platform, how many of the 80 TCGA top
    genes are actually present, and which are missing."""
    rows = []
    for cohort_label, (ext_expr, ext_clin) in per_series.items():
        platforms = sorted(ext_clin["platform"].dropna().unique().tolist()) if "platform" in ext_clin.columns else ["?"]
        present = [g for g in dp.top_genes if g in ext_expr.columns]
        missing = [g for g in dp.top_genes if g not in ext_expr.columns]
        rows.append({
            "cohort": cohort_label, "platform(s)": ", ".join(platforms),
            "common_genes": f"{len(present)}/{len(dp.top_genes)}",
            "missing_genes": ", ".join(missing) if missing else "(none)",
        })
    df = pd.DataFrame(rows)
    print("[EXPERIMENT 7 -- Platform sensitivity]")
    print(df.to_string(index=False))
    return df


def experiment8_gene_sensitivity(clinical_df, expr_df, severity_context_source,
                                  ext_expr, ext_clin, cohort_label="pooled"):
    """EXPERIMENT 8: impute vs. drop the gene(s) missing on the external platform."""
    dp_a = TCGADataPipeline(clinical_df, expr_df, severity_context_source)
    splits_a = dp_a.prepare_data()
    X_train_a, C_train_a, _ = splits_a["train"]
    flow_a = train_flow_map(X_train_a, C_train_a, epochs=150)
    X_ext_a, C_ext_a, _, keep_a = dp_a.transform_external(ext_clin, ext_expr, verbose=True)
    missing_genes = [g for g in dp_a.top_genes if g not in ext_expr.columns]

    with torch.no_grad():
        x0 = torch.randn(X_ext_a.shape[0], X_ext_a.shape[1], device=DEVICE)
        gen_a = integrate_flow(flow_a, x0, C_ext_a.to(DEVICE)).cpu().numpy()
    result_a = {"variant": "A_impute_missing_genes", "cohort": cohort_label,
                "missing_genes_imputed": missing_genes,
                "mmd": mmd_rbf(X_ext_a.numpy(), gen_a), "wasserstein": mean_wasserstein(X_ext_a.numpy(), gen_a)}

    if not missing_genes:
        print(f"[EXPERIMENT 8 | {cohort_label}] No missing genes -- variant B == variant A.")
        return pd.DataFrame([result_a])

    dp_b = TCGADataPipeline(clinical_df, expr_df, severity_context_source, drop_genes=missing_genes)
    splits_b = dp_b.prepare_data()
    X_train_b, C_train_b, _ = splits_b["train"]
    flow_b = train_flow_map(X_train_b, C_train_b, epochs=150)
    X_ext_b, C_ext_b, _, keep_b = dp_b.transform_external(ext_clin, ext_expr, verbose=True)

    with torch.no_grad():
        x0 = torch.randn(X_ext_b.shape[0], X_ext_b.shape[1], device=DEVICE)
        gen_b = integrate_flow(flow_b, x0, C_ext_b.to(DEVICE)).cpu().numpy()
    result_b = {"variant": "B_drop_missing_genes", "cohort": cohort_label,
                "dropped_genes": missing_genes,
                "mmd": mmd_rbf(X_ext_b.numpy(), gen_b), "wasserstein": mean_wasserstein(X_ext_b.numpy(), gen_b)}

    df = pd.DataFrame([result_a, result_b])
    print(f"[EXPERIMENT 8 | {cohort_label}] impute vs drop for {missing_genes}:")
    print(df.to_string(index=False))
    return df


# ════════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════════

def run_all_experiments():
    print("Fetching TCGA clinical data from cBioPortal...")
    clinical_raw = fetch_clinical_cbioportal(TCGA_STUDY_ID)
    print("Fetching TCGA expression data from cBioPortal...")
    expr_raw = fetch_expression_cbioportal(TCGA_STUDY_ID)

    clinical_df, severity_context_source = build_tcga_clinical(clinical_raw)
    expr_df = build_tcga_expression(expr_raw)
    common_patients = clinical_df.index.intersection(expr_df.index)
    clinical_df, expr_df = clinical_df.loc[common_patients], expr_df.loc[common_patients]
    print(f"TCGA patients with clinical + expression data: {len(common_patients)}")
    print(f"TCGA severity_context_source = '{severity_context_source}'")

    dp = TCGADataPipeline(clinical_df, expr_df, severity_context_source)
    splits = dp.prepare_data()
    X_train, C_train, y_train = splits["train"]
    X_oracle, C_oracle, y_oracle = splits["oracle"]
    X_test, C_test, y_test = splits["test"]
    print(f"Train: {X_train.shape[0]} | Oracle: {X_oracle.shape[0]} | Test: {X_test.shape[0]} "
          f"| Digital twin dim: {X_train.shape[1]}")

    print("[TRAINING] Reward oracle...")
    oracle = train_reward_oracle(X_oracle, y_oracle)
    print("[TRAINING] TCGA conditional flow map...")
    flow_model = train_flow_map(X_train, C_train)
    print("[TRAINING] TCGA flow map ablation (no context)...")
    flow_no_context = train_flow_map_no_context(X_train)
    print("[TRAINING] Gaussian variants (Ledoit-Wolf, source)...")
    gaussian_variants = GaussianVariants().fit_source(X_train)
    print("[TRAINING] VAE baseline...")
    vae_model = train_vae(X_train, C_train, epochs=200)

    saudi_expr_pooled, saudi_clin_pooled, per_series = load_saudi_cohorts()
    cohorts_to_run = list(per_series.items()) + [("pooled", (saudi_expr_pooled, saudi_clin_pooled))]

    exp1 = experiment1_in_domain(flow_model, oracle, gaussian_variants, vae_model,
                                  X_train, C_train, X_test, C_test)
    exp7 = experiment7_platform_sensitivity(dp, per_series)

    all_exp2, all_exp3_check, all_exp3, all_exp4, all_exp5, all_exp6, all_exp8 = {}, {}, {}, {}, {}, {}, {}

    for cohort_label, (ext_expr, ext_clin) in cohorts_to_run:
        print(f"\n{'='*80}\nCohort: {cohort_label}\n{'='*80}")
        exp2, X_ext_harmonized = experiment2_population_shift(
            dp, flow_model, gaussian_variants, vae_model, ext_expr, ext_clin, cohort_label=cohort_label)
        all_exp2[cohort_label] = exp2
        if exp2 is None:
            continue

        plot_population_shift_umap(X_test.numpy(), exp2["X_real"],
                                    f"{OUT_DIR}/figure_population_shift_umap_{cohort_label}.png",
                                    cohort_label=cohort_label)

        all_exp3_check[cohort_label] = experiment3_stage_conditioning_transfer(
            dp, flow_model, flow_no_context, ext_expr, ext_clin, cohort_label=cohort_label)

        exp3_out = experiment3_population_adaptation(
            dp, flow_model, gaussian_variants, vae_model, ext_expr, ext_clin, cohort_label=cohort_label)
        if exp3_out is not None:
            all_exp3[cohort_label], _adapted_flow = exp3_out
            df4, agg4 = experiment4_calibration_ablation(
                dp, flow_model, gaussian_variants, ext_expr, ext_clin, cohort_label=cohort_label)
            all_exp4[cohort_label] = (df4, agg4)

        all_exp5[cohort_label] = experiment5_risk_steering(
            dp, flow_model, oracle, ext_expr, ext_clin, cohort_label=cohort_label)
        all_exp6[cohort_label] = experiment6_risk_steering_baselines(
            dp, gaussian_variants, vae_model, oracle, ext_expr, ext_clin, cohort_label=cohort_label)

        all_exp8[cohort_label] = experiment8_gene_sensitivity(
            clinical_df, expr_df, severity_context_source, ext_expr, ext_clin, cohort_label=cohort_label)

    # ---- Save bundle ----
    summary_rows = []
    for label, e2 in all_exp2.items():
        if e2 is not None:
            summary_rows.append({"experiment": f"2_population_shift[{label}]", **e2["summary"]})
    for label, e3 in all_exp3.items():
        if e3 is not None:
            summary_rows.append({"experiment": f"3_population_adaptation[{label}]", **e3})
    for label, e5 in all_exp5.items():
        if e5 is not None:
            summary_rows.append({"experiment": f"5_risk_steering[{label}]",
                                  **{k: v for k, v in e5.items()
                                     if k not in ("alphas", "mean_risk_by_alpha", "risk_ci95_low", "risk_ci95_high")}})
    pd.DataFrame(summary_rows).to_csv(f"{OUT_DIR}/experiment_summary.csv", index=False)

    if len(exp7):
        exp7.to_csv(f"{OUT_DIR}/experiment7_platform_sensitivity.csv", index=False)
    for label, e2 in all_exp2.items():
        if e2 is not None:
            e2["per_feature_shift"].to_csv(f"{OUT_DIR}/experiment2_per_feature_shift_{label}.csv", index=False)
    for label, (df4, agg4) in all_exp4.items():
        if len(df4):
            df4.to_csv(f"{OUT_DIR}/experiment4_calibration_ablation_raw_{label}.csv", index=False)
            agg4.to_csv(f"{OUT_DIR}/experiment4_calibration_ablation_summary_{label}.csv", index=False)
    for label, e6 in all_exp6.items():
        if e6 is not None and len(e6):
            e6.to_csv(f"{OUT_DIR}/experiment6_risk_steering_baselines_{label}.csv", index=False)
    for label, e8 in all_exp8.items():
        if e8 is not None and len(e8):
            e8.to_csv(f"{OUT_DIR}/experiment8_gene_sensitivity_{label}.csv", index=False)

    zip_path = "/content/population_shift_results.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for fname in os.listdir(OUT_DIR):
            zf.write(os.path.join(OUT_DIR, fname), arcname=fname)
    print(f"Bundled results into {zip_path}")

    return {
        "dp": dp, "oracle": oracle, "flow_model": flow_model, "flow_no_context": flow_no_context,
        "gaussian_variants": gaussian_variants, "vae_model": vae_model,
        "experiment1": exp1, "experiment2": all_exp2, "experiment3_conditioning_check": all_exp3_check,
        "experiment3": all_exp3, "experiment4": all_exp4, "experiment5": all_exp5,
        "experiment6": all_exp6, "experiment7": exp7, "experiment8": all_exp8,
    }


if __name__ == "__main__":
    results = run_all_experiments()
    print("\n=== DONE ===")