# Machinery Maintenance Monitoring

Remaining useful life (RUL) prediction on the **NASA C-MAPSS** turbofan degradation dataset using a **PyTorch Transformer** and a preprocessing pipeline aligned with the reference paper in this repo (`Base Paper`). The main experiment notebook is [`MIOM.ipynb`](MIOM.ipynb).

**Repository:** [https://github.com/hrishavmba2027-byte/Machinery_Maintainance_Monitoring](https://github.com/hrishavmba2027-byte/Machinery_Maintainance_Monitoring.git)

---

## What this project does

- Loads C-MAPSS **`train_FD00X.txt`**, **`test_FD00X.txt`**, and **`RUL_FD00X.txt`** from `Data/raw/`.
- Keeps **14 informative sensors** (drops near-constant channels: s1, s5, s6, s10, s16, s18, s19).
- Applies **EWMA smoothing**, then fits **`MinMaxScaler`** on training features to **[-1, 1]** (per paper; not `StandardScaler`).
- Builds **training labels** with a **piecewise-linear RUL** capped at `RUL_CEILING` (default 125).
- Builds the **test evaluation set** as the **last cycle per engine**, with ground-truth RUL from **`RUL_FD00X.txt`** via a **`unit → RUL` lookup** (required when unit IDs are non-standard or namespaced).
- Trains a **Transformer** on **rolling windows** of length **`SEQ_LEN`** (default 30 for FD001-style settings).
- Reports **RMSE**, **asymmetric C-MAPSS score**, MAE, bias; saves **figures** and a **checkpoint** under `checkpoints/`.

Supported dataset variants: **`FD001`–`FD004`**. The notebook is set up to train **one subset at a time** by changing **`ACTIVE_SUBSET`**. The loader also supports optional **`unit_prefix`** renaming so multiple subsets could be combined without engine-ID collisions (see commented cells in `MIOM.ipynb` if you enable that workflow).

---

## Data layout

Place official C-MAPSS text files here:

```
Data/raw/
  train_FD001.txt  test_FD001.txt  RUL_FD001.txt
  train_FD002.txt  test_FD002.txt  RUL_FD002.txt
  ...
```

Processed artifacts (after running the notebook) may appear under `Data/processed/` and related folders.

---

## Environment

Python **3.10+** recommended (with **`torch`** matching your CUDA / CPU / Apple Silicon setup).

Install dependencies:

```bash
pip install -r requirements.txt
```

[`requirements.txt`](requirements.txt) includes: `numpy`, `pandas`, `scikit-learn`, `torch`, `matplotlib`.  


---

## How to run

1. Clone the repo and add C-MAPSS data under `Data/raw/` as above.

   ```bash
   git clone https://github.com/hrishavmba2027-byte/Machinery_Maintainance_Monitoring.git
   cd Machinery_Maintainance_Monitoring
   ```

2. Create a virtual environment (optional) and install requirements.

3. Open **`MIOM.ipynb`** in Jupyter, VS Code, or Cursor and run cells **top to bottom**.

4. In the first configuration cell, set **`ACTIVE_SUBSET`** to **`FD001`**, **`FD002`**, **`FD003`**, or **`FD004`** (and adjust **`SEQ_LEN`** / hyperparameters if you follow per-subset settings from the paper).

**Outputs (typical):**

- Processed CSVs and scaler under `Data/processed/` (paths depend on `ACTIVE_SUBSET`).
- `checkpoints/transformer_<SUBSET>.pt` — model + training metadata.
- `results_<SUBSET>.png` — evaluation plots.

---

## Key configuration (notebook)

| Setting | Role |
|--------|------|
| `DATA_DIR` | Raw data root (`Data/raw`) |
| `ACTIVE_SUBSET` | Which FD00X file group to train on |
| `SEQ_LEN` | Window length for the Transformer |
| `RUL_CEILING` | Piecewise RUL cap (e.g. 125) |
| `BATCH_SIZE`, `EPOCHS`, `LR`, `PATIENCE` | Training loop |

Device is selected automatically: **CUDA → MPS → CPU**.

---

## Reference

C-MAPSS is a standard benchmark for predictive maintenance; methodology and sensor choice in `MIOM.ipynb` follow the paper included in this repository as **`Base Paper`**.

---

## License / data

C-MAPSS data is distributed by NASA; ensure you comply with their usage terms when obtaining and sharing the raw files.
