# SPASHTA: Speech Posthoc Attribution with Salient Highlighting for Transparent Analysis

This repository contains the official implementation of **SPASHTA**, a model-agnostic posthoc explainability framework for Speech Emotion Recognition (SER). SPASHTA estimates frame-wise acoustic attribution by measuring prediction confidence degradation under localized temporal sliding-window perturbations. It reconstructs the most expressive, salient audio segments into a listenable waveform ($y_{\text{salient}}$), rendering explanations directly inspectable by human listeners and domain experts.

---

## Repository Structure

```
├── app.py                            # Interactive Streamlit Web Application
│
├── original_model.py                 # CNN-BiGRU-Attention architecture
├── original_utils.py                 # 374-D acoustic feature extraction (MFCC, Mel, Chroma, Pitch, Energy)
├── original_train.py                 # Training script for CNN-BiGRU model
│
├── cnn_lstm_model.py                 # Hybrid 1D-CNN-BiLSTM architecture
├── cnn_lstm_features.py              # Sequential feature extraction (161 x 125)
├── cnn_lstm_preprocess.py            # Speech enhancement pipeline (Kalman -> NLMS -> Wiener filter)
├── cnn_lstm_train.py                 # Training script for Hybrid CNN-LSTM model
│
├── se_resnet_model.py                # SE-ResNet-BiGRU-Attention architecture
├── se_resnet_features.py             # 3-channel spectrogram feature extraction (3 x 64 x 501)
├── se_resnet_train.py                # Training script for SE-ResNet model
│
├── benchmark_explainers.py           # Quantitative XAI benchmark (SPASHTA vs. LIME vs. SHAP)
├── explainers_stability_analysis.py # Noise stability & reproducibility analyzer (Cosine & Pearson)
├── validate_spashta_models.py        # Cross-architecture model validation suite
├── spashta_threshold_analysis.py     # Threshold ratio (tau) sensitivity grid search
├── lime_explain_cnn_lstm.py          # Standalone LIME audio diagnostic runner
├── shap_explain_cnn_lstm.py          # Standalone GradientSHAP audio diagnostic runner
│
├── requirements_sparsha.txt          # Python package dependencies
├── LICENSE                           # Apache 2.0 License
└── README.md                         # Project documentation
```

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/durgesh088/SPASHTA-SER.git
cd SPASHTA-SER
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements_sparsha.txt
```

> **Note**: Ensure the pre-trained weight checkpoints (`.pt`), scalers (`.pkl`), and label encoders (`.pkl`) are located in the repository root directory before running the application or benchmark scripts.

---

## Pre-trained Checkpoints & Dataset Download

Due to file size constraints, trained model weights (`.pt`), scaler artifacts (`.pkl`), label encoders (`.pkl`), and audio dataset files are hosted on Google Drive:

📥 **[Download Model Checkpoints, Scalers & Dataset (Google Drive)](https://drive.google.com/drive/folders/1l7BRMD9U4baM0aQhWN5xCYxQTFVqhwW8)**

### Required Files in Drive:
* **Model Checkpoints**: `original_best_model.pt`, `cnn_lstm_best_model.pt`, `se_resnet_best_model.pt`
* **Scalers & Encoders**: `original_scaler.pkl`, `original_label_encoder.pkl`, `cnn_lstm_scaler.pkl`, `cnn_lstm_label_encoder.pkl`, `se_resnet_scaler.pkl`, `se_resnet_label_encoder.pkl`
* **Dataset Directory**: `hindi_dataset/`

*After downloading, place all `.pt` and `.pkl` files directly into the project root directory.*

---

## Usage

###  Running the Streamlit Web Application

Launch the interactive dashboard locally:
```bash
python -m streamlit run app.py
```

Available modes in the web dashboard:
1. **Upload File**: Upload custom `.wav` speech samples for instant emotion prediction, visual waveform/spectrogram rendering, and SPASHTA salience analysis.
2. **Record Audio**: Capture live speech directly from your microphone and view real-time emotion predictions & listenable explanations.
3. **Run Test Dataset**: Batch evaluate samples from `hindi_dataset/`, generate confusion matrix heatmaps, compute classification accuracy, and view computational runtime overhead statistics.

---

### 🔬 Running Research & Benchmarking Scripts

To reproduce the experimental results:

```bash
# 1. Comparative XAI Benchmark (SPASHTA vs. LIME vs. SHAP)
python benchmark_explainers.py

# 2. Explanation Stability & Robustness Analysis (5 Independent Stochastic Noise Runs)
python explainers_stability_analysis.py

# 3. Model-Agnostic Validation (Evaluating SPASHTA Across All 3 Architectures)
python validate_spashta_models.py

# 4. Threshold Sensitivity Grid Search (Window size and tau optimization)
python spashta_threshold_analysis.py
```

---

## Model Architectures & Preprocessing

| Architecture | Input Dimensions | Feature Type | Preprocessing Filters |
| :--- | :--- | :--- | :--- |
| **CNN-BiGRU-Attention** | $1 \times 374$ | Pooled MFCCs, Mel, Chroma, Pitch, Energy | Standard Scaling |
| **Hybrid 1D-CNN-BiLSTM** | $161 \times 125$ | Sequential Mel Spectrogram Matrices | Kalman $\rightarrow$ NLMS $\rightarrow$ Wiener Filter Cascade |
| **SE-ResNet-BiGRU-Attention** | $3 \times 64 \times 501$ | Log-Mel + 1st Order Delta + 2nd Order Delta | Spectrogram Channel-wise Scaler |

---

## License

This project is licensed under the **Apache 2.0 License**. See the [LICENSE](LICENSE) file for details.
