# Vehicle CAN Network Intrusion Detection System (IDS)

## 🚗 Why This Project?

Modern vehicles rely heavily on the Controller Area Network (CAN bus) to enable communication between Electronic Control Units (ECUs). However, the CAN bus protocol was **not originally designed with built-in security**, which leaves it vulnerable to attacks such as **spoofing, Denial-of-Service (DoS), and replay attacks**.

This project demonstrates how **machine learning and deep learning techniques** can be applied to detect such intrusions in real time, thereby enhancing vehicle cybersecurity.

---

## 🔍 Project Overview

- Simulated various CAN bus attacks including **spoofing, DoS, and replay attacks**, and collected realistic CAN log data.
- Performed data processing and cleaning on the CAN datasets to prepare them for machine learning training.
- Extracted relevant features using a **window-based parameter extraction** method for efficient real-time intrusion detection.
- Trained and compared multiple models including:
  - Traditional ML: **Random Forest, Extra Trees, LightGBM, XGBoost**
  - Deep Learning: **CNN, DNN, LSTM**
- Evaluated model performance using metrics such as **precision, recall, and F1-score**.
- Provided trained model files (`.joblib`, `.pkl`, `.h5`) for reuse and further experimentation.

---

## 📂 Repository Structure

Vehicle-CAN-IDS/
│
├── csv_files/           # Raw and processed CAN bus datasets
├── dnn_tuning/          # DNN model tuning experiments
├── dnn_tuning_v2/       # Updated DNN tuning experiments
├── ipynb/               # Jupyter notebooks for attack simulation and model training
├── models/              # Saved trained models
├── docs/                # Project images, flowcharts, and documentation
├── parsed_can_log.csv   # Sample parsed CAN log data
├── README.md            # This readme file


---

## 🧠 Key Notebooks

- `dos.ipynb` — DoS attack simulation and detection workflow.
- `experiment1.ipynb` — Baseline machine learning 1st experiment(Reseach papers approach).
- `experiment2.ipynb` — Baseline machine learning 2nd experiment (windowing approach which is also the novelty of our project).
- `fuzzy.ipynb`, `gear.ipynb`, `RPM.ipynb` — Additional CAN bus scenarios.
- `cnn_tuning/`, `dnn_tuning/` — Deep learning model tuning and experiments.

---

## 🏷️ Tech Stack

- **Languages & Libraries:** Python, Pandas, NumPy, scikit-learn, TensorFlow  
- **Hardware:** ESP32 CAN setup (hardware code not included in this repository)  
- **Environment:** Jupyter Notebook for simulation, training, and experimentation

---
## 🎥 Project Video Explanation

You can watch a detailed **video explanation** of the Vehicle CAN Network Intrusion Detection System project here:

[https://youtu.be/SFmveqE85Rk](https://youtu.be/SFmveqE85Rk)

