# 🔬 Cancer Diagnosis Predictor – KNN

A [Streamlit](https://streamlit.io/) web application that predicts cancer diagnosis using a **K-Nearest Neighbours** (KNN) classifier trained on patient health data.

---

## 📂 Project Structure

```
.
├── KNN (2).ipynb          # Original Jupyter notebook
├── cancer_data.csv        # Training dataset (required)
├── generate_model.py      # Trains the model and saves pickle files
├── app.py                 # Streamlit web application
├── knn_model.pkl          # Saved KNN model  (generated)
├── scaler.pkl             # Saved StandardScaler (generated)
├── requirements.txt       # Python dependencies
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add the dataset
Place **`cancer_data.csv`** in the project root directory.  
The CSV must contain these columns:

| Column | Type | Description |
|---|---|---|
| `age` | int | Patient age (dropped during training) |
| `gender` | int | 0 = Female, 1 = Male (dropped during training) |
| `bmi` | float | Body Mass Index |
| `smoking` | int | 0 = No, 1 = Yes |
| `genetic_risk` | int | 0 = Low, 1 = Medium, 2 = High |
| `physical_activity` | float | Hours of activity per week |
| `alcohol_intake` | float | Units of alcohol per week |
| `cancer_history` | int | 0 = No, 1 = Yes |
| `diagnosis` | int | **Target** — 0 = No Cancer, 1 = Cancer |

---

## 🚀 Running the App

### Step 1 – Generate the model pickle files
> Only needs to be done **once**.

```bash
python generate_model.py
```

This will print the model accuracy (~80%) and create:
- `knn_model.pkl`
- `scaler.pkl`

### Step 2 – Launch the Streamlit app

```bash
streamlit run app.py
```

The app will open automatically at **http://localhost:8501**.

---

## 🧠 Model Details

| Parameter | Value |
|---|---|
| Algorithm | K-Nearest Neighbours |
| `n_neighbors` | 5 |
| Scaler | StandardScaler |
| Train/Test split | 80 / 20 |
| Test Accuracy | ≈ 80 % |

**Features used for prediction:**

- BMI
- Smoking (Yes / No)
- Genetic Risk (Low / Medium / High)
- Physical Activity (hrs/week)
- Alcohol Intake (units/week)
- Cancer History (Yes / No)

---

## 📝 License
This project is for educational purposes.
