import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

try:
    import xgboost as xgb
except ImportError:
    xgb = None

FEATURE_NAMES = [
    "ejection_fraction",
    "heart_rate_bpm",
    "irregularity_index",
    "sdnn_ms",
    "rmssd_ms",
    "pnn50",
    "lf_power",
    "hf_power",
    "lf_hf_ratio",
    "sd1",
    "sd2",
    "sd_ratio",
    "dyssynchrony_index_ms",
    "motion_irregularity",
    "hr_x_dyssynchrony"
]

def generate_patient_features(rhythm_type: str, n_samples: int = 200) -> np.ndarray:
    """Generates synthetic patient feature vectors with typical physiological distributions and noise."""
    np.random.seed(42)
    features = []
    
    for _ in range(n_samples):
        if rhythm_type == "nsr":
            ef = np.random.normal(58.0, 5.0)
            hr = np.random.normal(75.0, 8.0)
            irreg = np.random.uniform(0.01, 0.06)
            sdnn = np.random.normal(45.0, 10.0)
            rmssd = np.random.normal(30.0, 8.0)
            pnn50 = np.random.normal(12.0, 5.0)
            lf = np.random.normal(450.0, 100.0)
            hf = np.random.normal(250.0, 60.0)
            lf_hf = lf / hf
            sd1 = np.random.normal(22.0, 5.0)
            sd2 = np.random.normal(60.0, 12.0)
            sd_ratio = sd1 / sd2
            dys = np.random.normal(35.0, 10.0)
            mot_irreg = np.random.normal(0.14, 0.03)
            
        elif rhythm_type == "afib":
            ef = np.random.normal(48.0, 8.0)
            hr = np.random.normal(110.0, 20.0)
            irreg = np.random.uniform(0.18, 0.35)
            sdnn = np.random.normal(115.0, 20.0)
            rmssd = np.random.normal(90.0, 18.0)
            pnn50 = np.random.normal(35.0, 8.0)
            lf = np.random.normal(150.0, 50.0)
            lf_hf = np.random.uniform(0.15, 0.50)
            hf = lf / lf_hf
            sd1 = np.random.normal(65.0, 12.0)
            sd2 = np.random.normal(85.0, 15.0)
            sd_ratio = sd1 / sd2
            dys = np.random.normal(115.0, 20.0)
            mot_irreg = np.random.normal(0.42, 0.08)
            
        elif rhythm_type == "pvc":
            ef = np.random.normal(52.0, 6.0)
            hr = np.random.normal(74.0, 10.0)
            irreg = np.random.uniform(0.08, 0.14)
            sdnn = np.random.normal(75.0, 15.0)
            rmssd = np.random.normal(70.0, 12.0)
            pnn50 = np.random.normal(22.0, 6.0)
            lf = np.random.normal(350.0, 80.0)
            hf = np.random.normal(280.0, 70.0)
            lf_hf = lf / hf
            sd1 = np.random.normal(48.0, 8.0)
            sd2 = np.random.normal(80.0, 15.0)
            sd_ratio = sd1 / sd2
            dys = np.random.normal(130.0, 25.0)
            mot_irreg = np.random.normal(0.35, 0.06)
            
        elif rhythm_type == "bradycardia":
            ef = np.random.normal(56.0, 5.0)
            hr = np.random.normal(48.0, 5.0)
            irreg = np.random.uniform(0.01, 0.06)
            sdnn = np.random.normal(48.0, 10.0)
            rmssd = np.random.normal(32.0, 8.0)
            pnn50 = np.random.normal(13.0, 5.0)
            lf = np.random.normal(480.0, 100.0)
            hf = np.random.normal(260.0, 60.0)
            lf_hf = lf / hf
            sd1 = np.random.normal(23.0, 5.0)
            sd2 = np.random.normal(63.0, 12.0)
            sd_ratio = sd1 / sd2
            dys = np.random.normal(35.0, 10.0)
            mot_irreg = np.random.normal(0.14, 0.03)
            
        elif rhythm_type == "tachycardia":
            ef = np.random.normal(54.0, 6.0)
            hr = np.random.normal(125.0, 12.0)
            irreg = np.random.uniform(0.01, 0.06)
            sdnn = np.random.normal(30.0, 8.0)
            rmssd = np.random.normal(18.0, 5.0)
            pnn50 = np.random.normal(5.0, 3.0)
            lf = np.random.normal(300.0, 70.0)
            hf = np.random.normal(180.0, 40.0)
            lf_hf = lf / hf
            sd1 = np.random.normal(13.0, 4.0)
            sd2 = np.random.normal(42.0, 9.0)
            sd_ratio = sd1 / sd2
            dys = np.random.normal(30.0, 8.0)
            mot_irreg = np.random.normal(0.16, 0.03)
            
        hr_x_dys = hr * dys
        
        vec = [
            ef, hr, irreg, sdnn, rmssd, pnn50, lf, hf, lf_hf, sd1, sd2, sd_ratio, dys, mot_irreg, hr_x_dys
        ]
        features.append(vec)
        
    return np.array(features)

def train():
    """Generates synthetic dataset, fits scaler, trains XGBoost booster, and saves outputs."""
    if xgb is None:
        print("Error: XGBoost is not installed. Cannot train classifier.")
        return
        
    print("Generating synthetic patient dataset for arrhythmia classification...")
    classes = ["normal_sinus_rhythm", "atrial_fibrillation", "pvc", "bradycardia", "tachycardia"]
    class_map = {cls: idx for idx, cls in enumerate(classes)}
    
    X_list = []
    y_list = []
    
    rhythm_map = {
        "normal_sinus_rhythm": "nsr",
        "atrial_fibrillation": "afib",
        "pvc": "pvc",
        "bradycardia": "bradycardia",
        "tachycardia": "tachycardia"
    }
    for cls in classes:
        cls_features = generate_patient_features(
            rhythm_map[cls],
            n_samples=250
        )
        X_list.append(cls_features)
        y_list.append(np.full(len(cls_features), class_map[cls]))
        
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    
    print("Fitting feature normalizer...")
    scaler = StandardScaler()
    scaler.fit(X)
    
    scaler_params = {}
    for i, name in enumerate(FEATURE_NAMES):
        scaler_params[name] = {
            "mean": float(scaler.mean_[i]),
            "std": float(scaler.scale_[i])
        }
        
    os.makedirs("models", exist_ok=True)
    with open("models/scaler_params.json", "w") as f:
        json.dump(scaler_params, f, indent=4)
    print("Saved normalizer parameters to models/scaler_params.json")
    
    X_norm = scaler.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training XGBoost Classifier...")
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    params = {
        'max_depth': 4,
        'eta': 0.1,
        'objective': 'multi:softprob',
        'num_class': 5,
        'eval_metric': 'mlogloss',
        'seed': 42
    }
    
    evallist = [(dtrain, 'train'), (dtest, 'eval')]
    num_round = 50
    
    bst = xgb.train(params, dtrain, num_round, evallist, verbose_eval=False)
    preds_prob = bst.predict(dtest)
    preds = np.argmax(preds_prob, axis=1)
    
    acc = accuracy_score(y_test, preds)
    print(f"\nModel Training Complete. Test Accuracy: {acc*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=classes))
    
    bst.save_model("models/xgboost_arrhythmia.json")
    print("Saved trained XGBoost model to models/xgboost_arrhythmia.json")

if __name__ == "__main__":
    train()
