import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import pyswarms as ps
import matplotlib.pyplot as plt  # Added for image visualization

# ========== CONFIGURATION ==========

IMAGE_SIZE = (100, 100)  # Reduced from original 800x800
PATHS = {
    'train_csv': r'C:\Programming\DR Detection using ML\Messidor\train.csv',
    'test_csv': r'C:\Programming\DR Detection using ML\Messidor\test.csv',
    'train_img': r'C:\Programming\DR Detection using ML\Messidor\trainimg',
    'test_img': r'C:\Programming\DR Detection using ML\Messidor\testimg'
}

# Column name standardization for diabetic retinopathy binary classification
COLUMN_MAPPING = {
    'image': ['image', 'img', 'image_name', 'image_id', 'Image'],  # Added 'Image' based on your dataset
    'target': ['risk_of_macular_edema', 'macular_edema_risk', 'retinal_risk', 'diagnosis', 'Risk of macular edema ']  # Added 'Risk of macular edema' from your dataset
}


# PSO parameters
PSO_CONFIG = {
    'n_particles': 10,
    'iterations': 10,
    'options': {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
}

# ========== DATA LOADING & VALIDATION ==========

def load_and_validate_data(path):
    """Load CSV with automatic column standardization for diabetic retinopathy"""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    # Find matching columns
    img_col = next((col for col in COLUMN_MAPPING['image'] if col in df.columns), None)
    target_col = next((col for col in COLUMN_MAPPING['target'] if col in df.columns), None)

    if not img_col or not target_col:
        available = df.columns.tolist()
        raise ValueError(f"Columns not found. Available columns: {available}")

    # Return standardized dataframe
    return df[[img_col, target_col]].rename(columns={img_col: 'image', target_col: 'target'})

def relabel_classes(row):
    """Relabels target classes for binary classification"""
    if row['target'] in [0]:  # Negative class
        return 0
    elif row['target'] in [1, 2, 3]:  # Positive class
        return 1

# Load datasets
try:
    train_df = load_and_validate_data(PATHS['train_csv'])
    test_df = load_and_validate_data(PATHS['test_csv'])
    print("Data loaded successfully. Columns standardized to ['image', 'target']")

    # Apply relabeling for binary classification
    train_df['target'] = train_df.apply(relabel_classes, axis=1)
    test_df['target'] = test_df.apply(relabel_classes, axis=1)
except Exception as e:
    print(f"Data loading error: {str(e)}")
    raise

# ========== IMAGE PROCESSING ==========

def preprocess_image(img_path):
    """Process images with error handling"""
    try:
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Display the original image
        plt.figure(figsize=(10, 10))
        plt.subplot(2, 2, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')

        # Green channel extraction
        green_channel = img[:, :, 1]
        plt.subplot(2, 2, 2)
        plt.imshow(green_channel, cmap='gray')
        plt.title('Green Channel')

        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(green_channel)
        plt.subplot(2, 2, 3)
        plt.imshow(enhanced, cmap='gray')
        plt.title('After CLAHE')

        # Gaussian Blur
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        plt.subplot(2, 2, 4)
        plt.imshow(blurred, cmap='gray')
        plt.title('After Gaussian Filter')

        plt.show()

        return blurred.flatten()
    except Exception as e:
        print(f"Error processing {img_path}: {str(e)}")
        return None

# ========== DATASET PREPARATION ==========

def build_dataset(df, img_dir):
    """Create normalized dataset"""
    X, y = [], []
    for _, row in df.iterrows():
        img_path = os.path.join(img_dir, row['image'])
        features = preprocess_image(img_path)
        if features is not None:
            X.append(features)
            y.append(row['target'])

    X_array = np.array(X) / 255.0  # Normalization
    y_array = np.array(y)

    print(f"Created dataset with {X_array.shape[0]} samples")
    return X_array, y_array

# Build datasets
try:
    X_train, y_train = build_dataset(train_df, PATHS['train_img'])
    X_test, y_test = build_dataset(test_df, PATHS['test_img'])
except Exception as e:
    print(f"Dataset creation failed: {str(e)}")
    raise

# ========== PSO FEATURE SELECTION ==========

def pso_fitness(masks):
    """PSO-compatible fitness function"""
    scores = []
    for mask in masks:
        binary_mask = mask > 0.5
        if not binary_mask.any():
            scores.append(1.0)
            continue

        try:
            X_subset = X_train[:, binary_mask]
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                X_subset, y_train, test_size=0.2, random_state=42
            )
            clf = SVC(kernel='linear')
            clf.fit(X_train_split, y_train_split)
            score = accuracy_score(y_val_split, clf.predict(X_val_split))
            scores.append(1 - score)
        except:
            scores.append(1.0)
    return np.array(scores)

# Run PSO
print(f"\nStarting PSO with {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} images ({X_train.shape[1]} features)")
optimizer = ps.single.GlobalBestPSO(
    n_particles=PSO_CONFIG['n_particles'],
    dimensions=X_train.shape[1],
    options=PSO_CONFIG['options'],
    bounds=([0]*X_train.shape[1], [1]*X_train.shape[1])
)
best_cost, best_pos = optimizer.optimize(pso_fitness, iters=PSO_CONFIG['iterations'])

# Apply feature selection
feature_mask = best_pos > 0.5
print(f"Selected {feature_mask.sum()} features ({(feature_mask.sum()/len(feature_mask))*100:.1f}%)")

X_train_selected = X_train[:, feature_mask]
X_test_selected = X_test[:, feature_mask]

# ========== CLASSIFIER EVALUATION ==========

def evaluate_classifier(name, model):
    """Evaluate and report classifier performance"""
    try:
        model.fit(X_train_selected, y_train)
        preds = model.predict(X_test_selected)

        metrics = {
            'accuracy': accuracy_score(y_test, preds),
            'f1': f1_score(y_test, preds, average='weighted'),
            'cm': confusion_matrix(y_test, preds)
        }
        metrics['sensitivity'] = metrics['cm'][1,1] / (metrics['cm'][1,1] + metrics['cm'][1,0])
        metrics['specificity'] = metrics['cm'][0,0] / (metrics['cm'][0,0] + metrics['cm'][0,1])

        print(f"\n{name} Results:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Sensitivity: {metrics['sensitivity']:.4f}")
        print(f"Specificity: {metrics['specificity']:.4f}")
        print("Confusion Matrix:")
        print(metrics['cm'])
    except Exception as e:
        print(f"Error evaluating {name}: {str(e)}")

# Evaluate classifiers
evaluate_classifier("SVM", SVC(kernel='linear'))
evaluate_classifier("KNN", KNeighborsClassifier(n_neighbors=5))

print("\nExecution completed successfully!")
