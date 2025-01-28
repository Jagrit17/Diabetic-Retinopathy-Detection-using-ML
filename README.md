Diabetic Retinopathy Detection using Machine Learning

This repository implements a pipeline to detect diabetic retinopathy using feature selection, image preprocessing, and machine learning models. The project uses two datasets: Messidor I and Messidor II.

Features

- Image Preprocessing: Includes green channel extraction, CLAHE (Contrast Limited Adaptive Histogram Equalization), and Gaussian filtering.
- Feature Selection: Utilizes Particle Swarm Optimization (PSO) for dimensionality reduction.
- Classification Models: Support Vector Machines (SVM) and K-Nearest Neighbors (KNN).
- Metrics: Calculates accuracy, F1 score, sensitivity, specificity, and confusion matrices.
- Visualization: Displays intermediate image processing steps.

Prerequisites
Dataset Links

Download the datasets from Kaggle:
Messidor I Dataset: https://www.kaggle.com/datasets/hanhan2010/messidor
Messidor II Dataset: https://www.kaggle.com/datasets/mariaherrerot/messidor2preprocess

Libraries Used:

- Python 3.8+
- OpenCV
- NumPy
- Pandas
- scikit-learn
- PySwarms
- Matplotlib

Install the dependencies using:
pip install -r requirements.txt

Folder Structure

├── M1image.py       # Code for Messidor I dataset image preprocessing
├── M1result.py      # Classification and feature selection for Messidor I dataset
├── M2image.py       # Code for Messidor II dataset image preprocessing
├── M2result.py      # Classification and feature selection for Messidor II dataset
├── trainimg/        # Folder containing training images
├── testimg/         # Folder containing test images
├── train.csv        # Training dataset CSV file
├── test.csv         # Test dataset CSV file

How to Use

1. Data Setup: Ensure the datasets (CSV files) and images are in the correct paths as specified in the code. Update the `PATHS` dictionary in the scripts if necessary.
2. Image Preprocessing: Run `M1image.py` or `M2image.py` to preprocess the images and visualize the processing steps.
3. Feature Selection and Classification: Run `M1result.py` or `M2result.py` for feature selection using PSO and classification using SVM and KNN.
4. Results:
- Outputs the selected features, accuracy, F1 score, sensitivity, specificity, and confusion matrix for the classifiers.
- Displays results for both SVM and KNN models.


Notes

1. Ensure the `train.csv` and `test.csv` files have appropriate column names for `image` and `target`.
2. Update `COLUMN_MAPPING` in the scripts to standardize the dataset columns.
3. Intermediate processing steps can be visualized for the first few images using the `display_samples` parameter.

