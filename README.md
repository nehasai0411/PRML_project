# CIFAR-10 Image Classification

This project compares several classical machine learning and deep learning models for image classification on the CIFAR-10 dataset. Each method is evaluated based on test accuracy and generalization performance.

## 📂 Methods Used

### 1. Convolutional Neural Network (CNN)
Two convolutional blocks with pooling and dropout followed by a dense layer and softmax.  
**Test Accuracy**: **77.44%**  
Shows strong learning and generalization. Dropout mitigates slight overfitting.

### 2. Support Vector Machine (SVM)
PCA reduces input dimensions before classification with SVM.  
**Test Accuracy**: **51.50%**  
Performs well among classical models but lacks spatial feature learning.

### 3. Multilayer Perceptron (MLP)
Dense layers trained on flattened inputs with dropout regularization.  
**Test Accuracy**: **48.97%**  
Performs moderately well; better than tree-based models, worse than CNN.

### 4. Gradient Boosting
Boosted decision trees trained on PCA-reduced data.  
**Test Accuracy**: **43.00%**  
Handles class separation better than Random Forest.

### 5. Random Forest
Trained on PCA-reduced inputs for efficiency.  
**Test Accuracy**: **45.75%**  
Good interpretability and general performance.

### 6. K-Nearest Neighbors (KNN)
Applied on PCA-reduced feature vectors.  
**Test Accuracy**: **37.64%**  
Simple method; not optimal for high-dimensional image data.

### 7. Naive Bayes
Assumes feature independence, trained on PCA-reduced inputs.  
**Test Accuracy**: **32.75%**  
Baseline model with lowest performance; unable to capture image complexity.

## 📊 Evaluation
Models were evaluated using test accuracy and confusion matrices. CNN outperformed all others, demonstrating the power of deep learning in image classification tasks.

## 📁 Notebooks
- `CNN.ipynb`
- `SVM.ipynb`
- `MLP.ipynb`
- `GradientBoosting.ipynb`
- `RandomForest.ipynb`
- `KNN.ipynb`
- `NaiveBayes.ipynb`

