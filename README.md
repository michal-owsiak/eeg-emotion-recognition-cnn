## **EEG-based Binary Classification of Emotional Arousal: Exploratory Analysis and Machine Learning on DEAP Dataset**

This repository contains a [Jupyter Notebook](https://github.com/michal-owsiak/eeg-emotional-arousal-classification/blob/main/index.ipynb) that presents a comparative study of neural network and classical machine learning algorithms applied to the binary classification of EEG data. The aim was to assess how well various models can distinguish between two mental states based solely on bandpower features extracted from EEG signals.

The notebook demonstrates a complete end-to-end workflow, including preprocessing, feature extraction, model training, evaluation, and visualization.

---

## **Abstract**

**Background**:  
Electroencephalography (EEG) provides a non-invasive window into brain activity and is increasingly explored in affective computing. Emotional arousal — a key dimension of emotional experience — can potentially be inferred from EEG features, though its neural signatures remain subtle and difficult to extract.

**Aims**:  
This study evaluates the feasibility of classifying binary emotional arousal (high vs. low) based on EEG recordings from the [DEAP dataset](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/index.html). It was aimed to assess how well bandpower features from multiple frequency bands and scalp locations can distinguish arousal levels using both neural and classical machine learning classifiers.

**Methods:**  
EEG signals from 32 participants, each exposed to 40 music video trials (63 seconds each), were segmented into overlapping windows of 5 seconds with a 2.5-second step (50% overlap). For each window, absolute and relative bandpower was computed across four frequency bands (theta, alpha, beta, gamma) from 32 EEG channels, yielding 256 features per window. The resulting dataset contained 30,720 labeled windows (training + test). Feature selection was performed using random forest importance scores, and the top 50 features were used to train a neural network, along with logistic regression, random forest, SVM, and XGBoost classifiers. Performance was evaluated using accuracy and ROC AUC metrics.

**Results:**  
All models showed comparable, limited performance that exceeded random guessing, though the overall predictive power remained modest. Test accuracy ranged from 0.59 to 0.63, and ROC AUC from 0.64 to 0.65. The highest accuracy was achieved by the SVM classifier (63%), while the best ROC AUC score was obtained by the neural network model (65%).

**Conclusions:**  
The results suggest that EEG bandpower features carry only limited and hard-to-extract information about arousal, leading to modest model performance across classifiers. These findings highlight both the potential and limitations of EEG-based emotion recognition and suggest the need for more sophisticated feature engineering or multimodal data to improve performance in affective computing tasks.

---

## **Tools used**

**Languages, frameworks and environments:**
```
> Python
> Jupyter Notebook
> Markdown
```
**Libraries and packages:**
```
> TensorFlow / Keras
> scikit-learn
> XGBoost
> SciPy
> NumPy
> pandas
> matplotlib
> seaborn
> gdown
> os
> IPython
```
## **Data preparation**

The EEG data were preprocessed into numerical features as follows:

- **Bandpower computation:** Absolute and relative power calculated for θ, α, β, and γ bands for each of 32 EEG channels.
- **Data splitting:** Train/test split with stratified sampling.
- **Scaling:** Standardization applied to features prior to model training.

## **Modeling and evaluation**

### **Neural Network (MLP)**  
A shallow dense network (multi-layer perceptron) with regularization and droput was trained using early stopping to avoid overfitting.  


### **Classical Machine Learning Models**  
Four traditional models were tested for comparison:  
- Logistic Regression  
- Random Forest  
- Support Vector Machine (SVM)  
- XGBoost  

Each model was evaluated using accuracy, ROC AUC, precision, recall, and F1-score.

**Visualizations include:**
- Raw sample EEG signal along with spectrogram
- Histogram depicting distribution of the arousal self-assesments
- EEG bandpower and feature importance heatmaps
- Accuracy and loss curves (for MLP) 

---

## **Results and conclusions**

| Model               | Accuracy | ROC AUC |
|--------------------|----------|---------|
| Neural network (MLP) | 0.61     | 0.65    |
| Logistic regression  | 0.59     | 0.64    |
| Random forest        | 0.61     | 0.64    |
| SVM                  | 0.63     | 0.64    |
| XGBoost              | 0.60     | 0.64    |

---

### **Key observations:**

- All models perform **only slightly better than random guessing** (which would yield ~0.50 accuracy and ROC AUC).
- The **neural network (MLP)** achieves the highest **ROC AUC (0.65)**, indicating a marginally better ability to distinguish between the two classes.
- **SVM** obtains the highest **accuracy (0.63)**, though its ROC AUC is the same as other classical models.
- The **differences in performance across models are minimal**, suggesting that the EEG-based classification task is inherently difficult and possibly limited by the signal-to-noise ratio or data representation.
- The consistent performance across models indicates that no single algorithm provides a decisive advantage for this task.

---

Overall, the results suggest that more advanced feature engineering, temporal modeling, or deep architectures (e.g., CNNs, RNNs) may be necessary to meaningfully improve classification performance in this domain.

---

**Author of the notebook**  

Michał Owsiak
