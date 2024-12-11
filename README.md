# **BerTaNER - A Contemporary Approach for Named Entity Recognition in Tamil**  

BerTaNER is a machine learning project aimed at improving Named Entity Recognition (NER) for Tamil, a morphologically rich and low-resource language. By fine-tuning transformer-based models, TamilBERT and Multilingual BERT (mBERT), BerTaNER provides a robust solution for identifying named entities in Tamil text.  

---

## **Table of Contents**  
1. [Introduction](#introduction)  
2. [Features](#features)  
3. [Project Workflow](#project-workflow)  
4. [Datasets](#datasets)  
5. [Model Configurations](#model-configurations)  
6. [Installation and Setup](#installation-and-setup)  
7. [Usage](#usage)  
8. [Results](#results)  
9. [Future Work](#future-work)  
10. [Contributing](#contributing)  
11. [License](#license)  

---

## **Introduction**  
Named Entity Recognition (NER) is a key task in natural language processing (NLP), which identifies and classifies entities such as names, locations, and organizations in text.  
### **Challenges in Tamil NER**  
- Tamil's rich morphology and complex structure.  
- Low availability of annotated datasets for training.  
- Handling noisy and colloquial Tamil text.  

BerTaNER focuses on overcoming these challenges by leveraging transformer-based language models fine-tuned specifically for Tamil.  

---

## **Features**  
- **Pretrained Models**: Utilizes TamilBERT and Multilingual BERT for NER tasks.  
- **Advanced Fine-Tuning Techniques**: Implements Layer-wise Learning Rate Decay (LLRD) and dropout regularization.  
- **Token Classification**: Efficiently classifies entities at the token level.  
- **Customizable Pipeline**: Modular design for data preprocessing, model training, and evaluation.  

---

## **Project Workflow**  
The workflow consists of the following steps:  
1. **Data Preprocessing**  
   - Cleaning and tokenizing the datasets.  
   - Aligning tokens with the respective labels.  

2. **Model Training**  
   - Fine-tuning TamilBERT and mBERT with customized configurations.  
   - Regularization with dropout and LLRD.  

3. **Evaluation**  
   - Measuring loss, accuracy, precision, recall, and F1 score.  

4. **Comparison**  
   - Comparative performance analysis of TamilBERT and mBERT.  

---

## **Datasets**  
### **WikiANN Dataset**  
- Multilingual NER dataset used for TamilBERT fine-tuning.  
- Includes annotations for person, location, and organization entities.  

### **Google Xtreme Dataset**  
- Another multilingual dataset used for mBERT fine-tuning.  
- Provides additional contextual data for better model generalization.  

---

## **Model Configurations**  
### **TamilBERT**  
- Pretrained model: `l3cube-pune/tamil-bert`.  
- Learning Rate: `1e-5`.  
- Batch Size: `16` and `32`.  
- Dropout: `0.3` and `0.4`.  

### **Multilingual BERT (mBERT)**  
- Pretrained model: `bert-base-multilingual-cased`.  
- Learning Rate: `2e-5`.  
- Batch Size: `16` and `32`.  
- Dropout: `0.3` and `0.4`.  

Both models use **Layer-wise Learning Rate Decay (LLRD)** for stable and effective fine-tuning of deeper layers.  

---

## **Installation and Setup**

### **Install Dependencies**
Install all required dependencies:
```bash
pip install -r requirements.txt
```

### **Environment Configuration**
Ensure that you have:
- Python 3.8+
- Libraries: `transformers`, `torch`, `sklearn`

---


## **Usage**

### **Data Preprocessing**
Prepare the dataset for training using the following command:
```bash
python preprocess.py --dataset [WikiANN|Xtreme]
```

### **Model Training**
Train the selected model with configurations defined in JSON files:
```bash
python train.py --model [TamilBERT|mBERT] --config configs/model_config.json
```

### **Evaluation**
Evaluate the trained model on the test dataset:
```bash
python evaluate.py --model [TamilBERT|mBERT]
```

---

## **Results**

| **Model**            | **Learning Rate** | **Batch Size** | **Dropout** | **Eval Loss** | **Precision** | **Recall** | **F1 Score** |
|-----------------------|-------------------|----------------|-------------|---------------|---------------|------------|--------------|
| TamilBERT             | 0.00001          | 16             | 0.3         | 1.214         | 0.679         | 0.655      | 0.669        |
| TamilBERT             | 0.00001          | 32             | 0.3         | 1.499         | 0.541         | 0.521      | 0.531        |
| TamilBERT             | 0.00005          | 16             | 0.4         | 1.396         | 0.455         | 0.436      | 0.445        |
| Multilingual BERT     | 0.00002          | 16             | 0.3         | 1.190         | 0.734         | 0.691      | 0.712        |

---

## **Future Work**

1. Incorporating noisy Tamil text to improve model generalization.
2. Exploring hybrid models that combine rule-based and machine learning approaches.
3. Extending the framework to support other low-resource languages.

---

## **Contributing**

Contributions are welcome! Follow these steps:

1. **Fork the Repository**  
   Create your own copy of the repository.

2. **Create a New Branch**
   ```bash
   git checkout -b feature-name
   ```

3. **Commit Your Changes**
   ```bash
   git commit -m "Add feature"
   ```

4. **Push to the Branch**
   ```bash
   git push origin feature-name
   ```

5. **Submit a Pull Request**
   Open a pull request to the main repository.

---

## **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## **Acknowledgements**

- **L3Cube Pune** for the TamilBERT model.
- **Google** for the Xtreme dataset.
- Open-source contributors for tools and resources used in this project.
