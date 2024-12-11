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
### **Clone the Repository**  
```bash
git clone https://github.com/your-username/BerTaNER-Ta.git
cd BerTaNER-Ta
