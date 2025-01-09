# **User Manual: G.R.O.O.T (Guided Readmission & Orchestrated Observation Text)**

Welcome to the comprehensive user manual for our **G.R.O.O.T (Guided Readmission & Orchestrated Observation Text)** and **Retrieval-Augmented Generation (RAG) Summaries** project. This manual aims to explain:

1. **Why** this project exists.  
2. **Who** can use it.  
3. **Why** we chose the MIMIC dataset.  
4. **Where** the data came from.  
5. **What** the entire code base does, **how** it does it, and **when** you might want to use certain features.  

We’ll keep the language straightforward and highlight motivations, folder structures, and code commentary extensively.

---

## **Contents**

1. Overview
2. Data
3. Folder Structure
4. How to Use
5. Codex Manual

---

## **1. Overview**

### **1.1 What Is This Project?**

- We want to **predict 30-day readmission risk** for patients by using a real-world medical dataset called **MIMIC** (Medical Information Mart for Intensive Care).
- We also incorporate a **RAG (Retrieval-Augmented Generation)** pipeline to provide **short text summaries** of the risk factors and best practices, leveraging **Local LLM** or **Hugging Face** models.

### **1.2 Why We Need This Project**

- **Hospitals** often care about which patients are more likely to come back (“readmitted”) within 30 days because it affects costs, patient health outcomes, and resource planning.
- By **predicting** who’s at high risk, doctors/nurses can intervene earlier, reduce complications, and **focus** on better care transitions.
- The **RAG Summaries** help automatically write a short note or plan about these risk factors, so clinicians can quickly see the main issues.

### **1.3 Who Can Use This Project**

1. **Medical Researchers**: They can try custom features, test readmission models, and refine knowledge-based summarization.
2. **Data Scientists**: They can learn how to do end-to-end ingestion, modeling, retrieval, and summarization with large language models (LLMs).
3. **Clinicians or IT Staff**: Potentially adapt the pipeline to see risk predictions for real patients (with the correct compliance and anonymization).
4. **Curious Learners**: Even someone new can run the pipeline, pick conditions, and see how the model responds.

### **1.4 Motivation for the Project**

- Real hospitals pay close attention to 30-day readmission rates (like for heart failure, sepsis, etc.). We want a **clinical decision support tool** that:
  1. Predicts readmission risk.  
  2. Summarizes key risk factors & guidelines to help clinicians or data scientists see **why** a patient might be readmitted.

---

## **2. Data**

### **2.1 What Is MIMIC?**

**MIMIC (Medical Information Mart for Intensive Care)** is a large, publicly available, de-identified clinical database developed by the MIT Lab for Computational Physiology. It contains detailed data from critical care units of the Beth Israel Deaconess Medical Center. Because the data is rigorously de-identified, researchers and data scientists worldwide use it to develop and evaluate algorithms in healthcare machine learning.

### **2.2 Origin of MIMIC**

- **MIMIC** is hosted on [PhysioNet.org](https://physionet.org/).  
- Anyone can sign up for an account, complete a short certification course on data usage, and then request access to the MIMIC dataset for research or educational purposes.

### **2.3 Why MIMIC Dataset?**

- **Breadth and Depth**: MIMIC includes **patient admissions**, **discharge summaries**, ICU chart events, lab data, and more, making it ideal for modeling real-world hospital scenarios.  
- **Free and Open**: Although you need credentialing, there’s no cost to access MIMIC once you’ve met the requirements, and the data is widely used in academic research.  
- **Rich in Clinical Variables**: The dataset spans multiple ICU stays, enabling the creation of advanced prediction tasks such as **30-day readmission** and **clinical summarization**.  
- **NLP-Friendly**: MIMIC provides **clinical notes** for tasks like discharge summary analysis, which we leverage to build an **NLP** pipeline for summarizing risk factors and recommended guidelines.

---

## **3. Folder & File Structure**

Here is a typical layout:
