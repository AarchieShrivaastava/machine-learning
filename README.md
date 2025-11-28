# 📧 AIML-Based Spam Email Detector  

---

## 📌 Overview  
This project is a simple **Spam Email Detection System** built using **Machine Learning (AIML)** in Python.  
It classifies given email text as either:

- **Spam** (unwanted, promotional, fraudulent emails)  
- **Ham** (legitimate, normal emails)

The model is based on **Multinomial Naive Bayes**, a popular algorithm for text classification.

---

## 🎯 Features  
- ✔ Detects **Spam / Not Spam**
- ✔ Uses **CountVectorizer** (Bag of Words)
- ✔ Machine Learning model using **MultinomialNB**
- ✔ Shows **accuracy & performance**
- ✔ **Interactive mode** for testing emails
- ✔ Very beginner-friendly

---

## 🧠 Technologies Used  
- Python 3.x  
- scikit-learn  
- CountVectorizer  
- Multinomial Naive Bayes  

---

## 📂 Project Structure

📁 SpamEmailDetector
│
├── spam_email_detector.py # Main ML program
├── README.md # Documentation
└── dataset.txt (optional) # Custom dataset if extended


---

## ⚙️ Installation & Setup

### 1️⃣ Install Python Dependencies

```bash
pip install scikit-learn
If pip does not work, reinstall Python and enable:
✔ Add Python to PATH

### ▶️ How to Run
python spam_email_detector.py


After running, you will see:

Model accuracy

Email prediction interface

Example:

Enter email text: You won a free iPhone!
👉 Prediction: This email is likely SPAM.

### 📘 How It Works
1. Dataset Preparation

A small hardcoded dataset of email samples labeled as:

spam

ham

2. Text Vectorization

CountVectorizer converts words → numeric vectors.

3. Model Training

Trained using Multinomial Naive Bayes.

4. Model Evaluation

Accuracy + classification report printed.

5. User Input Prediction

User types an email → model predicts spam or ham.

### 📊 Sample Output
=== MODEL EVALUATION ===
Accuracy: 0.75

=== SPAM EMAIL DETECTOR ===
Type an email to check if it is SPAM or NOT SPAM.

### 🔧 Future Enhancements

Add a larger dataset

Use TF-IDF instead of CountVectorizer

Build GUI using Tkinter

Implement deep learning models (LSTM, BERT)

Connect with real email APIs

### 📝 Author

AIML Spam Email Detector – B.Tech CSE First Year Mini Project
