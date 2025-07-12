# fake-news-detector-and-generator


# 🧠 Fake News Detection & Generation App

An intelligent web app that **detects fake news** using BERT and **generates realistic fake articles** using GPT-2. Built with an interactive UI in **Streamlit**, the app also fetches **real-time headlines** using NewsAPI and classifies them instantly.

---

## 📁 Project Structure

Fake-News-Detection/
├── app.py # Main Streamlit application
├── requirements.txt # Python dependencies
├── Fake.csv # Dataset for fake news
├── True.csv # Dataset for real news
├── lr_model.pkl 
├── vectorizer.pkl # TF-IDF vectorizer
├── README.md # This documentation file
├── venv/ # Virtual environment 

---

## 🚀 Features

- 📝 Manual fake/real news classification using BERT
- 🌐 Real-time headline analysis with NewsAPI
- 🧬 GPT-2-based fake news generation
- 🎨 Animated Streamlit UI with glowing icons and cards
- ⚡ Confidence scores and result highlights
- 🔒 Secure API integration

---

## 🧪 Tech Stack

| Component         | Technology                         |
|------------------|-------------------------------------|
| **Frontend**      | Streamlit (Python-based UI)         |
| **ML Models**     | BERT (`bert-base-uncased`), GPT-2   |
| **Libraries**     | HuggingFace Transformers, PyTorch   |
| **Data**          | Fake.csv, True.csv from Kaggle      |
| **API**           | [NewsAPI.org](https://newsapi.org)  |
| **Styling**       | HTML/CSS inside Streamlit markdown  |

---

## 📦 Installation

### 1. Clone the Repo
```bash
git clone https://github.com/your-username/Fake-News-Detection.git
cd Fake-News-Detection

Create & Activate a Virtual Environment
python -m venv venv
source venv/bin/activate       # Mac/Linux
venv\Scripts\activate          # Windows

Install Dependencies
pip install -r requirements.txt

 Run the App
 streamlit run app.py


📊 Dataset Information
The model uses a labeled dataset combining Fake.csv and True.csv, each containing news headlines and articles sourced from Kaggle's fake news challenge dataset.

📖 How It Works
🔍 Manual News Check
Paste any headline or article and the model classifies it as Real or Fake with a confidence score using a BERT model.

🌐 Real-Time Detection
It fetches live news from NewsAPI, evaluates them instantly, and displays predictions.

🎭 Fake News Generation
Enter any topic and it uses GPT-2 to generate a fake article, which is then evaluated to see whether it's convincing enough to be classified as fake or real.

🛠 Requirements
Python 3.8+

Streamlit

torch

transformers

newsapi-python

scikit-learn

pandas

🤝 Acknowledgements
🤖 HuggingFace Transformers

📰 NewsAPI.org

🎨 Streamlit

📊 Kaggle for the dataset

📜 License
This project is licensed under the MIT License. Feel free to use, modify, or distribute with attribution.



---
