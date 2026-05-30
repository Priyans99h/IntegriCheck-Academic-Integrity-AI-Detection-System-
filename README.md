# IntegriCheck — AI & Plagiarism Detection System

**MSc Applied Statistics — Final Year Project**

A dual-mode academic integrity platform detecting plagiarism and AI-generated content.

---

## Folder Structure
```
integricheck/
├── notebooks/                  ← Run these IN ORDER
│   ├── step1_environment_setup.ipynb
│   ├── step2_data_collection.ipynb
│   ├── step3_plagiarism_engine.ipynb
│   ├── step4_ai_detection_engine.ipynb
│   ├── step5_flask_app.ipynb
│   └── step6_evaluation_benchmarking.ipynb
├── src/
│   ├── plagiarism/engine.py    ← Plagiarism detection module
│   ├── ai_detection/engine.py  ← AI detection module
│   └── utils/
├── flask_app/
│   ├── app.py                  ← Main Flask application
│   └── templates/
│       ├── index.html          ← Main UI
│       └── dashboard.html      ← University dashboard
├── data/
│   ├── raw/                    ← Downloaded raw data (Step 2)
│   ├── processed/              ← Clean data + feature plots
│   └── models/                 ← Trained models (Step 3, 4)
├── reports/                    ← Generated PDF reports
├── config.py                   ← Central configuration
└── requirements.txt
```

## How to Run (Follow Steps IN ORDER)

```bash
pip install -r requirements.txt

# Open Jupyter and run each notebook:
# Step 1 → Step 2 → Step 3 → Step 4 → Step 5 → Step 6
jupyter notebook notebooks/
```

## Tech Stack
- **ML**: scikit-learn, sentence-transformers, PyTorch
- **NLP**: NLTK, HuggingFace datasets
- **Backend**: Flask 3.0
- **Reports**: FPDF2
