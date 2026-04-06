---
title: SmartPrice Engine
emoji: ⚡
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
app_port: 7860
---

# ⚡ SmartPrice Engine

**Dynamic Pricing API powered by XGBoost + FastAPI**

---

## 🖥️ Run Locally

### 1. نزّل الداتا
> https://www.kaggle.com/datasets/arashnic/dynamic-pricing-dataset

نزّل `dynamic_pricing.csv` وحطه في:
```
smartprice-engine/data/dynamic_pricing.csv
```

### 2. Install
```bash
pip install -r requirements.txt
```

### 3. شغّل
```bash
python run.py
```

افتح: **http://localhost:8000**

---

## 🐳 Docker (بعد التدريب المحلي)

```bash
python run.py        # دربّ الموديل أولاً ثم Ctrl+C
docker build -t smartprice .
docker run -p 7860:7860 smartprice
```

---

## 🚀 Hugging Face Deploy

```bash
git init && git add . && git commit -m "init"
git remote add origin https://huggingface.co/spaces/<username>/smartprice
git push
```

> تأكد إن models/*.joblib مش في .gitignore قبل الـ push

---

## Tech Stack
`XGBoost` · `FastAPI` · `Docker` · `Hugging Face Spaces`
