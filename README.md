# 📚 Book Clustering & Genre Prediction

This project automatically groups books into meaningful clusters and predicts genres using **unsupervised machine learning (KMeans)**.  
The dataset is created by scraping 50k+ book records (titles, authors, ratings, reviews) using **BeautifulSoup** and **Requests**.  
Predictions are served via a **Flask API** with a simple HTML/JS frontend for **real-time genre classification**.

---

## Project Structure

- **Data Collection:** Scrapes book titles, authors, ratings, and reviews from online sources  
- **Data Cleaning & Feature Engineering:** Processes textual and numerical features, applies TF-IDF vectorization and scaling  
- **Clustering:** Uses KMeans to automatically group books into clusters  
- **Model Deployment:** Saves TF-IDF, scaler, and KMeans models with `joblib` for reuse  
- **API & Frontend:** Flask API serves predictions; HTML/JS frontend provides interactive input/output  
- **Visualization:** PCA-based plots and histograms for cluster distributions and rating patterns  

---

## Requirements


Install dependencies with:

```bash
pip install -r requirements.txt
