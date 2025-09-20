from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from scipy.sparse import hstack

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Load models
tfidf = joblib.load('models/tfidf_vectorizer.pkl')
scaler = joblib.load('models/scaler.pkl')
kmeans = joblib.load('models/kmeans_model.pkl')


# Cluster labels mapping
cluster_labels = {
    0: "Mystery & Thriller",
    1: "Horror / Suspense",
    2: "Fiction & Fantasy",
    3: "Fantasy & Sci-Fi",
    4: "YA & Adventure"
}

@app.route('/', methods=['POST'])
def predict_cluster():
    data = request.json
    title = data.get('title')
    author = data.get('author')
    avg_rating = float(data.get('avg_rating', 0))
    num_rating = int(data.get('num_rating', 0))

    text_features = tfidf.transform([title + " " + author])
    num_features = scaler.transform([[avg_rating, num_rating]])
    X_input = hstack([text_features, num_features])

    cluster = int(kmeans.predict(X_input)[0])
    category = cluster_labels.get(cluster, "Unknown")

    return jsonify({'cluster': cluster, 'category': category})

if __name__ == '__main__':
    app.run(debug=True)
