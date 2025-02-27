from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  # To handle CORS issues
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Load models and resources once when the app starts
STOPWORDS = set(stopwords.words("english"))
predictor = pickle.load(open(r"venv\Model\model_xgb.pkl", "rb"))
scaler = pickle.load(open(r"scaler.pkl", "rb"))
cv = pickle.load(open(r"countVectorizer.pkl", "rb"))

@app.route("/test", methods=["GET"])
def test():
    return "Test request received successfully. Service is running."

@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("landing.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "text" in request.json:
            # Single string prediction
            text_input = request.json["text"]
            predicted_sentiment = single_prediction(text_input)
            return jsonify({"prediction": predicted_sentiment})
        else:
            return jsonify({"error": "No text input provided."})
    except Exception as e:
        return jsonify({"error": str(e)})

def single_prediction(text_input):
    stemmer = PorterStemmer()
    
    # Text preprocessing
    review = re.sub("[^a-zA-Z]", " ", text_input)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if not word in STOPWORDS]
    review = " ".join(review)
    
    print(f"Preprocessed Review: {review}")
    
    # Transform the text into features
    corpus = [review]
    X_prediction = cv.transform(corpus).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    
    print(f"Feature Vector: {X_prediction_scl}")
    
    # Get prediction (proba -> argmax for the sentiment class)
    y_predictions = predictor.predict_proba(X_prediction_scl)
    print(f"Prediction Probabilities: {y_predictions}")
    y_predictions = y_predictions.argmax(axis=1)[0]
    
    return "Positive" if y_predictions == 1 else "Negative"

if __name__ == "__main__":
    app.run(port=5000, debug=True)