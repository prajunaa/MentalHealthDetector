from flask import Flask, render_template, request
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from PIL import Image, ImageOps, ImageFilter
import pytesseract

pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

nlp = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
nlp.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# TRAINING TTRAING TRQIN TRAIN TRAIN TRAN TRAIN
df = pd.read_csv("mental_health.csv")
df = df.dropna()

print("First 5 records:", df.head())

# Normalize text by converting all characters to lowercase
# This makes sure our feature extraction stuff works likje for example "Happy" == "happy"
df["text"] = df["statement"].str.lower()
df["label"] = df["status"]

# ml models need number labels, so we convert them with label encoder and make sure it works!!!!!!!!! heh
le = LabelEncoder()
df["label_id"] = le.fit_transform(df["label"])

label_names = list(le.classes_)
print(label_names)

# 80 / 20 train test split, random state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label_id"], test_size=0.2, random_state=42
)

# Create a machine learning pipeline that: converts text into TF-IDF feature vectors, trains a Logistic Regression classifier on those features
model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=10000)),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")) # make this balanced so everything isn't 'normal', i want hidden messages to POP
])

model.fit(X_train, y_train)

# find severity of the situation based on our label.
def map_to_risk(label):
    if label == "Suicidal":
        return "Acute"
    elif label == "Normal":
        return "Low"
    else:
        return "Elevated"

#take text from our screenshot
def extract_text_from_image(image_path: str) -> str:
    """
    Read a screenshot and return extracted text.
    """

    img = Image.open(image_path)

    # Basic preprocessing to help OCR on screenshots
    img = ImageOps.grayscale(img)
    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.SHARPEN)

    # Simple thresholding to make text stand out more
    img = img.point(lambda x: 0 if x < 160 else 255, mode="1")

    text = pytesseract.image_to_string(img, lang="eng")
    return text.strip()

#prediction algo
def predict_full(text):
    text = text.lower()
    # Get probability distribution across all classes and chooses one with highest probs
    probs = model.predict_proba([text])[0]
    pred_id = probs.argmax()
    pred_label = label_names[pred_id]
    
    risk = map_to_risk(pred_label)
    
    print("Mental Health Category:", pred_label)
    print("Risk Level:", risk)
    
    print("\nLikelihoods:")
    likelihoods = {}
    for i, p in enumerate(probs):
        likelihoods[label_names[i]] = round(float(p), 3)
        print(f"{label_names[i]}: {round(p, 3)}")
    
    return {
        "extracted_text": text,
        "category": pred_label,
        "risk": risk,
        "likelihoods": likelihoods
    }


def predict_from_screenshot(image_path: str):
    extracted_text = extract_text_from_image(image_path)

    if not extracted_text:
        print("No text was found in the image.")
        return {
            "extracted_text": "",
            "category": "No text found",
            "risk": "Unknown",
            "likelihoods": {}
        }

    print(extracted_text)
    return predict_full(extracted_text)


@nlp.route("/")
def index():
    return render_template("index.html")

@nlp.route("/upload", methods=["GET", "POST"])
def upload():
    if "image" not in request.files:
        return "No file uploaded"

    file = request.files["image"]

    if file.filename == "":
        return "No selected file"

    filepath = os.path.join(nlp.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    result = predict_from_screenshot(filepath)

    return render_template(
        "index.html",
        extracted_text=result["extracted_text"],
        category=result["category"],
        risk=result["risk"],
        likelihoods=result["likelihoods"],
        image_path="/"+filepath
    )

if __name__ == "__main__":
    nlp.run(debug=True)
