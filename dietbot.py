#!/usr/bin/env python3
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # reduces TensorFlow info messages

import re
import json
import aiml
import wikipedia
import pandas as pd
import numpy as np
import nltk
import tensorflow as tf

from tkinter import Tk
from tkinter.filedialog import askopenfilename

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nltk.sem import Expression
from nltk.inference import ResolutionProver


def ensure_nltk():
    packages = [
        ("punkt", "tokenizers/punkt"),
        ("punkt_tab", "tokenizers/punkt_tab"),
        ("wordnet", "corpora/wordnet"),
        ("omw-1.4", "corpora/omw-1.4"),
    ]
    for pkg, path in packages:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


lemmatizer = WordNetLemmatizer()
read_expr = Expression.fromstring


def normalise_for_similarity(text: str) -> str:
    text = text.lower().strip()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha()]
    lemmas = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(lemmas)


def clean_entity(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", "", text)
    text = text.replace("_", "")
    if not text:
        return text
    return text[0].upper() + text[1:].lower()


def clean_property(text: str) -> str:
    text = text.strip().lower()
    text = text.replace("-", " ")
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"_+", "_", text)

    fixes = {
        "high_protine": "high_protein",
        "protine": "protein",
        "protin": "protein",
        "suger": "sugar",
    }
    return fixes.get(text, text)


def keyify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


class SimilarityQA:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path, encoding="utf-8", engine="python")
        if "question" not in self.df.columns or "answer" not in self.df.columns:
            raise ValueError("qa_pairs.csv must have headers: question,answer")

        self.questions_raw = self.df["question"].astype(str).tolist()
        self.answers = self.df["answer"].astype(str).tolist()

        self.questions_norm = [normalise_for_similarity(q) for q in self.questions_raw]
        self.vectorizer = TfidfVectorizer()
        self.q_matrix = self.vectorizer.fit_transform(self.questions_norm)

    def answer(self, user_text: str, threshold: float = 0.32):
        user_norm = normalise_for_similarity(user_text)
        if not user_norm:
            return "", 0.0

        u_vec = self.vectorizer.transform([user_norm])
        sims = cosine_similarity(u_vec, self.q_matrix)[0]
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])

        if best_score < threshold:
            return "", best_score
        return self.answers[best_idx], best_score


def load_kb(kb_path: str):
    kb = []
    with open(kb_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                kb.append(read_expr(line))
    return kb


def kb_contradicts(kb, expr):
    neg = read_expr(f"-({expr})")
    return ResolutionProver().prove(neg, kb)


def check_kb_integrity(kb):
    for i, expr in enumerate(kb):
        rest = kb[:i] + kb[i + 1:]
        if kb_contradicts(rest, expr):
            return False, expr
    return True, None


def check_fact(kb, expr):
    if ResolutionProver().prove(expr, kb):
        return "Correct"
    neg = read_expr(f"-({expr})")
    if ResolutionProver().prove(neg, kb):
        return "Incorrect"
    return "Sorry, I don't know"


def pick_image_file():
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = askopenfilename(
        title="Select an image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    root.destroy()
    return path


def predict_image(model, labels, image_path, img_size=(100, 100)):
    img = tf.keras.utils.load_img(image_path, target_size=img_size)
    arr = tf.keras.utils.img_to_array(img)
    arr = tf.expand_dims(arr, 0) / 255.0

    probs = model.predict(arr, verbose=0)[0]
    top_idx = int(np.argmax(probs))
    top_label = labels[top_idx]
    top_conf = float(probs[top_idx])

    top3_idx = np.argsort(probs)[::-1][:3]
    top3 = [(labels[i], float(probs[i])) for i in top3_idx]
    return top_label, top_conf, top3


def find_sample_image(term: str, sample_dir: str):
    if not os.path.isdir(sample_dir):
        return None

    want = keyify(term)
    if not want:
        return None

    for fn in os.listdir(sample_dir):
        low = fn.lower()
        if not low.endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue
        name_no_ext = os.path.splitext(fn)[0]
        if want in keyify(name_no_ext):
            return os.path.join(sample_dir, fn)

    return None


def open_image_file(path: str):
    try:
        os.startfile(path)  # Windows
    except Exception:
        pass


def main():
    ensure_nltk()

    base_dir = os.path.dirname(__file__)
    aiml_path = os.path.join(base_dir, "aiml", "dietbot.aiml")
    csv_path = os.path.join(base_dir, "data", "qa_pairs.csv")
    kb_path = os.path.join(base_dir, "data", "kb.csv")

    model_path = os.path.join(base_dir, "models", "fruit_model.h5")
    labels_path = os.path.join(base_dir, "models", "labels.json")

    sample_dir = os.path.join(base_dir, "sample_images")

    kern = aiml.Kernel()
    kern.learn(aiml_path)

    sim_qa = SimilarityQA(csv_path)

    kb = load_kb(kb_path)
    ok, bad_expr = check_kb_integrity(kb)
    if not ok:
        print("Error: the initial knowledge base has a contradiction.")
        print("Problem statement:", bad_expr)
        return

    fruit_model = None
    fruit_labels = None
    if os.path.exists(model_path) and os.path.exists(labels_path):
        fruit_model = tf.keras.models.load_model(model_path)
        with open(labels_path, "r", encoding="utf-8") as f:
            fruit_labels = json.load(f)

    print("Welcome to DietBot. Type 'help' for options. Type 'bye' to exit.")

    while True:
        try:
            user_input = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not user_input:
            continue

        answer = kern.respond(user_input) or ""

        if answer.startswith("#"):
            params = answer[1:].split("$")
            cmd = params[0] if params else ""

            if cmd == "0":
                print(params[1] if len(params) > 1 else "Bye!")
                break

            if cmd == "1":
                try:
                    topic = params[1] if len(params) > 1 else ""
                    topic = topic.strip()
                    if not topic:
                        print("Tell me what topic to search on Wikipedia.")
                    else:
                        print(wikipedia.summary(topic, sentences=3, auto_suggest=True))
                except Exception:
                    print("Sorry, I do not know that. Be more specific!")
                continue

            if cmd == "SIM":
                best_ans, score = sim_qa.answer(user_input, threshold=0.32)
                if best_ans:
                    print(best_ans)
                    print("Question: want a meal plan for your goal? (lose weight / gain muscle / maintain)")
                else:
                    print("I’m not sure about that. Try rephrasing or ask about diet goals.")
                continue

            if cmd == "KBADD":
                if len(params) < 3:
                    print("Please use: I know that X is Y")
                    continue

                entity = clean_entity(params[1])
                prop = clean_property(params[2])

                if not entity or not prop:
                    print("Please use: I know that X is Y")
                    continue

                new_expr = read_expr(f"{prop}({entity})")

                if kb_contradicts(kb, new_expr):
                    print("I cannot add that because it contradicts what I already know.")
                else:
                    kb.append(new_expr)
                    print(f"OK, I'll remember that {entity} is {prop}.")
                continue

            if cmd == "KBCHECK":
                if len(params) < 3:
                    print("Please use: Check that X is Y")
                    continue

                entity = clean_entity(params[1])
                prop = clean_property(params[2])

                if not entity or not prop:
                    print("Please use: Check that X is Y")
                    continue

                expr = read_expr(f"{prop}({entity})")
                result = check_fact(kb, expr)

                # required outputs first
                print(result)

                # clearer message after
                if result == "Correct":
                    if prop == "high_protein":
                        print("Tip: High protein helps muscle gain and keeps you full.")
                    elif prop == "high_fiber":
                        print("Tip: High fiber helps satiety and supports weight loss.")
                    elif prop == "high_sugar":
                        print("Tip: High sugar foods are easy to overeat. Try swaps like fruit or yogurt.")
                    elif prop == "good_for_weight_loss":
                        print("Tip: Combine this with protein to stay full.")
                    elif prop == "bad_for_weight_loss":
                        print("Tip: Reduce portion and swap to a healthier option.")
                print("Question: want to check another fact?")
                continue

            if cmd == "SHOWIMG":
                term = params[1].strip() if len(params) > 1 else ""
                if not term:
                    print("Type: show image of chicken")
                    continue

                sample_path = find_sample_image(term, sample_dir)
                if sample_path:
                    print(f"Opening sample image for: {term}")
                    open_image_file(sample_path)
                else:
                    print("No matching image found in sample_images.")
                    print("Tip: add a file like chicken.jpg or salmon.jpg to sample_images.")
                continue

            if cmd == "IMG":
                if fruit_model is None or fruit_labels is None:
                    print("Image model not found. Make sure models/fruit_model.h5 and models/labels.json exist.")
                    continue

                print("Press Enter to choose a file, or paste a full image path and press Enter.")
                typed_path = input("Image path (optional): ").strip()

                img_path = typed_path if typed_path else pick_image_file()
                if not img_path:
                    print("No image selected.")
                    continue

                label, conf, top3 = predict_image(fruit_model, fruit_labels, img_path, img_size=(100, 100))

                print(f"The image contains: {label}")
                for name, p in top3:
                    print(f"{name}: {p*100:.2f}%")

                tips = {
                    "Avocado": "Healthy fats, good for satiety. Keep portion moderate if cutting.",
                    "Banana": "Good pre workout carbs.",
                    "Dates": "High energy carbs. Good before training, watch quantity.",
                    "Nut": "High calories and healthy fats. Use small portions.",
                    "Potato Red": "Good carbs for training. Combine with protein for balance."
                }
                if label in tips:
                    print("Tip:", tips[label])

                # Extra: confidence fallback
                if conf < 0.60 and len(top3) >= 2:
                    print(f"Sorry, I am not confident. It might be {top3[0][0]} or {top3[1][0]}.")

                # Optional: open a matching sample image if you named files like avocado_1.jpg etc
                sample_path = find_sample_image(label, sample_dir)
                if sample_path:
                    print("Opening a sample image from sample_images...")
                    open_image_file(sample_path)

                print("Question: want to classify another image? type: image")
                continue

            print("I did not get that, please try again.")
            continue

        if answer.strip():
            print(answer)
        else:
            best_ans, score = sim_qa.answer(user_input, threshold=0.32)
            print(best_ans if best_ans else "I’m not sure about that. Try rephrasing or ask about diet goals.")


if __name__ == "__main__":
    main()