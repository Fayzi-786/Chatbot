#!/usr/bin/env python3
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

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


# ---------- clean printing ----------
def say(text: str):
    print(str(text).strip())


def menu_text(code: str) -> str:
    code = (code or "").upper().strip()

    if code == "HOME":
        return (
            "DietBot menu:\n"
            "1) diet\n"
            "2) meals\n"
            "3) logic\n"
            "4) image\n"
            "5) define <topic>\n"
            "\n"
            "Example: diet"
        )

    if code == "HELP":
        return (
            "DietBot features:\n"
            "- Task A: AIML + similarity (TF-IDF + cosine)\n"
            "- Task B: logic KB (I know that... / Check that...)\n"
            "- Task C: image classifier (type: image)\n"
            "\n"
            "Extras:\n"
            "- Wikipedia: define / what is / who is\n"
            "- show image of <thing>\n"
            "\n"
            "Type: diet, meals, logic, image"
        )

    if code == "DIET":
        return (
            "Diet options:\n"
            "- lose weight\n"
            "- gain muscle\n"
            "- maintenance\n"
            "- high protein foods\n"
            "- healthy snacks\n"
            "\n"
            "Example: lose weight"
        )

    if code == "DIET_TIPS":
        return (
            "Diet tips:\n"
            "- high protein foods\n"
            "- healthy snacks\n"
            "\n"
            "Example: high protein foods"
        )

    if code == "MEALS":
        return (
            "Meals:\n"
            "- pre workout\n"
            "- post workout\n"
            "\n"
            "Example: pre workout"
        )

    if code == "LOGIC":
        return (
            "Logic examples:\n"
            "- I know that Chicken is high protein\n"
            "- Check that Oats is good for weight loss\n"
            "\n"
            "Tip: spelling can be a bit flexible (is / ia / =)."
        )

    if code == "LOSE":
        return (
            "Weight loss basics:\n"
            "- small calorie deficit\n"
            "- high protein + high fiber\n"
            "- strength training + steps\n"
            "\n"
            "Try: healthy snacks"
        )

    if code == "GAIN":
        return (
            "Muscle gain basics:\n"
            "- small calorie surplus\n"
            "- enough protein daily\n"
            "- progressive overload training\n"
            "\n"
            "Try: high protein foods"
        )

    if code == "MAINT":
        return (
            "Maintenance basics:\n"
            "- eat around maintenance calories\n"
            "- keep protein consistent\n"
            "- keep activity stable\n"
        )

    if code == "PROTEIN":
        return (
            "High protein foods:\n"
            "- chicken, eggs, tuna\n"
            "- Greek yogurt, lentils, tofu\n"
            "\n"
            "Question: vegetarian or non veg?"
        )

    if code == "SNACKS":
        return (
            "Healthy snacks:\n"
            "- fruit, Greek yogurt\n"
            "- boiled eggs\n"
            "- hummus with carrots\n"
            "\n"
            "Question: any allergies?"
        )

    if code == "PRE":
        return (
            "Pre workout:\n"
            "- carbs + a little protein\n"
            "- banana and yogurt, oats, rice and chicken\n"
            "\n"
            "Question: training in 30 minutes or 2 hours?"
        )

    if code == "POST":
        return (
            "Post workout:\n"
            "- protein + carbs\n"
            "- chicken and rice, tuna sandwich, whey and banana\n"
            "\n"
            "Question: goal is muscle gain or fat loss?"
        )

    return "Type: help"


# ---------- NLTK ----------
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


# ---------- Task A similarity ----------
def normalise_for_similarity(text: str) -> str:
    text = text.lower().strip()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha()]
    lemmas = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(lemmas)


class SimilarityQA:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path, encoding="utf-8", engine="python")
        self.questions_raw = self.df["question"].astype(str).tolist()
        self.answers = self.df["answer"].astype(str).tolist()

        self.questions_norm = [normalise_for_similarity(q) for q in self.questions_raw]
        self.vectorizer = TfidfVectorizer()
        self.q_matrix = self.vectorizer.fit_transform(self.questions_norm)

    def answer(self, user_text: str, threshold: float = 0.32):
        user_norm = normalise_for_similarity(user_text)
        if not user_norm:
            return ""

        u_vec = self.vectorizer.transform([user_norm])
        sims = cosine_similarity(u_vec, self.q_matrix)[0]
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])

        if best_score < threshold:
            return ""
        return self.answers[best_idx]


# ---------- Task B cleaning ----------
def clean_entity(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", "", text).replace("_", "")
    if not text:
        return text
    return text[0].upper() + text[1:].lower()


def clean_property(text: str) -> str:
    text = text.strip().lower()
    text = text.replace("-", " ")
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"_+", "_", text)

    fixes = {"high_protine": "high_protein", "protine": "protein", "protin": "protein", "suger": "sugar"}
    return fixes.get(text, text)


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


def parse_know_like(text: str):
    m = re.match(r"^\s*i\s+know\s+that\s+(.+?)\s+(is|ia|iz|=)\s+(.+)\s*$", text, re.IGNORECASE)
    if not m:
        return None
    return clean_entity(m.group(1)), clean_property(m.group(3))


def parse_check_like(text: str):
    m = re.match(r"^\s*check\s+that\s+(.+?)\s+(is|ia|iz|=)\s+(.+)\s*$", text, re.IGNORECASE)
    if not m:
        return None
    return clean_entity(m.group(1)), clean_property(m.group(3))


# ---------- Task C ----------
def pick_image_file():
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = askopenfilename(title="Select an image", filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
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


def keyify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def find_sample_image(term: str, sample_dir: str):
    if not os.path.isdir(sample_dir):
        return None
    want = keyify(term)
    for fn in os.listdir(sample_dir):
        if fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            if want in keyify(os.path.splitext(fn)[0]):
                return os.path.join(sample_dir, fn)
    return None


def open_image_file(path: str):
    try:
        os.startfile(path)
    except Exception:
        pass


# ---------- Wikipedia fix ----------
def wiki_summary_safe(query: str) -> str:
    query = (query or "").strip()
    if not query:
        return "Tell me what topic to search on Wikipedia."

    # Try normal summary
    try:
        return wikipedia.summary(query, sentences=3, auto_suggest=True, redirect=True)
    except Exception:
        pass

    # Try search-based fallback
    try:
        results = wikipedia.search(query, results=5)
        if results:
            return wikipedia.summary(results[0], sentences=3, auto_suggest=False, redirect=True)
    except Exception:
        pass

    # Try removing leading articles (a/an/the)
    try:
        cleaned = re.sub(r"^(a|an|the)\s+", "", query.lower()).strip()
        if cleaned and cleaned != query.lower():
            results = wikipedia.search(cleaned, results=5)
            if results:
                return wikipedia.summary(results[0], sentences=3, auto_suggest=False, redirect=True)
    except Exception:
        pass

    return "Sorry, I do not know that. Be more specific!"


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
        say("Error: the initial knowledge base has a contradiction.")
        say(f"Problem statement: {bad_expr}")
        return

    fruit_model = tf.keras.models.load_model(model_path)
    with open(labels_path, "r", encoding="utf-8") as f:
        fruit_labels = json.load(f)

    say("Welcome to DietBot. Type 'help' for options. Type 'bye' to exit.")

    while True:
        try:
            user_input = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            say("Bye!")
            break

        if not user_input:
            continue

        # Logic: accept ia / is / =
        know = parse_know_like(user_input)
        if know:
            entity, prop = know
            new_expr = read_expr(f"{prop}({entity})")
            if kb_contradicts(kb, new_expr):
                say("I cannot add that because it contradicts what I already know.")
            else:
                kb.append(new_expr)
                say(f"OK, I'll remember that {entity} is {prop}.")
            continue

        chk = parse_check_like(user_input)
        if chk:
            entity, prop = chk
            expr = read_expr(f"{prop}({entity})")
            say(check_fact(kb, expr))
            continue

        answer = kern.respond(user_input) or ""

        if answer.startswith("#"):
            params = answer[1:].split("$")
            cmd = params[0] if params else ""

            if cmd == "0":
                say(params[1] if len(params) > 1 else "Bye!")
                break

            if cmd == "MENU":
                code = params[1] if len(params) > 1 else ""
                say(menu_text(code))
                continue

            if cmd == "1":
                topic = params[1].strip() if len(params) > 1 else ""
                say(wiki_summary_safe(topic))
                continue

            if cmd == "SIM":
                best = sim_qa.answer(user_input, threshold=0.32)
                if best:
                    say(best)
                else:
                    say("I’m not sure. Type: help")
                continue

            if cmd == "SHOWIMG":
                term = params[1].strip() if len(params) > 1 else ""
                sample_path = find_sample_image(term, sample_dir)
                if sample_path:
                    say(f"Opening sample image for: {term}")
                    open_image_file(sample_path)
                else:
                    say("No matching image found in sample_images.")
                continue

            if cmd == "IMG":
                say("Select an image file now...")
                img_path = pick_image_file()
                if not img_path:
                    say("No image selected.")
                    continue

                label, conf, top3 = predict_image(fruit_model, fruit_labels, img_path, img_size=(100, 100))
                say(f"The image contains: {label}")
                for name, p in top3:
                    say(f"- {name}: {p*100:.2f}%")
                if conf < 0.60 and len(top3) >= 2:
                    say(f"Sorry, I am not confident. It might be {top3[0][0]} or {top3[1][0]}.")
                continue

            say("Type: help")
            continue

        if answer.strip():
            say(answer)
        else:
            say("Type: help")


if __name__ == "__main__":
    main()