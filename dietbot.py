#!/usr/bin/env python3
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # hide TensorFlow INFO logs

import re
import json
import time
import warnings
import logging
from pathlib import Path

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


# -----------------------------
# Quiet down noisy libraries
# -----------------------------
warnings.filterwarnings("ignore", message="No parser was explicitly specified")
tf.get_logger().setLevel(logging.ERROR)
try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
except Exception:
    pass


def say(text: str):
    """Print preserving internal newlines."""
    if text is None:
        return
    print(str(text).rstrip("\n"))


# -----------------------------
# Menu text (printed by Python)
# -----------------------------
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
            "6) rate meal protein=40 sugar=5 fiber=8\n"
            "\n"
            "Example: diet"
        )

    if code == "HELP":
        return (
            "You can ask me about:\n"
            "- lose weight / gain muscle / maintenance\n"
            "- high protein foods / healthy snacks\n"
            "- pre workout / post workout\n"
            "\n"
            "Logic:\n"
            "- I know that Chicken is high protein\n"
            "- Check that Oats is good for weight loss\n"
            "- Check if chicken is high protein\n"
            "\n"
            "Images:\n"
            "- image (baseline CNN)\n"
            "- image tl (MobileNetV2 transfer learning)\n"
            "\n"
            "Extra (Task B):\n"
            "- rate meal protein=40 sugar=5 fiber=8\n"
            "\n"
            "Extra (Task A):\n"
            "- define calorie deficit"
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
            "> I know that Chicken is high protein\n"
            "OK, I will remember that Chicken is high_protein.\n"
            "\n"
            "> Check that Oats is good for weight loss\n"
            "It may not be true... let me check...\n"
            "Correct\n"
            "\n"
            "You can also write:\n"
            "- Check if chicken is high protein"
        )

    if code == "LOSE":
        return (
            "Weight loss basics:\n"
            "- small calorie deficit\n"
            "- high protein + high fiber\n"
            "- strength training + steps"
        )

    if code == "GAIN":
        return (
            "Muscle gain basics:\n"
            "- small calorie surplus\n"
            "- enough protein daily\n"
            "- progressive overload training"
        )

    if code == "MAINT":
        return (
            "Maintenance basics:\n"
            "- eat around maintenance calories\n"
            "- keep protein consistent\n"
            "- keep activity stable"
        )

    if code == "PROTEIN":
        return (
            "High protein foods include:\n"
            "- chicken, eggs, tuna\n"
            "- Greek yogurt, lentils, tofu\n"
            "Question: vegetarian or non veg?"
        )

    if code == "SNACKS":
        return (
            "Healthy snacks:\n"
            "- fruit, Greek yogurt\n"
            "- boiled eggs\n"
            "- hummus with carrots"
        )

    if code == "PRE":
        return (
            "Pre workout:\n"
            "- carbs + a little protein\n"
            "Examples: banana and yogurt, oats, rice and chicken"
        )

    if code == "POST":
        return (
            "Post workout:\n"
            "- protein + carbs\n"
            "Examples: chicken and rice, tuna sandwich, whey and banana"
        )

    return "Type: help"


# -----------------------------
# NLTK
# -----------------------------
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


# -----------------------------
# Task A: similarity Q/A
# -----------------------------
def normalise_for_similarity(text: str) -> str:
    text = text.lower().strip()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha()]
    lemmas = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(lemmas)


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

    def answer(self, user_text: str, threshold: float = 0.32) -> str:
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


# -----------------------------
# Task B: logic KB
# -----------------------------
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
    fixes = {
        "high_protine": "high_protein",
        "protine": "protein",
        "protin": "protein",
        "suger": "sugar",
    }
    return fixes.get(text, text)


def strip_leading_the(text: str) -> str:
    return re.sub(r"^(the)\s+", "", text.strip(), flags=re.IGNORECASE)


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
    return "Sorry I don't know."


def parse_know_like(text: str):
    m = re.match(r"^\s*i\s+know\s+that\s+(.+?)\s+(is|ia|iz|=)\s+(.+)\s*$", text, re.IGNORECASE)
    if not m:
        return None
    entity_raw = strip_leading_the(m.group(1))
    return clean_entity(entity_raw), clean_property(m.group(3))


def parse_check_like(text: str):
    m = re.match(r"^\s*check\s+that\s+(.+?)\s+(is|ia|iz|=)\s+(.+)\s*$", text, re.IGNORECASE)
    if not m:
        return None
    entity_raw = strip_leading_the(m.group(1))
    return clean_entity(entity_raw), clean_property(m.group(3))


def parse_checkif_payload(payload: str):
    payload = payload.strip()
    payload = re.sub(r"^(the)\s+", "", payload, flags=re.IGNORECASE)
    m = re.match(r"^(.+?)\s+(is|ia|iz|=)\s+(.+)$", payload, flags=re.IGNORECASE)
    if not m:
        return None
    entity = clean_entity(m.group(1))
    prop = clean_property(m.group(3))
    return entity, prop


# -----------------------------
# Task B EXTRA: fuzzy meal score
# -----------------------------
def tri(x, a, b, c):
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a + 1e-9)
    return (c - x) / (c - b + 1e-9)


def fuzzy_meal_score(protein_g: float, sugar_g: float, fiber_g: float):
    p_low = tri(protein_g, 0, 0, 20)
    p_med = tri(protein_g, 15, 30, 45)
    p_high = tri(protein_g, 35, 60, 60)

    s_low = tri(sugar_g, 0, 0, 10)
    s_med = tri(sugar_g, 8, 20, 35)
    s_high = tri(sugar_g, 30, 60, 60)

    f_low = tri(fiber_g, 0, 0, 3)
    f_med = tri(fiber_g, 2, 6, 10)
    f_high = tri(fiber_g, 8, 20, 20)

    healthy = max(min(p_high, s_low, f_high), min(p_high, s_low, f_med))
    okay = max(min(p_med, s_low, f_med), min(p_med, s_med, f_med))
    unhealthy = max(s_high, min(p_low, f_low), min(s_med, p_low))

    denom = healthy + okay + unhealthy + 1e-9
    score = (healthy * 0.85 + okay * 0.55 + unhealthy * 0.20) / denom

    if score >= 0.70:
        label = "Healthy"
    elif score >= 0.40:
        label = "Okay"
    else:
        label = "Unhealthy"

    return label, score


def parse_rate_meal(payload: str):
    payload = payload.strip().lower()
    nums = re.findall(r"[-+]?\d*\.?\d+", payload)
    has_keys = ("protein" in payload) or ("sugar" in payload) or ("fiber" in payload)

    if has_keys:
        def get_val(key):
            m = re.search(rf"{key}\s*[:=]\s*([-+]?\d*\.?\d+)", payload)
            return float(m.group(1)) if m else None

        p = get_val("protein")
        s = get_val("sugar")
        f = get_val("fiber")
        if p is None or s is None or f is None:
            return None
        return p, s, f

    if len(nums) >= 3:
        return float(nums[0]), float(nums[1]), float(nums[2])

    return None


# -----------------------------
# Task C: image classification
# -----------------------------
def pick_image_file():
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = askopenfilename(
        title="Select image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    root.destroy()
    return path


def predict_probs(model, image_path, img_size):
    img = tf.keras.utils.load_img(image_path, target_size=img_size)
    arr = tf.keras.utils.img_to_array(img)
    arr = tf.expand_dims(arr, 0)  # models already include rescaling
    probs = model.predict(arr, verbose=0)[0]
    return probs


# -----------------------------
# Task A EXTRA: Wikipedia lookup
# -----------------------------
WIKI_ALIASES = {
    "ai": "Artificial intelligence",
    "ml": "Machine learning",
    "cpu": "Central processing unit",
    "gpu": "Graphics processing unit",
    "protein": "Protein",
}


def wiki_summary_safe(query: str) -> str:
    query = (query or "").strip()
    if not query:
        return "Sorry, I do not know that. Be more specific!"

    wikipedia.set_lang("en")

    key = query.lower().strip()
    if key in WIKI_ALIASES:
        query = WIKI_ALIASES[key]

    try:
        return wikipedia.summary(query, sentences=3, auto_suggest=False, redirect=True)
    except Exception:
        pass

    try:
        return wikipedia.summary(query, sentences=3, auto_suggest=True, redirect=True)
    except Exception:
        pass

    try:
        results = wikipedia.search(query, results=5)
        if results:
            return wikipedia.summary(results[0], sentences=3, auto_suggest=False, redirect=True)
    except Exception:
        pass

    return "Sorry, I do not know that. Be more specific!"


# -----------------------------
# Extra utility: show sample image
# -----------------------------
def keyify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def find_sample_image(term: str, sample_dir: str):
    if not os.path.isdir(sample_dir):
        return None
    want = keyify(term)
    for fn in os.listdir(sample_dir):
        if not fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue
        if want in keyify(Path(fn).stem):
            return os.path.join(sample_dir, fn)
    return None


def open_image_file(path: str):
    try:
        os.startfile(path)  # Windows
    except Exception:
        pass


def main():
    ensure_nltk()

    base_dir = Path(__file__).resolve().parent
    aiml_path = base_dir / "aiml" / "dietbot.aiml"
    csv_path = base_dir / "data" / "qa_pairs.csv"
    kb_path = base_dir / "data" / "kb.csv"

    model_path = base_dir / "models" / "fruit_model.h5"
    labels_path = base_dir / "models" / "labels.json"

    tl_model_path = base_dir / "models" / "fruit_mobilenet.h5"
    tl_labels_path = base_dir / "models" / "labels_mobilenet.json"

    sample_dir = base_dir / "sample_images"

    say("============= RESTART: dietbot.py =============")

    # Load AIML
    if not aiml_path.exists():
        say(f"AIML file missing: {aiml_path}")
        return

    kern = aiml.Kernel()
    t0 = time.perf_counter()
    kern.learn(str(aiml_path))
    t1 = time.perf_counter()
    say(f"Kernel bootstrap completed in {t1 - t0:.2f} seconds")

    sim_qa = SimilarityQA(str(csv_path))

    kb = load_kb(str(kb_path))
    ok, bad_expr = check_kb_integrity(kb)
    if not ok:
        say("Error: the initial knowledge base has a contradiction.")
        say(f"Problem statement: {bad_expr}")
        return

    if not model_path.exists() or not labels_path.exists():
        say("Image model not found. Run: python training/train_fruit_model.py")
        say(f"Missing: {model_path} or {labels_path}")
        return

    fruit_model = tf.keras.models.load_model(str(model_path))
    fruit_labels = json.loads(labels_path.read_text(encoding="utf-8"))

    tl_model = None
    tl_labels = None
    if tl_model_path.exists() and tl_labels_path.exists():
        tl_model = tf.keras.models.load_model(str(tl_model_path))
        tl_labels = json.loads(tl_labels_path.read_text(encoding="utf-8"))

    say("Welcome to DietBot.")
    say("Type 'help' to see examples.")
    say("Type 'bye' to exit.")

    while True:
        try:
            user_input = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            say("Bye!")
            break

        if not user_input:
            continue

        # Task B: robust "I know that..." / "Check that..."
        know = parse_know_like(user_input)
        if know:
            entity, prop = know
            new_expr = read_expr(f"{prop}({entity})")
            if kb_contradicts(kb, new_expr):
                say("Sorry this contradicts with what I know!")
            else:
                kb.append(new_expr)
                say(f"OK, I will remember that {entity} is {prop}.")
            continue

        chk = parse_check_like(user_input)
        if chk:
            entity, prop = chk
            expr = read_expr(f"{prop}({entity})")
            say("It may not be true... let me check...")
            say(check_fact(kb, expr))
            continue

        # AIML
        answer = kern.respond(user_input) or ""

        # ✅ FIX: if AIML returns normal text, print it
        if answer and not answer.startswith("#"):
            say(answer)
            continue

        # AIML commands
        if answer.startswith("#"):
            params = answer[1:].split("$")
            cmd = params[0] if params else ""

            if cmd == "0":
                say(params[1] if len(params) > 1 else "Bye!")
                break

            if cmd == "MENU":
                say(menu_text(params[1] if len(params) > 1 else "HOME"))
                continue

            if cmd in ("WIKI", "1"):
                topic = params[1] if len(params) > 1 else ""
                say(wiki_summary_safe(topic))
                continue

            if cmd == "KBADD":
                if len(params) < 3:
                    say("Please use: I know that X is Y")
                    continue
                entity = clean_entity(strip_leading_the(params[1]))
                prop = clean_property(params[2])
                new_expr = read_expr(f"{prop}({entity})")
                if kb_contradicts(kb, new_expr):
                    say("Sorry this contradicts with what I know!")
                else:
                    kb.append(new_expr)
                    say(f"OK, I will remember that {entity} is {prop}.")
                continue

            if cmd == "KBCHECK":
                if len(params) < 3:
                    say("Please use: Check that X is Y")
                    continue
                entity = clean_entity(strip_leading_the(params[1]))
                prop = clean_property(params[2])
                expr = read_expr(f"{prop}({entity})")
                say("It may not be true... let me check...")
                say(check_fact(kb, expr))
                continue

            if cmd == "SIM":
                best = sim_qa.answer(user_input, threshold=0.32)
                say(best if best else "I did not get that, please try again.")
                continue

            if cmd == "CHECKIF":
                payload = params[1] if len(params) > 1 else ""
                parsed = parse_checkif_payload(payload)
                if not parsed:
                    say("Please use: check if X is Y")
                    continue
                entity, prop = parsed
                expr = read_expr(f"{prop}({entity})")
                say("It may not be true... let me check...")
                say(check_fact(kb, expr))
                continue

            if cmd == "FUZZY":
                payload = params[1] if len(params) > 1 else ""
                parsed = parse_rate_meal(payload)
                if not parsed:
                    say("Please use:")
                    say("rate meal protein=40 sugar=5 fiber=8")
                    say("or: rate meal 40 5 8")
                    continue
                p, s, f = parsed
                label, score = fuzzy_meal_score(p, s, f)
                say(f"Meal score: {label} ({score:.2f})")
                say(f"- protein={p}g, sugar={s}g, fiber={f}g")
                continue

            if cmd == "SHOWIMG":
                term = params[1] if len(params) > 1 else ""
                sample_path = find_sample_image(term, str(sample_dir))
                if sample_path:
                    say(f"Opening image for: {term}")
                    open_image_file(sample_path)
                else:
                    say("Sorry, I cannot find that image in sample_images.")
                continue

            if cmd == "IMG":
                say("Select image file (a dialog will open)...")
                img_path = pick_image_file()
                if not img_path:
                    say("No image selected.")
                    continue

                probs = predict_probs(fruit_model, img_path, img_size=(100, 100))
                top_idx = int(np.argmax(probs))
                top_label = fruit_labels[top_idx]
                top_conf = float(probs[top_idx])

                say(f"The image contains: {top_label}")
                for i, name in enumerate(fruit_labels):
                    say(f"{name}: {probs[i]*100:.2f}%")

                # Task C "extra-like" behaviour: low-confidence warning
                if top_conf < 0.60:
                    say("Note: I am not confident. This image may be outside my dataset.")
                else:
                    say("Note: this model only knows these classes:")
                    say(", ".join(fruit_labels))
                continue

            if cmd == "IMGTL":
                if tl_model is None or tl_labels is None:
                    say("Transfer model not found. Run: python training/train_fruit_mobilenet.py")
                    continue

                say("Select image file (a dialog will open)...")
                img_path = pick_image_file()
                if not img_path:
                    say("No image selected.")
                    continue

                probs = predict_probs(tl_model, img_path, img_size=(160, 160))
                top_idx = int(np.argmax(probs))
                top_label = tl_labels[top_idx]
                top_conf = float(probs[top_idx])

                say(f"The image contains (MobileNetV2): {top_label}")
                for i, name in enumerate(tl_labels):
                    say(f"{name}: {probs[i]*100:.2f}%")

                if top_conf < 0.60:
                    say("Note: I am not confident. This image may be outside my dataset.")
                else:
                    say("Note: this model only knows these classes:")
                    say(", ".join(tl_labels))
                continue

            say("I did not get that, please try again.")
            continue

        # No AIML match → similarity fallback
        best = sim_qa.answer(user_input, threshold=0.32)
        say(best if best else "I did not get that, please try again.")


if __name__ == "__main__":
    main()