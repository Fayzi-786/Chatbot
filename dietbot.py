#!/usr/bin/env python3
import os
import re
import aiml
import wikipedia
import pandas as pd
import numpy as np
import nltk

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
    text = re.sub(r"\s+", "", text)      # remove spaces for better output
    text = text.replace("_", "")         # remove underscores for better output
    if not text:
        return text
    return text[0].upper() + text[1:].lower()


def clean_property(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"_+", "_", text)      # collapse multiple underscores
    text = text.replace("-", " ")

    fixes = {
        "high_protine": "high_protein",
        "protine": "protein",
    }

    return fixes.get(text, text)


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


def main():
    ensure_nltk()

    base_dir = os.path.dirname(__file__)
    aiml_path = os.path.join(base_dir, "aiml", "dietbot.aiml")
    csv_path = os.path.join(base_dir, "data", "qa_pairs.csv")
    kb_path = os.path.join(base_dir, "data", "kb.csv")

    kern = aiml.Kernel()
    kern.learn(aiml_path)

    sim_qa = SimilarityQA(csv_path)

    kb = load_kb(kb_path)
    ok, bad_expr = check_kb_integrity(kb)
    if not ok:
        print("Error: the initial knowledge base has a contradiction.")
        print("Problem statement:", bad_expr)
        return

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
                print(check_fact(kb, expr))
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
