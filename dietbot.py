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


# ---------- NLTK setup (downloads only once) ----------
def ensure_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    # Punkt sometimes needs this extra table
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab")

    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet")

    try:
        nltk.data.find("corpora/omw-1.4")
    except LookupError:
        nltk.download("omw-1.4")


lemmatizer = WordNetLemmatizer()


def normalise_for_similarity(text: str) -> str:
    """
    Lowercase -> tokenize -> keep alphabetic tokens -> lemmatise -> join back.
    This is what we feed to TF/IDF.
    """
    text = text.lower().strip()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha()]
    lemmas = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(lemmas)


# ---------- CSV Q/A Similarity Engine ----------
class SimilarityQA:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        if "question" not in self.df.columns or "answer" not in self.df.columns:
            raise ValueError("qa_pairs.csv must have headers: question,answer")

        self.questions_raw = self.df["question"].astype(str).tolist()
        self.answers = self.df["answer"].astype(str).tolist()

        # Pre-normalise questions
        self.questions_norm = [normalise_for_similarity(q) for q in self.questions_raw]

        # TF/IDF model trained on question list
        self.vectorizer = TfidfVectorizer()
        self.q_matrix = self.vectorizer.fit_transform(self.questions_norm)

    def answer(self, user_text: str, threshold: float = 0.25) -> tuple[str, float, str]:
        """
        Returns: (best_answer, best_score, matched_question)
        If score < threshold -> returns ("", score, matched_question)
        """
        user_norm = normalise_for_similarity(user_text)
        if not user_norm:
            return "", 0.0, ""

        u_vec = self.vectorizer.transform([user_norm])
        sims = cosine_similarity(u_vec, self.q_matrix)[0]
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])

        matched_q = self.questions_raw[best_idx]
        best_ans = self.answers[best_idx]

        if best_score < threshold:
            return "", best_score, matched_q
        return best_ans, best_score, matched_q


# ---------- Main Chatbot ----------
def main():
    ensure_nltk()

    base_dir = os.path.dirname(__file__)
    aiml_path = os.path.join(base_dir, "aiml", "dietbot.aiml")
    csv_path = os.path.join(base_dir, "data", "qa_pairs.csv")

    # AIML kernel
    kern = aiml.Kernel()
    kern.learn(aiml_path)

    # Similarity Q/A
    sim_qa = SimilarityQA(csv_path)

    print("Welcome to DietBot. Type 'help' for options. Type 'bye' to exit.")

    while True:
        try:
            user_input = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not user_input:
            continue

        # AIML response first
        answer = kern.respond(user_input) or ""

        # Handle special commands from AIML
        if answer.startswith("#"):
            params = answer[1:].split("$")
            cmd = params[0] if params else ""

            # Exit
            if cmd == "0":
                print(params[1] if len(params) > 1 else "Bye!")
                break

            # Wikipedia
            if cmd == "1":
                try:
                    topic = params[1] if len(params) > 1 else ""
                    if not topic.strip():
                        print("Tell me what topic to search on Wikipedia.")
                    else:
                        print(wikipedia.summary(topic, sentences=3, auto_suggest=True))
                except Exception:
                    print("Sorry, I do not know that. Be more specific!")
                continue

            # Similarity trigger from AIML fallback
            if cmd == "SIM":
                best_ans, score, matched_q = sim_qa.answer(user_input, threshold=0.25)
                if best_ans:
                    print(best_ans)
                else:
                    print("I’m not sure about that. Try rephrasing or ask about diet/workout goals.")
                continue

            # Unknown command
            print("I did not get that, please try again.")
            continue

        # Normal AIML output
        if answer.strip():
            print(answer)
        else:
            # If AIML returns empty (rare), fallback to similarity anyway
            best_ans, score, matched_q = sim_qa.answer(user_input, threshold=0.25)
            print(best_ans if best_ans else "I’m not sure about that. Try rephrasing or ask about diet/workout goals.")


if __name__ == "__main__":
    main()
