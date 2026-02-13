#!/usr/bin/env python3
import os
import aiml
import wikipedia

kern = aiml.Kernel()

# Load the AIML file that sits next to this .py
aiml_path = os.path.join(os.path.dirname(__file__), "mybot-basic.xml")
kern.learn(aiml_path)   # ← replaces bootstrap(); avoids time.clock()

print("Welcome to this chat bot. Please feel free to ask questions from me!")

while True:
    try:
        userInput = input("> ")
    except (KeyboardInterrupt, EOFError):
        print("Bye!")
        break

    answer = kern.respond(userInput) or ""   # avoid IndexError when empty

    if answer.startswith("#"):
        params = answer[1:].split("$")
        try:
            cmd = int(params[0])
        except ValueError:
            cmd = 99

        if cmd == 0:
            print(params[1] if len(params) > 1 else "Bye!")
            break
        elif cmd == 1:
            try:
                topic = params[1] if len(params) > 1 else ""
                print(wikipedia.summary(topic, sentences=3, auto_suggest=True))
            except Exception:
                print("Sorry, I do not know that. Be more specific!")
        else:
            print("I did not get that, please try again.")
    else:
        print(answer if answer else "I did not get that, please try again.")
