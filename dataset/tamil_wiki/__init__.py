import os
from bs4 import BeautifulSoup, NavigableString
import re


def sentences():
    for f in os.listdir(os.path.join(os.path.dirname(__file__), "data")):
        print(f)
        f = os.path.join(os.path.dirname(__file__), "data", f)
        if os.path.isfile(f):
            with open(f, "r", encoding="utf-8") as f:
                for sentence in process(BeautifulSoup(f.read(), "html.parser")):
                    yield sentence


def process(tag, level=0):
    print("  " * level + f"[{tag.name}]: {tag.attrs}")
    # Recursively iterate over child tags
    for child in tag.contents:  # Using tag.contents instead of tag.children
        if isinstance(child, NavigableString):  # Handle NavigableString safely
            if child.strip():  # Ignore empty strings
                for sentence in re.split(r"[.!?]", child.strip()):
                    sentence = sentence.strip()
                    if sentence:
                        yield sentence
        elif child and child.name:  # Only process tags (not NavigableStrings)
            yield from process(child, level + 1)
