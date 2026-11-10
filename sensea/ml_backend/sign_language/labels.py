LABELS = [
    "hello",
    "goodbye",
    "thank_you",
    "sorry",
    "please",
    "yes",
    "no",
    "help",
    "stop",
    "okay",
    "me",
    "you",
    "mother",
    "father",
    "friend",
    "home",
    "hospital",
    "eat",
    "drink",
    "water",
]

LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}

ID_TO_LABEL = {i: label for i, label in enumerate(LABELS)}
