import requests
import os

from dotenv import load_dotenv

load_dotenv()


BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "http://127.0.0.1:9000"
)


def save_permanent_memory(memory):

    url = (
        f"{BACKEND_URL}"
        "/memory/permanent"
    )

    response = requests.post(
        url,
        json=memory,
        timeout=30
    )

    response.raise_for_status()

    return response.json()