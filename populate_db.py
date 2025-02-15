import os

from dotenv import load_dotenv

from db import MilvusDB
from embedding import get_embedding

load_dotenv()
DB_PATH = os.getenv("DB_PATH", "./test.db")

anime_data = [
    "Attack on Titan was created by Hajime Isayama and first serialized in 2009.",
    "Naruto follows the story of Naruto Uzumaki, a ninja seeking recognition and aiming to become Hokage.",
    "One Piece is the longest-running anime series by Eiichiro Oda, following Monkey D. Luffy's adventure to find the One Piece treasure.",
    "Demon Slayer: Kimetsu no Yaiba follows Tanjiro Kamado, who fights demons to avenge his family.",
    "Death Note is a psychological thriller where Light Yagami discovers a notebook that lets him kill anyone by writing their name.",
    "Fullmetal Alchemist is about brothers Edward and Alphonse Elric, who use alchemy to search for the Philosopher's Stone.",
    "My Hero Academia is set in a world where most people have superpowers (quirks), following Izuku Midoriya's journey to becoming a hero.",
]


def populate_database():
    """Inserts anime-related knowledge into the database."""
    db = MilvusDB(DB_PATH)

    for text in anime_data:
        embedding = get_embedding(text)
        db.insert_text(text, embedding)
        print(f"Inserted: {text}")

    print("Database population complete!")


if __name__ == "__main__":
    populate_database()
