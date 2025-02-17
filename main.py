import os

import discord
import openai
from dotenv import load_dotenv

from db import MilvusDB
from embedding import get_embedding

load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")
DB_PATH = os.getenv("DB_PATH", "./test.db")

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)
vector_db = MilvusDB(DB_PATH)  # Initialize DB


def generate_response(question: str, retrieved_text: str) -> str:
    """
    Use GPT to generate a more natural response based on retrieved knowledge.
    """

    prompt = f"""
    You are an anime expert. A user asked: "{question}"
    Here is a fact related to their question: "{retrieved_text}"
    Formulate a clear and informative response using this information.
    """

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an anime chatbot."},
            {"role": "user", "content": prompt},
        ],
    )

    return response.choices[0].message.content


@client.event
async def on_ready():
    print(f"We have logged in as {client.user}")


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    query_text = message.content
    query_embedding = get_embedding(query_text)

    retrieved_text = vector_db.retrieve_text(query_embedding)

    if not retrieved_text:
        await message.channel.send("I don't know that yet!")
        return

    ai_response = generate_response(query_text, retrieved_text)
    await message.channel.send(ai_response)


client.run(TOKEN)
