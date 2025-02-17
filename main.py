import os

import discord
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

    await message.channel.send(f"Here's what I found: {retrieved_text}")


client.run(TOKEN)
