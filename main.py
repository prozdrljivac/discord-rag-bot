import os

import discord
from dotenv import load_dotenv

from db import MilvusDB

load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")
DB_PATH = os.getenv("DB_PATH", "./test.db")

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)
db = MilvusDB(DB_PATH)


@client.event
async def on_ready():
    print(f"We have logged in as {client.user}")


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    embedding = [0.1] * 1536  # Placeholder embedding (later use OpenAI API)
    retrieved_text = db.retrieve_text(embedding)

    if retrieved_text:
        await message.channel.send(f"Here's what I found: {retrieved_text}")
    else:
        await message.channel.send("I don't know that yet!")


client.run(TOKEN)
