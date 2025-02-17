import os

import discord
import openai
import redis
from dotenv import load_dotenv

from db import MilvusDB
from embedding import get_embedding

load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")
DB_PATH = os.getenv("DB_PATH", "./test.db")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)
vector_db = MilvusDB(DB_PATH)
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)


def store_message(user_id: str, role: str, message: str):
    """
    Store user and bot messages in Redis with a limit of 10 messages per user.
    """

    key = f"chat_history:{user_id}"
    redis_client.rpush(key, f"{role}: {message}")

    # Keep only the last 10 messages
    redis_client.ltrim(key, -10, -1)


def get_chat_history(user_id: str) -> list:
    """Retrieve the last 10 messages from Redis."""
    key = f"chat_history:{user_id}"
    return redis_client.lrange(key, 0, -1)


def generate_response(user_id: str, question: str, retrieved_text: str) -> str:
    """Use GPT to generate a response with conversation memory."""

    chat_history = get_chat_history(user_id)
    history_text = (
        "\n".join(chat_history) if chat_history else "No prior conversation."
    )

    prompt = f"""
    You are an anime expert having a conversation with a user.
    Here is the conversation history:
    {history_text}

    The user now asks: "{question}"
    Here is some information related to their question: "{retrieved_text}"

    Respond in a conversational manner while considering the context.
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

    user_id = str(message.author.id)
    query_text = message.content

    store_message(user_id, "User", query_text)
    query_embedding = get_embedding(query_text)

    retrieved_text = vector_db.retrieve_text(query_embedding)

    if not retrieved_text:
        await message.channel.send("I don't know that yet!")
        return

    ai_response = generate_response(user_id, query_text, retrieved_text)
    await message.channel.send(ai_response)
    store_message(user_id, "Bot", ai_response)


client.run(TOKEN)
