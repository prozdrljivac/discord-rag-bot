import discord
import os
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f"Bot is online as {client.user}")

@client.event
async def on_message(message):
    print(f"Message received: '{message.content}' from {message.author}")
    if message.author == client.user:
        return
    if message.content == "!hello":
        await message.channel.send("Hey there!")

client.run(TOKEN)