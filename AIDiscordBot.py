import json
import discord
from llama_cpp import Llama
from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.messages_formatter import MessagesFormatterType
main_model = Llama(
    "/home/oddmin/llama.cpp/models/8B/Meta-Llama-3-8B-Instruct.Q4_0.gguf",
    n_gpu_layers=0,
    f16_kv=True,
    use_mlock=False,
    embedding=False,
    n_threads=4,
    n_batch=1024,
    n_ctx=8192,
    last_n_tokens_size=128,
    verbose=True,
    seed=42,
)
wrapped_model = LlamaCppAgent(main_model, debug_output=True
                              ,system_prompt="You are a normall person with a side of r/sillygirlclub")
                              #, predefined_messages_formatter_type=MessagesFormatterType.CHATML)
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)
@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')
@client.event
async def on_message(message):
    if message.author == client.user or message.content.startswith('$'):
        return

    else:
        await message.channel.send(wrapped_model.get_chat_response(message.content, temperature=1,max_tokens=512))

client.run('')