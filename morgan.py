import os
import random
import discord
from huggingface_hub import InferenceClient

# 環境変数の読み込み
TOKEN = os.getenv("TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")

# 応答を許可するチャンネルIDのリスト（カンマ区切りで複数指定可能）
env_channels = os.getenv("ALLOWED_CHANNEL_ID", "")
ALLOWED_CHANNEL_IDS = [
    int(ch_id.strip()) for ch_id in env_channels.split(",") if ch_id.strip().isdigit()
]

# Hugging Face InferenceClient の初期化
hf_client = InferenceClient(
    model="Qwen/Qwen2.5-Coder-32B-Instruct",
    token=HF_TOKEN
)

# Discord Intents 設定
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# キャラクター設定
CATCHPHRASE = "魔法の授業はちゃんとうけなさい！"
JOKE_KEYWORDS = [
    "ばか", "アホ", "ふざけ", "うんこ", "ちんこ", "へんたい", "ヌード",
    "死ね", "うそつき", "ふざけるな", "あほ", "バカ"
]

# モーガン先生の指示文（システムプロンプト）
SYSTEM_PROMPT = """
あなたの名前は「モーガン先生」です。
性格や特徴は以下の通りです：
・真面目な委員長タイプです。
・基本的には丁寧な敬語で話します。
・ただし、毎回ではなく「たまに」砕けた口調（フレンドリーなタメ口など）が混ざります。
・雑談には親切に応じますが、ダラダラしすぎないよう学習や成長を促す姿勢を見せます。
・短く分かりやすく返答してください（長文になりすぎないように）。
"""

def is_joking(text: str) -> bool:
    """ふざけた質問や不適切な発言の判定"""
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in JOKE_KEYWORDS)

def generate_chat_response(user_text: str) -> str:
    """Hugging Face API (Qwen2.5) を使用してモーガン先生の雑談応答を生成"""
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text}
        ]
        
        response = hf_client.chat_completion(
            messages=messages,
            max_tokens=250,
            temperature=0.7
        )
        reply = response.choices[0].message.content.strip()
        
        # 20%の確率で文末に口癖を付与
        if random.random() < 0.2 and CATCHPHRASE not in reply:
            reply += f"\n{CATCHPHRASE}"
        return reply

    except Exception as e:
        print(f"Hugging Face API エラー: {e}")
        return "ふむ、少し集中が切れてしまいました。もう一度話しかけてくれますか？"

@client.event
async def on_ready():
    print(f"モーガン先生（morgan.py）が起動しました: {client.user}")
    print(f"応答許可チャンネルID: {ALLOWED_CHANNEL_IDS}")

@client.event
async def on_message(message: discord.Message):
    # Bot自身の発言は無視
    if message.author == client.user:
        return

    # 指定チャンネル以外は無視
    if ALLOWED_CHANNEL_IDS and message.channel.id not in ALLOWED_CHANNEL_IDS:
        return

    user_input = message.content.strip()
    if not user_input:
        return

    # 1. ふざけた質問への判定
    if is_joking(user_input):
        await message.reply("真面目にやりなさい")
        return

    # 2. 質問（QA系）への判定 ➔ 類推せず指定の規定文言のみを返す
    question_keywords = ["?", "？", "教え", "何", "どう", "なに", "誰", "いつ", "どこ", "なぜ", "理由", "仕様", "スキル"]
    if any(q_word in user_input for q_word in question_keywords):
        reply_msg = "次までに答えを準備しておきますね。次も答えがない場合はQAリクエストにリクエストを送ってください"
        if random.random() < 0.2:
            reply_msg += f"\n{CATCHPHRASE}"
        await message.reply(reply_msg)
        return

    # 3. 雑談への対応 ➔ Qwen2.5モデルでモーガン先生として生成
    async with message.channel.typing():
        bot_reply = generate_chat_response(user_input)
        await message.reply(bot_reply)

client.run(TOKEN)