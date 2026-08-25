import os
import requests
from dotenv import load_dotenv

# --------------------------------------------------
# 1. 環境変数の読み込みとチェック
# --------------------------------------------------
load_dotenv()

# os.getenv と os.environ.get の両方で確実に取得
TOKEN = os.getenv("TOKEN") or os.environ.get("TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN") or os.environ.get("HF_TOKEN")
ALLOWED_CHANNEL_ID = os.getenv("ALLOWED_CHANNEL_ID") or os.environ.get("ALLOWED_CHANNEL_ID")

print(f"--- [DEBUG] TOKEN Detected: {bool(TOKEN)} ---")
print(f"--- [DEBUG] HF_TOKEN Detected: {bool(HF_TOKEN)} ---")

if not TOKEN:
    raise ValueError("エラー: 環境変数 'TOKEN' が設定されていません。RailwayのVariablesタブで 'TOKEN' を設定してください。")

# ALLOWED_CHANNEL_ID を数値（int）に安全に変換
if ALLOWED_CHANNEL_ID:
    try:
        ALLOWED_CHANNEL_ID = int(str(ALLOWED_CHANNEL_ID).strip('"\''))
    except ValueError:
        print("警告: ALLOWED_CHANNEL_ID が正しい数値ではありません。チャンネル制限なしで動作します。")
        ALLOWED_CHANNEL_ID = None

# --------------------------------------------------
# 2. Discord Client の初期化設定
# --------------------------------------------------
import discord

intents = discord.Intents.default()
intents.message_content = True  # メッセージ内容の取得を許可（Discord Developer PortalでもONが必要）
client = discord.Client(intents=intents)

# --------------------------------------------------
# 3. Hugging Face API 呼び出し関数
# --------------------------------------------------
def query_huggingface(prompt):
    # 最新の推論APIエンドポイント（Mistral-7B-Instruct-v0.3）
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
    
    headers = {"Content-Type": "application/json"}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    
    payload = {
        "inputs": f"<s>[INST] あなたは親切で賢いアシスタント「モーガン先生」です。日本語で丁寧に答えてください。\n\n質問: {prompt} [/INST]",
        "parameters": {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "return_full_text": False
        }
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "返答の生成に失敗しました。")
            elif isinstance(result, dict) and "generated_text" in result:
                return result["generated_text"]
            return "レスポンスの解析に失敗しました。"
        else:
            print(f"[API Error Details]: {response.status_code} - {response.text}")
            return f"申し訳ありません、APIエラーが発生しました（コード: {response.status_code}）。"
            
    except Exception as e:
        print(f"[Connection Error]: {e}")
        return f"通信エラーが発生しました: {e}"

# --------------------------------------------------
# 4. Discord イベントハンドラ
# --------------------------------------------------
@client.event
async def on_ready():
    print(f"Logged in as {client.user} (ID: {client.user.id})")
    print("モーガン先生が正常に起動・オンラインになりました！")

@client.event
async def on_message(message):
    # Bot自身の発言は無視
    if message.author == client.user:
        return

    # 指定したチャンネル以外は無視
    if ALLOWED_CHANNEL_ID and message.channel.id != ALLOWED_CHANNEL_ID:
        return

    user_input = message.content.strip()
    if not user_input:
        return

    # 入力中アニメーションを表示しながらAI処理
    async with message.channel.typing():
        reply = query_huggingface(user_input)
        
        # Discordの2000文字制限対策（超える場合は分割または省略）
        if len(reply) > 1900:
            reply = reply[:1900] + "\n...(長文のため省略されました)"
            
        await message.channel.send(reply)

# --------------------------------------------------
# 5. Botの実行
# --------------------------------------------------
client.run(TOKEN)