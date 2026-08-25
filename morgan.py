import os
import requests
from dotenv import load_dotenv

# --------------------------------------------------
# 1. 環境変数の読み込みとチェック
# --------------------------------------------------
load_dotenv()

TOKEN = os.getenv("TOKEN") or os.environ.get("TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN") or os.environ.get("HF_TOKEN")
ALLOWED_CHANNEL_ID = os.getenv("ALLOWED_CHANNEL_ID") or os.environ.get("ALLOWED_CHANNEL_ID")

print(f"--- [DEBUG] TOKEN Detected: {bool(TOKEN)} ---")
print(f"--- [DEBUG] HF_TOKEN Detected: {bool(HF_TOKEN)} ---")

# Discord Token チェック
if not TOKEN:
    raise ValueError("エラー: 環境変数 'TOKEN' が設定されていません。RailwayのVariablesタブで 'TOKEN' を設定してください。")

# Hugging Face Token チェック（未設定または空文字の場合にエラー停止）
if not HF_TOKEN or len(HF_TOKEN.strip()) == 0:
    raise ValueError("エラー: 環境変数 'HF_TOKEN' が設定されていません。RailwayのVariablesタブで 'HF_TOKEN' を設定してください。")

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
intents.message_content = True  # メッセージ内容の取得を許可
client = discord.Client(intents=intents)

# --------------------------------------------------
# 3. Hugging Face API 呼び出し関数
# --------------------------------------------------
def query_huggingface(prompt):
    API_URL = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-7B-Instruct"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {HF_TOKEN.strip()}"
    }
    
    full_prompt = f"システム: あなたは親切で賢いアシスタント「モーガン先生」です。日本語で丁寧に回答してください。\nユーザー: {prompt}\nアシスタント:"
    
    payload = {
        "inputs": full_prompt,
        "parameters": {
            "max_new_tokens": 256,
            "temperature": 0.7,
            "return_full_text": False
        },
        "options": {
            "wait_for_model": True
        }
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=40)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                text = result[0].get("generated_text", "")
                return text.strip() if text else "返答が空でした。"
            elif isinstance(result, dict) and "generated_text" in result:
                return result["generated_text"].strip()
            return "レスポンスの解析に失敗しました。"
        else:
            err_detail = response.text[:300]
            print(f"[API Error Status]: {response.status_code}")
            print(f"[API Error Detail]: {response.text}")
            return f"APIエラー（コード: {response.status_code}）\n詳細: `{err_detail}`"
            
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
        
        # Discordの2000文字制限対策
        if len(reply) > 1900:
            reply = reply[:1900] + "\n...(長文のため省略されました)"
            
        await message.channel.send(reply)

# --------------------------------------------------
# 5. Botの実行
# --------------------------------------------------
client.run(TOKEN)