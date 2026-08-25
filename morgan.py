import os
import discord
import requests

# --------------------------------------------------
# 1. 環境変数の読み込みとチェック
# --------------------------------------------------
TOKEN = os.getenv("TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")
ALLOWED_CHANNEL_ID = os.getenv("ALLOWED_CHANNEL_ID")

# TOKENが設定されていない場合はわかりやすいエラーを出す
if not TOKEN:
    raise ValueError("エラー: 環境変数 'TOKEN' が設定されていません。RailwayのVariablesタブで 'TOKEN' を設定してください。")

# ALLOWED_CHANNEL_ID が設定されている場合は数値（int）に変換
if ALLOWED_CHANNEL_ID:
    try:
        ALLOWED_CHANNEL_ID = int(ALLOWED_CHANNEL_ID)
    except ValueError:
        print("警告: ALLOWED_CHANNEL_ID が正しい数値ではありません。チャンネル制限なしで動作します。")
        ALLOWED_CHANNEL_ID = None

# --------------------------------------------------
# 2. Discord Client の初期化設定
# --------------------------------------------------
intents = discord.Intents.default()
intents.message_content = True  # メッセージ内容の取得を許可
client = discord.Client(intents=intents)

# --------------------------------------------------
# 3. Hugging Face API 呼び出し関数
# --------------------------------------------------
def query_huggingface(prompt):
    # Hugging FaceのモデルURL（必要に応じてモデル名を変更してください）
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    
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
            return "レスポンスの形式が不正です。"
        else:
            return f"APIエラーが発生しました（ステータスコード: {response.status_code}）"
    except Exception as e:
        return f"通信エラーが発生しました: {e}"

# --------------------------------------------------
# 4. Discord イベントハンドラ
# --------------------------------------------------
@client.event
async def on_ready():
    print(f"Logged in as {client.user} (ID: {client.user.id})")
    print("モーガン先生が正常に起動しました！")

@client.event
async def on_message(message):
    # Bot自身の発言は無視
    if message.author == client.user:
        return

    # 指定したチャンネル以外は無視（ALLOWED_CHANNEL_IDが設定されている場合）
    if ALLOWED_CHANNEL_ID and message.channel.id != ALLOWED_CHANNEL_ID:
        return

    # メッセージの処理
    user_input = message.content.strip()
    if not user_input:
        return

    # 応答処理中であることを示す（入力中アニメーション）
    async with message.channel.typing():
        reply = query_huggingface(user_input)
        await message.channel.send(reply)

# --------------------------------------------------
# 5. Botの実行
# --------------------------------------------------
client.run(TOKEN)