import os
import asyncio
import discord
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# --------------------------------------------------
# 1. 環境変数の読み込みと起動時チェック
# --------------------------------------------------
load_dotenv()

TOKEN = os.getenv("TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")

print(f"--- [DEBUG] TOKEN Detected: {bool(TOKEN)} ---")
print(f"--- [DEBUG] HF_TOKEN Detected: {bool(HF_TOKEN)} ---")

if not TOKEN:
    raise ValueError("エラー: 環境変数 'TOKEN' が設定されていません。RailwayのVariablesタブで設定してください。")

if not HF_TOKEN or len(HF_TOKEN.strip()) == 0:
    raise ValueError("エラー: 環境変数 'HF_TOKEN' が設定されていません。RailwayのVariablesタブで設定してください。")

# 許可チャンネルIDの読み込み（カンマ区切り対応）
env_channels = os.getenv("ALLOWED_CHANNEL_ID", "")
ALLOWED_CHANNEL_IDS = [int(ch_id.strip()) for ch_id in env_channels.split(",") if ch_id.strip().isdigit()]

# --------------------------------------------------
# 2. Hugging Face クライアントの初期化
# --------------------------------------------------
hf_client = InferenceClient(
    model="Qwen/Qwen2.5-Coder-32B-Instruct",
    token=HF_TOKEN.strip()
)

# --------------------------------------------------
# 3. 知識ベース（qa.txt）の読み込み関数
# --------------------------------------------------
def load_knowledge_base() -> str:
    file_path = "qa.txt"
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            print(f"qa.txt読み込みエラー: {e}")
            return ""
    return ""

# --------------------------------------------------
# 4. AI（モーガン先生）の思考・回答関数
# --------------------------------------------------
async def ask_ai(prompt: str) -> str:
    knowledge = load_knowledge_base()
    
    # QAデータの準備状態を判定
    if knowledge:
        qa_status_prompt = f"【公式QAデータベース】\n{knowledge}"
    else:
        qa_status_prompt = "【公式QAデータベース】\n（現在QAは準備中・未登録の状態です）"

    # モーガン先生のキャラクター設定・応答ルール
    system_instruction = (
        "あなたの名前は「モーガン先生」です。Discordの特定チャンネルでユーザーの質問や雑談に応じるBOTです。\n\n"
        "【性格・基本スタンス】\n"
        "- 性格は真面目な委員長タイプです。\n"
        "- 基本的には丁寧な敬語（〜です、〜ます）で話しますが、感情が高ぶった際やふとした瞬間に、たまに砕けた口調（〜だよ、〜じゃない？等）が混ざります。\n"
        "- 口癖は「魔法の授業はちゃんとうけなさい！」です。挨拶や雑談の合間など、適したタイミングで自然に使ってください。\n\n"
        "【応答ルール】\n"
        "1. 雑談・日常会話には、親切かつ真面目に応じつつ雑談に付き合ってください。\n"
        "2. ふざけた質問、不真面目な問いかけ、煽りに対しては、一言「真面目にやりなさい」とだけ厳しく返答・一蹴してください。\n"
        "3. 公式QAデータベースに関する質問や具体的な確認事項について：\n"
        "   - 現在QAが準備中の場合は、「ただいまQAができるまで準備中となっております。今しばらくお待ちくださいね」といった案内を含めて回答してください。\n"
        "   - QAに記載がない質問や答えられない質問については、絶対に類推・推測・捏造して答えてはいけません。\n"
        "   - QAに記載のない質問への回答は、正確に「次までに答えを準備しておきますね。次も答えがない場合はQAリクエストにリクエストを送ってください」と答えてください。\n\n"
        f"{qa_status_prompt}"
    )

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": prompt}
    ]

    loop = asyncio.get_running_loop()
    for attempt in range(2):
        try:
            response = await loop.run_in_executor(
                None,
                lambda: hf_client.chat_completion(
                    messages=messages,
                    max_tokens=400,
                    temperature=0.6
                )
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[AI Error Attempt {attempt + 1}]: {e}")
            if attempt == 0:
                await asyncio.sleep(1)
            else:
                return "申し訳ありません、通信エラーが発生しました。時間を置いて再度お試しください。"

# --------------------------------------------------
# 5. Discord イベントハンドラ
# --------------------------------------------------
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f"Logged in as {client.user} (ID: {client.user.id})")
    print("モーガン先生（委員長モード）が正常に起動・オンラインになりました！")

@client.event
async def on_message(message: discord.Message):
    # Bot自身の発言は無視
    if message.author == client.user:
        return

    # 指定したチャンネル以外は無視
    if ALLOWED_CHANNEL_IDS and message.channel.id not in ALLOWED_CHANNEL_IDS:
        return

    user_input = message.content.strip()
    if not user_input:
        return

    # AI回答処理（考え中表示から更新）
    reply_msg = await message.reply("考え中… 🤔")
    ai_response = await ask_ai(user_input)
    
    # Discordの2000文字制限対策
    if len(ai_response) > 1900:
        ai_response = ai_response[:1900] + "\n...(長文のため省略されました)"
        
    await reply_msg.edit(content=ai_response)

# --------------------------------------------------
# 6. Botの実行
# --------------------------------------------------
client.run(TOKEN)