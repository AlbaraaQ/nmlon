import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
import requests
from together import Together  # تأكد من تثبيت مكتبة together
import base64
from io import BytesIO
from PIL import Image

# إعداد تسجيل الأخطاء والمعلومات
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

# إعداد مفتاح API لـ Together
client = Together(api_key="5ba5c96173d4c62eab6e81edc5abc3f32c4a8c2aa732ef6edbdb2135d27ffdeb")

# إعداد مفتاح API لـ Hugging Face
HUGGING_FACE_API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-3.5-large-turbo"
HUGGING_FACE_HEADERS = {"Authorization": "Bearer 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"}

# متغير لتحديد النموذج الذي اختاره المستخدم
user_selected_model = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [
            InlineKeyboardButton("Together (FLUX)", callback_data="flux"),
            InlineKeyboardButton("Hugging Face (Stable Diffusion)", callback_data="stable_diffusion")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("اختر النموذج الذي ترغب باستخدامه:", reply_markup=reply_markup)

async def set_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    selected_model = query.data
    user_selected_model[user_id] = selected_model
    await query.answer()
    await query.edit_message_text(text=f"تم اختيار النموذج: {selected_model}")

async def generate_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    user_prompt = update.message.text

    # تحقق من النموذج الذي اختاره المستخدم
    model = user_selected_model.get(user_id, "flux")

    if model == "flux":
        await update.message.reply_text("جاري إنشاء الصورة باستخدام نموذج Together (FLUX)... قد يستغرق ذلك بعض الوقت.")
        try:
            response = client.images.generate(
                prompt=user_prompt,
                model="black-forest-labs/FLUX.1-schnell-Free",
                width=1024,
                height=768,
                steps=1,
                n=1,
                response_format="b64_json"
            )
            if response.data:
                image_base64 = response.data[0].b64_json
                image_data = base64.b64decode(image_base64)
                image = Image.open(BytesIO(image_data))
                with BytesIO() as output:
                    image.save(output, format="PNG")
                    output.seek(0)
                    await context.bot.send_photo(chat_id=update.effective_chat.id, photo=output)
            else:
                await update.message.reply_text("عذرًا، لم أتمكن من توليد صورة باستخدام نموذج Together.")
        except Exception as e:
            await update.message.reply_text(f"حدث خطأ أثناء توليد الصورة باستخدام Together: {e}")

    elif model == "stable_diffusion":
        await update.message.reply_text("جاري إنشاء الصورة باستخدام نموذج Hugging Face... قد يستغرق ذلك بعض الوقت.")
        try:
            response = requests.post(HUGGING_FACE_API_URL, headers=HUGGING_FACE_HEADERS, json={"inputs": user_prompt})
            if response.status_code == 200:
                image_bytes = response.content
                image = Image.open(BytesIO(image_bytes))
                with BytesIO() as output:
                    image.save(output, format="PNG")
                    output.seek(0)
                    await context.bot.send_photo(chat_id=update.effective_chat.id, photo=output)
            else:
                await update.message.reply_text("عذرًا، لم أتمكن من توليد صورة باستخدام نموذج Hugging Face.")
        except Exception as e:
            await update.message.reply_text(f"حدث خطأ أثناء توليد الصورة باستخدام Hugging Face: {e}")

def main():
    # استبدل 'YOUR_TELEGRAM_BOT_TOKEN' بالتوكن الخاص بالبوت
    app = ApplicationBuilder().token('7865424971:AAF_Oe6lu8ZYAl5XIF1M6qU_8MK6GHWEll8').build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(set_model))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, generate_image))

    app.run_polling()

if __name__ == "__main__":
    main()
