import logging
import os
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
from huggingface_hub import login
import videogen_hub
import torch
import torchvision.io as io
from dotenv import load_dotenv

# تحميل متغيرات البيئة من ملف .env
load_dotenv()

# إعداد تسجيل الأخطاء والمعلومات
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

# تسجيل الدخول باستخدام Token الخاص بك (من متغيرات البيئة)
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    logging.error("HF_TOKEN is not set. Please set the token in your environment.")
    raise EnvironmentError("HF_TOKEN is not set. Check your .env file.")

login(token=HF_TOKEN)

# تحميل النموذج
try:
    model = videogen_hub.load('VideoCrafter2')
    logging.info("Model loaded successfully!")
except Exception as e:
    logging.error(f"Error loading the model: {e}")
    raise

# وظيفة لتوليد الفيديو بناءً على النص المدخل
def generate_video(prompt: str) -> str:
    try:
        video = model.infer_one_video(prompt=prompt)
        output_filename = "generated_video.mp4"
        io.write_video(output_filename, video.permute(0, 2, 3, 1).numpy(), fps=30)
        logging.info(f"Video generated and saved as {output_filename}")
        return output_filename
    except Exception as e:
        logging.error(f"Error in generating video: {e}")
        raise

# وظيفة لإرسال رسالة ترحيب عند استخدام أمر /start
async def start(update: Update, context: CallbackContext):
    logging.info("Received /start command")
    await update.message.reply_text(
        "مرحبًا بك في بوت توليد الفيديوهات! أرسل لي وصفًا نصيًا وسأقوم بإنشاء فيديو لك 🎥."
    )

# وظيفة التعامل مع الرسائل من تيليجرام
async def handle_message(update: Update, context: CallbackContext):
    user_prompt = update.message.text
    logging.info(f"Received message: {user_prompt}")
    await update.message.reply_text("جاري إنشاء الفيديو، يرجى الانتظار...")
    try:
        # توليد الفيديو بناءً على النص الذي أدخله المستخدم
        video_path = generate_video(user_prompt)

        # إرسال الفيديو إلى المستخدم
        with open(video_path, 'rb') as video_file:
            await update.message.reply_video(video=video_file)
            logging.info(f"Video sent to user: {update.message.chat.username}")
    except Exception as e:
        logging.error(f"Error sending video: {e}")
        await update.message.reply_text(f"حدث خطأ أثناء توليد الفيديو: {e}")

# إعداد البوت
def main():
    # استخدام متغير بيئة للحصول على توكن البوت
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
    if not TELEGRAM_TOKEN:
        logging.error("TELEGRAM_TOKEN is not set. Please set the token in your environment.")
        return

    # إعداد التطبيق
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # إعداد الـ Handlers للأوامر والرسائل
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    try:
        logging.info("Bot is starting...")
        application.run_polling()
    except Exception as e:
        logging.error(f"Error occurred during bot polling: {e}")

if __name__ == '__main__':
    main()
