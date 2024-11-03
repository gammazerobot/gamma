import os
import cv2
import numpy as np
import fitz
from aiogram import Bot, Dispatcher, types, F
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.fsm.state import State, StatesGroup
from aiogram.filters import Command
from aiogram.types import Message, FSInputFile
from aiogram.fsm.context import FSMContext
import asyncio
import logging
import tempfile
import shutil
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Form(StatesGroup):
    waiting_for_pdf = State()

API_TOKEN = '7611578010:AAHt2uEA-nHSCcxqFebhyzCsvhqzHZr84tM'
bot = Bot(token=API_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(storage=storage)
@dp.message(Command("start"))
async def start_command(message: Message, state: FSMContext):
    welcome_text = (
        "👋 Привет! Я бот для обработки PDF файлов.\n\n"
        "📝 Что я умею:\n"
        "• Обрабатывать PDF файлы\n"
        "• Удалять определенные элементы со страниц\n"
        "• Сохранять качество документа\n\n"
        "🚀 Просто отправьте мне PDF файл, и я начну обработку!\n\n"
        "⚠️ Важно: размер файла не должен превышать 20MB"
    )
    logger.info(f"Пользователь {message.from_user.id} запустил бота")
    await message.reply(welcome_text)
    await state.set_state(Form.waiting_for_pdf)

# Структура для хранения информации о пользователях
user_stats = defaultdict(lambda: {"username": None, "usage_count": 0})

def safe_remove(file_path):
    try:
        if os.path.exists(file_path):
            os.chmod(file_path, 0o777)  # Даем все права на файл
            os.remove(file_path)
    except Exception as e:
        logger.error(f"Failed to remove temporary file {file_path}: {e}")

class ImageProcessor:
    def __init__(self):
        self.template_paths = [
            'test3.png',
            'test.png'
        ]
        self.logger = logging.getLogger(__name__)

    def process_image(self, image):
        self.logger.info("Начало обработки изображения")
        processed_image = image.copy()

        for template_path in self.template_paths:
            processed_image = self.find_and_process_image(template_path, processed_image)

        self.logger.info("Завершение обработки изображения")
        return processed_image

    def find_and_process_image(self, template_path, image):
        try:
            small_image = cv2.imread(template_path)
            if small_image is None:
                self.logger.error(f"Не удалось загрузить шаблон: {template_path}")
                return image

            large_h, large_w = image.shape[:2]
            small_h, small_w = small_image.shape[:2]

            if small_h > large_h or small_w > large_w:
                scale = min(large_h/small_h, large_w/small_w) * 0.9
                new_width = int(small_w * scale)
                new_height = int(small_h * scale)
                small_image = cv2.resize(small_image, (new_width, new_height))
                self.logger.info(f"Шаблон масштабирован до {new_width}x{new_height}")

            h, w = small_image.shape[:2]
            result = cv2.matchTemplate(image, small_image, cv2.TM_CCOEFF_NORMED)
            threshold = 0.8
            yloc, xloc = np.where(result >= threshold)

            if len(xloc) > 0:
                self.logger.info(f"Найдено {len(xloc)} совпадений на изображении")
                
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                
                for (x, y) in zip(xloc, yloc):
                    y_start = max(y - 1, 0)
                    y_end = min(y + h + 1, large_h)
                    x_start = max(x - 1, 0)
                    x_end = min(x + w + 1, large_w)
                    mask[y_start:y_end, x_start:x_end] = 255

                processed = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
                return processed
            return image
        except Exception as e:
            self.logger.error(f"Ошибка при обработке шаблона {template_path}: {str(e)}")
            return image

image_processor = ImageProcessor()

def process_pdf(input_path, output_path):
    temp_files = []
    try:
        doc_in = fitz.open(input_path)
        doc_out = fitz.open()
        
        for page in doc_in:
            pix = page.get_pixmap()
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n)
            
            processed = image_processor.process_image(img)
            
            temp_img_path = tempfile.mktemp(suffix='.png')
            temp_files.append(temp_img_path)
            cv2.imwrite(temp_img_path, cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
            
            new_page = doc_out.new_page(width=page.rect.width,
                                      height=page.rect.height)
            new_page.insert_image(new_page.rect, filename=temp_img_path)
        
        doc_out.save(output_path)
        doc_in.close()
        doc_out.close()
        return True
        
    except Exception as e:
        logger.error(f"Error in process_pdf: {e}")
        return False
    finally:
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                logger.error(f"Failed to remove temporary file {temp_file}: {e}")

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
MAX_CONCURRENT_PROCESSING = 3
processing_semaphore = asyncio.Semaphore(MAX_CONCURRENT_PROCESSING)

@dp.message(F.document)
async def handle_document(message: Message, state: FSMContext):
    user_id = message.from_user.id
    username = message.from_user.username
    user_stats[user_id]["username"] = username
    user_stats[user_id]["usage_count"] += 1
    
    logger.info(f"Получен документ от пользователя {user_id}")
    
    status_message = await message.reply("Начинаю обработку файла... ⏳")
    
    input_pdf_path = f'input_{user_id}.pdf'
    output_pdf_path = f'output_{user_id}.pdf'
    
    try:
        if not message.document.file_name.lower().endswith('.pdf'):
            await status_message.edit_text("❌ Пожалуйста, отправьте файл в формате PDF.")
            return

        await status_message.edit_text("📥 Загружаю файл...")
        
        file_id = message.document.file_id
        file = await bot.get_file(file_id)
        await bot.download_file(file.file_path, input_pdf_path)
        
        await status_message.edit_text("✅ Файл успешно загружен\n⚙️ Начинаю обработку...")
        
        logger.info(f"Начало обработки PDF для пользователя {user_id}")
        
        with fitz.open(input_pdf_path) as doc, fitz.open() as output_doc:
            total_pages = len(doc)
            await status_message.edit_text(f"📄 Всего страниц: {total_pages}\n🔄 Обработка страницы 1/{total_pages}")
            
            for page_num, page in enumerate(doc):
                if page_num % 2 == 0 or page_num == total_pages - 1:
                    progress = "🔸" * ((page_num + 1) // (total_pages // 5 + 1)) + "⭕" * (5 - (page_num + 1) // (total_pages // 5 + 1))
                    await status_message.edit_text(
                        f"📄 Всего страниц: {total_pages}\n"
                        f"🔄 Обработка страницы {page_num + 1}/{total_pages}\n"
                        f"Прогресс: {progress}\n"
                        f"⏳ Пожалуйста, подождите..."
                    )
                
                logger.info(f"Обработка страницы {page_num + 1} для пользователя {user_id}")
                pix = page.get_pixmap()
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                processed_img = image_processor.process_image(img)
                
                img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                temp_png_path = f'temp_{user_id}_{page_num}.png'
                cv2.imwrite(temp_png_path, img_rgb)
                
                try:
                    new_page = output_doc.new_page(width=page.rect.width, height=page.rect.height)
                    new_page.insert_image(new_page.rect, filename=temp_png_path)
                finally:
                    if os.path.exists(temp_png_path):
                        os.remove(temp_png_path)

            await status_message.edit_text("💾 Сохраняю результат...")
            output_doc.save(output_pdf_path)

        await status_message.edit_text("📤 Отправляю обработанный файл...")
        doc_to_send = FSInputFile(output_pdf_path)
        await message.reply_document(
            doc_to_send,
            caption="✅ Обработка завершена успешно!\n\nОтправьте новый PDF файл для обработки или /start для начала."
        )
        await status_message.edit_text("✅ Готово!")

    except Exception as e:
        error_message = f"❌ Произошла ошибка при обработке файла:\n{str(e)}\n\nПопробуйте отправить файл еще раз или используйте /start"
        logger.error(f"Ошибка при обработке файла для пользователя {user_id}: {str(e)}")
        await status_message.edit_text(error_message)
    finally:
        for file_path in [input_pdf_path, output_pdf_path]:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"Временный файл удален: {file_path}")
            except Exception as e:
                logger.error(f"Ошибка при удалении временного файла: {str(e)}")

        await state.clear()
        logger.info(f"Состояние очищено для пользователя {user_id}")

@dp.message(Command("admin"))
async def admin_command(message: Message):
    if message.from_user.id == 1419048544:  # Replace with your admin ID
        user_count = len(user_stats)
        usage_counts = [user["usage_count"] for user in user_stats.values()]
        max_usage_count = max(usage_counts) if usage_counts else 0
        min_usage_count = min(usage_counts) if usage_counts else 0
        avg_usage_count = sum(usage_counts) / len(usage_counts) if usage_counts else 0
        
        admin_message = (
            f"Панель администратора:\n"
            f"Количество пользователей: {user_count}\n"
            f"Максимальное количество использований: {max_usage_count}\n"
            f"Минимальное количество использований: {min_usage_count}\n"
            f"Среднее количество использований: {avg_usage_count:.2f}\n"
            f"Список пользователей:\n"
        )
        
        for user_id, user in user_stats.items():
            admin_message += f"  - {user['username']} (ID: {user_id}): {user['usage_count']} использований\n"
        
        await message.reply(admin_message)
    else:
        await message.reply("Доступ запрещен")

temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')
os.makedirs(temp_dir, exist_ok=True)
tempfile.tempdir = temp_dir

async def main():
    logger.info("Bot started")
    await dp.start_polling(bot)

if __name__ == '__main__':
    asyncio.run(main())
