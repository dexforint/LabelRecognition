import os
from io import BytesIO
import asyncio
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, F, types, html  # executor
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import Message, ContentType

load_dotenv()


dp = Dispatcher()  # отвечает за асинхронность


async def on_startup(bot: Bot) -> None:
    pass


# устанавливаем функцию для запуска на старте
# @dp.message(F.content_type == ContentType.PHOTO)
@dp.message(F.photo)
async def handle_photo(message: Message, bot: Bot):
    """
    Обрабатывает сообщения с фотографиями, загружает их на Яндекс Диск.

    :param message: Объект сообщения, содержащий фотографию.
    :type message: Message
    :param bot: Объект бота для взаимодействия с Telegram API.
    :type bot: Bot
    """

    photo = message.photo[-1]
    file_info = await bot.get_file(photo.file_id)

    await bot.download_file(file_info.file_path, destination="./temp/photo.jpg")
    await message.answer("Фото успешно загружено! Обработка...")


# Обрабатываем команду /start
@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    message_text = f"Привет! Данный поможет тебе распознать маркировку детали на изображении и получить информацию о ней из базы.\n\nПросто отправь мне фото детали..."
    await message.answer(message_text)


async def main() -> None:
    # And the run events dispatching
    bot = Bot(
        token=os.getenv("TELEGRAM_TOKEN"),
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    await dp.start_polling(
        bot,
    )


if __name__ == "__main__":
    asyncio.run(main())
