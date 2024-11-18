"""
Losty – telegram bot for finding lost pets.

URL: https://t.me/losty_pets_bot

This script is the main entry point for the Losty telegram bot. It continuously updates the database
with new posts, handles incoming messages and sends matches to the user.

Author: Roman Kozlov
Github: https://github.com/donatorex
"""

import os
import random
import sqlite3
import threading
import time
from datetime import datetime

import telebot
from PIL import Image

from .main import LostyFinder


API_TOKEN = os.environ.get('TELEGRAM_API_TOKEN')
bot = telebot.TeleBot(API_TOKEN)

losty = LostyFinder()
user_data = {}


@bot.message_handler(commands=['start'])
def send_welcome(message: telebot.types.Message) -> None:
    """
    Handle the '/start' command and send a welcome message.
    """
    bot.send_message(message.chat.id, 'Отлично, давай начнём! Отправь мне фотографию хвостика🐾')


@bot.message_handler(commands=['test'])
def send_test(message):
    bot.reply_to(message, "Test mode")


@bot.message_handler(content_types=['photo'])
def handle_photo(message: telebot.types.Message) -> None:
    """
    Handle an incoming photo message, download the image and start a search for matches.

    :param message: The incoming message containing the photo.
    """
    try:
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        bot.send_message(message.chat.id, 'Ищу совпадения...')

        input_image_path = f"data/temp/{message.chat.id}_temp_image.jpg"
        with open(input_image_path, 'wb') as new_file:
            new_file.write(downloaded_file)

        pages = losty.find_matches(input_image_path)
        user_data[message.chat.id] = {'pages': pages, 'current_page': 1}

        send_matches(message.chat.id, pages, 1)
        os.remove(input_image_path)
    except Exception as e:
        print(f"Error occurred with bot: {e}")
        bot.send_message(message.chat.id, "Произошла техническая ошибка. Попробуйте позже.")


@bot.callback_query_handler(func=lambda call: call.data.startswith("next_page_"))
def handle_page_change(call: telebot.types.CallbackQuery) -> None:
    """
    Handle a callback query to change the page of matches.

    :param call: The incoming callback query.
    """
    page_number = int(call.data.split("_")[2])
    chat_id = call.message.chat.id
    if chat_id in user_data:
        pages = user_data[chat_id]['pages']
        send_matches(chat_id, pages, page_number)


def add_logo_to_image(image_path: str, group: str) -> Image:
    """
    Add a logo to an image.

    :param image_path: str - The file path to the image.
    :param group: str - The name of the group.
    :return: Image – The image with the added logo.
    """
    image = Image.open(image_path)

    logo = Image.open(f"{group}/{group}_profile_pic.jpg").convert('RGBA')

    logo_size_ratio = 0.15
    logo_width = int(image.width * logo_size_ratio)
    logo_height = int(logo.height * (logo_width / logo.width))

    logo_resized = logo.resize((logo_width, logo_height), Image.Resampling.LANCZOS)

    image.paste(logo_resized, (0, 0), logo_resized)

    return image


def send_matches(chat_id: int, pages: dict, page_number: int) -> None:
    """
    Send a message with matches found in the database.

    :param chat_id: int - The telegram chat id to send the message to.
    :param pages: dict - The dictionary of matches paginated by page number.
    :param page_number: int - The number of the page to send.
    """
    try:
        # Send a typing action to the user.
        bot.send_chat_action(chat_id, 'upload_photo')
        media = []

        # Iterate over the matches in the current page.
        for key, value in pages[page_number].items():
            shortcode = key
            group = value[0]
            date = datetime.fromisoformat(value[1]).astimezone()
            image_path = value[2]
            match_percentage = value[3]

            caption = get_caption(image_path)
            desc = build_description(group, shortcode, date, caption, match_percentage)

            image = add_logo_to_image(image_path, group)
            media.append(telebot.types.InputMediaPhoto(media=image, caption=desc))

        # Send the media group to the user.
        bot.send_media_group(chat_id, media)

        # Create the keyboard for the user.
        keyboard = []
        if page_number < len(pages):
            keyboard.append([
                telebot.types.InlineKeyboardButton("Найти еще совпадения", callback_data=f"next_page_{page_number + 1}")
            ])

        # Send the message with the keyboard.
        reply_markup = telebot.types.InlineKeyboardMarkup(keyboard)
        bot.send_message(chat_id, f"Страница {page_number} из 10", reply_markup=reply_markup)
    except Exception as e:
        print(f"Error occurred with bot: {e}")
        bot.send_message(chat_id, "Произошла техническая ошибка. Попробуйте позже.")


def get_caption(image_path: str) -> str:
    """
    Get the caption of an image from a file.

    :param image_path: str - The file path to the image.
    :return: str - The caption of the image.
    """
    caption_filename = image_path.split('_UTC')[0] + '_UTC.txt'
    if os.path.exists(caption_filename):
        with open(f"{caption_filename}", 'r', encoding='utf-8') as caption_file:
            return caption_file.read()
    return 'Описание отсутствует'


def build_description(group: str, shortcode: str, date: datetime, caption: str, match_percentage: float) -> str:
    """
    Build the description of a match.

    :param group: str - The name of the group.
    :param shortcode: str - The shortcode of the post.
    :param date: datetime - The date the post was published.
    :param caption: str - The caption of the post.
    :param match_percentage: float - The percentage of match.
    :return: str - The description of the match.
    """
    description = f"""
--- Группа ---
@{group}

--- Дата публикации ---
{date}

--- Ссылка на пост ---
https://www.instagram.com/{group}/p/{shortcode}

--- Совпадение ---
{match_percentage:.2%}

--- Описание ---
{caption}
"""

    if len(description) > 1024:
        return description[:1021] + '...'
    else:
        return description


def update_data() -> None:
    """
    Continuously update the database with new posts and log the number of posts added.

    :return: None
    """
    while True:
        conn = sqlite3.connect('data/losty_db.db')
        cur = conn.cursor()
        try:
            # Retrieve the minimum start date from posts.
            cur.execute('SELECT MAX(date) FROM posts GROUP BY group_id')
            start_date = min(cur.fetchall())[0]

            # Get the count of posts before update.
            cur.execute('SELECT COUNT(*) FROM posts;')
            count_before = cur.fetchone()[0]

            # Update data starting from the earliest post date or default to a full update.
            if start_date:
                losty.update_data(start_date=datetime.fromisoformat(start_date).replace(tzinfo=None))
            else:
                losty.update_data()

            # Get the count of posts after update.
            cur.execute('SELECT COUNT(*) FROM posts;')
            count_after = cur.fetchone()[0]

            # Log the number of posts added to the database.
            print(f"Database has been updated (added {int(count_after) - int(count_before)} post(s))")
            print('------------------------------')

        except Exception as e:
            print(f"An error occurred during the update: {e}")

        finally:
            # Ensure the cursor and connection are closed.
            cur.close()
            conn.close()

        # Sleep for a random interval before the next update cycle.
        time.sleep(random.randint(3400, 3800))


if __name__ == "__main__":
    update_thread = threading.Thread(target=update_data, daemon=True)
    update_thread.start()
    bot.infinity_polling()
