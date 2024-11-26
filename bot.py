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
from datetime import datetime, timedelta, timezone

import telebot
from PIL import Image

from losty import LostyFinder


API_TOKEN = os.environ.get('TELEGRAM_API_TOKEN')
AUTO_UPDATE_COMMAND = os.environ.get('AUTO_UPDATE_COMMAND')
DROP_SESSION_DATA_COMMAND = os.environ.get('DROP_SESSION_DATA_COMMAND')

AUTO_UPDATE = False
TEST_MODE = False

DATA_DIR = '/disk/data'
TEMP_DIR = '/disk/data/temp'
DB_PATH = '/disk/data/losty_db.db'

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
    global TEST_MODE
    TEST_MODE = not TEST_MODE
    bot.reply_to(message, f"Test mode {'enabled' if TEST_MODE else 'disabled'}")


@bot.message_handler(commands=[AUTO_UPDATE_COMMAND])
def auto_update(message):
    global AUTO_UPDATE
    AUTO_UPDATE = not AUTO_UPDATE
    bot.reply_to(message, f"Auto update {'enabled' if AUTO_UPDATE else 'disabled'}")
    print(f"Auto update {'enabled' if AUTO_UPDATE else 'disabled'}")


@bot.message_handler(commands=[DROP_SESSION_DATA_COMMAND])
def drop_session_data(message):
    session_file_path = os.path.join(DATA_DIR, 'session-inst')

    if os.path.exists(session_file_path):
        os.remove(session_file_path)
        losty.login()


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

        input_image_path = os.path.join(TEMP_DIR, f"{message.chat.id}_temp_image.jpg")
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

    logo_path = os.path.join(DATA_DIR, f"{group}/{group}_profile_pic.jpg")
    logo = Image.open(logo_path).convert('RGBA')

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
        bot.send_message(chat_id, f"Страница {page_number} из {len(pages)}", reply_markup=reply_markup)
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
    # Make sure the data and temp directories exists.
    os.makedirs(TEMP_DIR, exist_ok=True)

    # Create the database if it doesn't exist.
    if not os.path.exists(DB_PATH):
        create_database()

    # Refit the KNN model using embeddings from the database.
    losty.knn_refit()

    while True:

        if not AUTO_UPDATE:
            time.sleep(60)
            continue

        print('Starting update...')
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        try:
            # Get the count of posts before the update.
            cur.execute('SELECT COUNT(*) FROM posts;')
            count_before = cur.fetchone()[0]

            # Update data starting from the earliest post date or default to a full update.
            if losty.initial_update:
                # Retrieve the minimum start date from posts.
                cur.execute('SELECT MAX(date) FROM posts GROUP BY group_id')
                rows = cur.fetchall()
                losty.update_data(start_date=datetime.fromisoformat(min(rows)[0]).replace(tzinfo=None))
            else:
                losty.update_data()

            # Get the count of posts after update.
            cur.execute('SELECT COUNT(*) FROM posts;')
            count_after = cur.fetchone()[0]

            # Log the number of posts added to the database.
            print(f"Database has been updated (added {int(count_after) - int(count_before)} post(s))")

        except Exception as e:
            print(f"An error occurred during the update: {e}")

        finally:
            # Ensure the cursor and connection are closed.
            cur.close()
            conn.close()

            remaining_time = random.randint(7200, 10800)
            print(f'Next update on {datetime.now(tz=timezone.utc) + timedelta(seconds=remaining_time)} (UTC)')
            print('------------------------------')

            # Sleep for a random interval before the next update cycle.
            time.sleep(remaining_time)


def create_database() -> None:
    """
    Create the database if it doesn't exist.

    :return: None
    """
    db = sqlite3.connect(DB_PATH)
    cur = db.cursor()
    try:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS groups (
                group_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE
            );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS posts (
                post_id INTEGER PRIMARY KEY AUTOINCREMENT,
                group_id INTEGER,
                shortcode TEXT NOT NULL,
                date TIMESTAMP NOT NULL,
                caption TEXT,
                FOREIGN KEY (group_id) REFERENCES groups(group_id)
            );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS images (
                image_id INTEGER PRIMARY KEY AUTOINCREMENT,
                post_id INTEGER,
                image_path TEXT NOT NULL,
                embedding TEXT NOT NULL,
                FOREIGN KEY (post_id) REFERENCES posts(post_id)
            );
        """)
        db.commit()
    except Exception as e:
        print(f"An error occurred while creating the database: {e}")
        db.rollback()
    finally:
        cur.close()
        db.close()


if __name__ == "__main__":
    update_thread = threading.Thread(target=update_data, daemon=True)
    update_thread.start()
    bot.infinity_polling()
