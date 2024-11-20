"""
LostyFinder: Instagram-Based Image Matching and Database Management

This script defines a Python module for managing a database of Instagram posts and images,
primarily focused on lost and found pet groups. It utilizes Instagram scraping, image
embedding generation using a ResNet50 model, and K-Nearest Neighbors (KNN) for finding
similar images.

The module provides functionality to:

    — Upload data about posts in specified Instagram groups and auto-update in a set period.
    — Extract image embeddings using a pre-trained ResNet50 deep learning model.
    — Add and manage groups, posts, images and their embeddings in a SQLite database.
    — Cleaning the database and storage from old posts (older than half a year).
    — Search for the most relevant posts for an input image by embeddings using the unsupervised NearestNeighbors model.

Main Classes and Functions:
    — `LostyFinder`: Class responsible for downloading Instagram data, managing the database,
      generating image embeddings, and performing image matching.
    — `add_group`: Function to add a group to the database.
    — `add_post`: Function to add a post record to the database.
    — `add_image`: Function to add an image record to the database.
    — `check_shortcode`: Function to check if a shortcode exists in the database.
    — `cleanup_data`: Function to clean up old image and post data.
    — `knn_refit`: Method to refit the KNN model using embeddings from the database.
    — `find_matches`: Method to find and paginate image matches for a given input image.

Dependencies:
    — `instaloader`: For downloading Instagram posts and profile data.
    — `numpy`: For numerical operations and array handling.
    — `requests`: For handling HTTP requests.
    — `PIL`: For image processing.
    — `sklearn`: For machine learning models, including KNN.
    — `tensorflow.keras`: For deep learning model ResNet50.
    — `sqlite3`: For database interactions.

Note: This script assumes the existence of an SQLite database at 'data/losty_db.db' and
appropriate database schema for storing group, post and image records.

Author: Roman Kozlov
Github: https://github.com/donatorex
"""

import io
import json
import lzma
import os
import sqlite3
from datetime import datetime, timezone, timedelta

import instaloader
import numpy as np
import requests
from PIL import Image

from sklearn.neighbors import NearestNeighbors

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model


LOGIN = os.environ.get('LOGIN')
PASSWORD = os.environ.get('PASSWORD')
TIME_DELTA = timedelta(days=182)


def add_group(group: str) -> int:
    """
    Add a group to the database if it does not already exist.

    :param group: str – The name of the group to be added.
    :return: int – The ID of the group from the database.
    """
    db = sqlite3.connect('data/losty_db.db')
    cur = db.cursor()
    try:
        # Insert the group into the groups table, ignoring if it already exists.
        cur.execute('INSERT OR IGNORE INTO groups (name) VALUES (?);', (group,))
        db.commit()

        # Retrieve the group_id of the newly inserted or existing group.
        cur.execute('SELECT group_id FROM groups WHERE name = ?;', (group,))
        group_id = cur.fetchone()[0]
        return group_id
    except Exception as e:
        print(f"Error while adding a group: {e}")
        db.rollback()
    finally:
        cur.close()
        db.close()


def add_post(row: tuple) -> int:
    """
    Add a post record to the database.

    :param row: tuple – A tuple containing the following values:
        — group_id: int – The ID of the group the post belongs to.
        — shortcode: str – The shortcode of the post.
        — date: datetime.datetime – The date the post was published.
        — caption: str – The caption of the post.
    :return: int – The ID of the newly inserted post.
    """
    db = sqlite3.connect('data/losty_db.db')
    cur = db.cursor()
    try:
        # Insert the post into the posts table.
        cur.execute('''
            INSERT INTO posts (group_id, shortcode, date, caption) VALUES (?, ?, ?, ?);
        ''', row)
        post_id = cur.lastrowid
        db.commit()
        return post_id
    except Exception as e:
        print(f"Error while adding a post: {e}")
        db.rollback()
    finally:
        cur.close()
        db.close()


def add_image(post_id: int, image_path: str, embedding: list) -> None:
    """
    Add an image record to the database.

    :param post_id: int – The ID of the associated post.
    :param image_path: str – The file path to the image.
    :param embedding: list – The embedding of the image as a list.
    """
    db = sqlite3.connect('data/losty_db.db')
    cur = db.cursor()
    try:
        # Insert the image record into the images table.
        cur.execute('''
            INSERT INTO images (post_id, image_path, embedding) VALUES (?, ?, ?);
        ''', (post_id, image_path, json.dumps(embedding)))
        db.commit()
    except Exception as e:
        print(f"Error while adding an image: {e}")
        db.rollback()
    finally:
        cur.close()
        db.close()


def check_shortcode(shortcode: str) -> bool:
    """
    Check if a shortcode exists in the database.

    :param shortcode: str – The shortcode to be checked.
    :return: bool – True if the shortcode exists, False otherwise.
    """
    db = sqlite3.connect('data/losty_db.db')
    cur = db.cursor()
    try:
        # Check if the shortcode exists in the database.
        cur.execute('SELECT EXISTS (SELECT 1 FROM posts WHERE shortcode LIKE ?) AS FOUND;', (f'%{shortcode}%',))
        return cur.fetchone()[0]
    except Exception as e:
        print(f"Error while checking shortcode's from the database: {e}")
        db.rollback()
    finally:
        cur.close()
        db.close()


def cleanup_data() -> None:
    """
    Cleans up old image and post data from the database and file system.

    """
    db = sqlite3.connect('data/losty_db.db')
    cur = db.cursor()
    try:
        zero_date = datetime.now(tz=timezone.utc) - TIME_DELTA
        cur.execute("""
            SELECT i.post_id, MIN(i.image_path) AS image_path
            FROM images AS i
            INNER JOIN posts AS p ON i.post_id = p.post_id
            WHERE p.date < ?
            GROUP BY i.post_id;
        """, (zero_date.strftime('%Y-%m-%d %H:%M:%S'),))
        rows = cur.fetchall()
        if rows:
            # Delete images and posts from the database.
            cur.execute(f"DELETE FROM images WHERE post_id IN ({','.join('?' * len(rows))})", [row[0] for row in rows])
            cur.execute(f"DELETE FROM posts WHERE post_id IN ({','.join('?' * len(rows))})", [row[0] for row in rows])
            # Remove image files from the file system.
            for row in rows:
                directory = os.path.dirname(row[1])
                base_name = os.path.basename(row[1]).split('_UTC')[0]
                for file_name in os.listdir(directory):
                    if file_name.startswith(base_name):
                        file_path = os.path.join(directory, file_name)
                        os.remove(file_path)
        db.commit()
    except Exception as e:
        print(f"Error during cleanup: {e}")
        db.rollback()
    finally:
        cur.close()
        db.close()


class LostyFinder:
    """
    Class for updating database and finding matches for an image.
    """

    def __init__(self):
        """
        Initialize the class.

        :param self.groups: list – List of group names.
        :param self.loader: Instaloader object.
        :param self.model: ResNet50 model.
        :param self.knn: NearestNeighbors object.
        """
        self.groups = ['lost_found_pets_almaty', 'poteryashki_almaty', 'almaty_pomosh_zhivotnym', 'aulau_kyzmeti']
        self.loader = instaloader.Instaloader()
        self.login()

        base_model = ResNet50(weights='imagenet')
        self.model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
        self.knn = NearestNeighbors(metric='cosine')

    def login(self, relogin: bool = False) -> None:
        """
        Login to instagram and saving the session file.

        :param relogin: bool - If True, a relogin will be done with the session file saved.
        """
        try:
            if not relogin and os.path.exists('data/session-inst'):
                self.loader.load_session_from_file(LOGIN, 'data/session-inst')
            else:
                self.loader.login(user=LOGIN, passwd=PASSWORD)
                self.loader.save_session_to_file('data/session-inst')
                print(f"Successful {'re-login' if relogin else 'login'} attempt")
        except Exception as e:
            print(f"Unsuccessful {'re-login' if relogin else 'login'} attempt: {e}")

    def get_embedding(self, image_path: str) -> list:
        """
        Get the embedding of the given image using the ResNet50 model.

        :param image_path: str – The file path to the image.
        :return: list – The embedding of the image as a list.
        """
        try:
            img = image.load_img(image_path, target_size=(224, 224))
            img_data = image.img_to_array(img)
            img_data = np.expand_dims(img_data, axis=0)
            img_data = preprocess_input(img_data)
            embedding = self.model.predict(img_data, verbose=0).flatten()
            return embedding.tolist()
        except Exception as e:
            print(f"Error while getting embedding: {e}")
            return []

    def knn_refit(self) -> None:
        """
        Refit the KNN model using embeddings from the database.

        :return: None.
        """
        db = sqlite3.connect('data/losty_db.db')
        cur = db.cursor()
        try:
            embeddings_data = []
            # Retrieve all embeddings from the images table.
            cur.execute('SELECT embedding FROM images')
            for row in cur.fetchall():
                # Deserialize the embedding from JSON format and append to list.
                embedding = json.loads(row[0])
                embeddings_data.append(embedding)

            # If embeddings are found, fit the KNN model.
            if embeddings_data:
                embeddings_array = np.array(embeddings_data)
                self.knn.fit(embeddings_array)

        except Exception as e:
            print(f"An error occurred while refitting KNN model: {e}")
            db.rollback()
        finally:
            cur.close()
            db.close()

    def find_matches(self, input_image_path: str) -> dict:
        """
        Find matches for the given image by calculating the distances between its embedding
        and the embeddings of all images in the database. The matches are paginated and returned
        as a dictionary where each key is a page number and the value is a dictionary where
        each key is a shortcode, and the value is a list of the following format:
        [image_path, date, image_path, match_percentage].

        :param input_image_path: The file path to the image to find matches for.
        :return: A dictionary of matches paginated by page number.
        """

        # Retrieve the embedding of the given image.
        input_embedding = self.get_embedding(input_image_path)
        input_embedding_array = np.array(input_embedding).reshape(1, -1)
        try:
            # Calculate the distances between the input embedding and all embeddings in the database.
            distances, indices = self.knn.kneighbors(input_embedding_array, n_neighbors=self.knn.n_samples_fit_)
        except Exception as e:
            print(f"An error occurred while finding matches: {e}")
            return {}

        # Initialize an empty dictionary to store the matches.
        pages = {}
        # Define the page size.
        page_size = 10
        # Initialize the page number.
        current_page = 1

        db = sqlite3.connect('data/losty_db.db')
        cur = db.cursor()

        try:
            # Iterate over the indices of the nearest neighbors.
            for index in indices[0]:
                index = int(index)
                # Retrieve the post_id and image_path of the current nearest neighbor.
                cur.execute('''
                        SELECT post_id, image_path FROM images LIMIT 1 OFFSET ?;
                    ''', (index,))
                match = cur.fetchone()
                if match:
                    post_id, image_path = match
                    # Retrieve the shortcode and date of the post.
                    cur.execute('''
                            SELECT shortcode, date FROM posts WHERE post_id = ?;
                        ''', (post_id,))
                    row = cur.fetchone()
                    shortcode = row[0]
                    date = row[1]
                    # If the current page is not in the pages dictionary, add it.
                    if current_page not in pages:
                        pages[current_page] = {}
                    # If the shortcode is not already in the pages dictionary, add it.
                    if not any(shortcode in d for d in pages.values()):
                        # Calculate the match percentage.
                        match_percentage = (1 - distances[0][list(indices[0]).index(index)])
                        # Add the match to the pages dictionary.
                        pages[current_page][shortcode] = [
                            os.path.basename(os.path.dirname(image_path)),  # group
                            date,
                            image_path,
                            match_percentage
                        ]
                    # If the current page is full, increment the page number.
                    if len(pages[current_page]) >= page_size:
                        current_page += 1
            return pages
        except Exception as e:
            print(f"Error while finding matches: {e}")
            db.rollback()
            return {}
        finally:
            cur.close()
            db.close()

    def update_data(self, start_date: datetime = None) -> None:
        """
        Update the database with new data from the given groups.

        :param start_date: datetime.datetime - The start date of the posts to be downloaded.
        :return: None.
        """

        if start_date is None:
            start_date = datetime.now(tz=timezone.utc).replace(tzinfo=None) - TIME_DELTA

        for group in self.groups:
            try:
                # Load profile from Instagram.
                profile = instaloader.Profile.from_username(self.loader.context, group)

                # Ensure the group's directory exists.
                if not os.path.exists(f"data/groups/{group}"):
                    os.mkdir(f"data/groups/{group}")

                # Download posts from the profile, filtering by the start date, in 'data/groups' directory.
                original_dir = os.getcwd()
                os.chdir('data/groups')

                self.loader.posts_download_loop(
                    posts=profile.get_posts(),
                    target=group,
                    fast_update=True,
                    takewhile=lambda post: post.date_utc > start_date,
                    possibly_pinned=3
                )
                os.chdir(original_dir)

                # Save profile picture if it doesn't exist.
                if not os.path.exists(f"data/groups/{group}/{group}_profile_pic.jpg"):
                    profile_pic_url = profile.profile_pic_url_no_iphone
                    image_b = requests.get(profile_pic_url).content
                    image = Image.open(io.BytesIO(image_b))
                    image.save(f"data/groups/{group}/{group}_profile_pic.jpg")

            except Exception as e:
                print(f"Attempt to authorization or posts download aborted: {e}")
                self.knn_refit()
                self.login(relogin=True)
                return

            # Add the group to the database.
            group_id = add_group(group)

            try:
                # Process each file in the group's directory.
                for file_name in os.listdir(f"data/groups/{group}/"):
                    # Delete .mp4 files.
                    if file_name.endswith('.mp4'):
                        os.remove(os.path.join('data', 'groups', group, file_name))
                        continue

                    if file_name.endswith('.json.xz'):
                        with lzma.open(os.path.join('data', 'groups', group, file_name), 'rt') as file:
                            data = json.load(file)['node']

                        shortcode = data['shortcode']

                        # Check if the post is already in the database.
                        if not check_shortcode(shortcode):
                            date = datetime.fromtimestamp(data['date'], tz=timezone.utc)
                            caption = data['caption']
                            filename = date.strftime('%Y-%m-%d_%H-%M-%S_UTC')

                            # Add the post to the database.
                            post_id = add_post((group_id, shortcode, date, caption))

                            # Determine the type and process accordingly.
                            if data['__typename'] in ('GraphImage', 'GraphVideo'):
                                image_path = os.path.join('data', 'groups', group, f"{filename}.jpg")
                                embedding = self.get_embedding(image_path)
                                add_image(post_id, image_path, embedding)
                            elif data['__typename'] == 'GraphSidecar':
                                for i in range(1, len(data['edge_sidecar_to_children']['edges']) + 1):
                                    image_path = os.path.join('data', 'groups', group, f"{filename}_{i}.jpg")
                                    embedding = self.get_embedding(image_path)
                                    add_image(post_id, image_path, embedding)
                            else:
                                continue
            except Exception as e:
                print(f"Error processing files for group {group}: {e}")

            # Log successful update.
            print(f"Data from group @{group} updated")

        # Refit KNN model and cleanup old data.
        self.knn_refit()
        cleanup_data()
