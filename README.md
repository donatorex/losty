# Losty – telegram bot for finding lost pets.
Losty is a Telegram bot designed to assist in finding lost pets by leveraging Instagram data. It fetches posts from specific Instagram groups related to lost and found pets, analyzes images using machine learning, and matches them with images provided by users.

URL: https://t.me/losty_pets_bot

This script defines a Python module for managing a database of Instagram posts and images,
primarily focused on lost and found pet groups. It utilizes Instagram scraping, image
embedding generation using a ResNet50 model, and K-Nearest Neighbors (KNN) for finding
similar images.

The module provides functionality to:

    — Upload data about posts in specified Instagram groups and auto-update in a set period.
    — Extract image embeddings using a pre-trained ResNet50 deep learning model.
    — Add and manage groups, posts, images and their embeddings in a SQLite database.
    — Cleaning the database and storage from old posts.
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

The bot works in Almaty, Kazakhstan.
