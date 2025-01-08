# Grailed Fashion Price Predictor

ID2223 final project - a Machine Learning (ML) pipeline for predicting the price of a fashion items listed on the online marketplace Grailed.

Created by Samuel Horacek (horacek@kth.se) and Eugene Park (epark@kth.se).

## How to run

1. Install dependencies, e.g. using `pip install -r requirements.txt`.
2. Fill the `.env` file with required secrets, defined in the `.env.example` file.

## Introduction and problem overview

Grailed is a platform for buying and selling used fashion items. These fashion items can be designer garments, rare and archive pieces, or streetwear. What is often common for these items is that they are possibly not so easily found in stores, and thus the price is not so easily determined, since it can range from a few dollars to a few thousand.

Currently, there are million of live listings on the platform, and the number is growing. Every time a user wants to create a new listing, they have to fill out the relevant information about the item. This includes the brand, the size, the condition, color and others. Finally, a user has to set a price for the item. This can be tricky, and Grailed only provides a suggested price range, which can in some cases be quite inaccurate. This "Price range is calculated based on what most similar items recently sold for over the past few months (excluding shipping).". We think we can do better.

Our goal is to train a predictive model that can help users price their items accordingly. Using both historical data and the relevant information of the current listing, we want to predict the price of the item. For the suggestions to be as accurate as possible, we make use of information about previously sold items on Grailed.

### Data

There are a few approaches to getting this data. Since Grailed doesn't have an open API, we had to find another way to get the data. A common approach is to scrape the website, but this would require us to create such a tool from scratch.

Instead, we have used the [grailed_api](https://github.com/pznamir00/Grailed-API) Python library, which is a convenient wrapper for fetching data from Grailed. To do so, it creates and then sends requests to the Grailed's Algolia instance.

To train our model, we need to get the previously sold items. Since the Algolia instance/Grailed API allows only to query for a maximum of 1000 items, we consider the initial 1000 items as our original dataset (used in the backfill pipeline, see below). The dataset is being added to by an online pipeline.

### Feature engineering

To train our model, we need to abstract the data into features. We have created a feature engineering pipeline that takes the raw data and transforms it into a suitable format.

When selecting features, we were guided by our personal experience with the platform and by a [project](https://github.com/kirill-rubashevskiy/graildient-descent) we came across with a similar focus. 

We set on the following features, already present in the dataset:
- `category_path`: The category path of the item, e.g. `accessories.hats`.
- `color`: The color of the item, e.g. `Black`.
- `condition`: The condition of the item, e.g. `is_new`.
- `designer_names`: The brands/designers of the item, e.g. `Gucci`.
- `followerno`: The number of followers of the seller, e.g. `100`.
- `hashtags`: The hashtags of the item, e.g. `#gucci`.
- `sold_price`: The price of the item (in USD) and the target variable, e.g. `100`.
- `title`: The title of the item, e.g. `Gucci hat`.

There are other features we could have considered. We wanted to, but had issues with (`size` and `userScore`).
Finally, we consider features that help identify and sort the items - `id` and `sold_at`, but we did not use them to train our model.

Since a lot of these features are categorical, we had to come up with a way of representing them in a format understandable by our predictive model - as numerical values:

- `category_path`, `color` were represented using label encoding.
- `condition` was represented using one-hot encoding.
- `designer_names`, `description`, `title`, `hashtags` were represented using embeddings (using the [fastembed](https://github.com/qdrant/fastembed) library).

## Architecture



## Predicting prices

### Evaluation 

## Conclusion, future work

### Problems encountered

### Future work