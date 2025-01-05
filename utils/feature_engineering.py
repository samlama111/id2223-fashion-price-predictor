import polars as pl
from datetime import datetime
from grailed_api import GrailedAPIClient
from dotenv import load_dotenv
from fastembed import TextEmbedding

load_dotenv()

# Premade client from https://github.com/pznamir00/Grailed-API
client = GrailedAPIClient()

# This will trigger the model download and initialization
embedding_model = TextEmbedding()


def __embed_text(documents: list[str]) -> list[list[float]]:
    embeddings_generator = embedding_model.embed(
        documents
    )  # reminder this is a generator
    embeddings_list = list(embeddings_generator)
    # you can also convert the generator to a list, and that to a numpy array

    return embeddings_list


def __get_latest_sold_products(no_of_hits=100):
    # Per default, includes only Men's products and all brands/items
    products = client.find_products(sold=True, hits_per_page=no_of_hits)
    return products[
        no_of_hits:
    ]  # First no_of_hits entries are the most recent, not sold


def __filter_products(products: list[dict], keys) -> list[dict]:
    return [
        {key: product[key] for key in keys if key in product} for product in products
    ]


def __dict_to_polars(products):
    return pl.DataFrame(products)


def __item_condition_to_ordinal(condition):
    # is_worn -> 0, is_used -> 1, is_gently_used -> 2, is_new -> 3
    if condition == "is_worn":
        return 0
    elif condition == "is_used":
        return 1
    elif condition == "is_gently_used":
        return 2
    elif condition == "is_new":
        return 3
    else:
        raise ValueError(f"Invalid condition: {condition}")


def transform_features(products: list[dict]) -> list[dict]:
    transformed_products = []

    print("embedding designer names")
    designer_names = [
        " ".join(
            sorted([names.strip() for names in product["designer_names"].split("x")])
        )
        for product in products
    ]
    embedded_designer_names = __embed_text(designer_names)

    print("embedding descriptions")
    descriptions = [product["description"] for product in products]
    embedded_description = __embed_text(descriptions)

    print("embedding titles")
    titles = [product["title"] for product in products]
    embedded_titles = __embed_text(titles)

    print("embedding hashtags")
    hashtags = [" ".join(sorted(product["hashtags"])) for product in products]
    embedded_hashtags = __embed_text(hashtags)

    for i, product in enumerate(products):
        transformed_product = {
            **product,
            "designer_names": embedded_designer_names[i],
            "description": embedded_description[i],
            "embedded_hashtags": embedded_hashtags[i],
            "title": embedded_titles[i],
            "condition": __item_condition_to_ordinal(product["condition"]),
        }
        transformed_products.append(transformed_product)
    return transformed_products


def pipeline(no_of_hits=100):
    products = __get_latest_sold_products(no_of_hits=no_of_hits)
    # Check that we got the right number of products
    assert (
        len(products) == no_of_hits
    ), f"Expected {no_of_hits} products, got {len(products)}"

    ## Filter out keys we don't need or are out of scope (e.g. user info, etc.)
    dimension_labels = ["id", "sold_at"]
    x_labels = [
        "designer_names",
        "description",
        "title",
        "hashtags",
        "category_path",
        "condition",
        "size",
        "color",
        "followerno",
        "userScore",
    ]
    y_label = ["sold_price"]
    labels = dimension_labels + x_labels + y_label
    filtered_products = __filter_products(products, labels)
    transformed_products = transform_features(filtered_products)

    # Convert to polars df
    df = __dict_to_polars(transformed_products)

    # Cast time columns to DateTime
    df = df.with_columns(pl.col("sold_at").cast(pl.Datetime))

    # For now, just drop any items with null values
    df = df.drop_nulls()

    # Add expectation that sold_price > 0 in all rows
    assert df.filter(pl.col("sold_price") <= 0).is_empty()
    # Check that no column has any null values
    df_missing = df.filter(pl.any_horizontal(pl.all().is_null()))
    assert df_missing.is_empty()
    # Check that sold_at is not null, is in the past and of DateTime type
    current_time = datetime.now()
    assert df.filter(pl.col("sold_at") >= current_time).is_empty()

    # Clean the df, etc.
    # Drop unused columns
    # df = df.drop(["designers", "title"])

    print(df)

    return df
