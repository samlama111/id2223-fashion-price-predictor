import os

import polars as pl
import google.generativeai as genai
from grailed_api import GrailedAPIClient
from dotenv import load_dotenv

load_dotenv()

# Premade client from https://github.com/pznamir00/Grailed-API
client = GrailedAPIClient()
# This and the embed_text function are taken from https://ai.google.dev/gemini-api/docs/embeddings#generate-embeddings
genai.configure(api_key=os.environ["GOOGLE_AI_STUDIO_API_KEY"])

def get_latest_sold_products(no_of_hits=100):
    # Per default, includes only Men's products and all brands/items
    products = client.find_products(
        sold=True,
        # page=1,
        hits_per_page=no_of_hits,
    )
    return products[no_of_hits:] # First no_of_hits entries are the most recent, not sold

def get_latest_product():
    # Per default, includes only Men's products and all brands/items
    products = client.find_products(
        sold=False,
        # page=1,
        hits_per_page=1,
    )
    return products

def filter_item_keys(product, relevant_labels):
    return {key: product[key] for key in relevant_labels if key in product}

def dict_to_polars(products):
    return pl.DataFrame(products)

def embed_text(text):
    # Embeddings are free, as defined at https://ai.google.dev/pricing#text-embedding004
    result = genai.embed_content(
            model="models/text-embedding-004",
            content=text)

    return result['embedding']

def pipeline():
    no_of_hits = 100
    products = get_latest_sold_products(no_of_hits)
    
    ## Filter out keys we don't
    # Not needed are: price, badges, bumped_at, dropped??, marketplace?, price_doprs, price_i??, price_updated_at, traits, user, shipping
    # Ones we could consider are: buynow, category_path_size, category, color, location, makeoffer, traits, _rankingInfo?
    # Should haves: description, strata, heat_recency?    
    practical_labels = ['id', 'created_at', 'sold_at']
    X_labels = ['designers', 'title', 'condition', 'category_path_size']
    Y_label = 'sold_price'
    relevant_labels = practical_labels + X_labels + [Y_label]
        
    filtered_products = [filter_item_keys(product, relevant_labels) for product in products]
    
    # For now the designers are represented using a joined string
    # primitive_products = [{**product, 'designers': '<sep>'.join([designer['name'] for designer in product['designers']])} 
    primitive_products = [{**product, 'designers': ' '.join([designer['name'] for designer in product['designers']])} 
                          for product in filtered_products]
    
    # Combine relevant columns (designers and title)
    enriched_products = [{**product, 'designers_title': f"{product['designers']}: {product['title']}"} 
                          for product in primitive_products]
        
    # Convert to polars df
    df = dict_to_polars(enriched_products)
    
    # Change created_at and sold_at to DateTime
    df = df.with_columns(
        pl.col('created_at').cast(pl.Datetime),
        pl.col('sold_at').cast(pl.Datetime)
    )
    
    # Add expectation that sold_price > 0 in all rows
    assert df.filter(pl.col('sold_price') <= 0).is_empty()
    # TODO: Check that no column has any null values
    # TODO: Check that created_at and sold_at are not null, are in the past and of DateTime type

    # Clean the df, etc.
    # Drop unused columns
    df = df.drop(['designers', 'title'])
    
    # TODO: Represent categorical variables (we will most likely need a feature store for this, since we need to store the mappings)
    # For now let's only take into account designers_title
    # In future, account for category_path_size (as embeddings) and potentially separate embeddings for designers and title
    # and 'condition' using one-hot encoding or as an ordinal number
    df = df.with_columns(
        pl.col('designers_title')
        .map_elements(embed_text, return_dtype=pl.List(pl.Float32))
        .alias('designers_title_embedding')
    )
    print(df)
    
    return df
    
if __name__ == "__main__":
    pipeline()
