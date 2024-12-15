import polars as pl
from grailed_api import GrailedAPIClient

# Premade client from https://github.com/pznamir00/Grailed-API
client = GrailedAPIClient()

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

if __name__ == "__main__":
    no_of_hits = 100
    products = get_latest_sold_products(no_of_hits)
    
    ## Filter out keys we don't
    # Not needed are: price, badges, bumped_at, dropped??, marketplace?, price_doprs, price_i??, price_updated_at, traits, user, shipping
    # Ones we could consider are: buynow, category_path_size, category, color, location, makeoffer, traits, _rankingInfo?
    # Should haves: description, strata, heat_recency?, sold_at    
    practical_labels = ['id', 'created_at']
    X_labels = ['designers', 'title', 'condition', 'category_path_size']
    Y_label = 'sold_price'
    relevant_labels = practical_labels + X_labels + [Y_label]
        
    filtered_products = [filter_item_keys(product, relevant_labels) for product in products]
    
    # For now the designers are represented using a joined string
    primitive_products = [{**product, 'designers': '<sep>'.join([designer['name'] for designer in product['designers']])} 
                          for product in filtered_products]
    # print(primitive_products[0])
        
    # Convert to polars df
    df = dict_to_polars(primitive_products)
    print(df)
    
    # Clean the df, etc.
    # Add expectation that sold_price > 0 in all rows
    assert df.filter(pl.col('sold_price') <= 0).is_empty()
    # TODO: Check that no column has any null values

    # TODO: Represent categorical variables (we will most likely need a feature store for this, since we need to store the mappings)
    # IMO, let's represent designers, title & (most likely) category_path_size cat. vars as embeddings
    # condition we can represent using one-hot encoding or as an ordinal number