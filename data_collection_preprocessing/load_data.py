import polars as pl
from grailed_api import GrailedAPIClient
from dotenv import load_dotenv

load_dotenv()

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

# TODO: Fix this
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

def pipeline(no_of_hits=100):
    products = get_latest_sold_products(no_of_hits=no_of_hits)
    # Check that we got the right number of products
    assert len(products) == no_of_hits
    
    ## Filter out keys we don't need or are out of scope (e.g. user info, etc.)
    practical_labels = ['id', 'sold_at']
    X_labels = ['designers', 'title', 'subcategory', 'category', 'condition', 'description', 'size', 'color']
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
    
    # Cast time columns to DateTime
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

    print(df)
    
    return df
