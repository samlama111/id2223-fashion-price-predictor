from grailed_api import GrailedAPIClient

# Premade client from https://github.com/pznamir00/Grailed-API
client = GrailedAPIClient()

def get_latest_products():
    # Per default, includes only Men's products and all brands/items
    products = client.find_products(
        sold=True,
        # page=1,
        hits_per_page=101, # Returns twice as many?
    )
    return products

if __name__ == "__main__":
    products = get_latest_products()
    print(type(products), len(products))
    print(products[0])