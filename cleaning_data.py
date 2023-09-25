import pandas as pd

def load_clean_data():
    dtypes = {
        "'Unnamed: 0'": "int64",
        "DEALKEY": "float64",
        "DEALDETKEY": "float64",
        "CUSTOMERID": "object",
        "CHANNELID": "category",
        "CHANNELDSC": "object",
        "HAS_REAL_ORDER": "category",
        "DEALVALUE": "float64",
        "DEALQTY": "float64",
        "ORDER_PRICE": "float64",
        "ORIGINAL_PRICE": "float64",
        "ORIGINAL_PRICE_NO_DISCOUNT_PERS": "float64",
        "CUSTOMER_DISCOUNT_PERS": "float64",
        "DISCOUNT_PCT": "float64",
        "ARTID": "object",
        "ARTDSC": "object",
        "BRANDID": "object",
        "CATEGORYID": "object",
        "CATEGORYDSC": "object",
        "BRANDDSC": "object",
    }
    date_cols = ["QUOTATION_DATE", "ARTID_DATE_CREATE","CUSTOMER_CREATE_DATE"]

    dataset = pd.read_csv(
        "case_study_anonymized.csv",
        sep="|",
        encoding="latin",
        dtype=dtypes,
        parse_dates=date_cols,
    )
    dataset.rename(columns={'Unnamed: 0': 'Unnamed'}, inplace=True)
    return dataset
