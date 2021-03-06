'''
AS Watson Hackathon
Team 8
Sheng Zhang
'''

# Import libraries
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

## Association Rules
# Read in transaction-level data
transaction_data = pd.read_csv("Data/test_trans.csv", dtype={'CONTACT_ID': str, 'SEG_NAME': str, 'SEG_NUM': str, 'SEG_START_DT': str, 'SEG_END_DT': str}, encoding = "ISO-8859-1")

# Construct the basket object used for AR algorithm
basket = (transaction_data.groupby(['ORDER_NUMBER', 'PRODUCT_NAME'])['ITEM_QUANTITY']
          .sum().unstack().reset_index().fillna(0)
          .set_index('ORDER_NUMBER'))

# Transform purchase amount to binary
def transform_binary(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1
basket_sets = basket.applymap(transform_binary)

# Find out the frequent product sets based on the Support metrics
frequent_product_sets = apriori(basket_sets, min_support=0.07, use_colnames=True)# Figure out association rules for frequent product sets

# Determine the association rules for product recommendations
rules = association_rules(frequent_product_sets, metric="lift", min_threshold=1)


## Product Unit Margin
# Read in relevant data tables
product_margin = pd.read_csv("Data/product_margin.csv", encoding = "ISO-8859-1")
product = pd.read_csv("Data/product.csv", encoding = "ISO-8859-1")

# Compute price, unit cost, and unit profit
product_margin["PRICE"] = product_margin["SALES_AMOUNT"] / product_margin["SALES_UNIT"]
product_margin["AVG_MARGIN"] = product_margin["GROSS_MARGIN"] / product_margin["SALES_UNIT"]
product_margin["UNIT_COST"] = product_margin["PRICE"] - product_margin["AVG_MARGIN"]
product_margin["MARGIN_RATE"] = product_margin["AVG_MARGIN"] / product_margin["PRICE"]

# Merge tables and deal with missing values
product_margin_no_na = product_margin.dropna()   # should check for na counts with pd.isnull(product_margin).sum() before dropping
product_merge = pd.merge(product_margin_no_na,product,on = "PRODUCT_ID",how = "left")
product_merge_no_na = product_merge.dropna()

# Exclude invalid entries
product_merge_valid = product_merge_no_na[(~product_merge_no_na["BRAND_NAME"].isin(["COUPON"])) & (product_merge_no_na["SALES_AMOUNT"] > product_merge_no_na["GROSS_MARGIN"])]



'''
Need to:
- Handle user input of customer IDs
- Find out all past products of the inputed customer
- Sort these products by the # of times bought
- Output a product list
'''


# Find out the top product to cross-sell based on association rules


# Find out the top substitute (same product hierarchy 3) in terms of margin
top_product_hier3 = product_merge_valid.sort_values("AVG_MARGIN",ascending = False)["PRODUCT_NAME"][0]








