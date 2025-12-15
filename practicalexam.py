
import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori,association_rules

#part A DATA PREPARATION

data = {
    'Transaction_ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Items': [
        ['Bread', 'Milk', 'Eggs'],
        ['Bread', 'Butter'],
        ['Milk', 'Diapers', 'Beer'],
        ['Bread', 'Milk', 'Butter'],
        ['Milk', 'Diapers', 'Bread'],
        ['Beer', 'Diapers'],
        ['Bread', 'Milk', 'Eggs', 'Butter'],
        ['Eggs', 'Milk'],
        ['Bread', 'Diapers', 'Beer'],
        ['Milk', 'Butter']
    ]
}

df = pd.DataFrame(data)
print("Original Dataset:")
print(df)
transactions = df['Items'].tolist()

print("\nTransaction Format:")
print(transactions)
# 2. Encoding  the transaction data into one-hot encoded format
te = TransactionEncoder()
te_ary = te.fit(df['Items']).transform(df['Items'])
encoded_df = pd.DataFrame(te_ary, columns=te.columns_)

print("Encoded Transaction Data (0=false,1=true):")
print(encoded_df.astype(int))
print("\n")

# Part B: Apriori Algorithm

# Generate frequent itemsets with min_support = 0.2
frequent_itemsets = apriori(encoded_df, min_support=0.2, use_colnames=True)

print("Frequent Itemsets (min_support = 0.2):")
print(frequent_itemsets.sort_values('support', ascending=False))
print("\n")

# Generate association rules with min_confidence = 0.5
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

print("Association Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values('lift', ascending=False))
print("\n")

