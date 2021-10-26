import numpy as np
import pandas as pd

# create Series Object:
prices_ds = pd.Series([1.5, 2, 2.5, 3],
            index=["apples", "oranges", "bananas", "strawberries"])

# create DataFrame Object from prices Series:
prices_df = pd.DataFrame(prices_ds)
print(prices_df)