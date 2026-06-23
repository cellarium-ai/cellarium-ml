# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import itertools

import numpy as np
import pandas as pd

from cellarium.ml.utilities.data import categories_to_codes, categories_to_product_codes


def test_categories_to_codes():
    df = pd.DataFrame(
        {
            "A": ["a", "b", "a", "c", "b"],
            "B": ["x", "y", "x", "z", "y"],
            "C": [1, 2, 1, 3, 2],
        }
    ).apply(pd.Categorical)
    df_codes = categories_to_codes(df)
    np.testing.assert_array_equal(df_codes, np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0], [2, 2, 2], [1, 1, 1]]))


def enumerate_combinations(categories_per_column: list[int]) -> pd.DataFrame:
    categories_per_column = categories_per_column[::-1]

    # Generate the range of values for each column based on the number of categories
    category_ranges = [range(c) for c in categories_per_column]

    # Use itertools.product to get all possible combinations
    combinations = list(itertools.product(*category_ranges))

    # Convert to DataFrame for easy handling and return
    df = pd.DataFrame(combinations, columns=[f"col_{i}" for i in range(len(categories_per_column))])
    df = df[reversed(df.columns)]
    df.columns = reversed(df.columns)
    return df


def test_categories_to_product_codes():
    """
    This is the case where we are using (potentially multiple) categoricals
    to create a single code for each cell. This involves on-the-fly
    computation of new categorical identities.
    """
    n_cats = [3, 2, 2]
    df = enumerate_combinations([3, 2, 2]).apply(pd.Categorical)
    print(df)

    # all categories present in the dataframe slice presented
    df_codes = categories_to_product_codes(df)
    print(df_codes)
    np.testing.assert_array_equal(df_codes, np.arange(len(df)))

    # skip last column
    sub_df = df[df.columns[:-1]].iloc[: len(df) // n_cats[-1]]
    print(sub_df)
    df_codes2 = categories_to_product_codes(sub_df)
    print(df_codes2)
    np.testing.assert_array_equal(df_codes2, np.arange(len(sub_df)))

    # same as above but brute force compute categorical codes using pandas
    sub_df["brute_force"] = (sub_df["col_1"].astype(str) + "__" + sub_df["col_0"].astype(str)).astype("category")
    print(sub_df)
    print("brute force codes", sub_df["brute_force"].cat.codes.values)
    np.testing.assert_array_equal(df_codes2, sub_df["brute_force"].cat.codes.values)

    # some categories are missing from the slice presented
    df2 = enumerate_combinations([2, 2]).apply(pd.Categorical)
    print(df2)
    df2["col_0"] = df2["col_0"].cat.add_categories([2])
    print("modified col_0 categories:")
    print(df2["col_0"].cat.categories)
    df_codes4 = categories_to_product_codes(df2)
    print(df_codes4)
    lookup_df = enumerate_combinations([3, 2]).apply(pd.Categorical)
    inds = []
    for ind in range(len(df2)):
        # figure out where the row of df2 is in lookup_df
        row = df2.iloc[ind]
        lookup = lookup_df[(lookup_df["col_0"] == row["col_0"]) & (lookup_df["col_1"] == row["col_1"])]
        inds.append(lookup.index[0])
    print(inds)
    np.testing.assert_array_equal(df_codes4, np.array(inds))

    # dataframe with one column versus series
    df = pd.DataFrame({"A": ["a", "b", "a", "c", "b"]}).apply(pd.Categorical)
    df_codes5 = categories_to_product_codes(df)
    print("dataframe", df_codes5)
    df_codes6 = categories_to_product_codes(df["A"])
    print("series", df_codes6)
    np.testing.assert_array_equal(df_codes5, df_codes6)
