import numpy as np
import pandas as pd


def save_pred(test_df, y, sub_id):
    original_index = test_df.index,
    output_df = pd.DataFrame(y, index=original_index, columns=["LABELS"], dtype=int)
    output_df.to_csv(f"./submission/submission{sub_id}.csv")