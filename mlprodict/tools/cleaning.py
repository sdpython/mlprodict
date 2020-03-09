"""
@file
@brief Better display.
"""


def clean_error_msg(df):
    """
    Removes EOL from error messages in dataframes.

    @param      df      dataframe
    @return     df      dataframe
    """
    def clean_eol(value):
        if isinstance(value, str):
            return value.replace("\n", " -- ")
        return value

    df = df.copy()
    for c in df.columns:
        if "ERROR" in c:
            df[c] = df[c].apply(clean_eol)
    return df
