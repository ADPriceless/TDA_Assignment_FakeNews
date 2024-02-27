"""A file to clean the news article dataset to be ready to
preprocess for a model"""

import pandas as pd


def clean_data(file_path: str) -> pd.DataFrame:
    """Function to clean the dataset"""
    df = _read_data(file_path)
    # remove German articles
    df = _drop_rows(df, 'language', 'german')
    df = _keep_columns(df, ['title', 'text', 'label'])
    df = _drop_null(df)
    df = _map_rename_label(df, 'label', 'is_fake', {'Real': 0, 'Fake': 1})
    return df


def _read_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)


def _drop_rows(df: pd.DataFrame, column: str, value: str) -> pd.DataFrame:
    return df[df[column] != value]


def _keep_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    return df[columns]


def _drop_null(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna()


def _map_rename_label(
    df: pd.DataFrame,
    old_label: str,
    new_label: str,
    mapping: dict
) -> pd.DataFrame:
    # map to real = 0, fake = 0
    df[old_label] = df[old_label].map(mapping)
    # rename label to is_fake
    return df.rename(columns={old_label: new_label})




if __name__ == '__main__':
    # clean the dataset
    df_clean = clean_data('data/news_articles (2).csv')
    # save the clean dataset
    df_clean.to_csv('data/news_articles_clean.csv', index=False)
