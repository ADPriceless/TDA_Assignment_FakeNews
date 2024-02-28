# pylint: disable=missing-function-docstring,redefined-outer-name


"""A file to clean the news article dataset to be ready to
preprocess for a model"""


import pandas as pd
from sklearn.model_selection import train_test_split


def clean_data(file_path: str) -> pd.DataFrame:
    """Function to clean the dataset"""
    df = _read_data(file_path)
    # remove German articles
    df = _drop_rows(df, 'language', 'german')
    df = _keep_columns(df, ['title', 'text', 'label'])
    df = _drop_null(df)
    df = _remove_newline_characters(df, 'title')
    df = _remove_newline_characters(df, 'text')
    df = _combine_columns(df, ['title', 'text'], 'text')
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


def _remove_newline_characters(df: pd.DataFrame, column: str) -> pd.DataFrame:
    removed = df[column].str.replace('\n','')
    df[column] = removed
    return df


def _combine_columns(df: pd.DataFrame, columns: list[str], new_column: str) -> pd.DataFrame:
    df[new_column] = df[columns[0]] + ' ' + df[columns[1]]
    if new_column in columns:
        columns.remove(new_column)
    return df.drop(columns=columns)


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


def split_into_test_and_train_datasets(df: pd.DataFrame) -> tuple:
    # split into train and test
    df_train, df_test = train_test_split(
        df, test_size=0.2, stratify=df['is_fake']
    )
    return df_train, df_test


def save_dataset(df: pd.DataFrame, file_path: str) -> None:
    df.to_csv(file_path, index=False)


if __name__ == '__main__':
    # clean the dataset
    df_clean = clean_data('data/news_articles (2).csv')
    # split into train and test
    df_train, df_test = split_into_test_and_train_datasets(df_clean)
    # save the clean, train and test datasets
    save_dataset(df_clean, 'data/news_articles_clean.csv')
    save_dataset(df_train, 'data/train.csv')
    save_dataset(df_test, 'data/test.csv')
