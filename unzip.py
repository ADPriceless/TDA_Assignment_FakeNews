"""Unzip the data files for Paperspace"""

import zipfile


def main():
    """Unzip 'data/text_only.zip'"""
    with zipfile.ZipFile('data/text_only.zip', 'r') as zip_ref:
        zip_ref.extractall('data/')


if __name__ == '__main__':
    main()
