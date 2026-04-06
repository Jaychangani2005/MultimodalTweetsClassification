from __future__ import annotations

from pathlib import Path
import tarfile
import zipfile

import pandas as pd

from exp.external import aidrtokenize

# function to untar data and unzip agreed label


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def untar_data_and_unzip_label(tar_data_file: str, zip_agreed_label_file: str) -> None:
    data_dir = PROJECT_ROOT / 'data'
    with tarfile.open(data_dir / tar_data_file) as tf:
        tf.extractall(data_dir)
    with zipfile.ZipFile(data_dir / zip_agreed_label_file) as zf:
        zf.extractall(data_dir)


# function to remove non-ASCII chars from data
def clean_ascii(text):
    return ''.join(i for i in text if ord(i) < 128)


# dictionary of tsv files of all three train,dev, and test split for both task
tsv_data_files = {'humanitarian_task_tsv_files':
                  (str(PROJECT_ROOT / 'sample_data' / 'task-humanitarian-text-img-agreed-lab-train.tsv'),
                   str(PROJECT_ROOT / 'sample_data' / 'task-humanitarian-text-img-agreed-lab-dev.tsv'),
                   str(PROJECT_ROOT / 'sample_data' / 'task-humanitarian-text-img-agreed-lab-test.tsv')),
                  'Informativeness_task_tsv_files':
                  (str(PROJECT_ROOT / 'sample_data' / 'task_informative_text_img_agreed_lab_train.tsv'),
                   str(PROJECT_ROOT / 'sample_data' / 'task_informative_text_img_agreed_lab_dev.tsv'),
                   str(PROJECT_ROOT / 'sample_data' / 'task_informative_text_img_agreed_lab_test.tsv')),

                  }

# function to get tsv files of all three splits for a particular task


def get_tsv_data_files(task_name_tsv_files: str):
    if task_name_tsv_files == 'Informativeness_task_tsv_files':
        info = True
    else:
        info = False
    return list(tsv_data_files[task_name_tsv_files]) + [info]


def get_dataframe(train_tsv: str, dev_tsv: str, test_tsv: str, info: bool, path: Path):
    print('reading data and preprocessing it.....')
    train = pd.read_csv(train_tsv, delimiter='\t', encoding='utf-8')
    if info:
        train = train.drop(0, axis=0)
    dev = pd.read_csv(dev_tsv, delimiter='\t', encoding='utf-8')
    test = pd.read_csv(test_tsv, delimiter='\t', encoding='utf-8')

    # to remove redundant links,hashtag sign, and non-ASCII characters
    train['tweet_text'] = train['tweet_text'].apply(
        lambda x: aidrtokenize.tokenize(x))
    dev['tweet_text'] = dev['tweet_text'].apply(
        lambda x: aidrtokenize.tokenize(x))
    test['tweet_text'] = test['tweet_text'].apply(
        lambda x: aidrtokenize.tokenize(x))

    train['tweet_text'] = train['tweet_text'].apply(lambda x: clean_ascii(x))
    dev['tweet_text'] = dev['tweet_text'].apply(lambda x: clean_ascii(x))
    test['tweet_text'] = test['tweet_text'].apply(lambda x: clean_ascii(x))

    train['is_valid'] = False
    dev['is_valid'] = True

    data = pd.concat([train, dev], axis=0).reset_index()
    data = data.drop(['index'], axis=1)
    test_data = test
    print("done!!")
    return data, test_data
