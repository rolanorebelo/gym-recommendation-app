#!/bin/bash
pip install --upgrade pip
pip install -r requirements.txt
python -m textblob.download_corpora
python -m nltk.downloader punkt
python -m nltk.downloader averaged_perceptron_tagger