#!/bin/bash

# Download NLTK data
python -m nltk.downloader punkt
python -m nltk.downloader averaged_perceptron_tagger
python -m nltk.downloader all
python -m textblob.download_corpora