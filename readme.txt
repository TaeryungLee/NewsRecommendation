NYT Comments Dataset: https://www.kaggle.com/aashita/nyt-comments

Unzip nyt.zip, txt_sentoken.zip into main working directory.

Code based on python 3.8.

requirements can be installed by "pip install -r requirements.txt".

NLTK components must be downloaded before running the code.

Can be downloaded in python code:
         import nltk
         nltk.download()

Time-consuming works are saved in pickle files, automatically loaded while running main.py.

Delete pickle files to run from base.

Takes about 10 hours using 10 multiprocessors.

* Added: downloading nyt.zip, txt_sentoken.zip fails by git clone.
         Manually download it.
