from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import os
import re
import sklearn.datasets as _SkData

_PATH = './assets/review_polarity/txt_sentoken'
_REPLACE_NO_SPACE = "[.;:!\'?,\"()\[\]]"



