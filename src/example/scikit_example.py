import sklearn.datasets as _SkData

_PATH = './assets/review_polarity/txt_sentoken'
_POS_PATH = './assets/review_polarity/txt_sentoken/pos/'
_NEG_PATH = './assets/review_polarity/txt_sentoken/neg/'

# Load data from pos and neg
posData = _SkData.load_files(_PATH, categories='pos')
negData = _SkData.load_files(_PATH, categories='neg')
print('posData = ', posData)
print('negData = ', negData)


