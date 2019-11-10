# This lib is for the POS-tagging and other
import nltk

# Download nltk dependencies
# if you "nltk.download()" wihout param
# it will open gui for downloading list of other
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Test the nltk
tokens = nltk.word_tokenize(
    "Can you please buy me an Arizona Ice Tea? It's $0.99")
print("Part of Speech: ", nltk.pos_tag(tokens))
