# Author: Vicky Mohammad
# Description: File for proccessing the POS tagging.

# This lib is for the POS-tagging and other
import nltk as _pos

# Download nltk dependencies
# if you 'nltk.download()' without param
# it will open gui for downloading list of other
_pos.download('punkt')
_pos.download('averaged_perceptron_tagger')


class POSTagging:
    def __init__(self):
        super().__init__()

    def runExample(self):
        tokens = _pos.word_tokenize(
            'Can you please buy me an Arizona Ice Tea? It\'s $0.99')
        print('POSTagging.runExample(): tokens = ', tokens)
        print('POSTagging.runExample(): pos_tag() = ', _pos.pos_tag(tokens))

    def run(self, doc):
        """Run the POS tagging.
        @doc document strings to be tagged.
        return list of tuples, index 0 is the word, 1 is the tag."""
        return _pos.pos_tag(_pos.word_tokenize(doc))
