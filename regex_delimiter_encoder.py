import re
from functools import partial

from torchnlp.encoders.text.static_tokenizer_encoder import StaticTokenizerEncoder


def _tokenize(s, delimiter):
    return re.split(r"(" + delimiter + r")", s)

def _detokenize(s, delimiter):
    return "".join(s)

class RegexDelimiterEncoder(StaticTokenizerEncoder):
    def __init__(self, delimiter, *args, **kwargs):
        if 'tokenize' in kwargs:
            raise TypeError('``DelimiterEncoder`` does not take keyword argument ``tokenize``.')

        if 'detokenize' in kwargs:
            raise TypeError('``DelimiterEncoder`` does not take keyword argument ``detokenize``.')
        
        self.delimiter = delimiter
        
        super().__init__(
            *args,
            tokenize=partial(_tokenize, delimiter=self.delimiter),
            detokenize=partial(_detokenize, delimiter=self.delimiter),
            **kwargs)
