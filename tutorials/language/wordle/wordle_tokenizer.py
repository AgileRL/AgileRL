import re

from agilerl.data.tokenizer import Tokenizer


class WordleTokenizer(Tokenizer):
    def __init__(self):
        self.special_vocab = [
            "<g>",
            "<b>",
            "<y>",
            "<|pad|>",
            "</a>",
            "</s>",
            "<s>",
            "<a>",
            "</eod>",
        ]
        self.vocab = list("abcdefghijklmnopqrstuvwxyz") + self.special_vocab
        self.t2i = {w: i for i, w in enumerate(self.vocab)}
        super().__init__(
            self.token_to_id("<|pad|>"),
            self.token_to_id("</s>"),
            self.token_to_id("</a>"),
            self.token_to_id("<s>"),
            self.token_to_id("<a>"),
            self.token_to_id("</eod>"),
        )

    def encode(self, str_, **kwargs):
        if isinstance(str_, str):
            special_idxs = []
            for special_char in self.special_vocab:
                special_idxs += list(
                    map(
                        lambda x: (x.start(), x.end(), self.token_to_id(special_char)),
                        re.finditer(re.escape(special_char), str_),
                    )
                )
            special_idxs.sort(key=lambda x: x[0])
            tokens = []
            curr = 0
            for s, e, tok in special_idxs:
                tokens.extend([self.token_to_id(c) for c in str_[curr:s]])
                tokens.append(tok)
                curr = e
            tokens.extend([self.token_to_id(c) for c in str_[curr:]])
            return tokens, [int(t != self.pad_token_id) for t in tokens]
        elif isinstance(str_, list):
            tokens, pads = zip(*[self.encode(item) for item in str_])
            max_len = max(map(len, tokens))
            return [
                list(item) + ([self.pad_token_id] * (max_len - len(item)))
                for item in tokens
            ], [list(item) + ([0] * (max_len - len(item))) for item in pads]
        else:
            raise ValueError("str_ must be a string or a list of strings")

    def decode(self, tokens, **kwargs):
        if len(tokens) == 0:
            return ""
        if not isinstance(tokens[0], list):
            return "".join([self.id_to_token(item) for item in tokens])
        elif isinstance(tokens[0], list):
            return [self.decode(item) for item in tokens]
        else:
            raise ValueError("tokens must be a list of ints or a list of lists of ints")

    def num_tokens(self):
        return len(self.vocab)

    def id_to_token(self, id_):
        return self.vocab[id_]

    def token_to_id(self, token):
        return self.t2i[token]

    def get_vocab(self):
        return self.vocab
