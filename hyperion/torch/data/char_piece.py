import logging

class CharPieceProcessor:
    def __init__(self):
        self.token2id = {}
        self.id2token = {}
        
    def load(self, token_list):
        for idx, token in enumerate(token_list):
            self.token2id[token] = idx
            self.id2token[idx] = token
        logging.info("Loaded {} tokens".format(len(self.token2id)))
        logging.info("First 10 tokens: {}".format(list(self.token2id.keys())[:10]))
        return True


    def piece_to_id(self, token):
        return self.token2id.get(token, self.token2id["<unk>"])

    def id_to_piece(self, idx):
        return self.id2token.get(idx, "<unk>")

    def encode_as_pieces(self, text):
        return [char for char in text]

    def encode(self, text, out_type=int):
        assert out_type in [int]
        return [self.piece_to_id(char) for char in text]

    def decode(self, ids):
        return ''.join([self.id_to_piece(idx) for idx in ids])

    def get_piece_size(self):
        return len(self.token2id)
