import sys
import struct
from transformers import AutoTokenizer
from sentencepiece import SentencePieceProcessor

class Tokenizer:
    def __init__(self, tokenizer):
        self.sp_model = SentencePieceProcessor(tokenizer.vocab_file)

        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        print(f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}")
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def export(self, exportPath):

        # get all the tokens (postprocessed) and their scores as floats
        tokens, scores = [], []
        for i in range(self.n_words):

            # decode the token and light postprocessing
            t = self.sp_model.id_to_piece(i)
            s = self.sp_model.get_score(i)
            if i == self.bos_id:
                t = '\n<s>\n'
            elif i == self.eos_id:
                t = '\n</s>\n'
            elif len(t) == 6 and t.startswith('<0x') and t.endswith('>'):
                t = chr(int(t[3:5], 16)) # e.g. make '<0x01>' into '\x01'
            t = t.replace('▁', ' ') # sentencepiece uses this character as whitespace
            b = t.encode('utf-8') # bytes of this token, utf-8 encoded

            tokens.append(b)
            scores.append(s)

        # write to a binary file
        with open(exportPath, 'wb') as f:
            f.write(struct.pack("I", self.n_words))
            f.write(struct.pack("I", self.bos_id))
            f.write(struct.pack("I", self.eos_id))
            for bytes, score in zip(tokens, scores):
                f.write(struct.pack("fI", score, len(bytes)))
                f.write(bytes)

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", trust_remote_code=True)

    exportPath = sys.argv[1]
    t = Tokenizer(tokenizer)
    t.export(exportPath)