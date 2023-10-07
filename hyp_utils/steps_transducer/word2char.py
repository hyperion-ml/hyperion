import os
import sys

word_file = sys.argv[1] # "data/it_test_proc_audio/text"
char_file = sys.argv[2] # "data/it_test_proc_audio/text_char"


# word_file = "exp/transducer/wav2vec2xlsr300m_transducer_v3.3_it.s1/it_test_proc_audio/transducer.text"
# char_file = "exp/transducer/wav2vec2xlsr300m_transducer_v3.3_it.s1/it_test_proc_audio/transducer_char.text"

output_chars = []
with open(word_file, "r") as fi:
    for line in fi.readlines():
        words = line.split(" ")
        chars = [words[0]]
        for wrd in words[1:]:
            for c in wrd:
                chars.append(c)
        output_chars.append(chars)

with open(char_file, "w") as fo:
    for chars in output_chars:
        fo.writelines(" ".join(chars))

