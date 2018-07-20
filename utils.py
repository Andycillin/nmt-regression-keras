'''
Author: Thien Phuc Tran
'''
import string
from numpy import array

def load_doc(filename):
    file = open(filename, mode='rt', encoding='utf-8')
    text = file.read()
    file.close()
    return text


def to_pairs(doc):
    lines = doc.strip().split('\n')
    pairs = [line.split('\t') for line in lines]
    return pairs


def clean_pairs(lines):
    cleaned = list()
    # re_print = re.compile('[^%s]' % re.escape(string.printable))
    table = str.maketrans('', '', string.punctuation)
    for pair in lines:
        clean_pair = list()
        for line in pair:
            # line = normalize('NFD', line).encode('ascii', 'ignore')
            # line = line.decode('UTF-8')
            line = line.split()
            line = [word.lower() for word in line]
            line = [word.translate(table) for word in line]
            # line = [re_print.sub('', w) for w in line]
            line = [word for word in line if word.isalpha()]
            clean_pair.append(' '.join(line))
        cleaned.append(clean_pair)
    return array(cleaned)


def load_dataset(filename):
    doc = load_doc(filename)
    pairs = to_pairs(doc)
    c_pairs = clean_pairs(pairs)
    return c_pairs[:, 0], c_pairs[:, 1]


def pad_sentence(sent, length, padelem):
    padded = sent + [padelem] * (length - len(sent))
    return padded
