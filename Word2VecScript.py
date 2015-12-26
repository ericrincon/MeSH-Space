import getopt
import sys
import multiprocessing

from gensim.models import Word2Vec
from random import sample

from nltk.tokenize import sent_tokenize

total_documents = 21850751


class Sentences(object):
    def __init__(self, source):
        self.source = source

    def __iter__(self):
        with open(self.source, 'r') as file:
            for line in file:
                yield line.split()


def main():
    # Get args for running
    input_file = None
    output_file = None
    model_name = None
    vector_size = 200
    sample_size = None
    preprocess_text = False
    min_c = 3
    window_size = 10
    epochs = 20

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'i:o:m:n:s:p:c:w:e:', ['input_file=', 'output_file=', 'model_name=',
                                                                      'vector_size=', 'sample_size=', 'preprocess=',
                                                                      'min_count=', 'window_size=', 'epochs='])
    except getopt.GetoptError:
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-i', '--input_file'):
            input_file = arg
        elif opt in ('-o', '--output_file'):
            output_file = arg
        elif opt in ('-m', '--model_name'):
            model_name = arg
        elif opt in ('-n', '--vector_size'):
            vector_size = int(arg)
        elif opt in ('-s', '--sample_size'):
            sample_size = int(arg)
        elif opt in ('-p', '--preprocess_text'):
            option = int(arg)

            if option == 1:
                preprocess_text = True
        elif opt in ('-c', '--min_count'):
            min_c = int(arg)

        elif opt in ('-w', '--window_size'):
            window_size = int(arg)
        elif opt in ('-e', '--epochs'):
            epochs = int(arg)
        else:
            sys.exit(2)

    if preprocess_text:
        preprocess(input_file, output_file, sample_size)
    else:
        cores = multiprocessing.cpu_count()

        sentences = Sentences(input_file)
        print('...building vocabulary')
        w2v_model = Word2Vec(sentences, min_count=min_c, size=vector_size, workers=cores, window=window_size,
                             iter=epochs)

        print('...training')
        w2v_model.train(sentences)
        w2v_model.save(model_name)

def preprocess(text_path, output_file, limit):
    file = open(text_path, 'r')

    if limit:
        lines_to_sample = sample(range(total_documents), limit * 2)
    output_file = open(output_file, 'w+')

    i = 0

    for n_line, line in enumerate(file):
        if limit:
            if i == limit:
                break
        if limit:
            if n_line not in lines_to_sample:
                continue

        split_line = line.split('||')

        if len(split_line) == 3:
            title, abstract, mesh = split_line
        else:
            continue

        if not abstract.strip() == 'Abstract available from the publisher.':
            abstract_sentences = sent_tokenize(abstract)

            for sentence in abstract_sentences:
                punctuation = "`~!@#$%^&*()_-=+[]{}\|;:'\"|<>,./?åαβ"
                numbers = "1234567890"
                number_replacement = create_whitespace(len(numbers))
                spacing = create_whitespace(len(punctuation))

                lowercase_line = sentence.lower()
                translation_table = str.maketrans(punctuation, spacing)
                translated_line = lowercase_line.translate(translation_table)
                translation_table_numbers = str.maketrans(numbers, number_replacement)
                final_translation = translated_line.translate(translation_table_numbers)

                output_file.write(final_translation + '\n')


def create_whitespace(length):
    whitespace = ''

    for i in range(length):
        whitespace += ' '

    return whitespace

if __name__ == '__main__':
    main()
