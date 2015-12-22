__author__ = 'ericrincon'

import sys
import getopt
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer

def main():
    text_path = None
    sample_size = None
    feature_limit = 50000


    try:
        opts, args = getopt.getopt(sys.argv[1:], 'i:l:s:', ['text_path=', 'feature_limit=', 'sample_size='])
    except getopt.GetoptError:
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-i', '--text_path'):
            text_path = arg
        elif opt in ('-s', '--sample_size'):
            sample_size = int(arg)
        elif opt in ('-l', '--word_limit'):
            feature_limit = int(arg)
        else:
            sys.exit(2)

    assert text_path is not None, 'Provide a path to the text file!'

    tfidf_vectorizer = TfidfVectorizer(min_df=3, max_features=feature_limit)
    abstracts = []

    i = 0

    with open(text_path) as file:
            for line in file:
                if sample_size:
                    if i == sample_size:
                        break

                split_line = line.split('||')

                if len(split_line) == 3:
                    title, abstract, mesh = split_line
                else:
                    continue

                if not abstract.strip() == 'Abstract available from the publisher.':
                    text = title + abstract
                    abstracts.append(text)
                    i += 1
    tfidf_vectorizer.fit(abstracts)

    pickle.dump(tfidf_vectorizer, open('tfidf.p', 'wb'))


if __name__ == '__main__':
    main()