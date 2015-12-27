__author__ = 'ericrincon'

import matplotlib

# So that matplotlib can plot the training error and validation error of network.
matplotlib.use('Agg')

import ANNMesh
import zipfile
import os
import keras
import sys
import getopt
import h5py
import pickle

import matplotlib.pyplot as plt

from random import sample
from lxml import etree
import numpy

from sklearn.externals import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from keras.callbacks import ModelCheckpoint

#Number of abstracts in dataset
total_documents = 21850751

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.batch_losses = []
        self.losses = []
        self.val_losses = []

    def on_batch_end(self, batch, logs={}):
        self.batch_losses.append(logs.get('loss'))


    def on_epoch_end(self, epoch, logs={}):
        loss_sum = 0

        for mini_batch_loss in self.batch_losses:
            loss_sum += mini_batch_loss
        average_loss = loss_sum/len(self.batch_losses)

        self.losses.append(average_loss)
        self.val_losses.append(logs.get('val_loss'))
        self.batch_losses.clear()

class BatchGenerator:
    def __init__(self, n_examples, source, n_per_batch, target_dict, X_train, Y_train):
        self.examples = n_examples
        self.file_source = source
        self.n_batch = n_per_batch
        self.target_dict = target_dict
        self.X_train = X_train
        self.Y_train = Y_train

    def __iter__(self):
        for n in range(int(self.examples/self.n_batch)):
            yield self.X_train['data'][n * self.n_batch: (n + 1)* self.n_batch, :],\
                  self.Y_train['data'][n * self.n_batch: (n + 1)* self.n_batch, :]


def train(epochs, model, n_examples, file_path, checkpointer, loss_history, n_per_batch, target_dict, mini_batch,
          X_train, Y_train):
    batch_generator = BatchGenerator(n_examples, file_path, n_per_batch, target_dict, X_train, Y_train)
    train_loss = []
    valid_loss = []


    for epoch in range(epochs):
        print('epoch: ', epoch + 1)

        for X_train, Y_train in batch_generator:
            model.fit(X_train, Y_train, validation_split=.1, show_accuracy=True, callbacks=[checkpointer, loss_history],
                      nb_epoch=1, batch_size=mini_batch)

        train_loss += loss_history.losses
        valid_loss += loss_history.val_losses

    return model, train_loss, valid_loss


def get_mesh_term_matrix(target_dict, mesh_list, n):
    target_matrix = numpy.zeros((n, len(target_dict)))

    for i, mesh_descriptors in enumerate(mesh_list):
        mesh_descriptor_indices = []
        target_vector = numpy.zeros((len(target_dict)))

        for mesh_descriptor in mesh_descriptors:
            mesh_descriptor = preprocess_mesh_term(mesh_descriptor)

            if mesh_descriptor in target_dict:
                mesh_descriptor_indices.append(target_dict[mesh_descriptor])
        target_vector[mesh_descriptor_indices] = 1
        target_matrix[i, :] = target_vector

    return target_matrix


def run_preprocess(all_mesh_terms_path, sample_size, full_path, abstracts_path):
    target_dict = parse_mesh(all_mesh_terms_path, False)
    process_save_data(sample_size, target_dict, full_path, abstracts_path, True)


def run_train_model(all_mesh_terms_path, abstracts_path, model_name, epochs, n_hidden_units, sample_size,
                    batch_size, mini_batch_size, full_path):

    print('..loading data')
    target_dict = parse_mesh(all_mesh_terms_path, False)
    X_train, Y_train = process_save_data(sample_size, target_dict, full_path, abstracts_path)

    model = ANNMesh.create_model(X_train['data'].shape[1], n_hidden_units, len(target_dict))

    if os.path.exists(model_name):
        model.load_weights(model_name)

    print('..training')

    checkpointer = ModelCheckpoint(filepath=model_name, verbose=1, save_best_only=True)
    loss_history = LossHistory()

    model, train_loss_history, valid_loss_history = train(epochs, model, sample_size, abstracts_path, checkpointer,
                                                     loss_history, batch_size, target_dict, mini_batch_size, X_train,
                                                     Y_train)
    print(len(train_loss_history), len(valid_loss_history))
    print('..saving model')
    model.save_weights("final_" + model_name)

    #Plot the loss curves
    epochs_axis = numpy.arange(1, epochs + 1)
    train_loss, = plt.plot(epochs_axis, train_loss_history, label='Train')
    val_loss, = plt.plot(epochs_axis, valid_loss_history, label='Validation')
    plt.legend(handles=[train_loss, val_loss])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    plt.savefig(model_name + '_loss_plot.png')


def main():
    #Get args for running
    all_mesh_terms_path = ''
    abstracts_path = ''
    model_name = ''
    epochs = 100
    n_hidden_units = 300
    sample_size = 50000
    preprocess = False
    #Batch size is not mini batch size but the size of examples too load at once
    batch_size = 5000

    mini_batch_size = 64

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'm:d:n:e:h:s:b:mb:p:v:', ['mesh_terms=', 'abstract_data','model_name',
                                                                           'epochs', 'hidden_units', 'sample_size',
                                                                           'batch_size', 'preprocess', 'verbose=',
                                                                           'hidden_layers', 'headless_plot=',])
    except getopt.GetoptError:
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-m', '--mesh_terms'):
            all_mesh_terms_path = arg
        elif opt in ('-d', '--abstract_data'):
            abstracts_path = arg
        elif opt in ('-n', '--model_name'):
            model_name = arg
        elif opt in ('-e', '--epochs'):
            epochs = int(arg)
        elif opt in ('-h', 'hidden_units'):
            n_hidden_units = int(arg)
        elif opt in ('-s', 'sample_size'):
            sample_size = int(arg)
        elif opt in ('-b', 'batch_size'):
            batch_size = int(arg)
        elif opt in ('-mb', 'mini_batch_size'):
            mini_batch_size = int(arg)
        elif opt in ('-p', 'prepreprocess'):
            option = int(arg)

            if option == 1:
                preprocess = True

        else:
            sys.exit(2)

    split_path = abstracts_path.split('/')
    split_path.pop()
    full_path = ''

    if not len(split_path) == 0:
        full_path = split_path.pop(0) + '/'
        for p in split_path:
            full_path += '/' + p
        full_path += '/'

    if preprocess:
        run_preprocess(all_mesh_terms_path, sample_size, full_path, abstracts_path)
    else:
        run_train_model(all_mesh_terms_path, abstracts_path, model_name, epochs, n_hidden_units, sample_size,
                        batch_size, mini_batch_size, full_path)




def transform_data(path, save, limit=None):
    corpus = []
    mesh_list = []

    with open(path) as file:
        for i, line in enumerate(file):
            if limit:
                if i == limit:
                    break

            split_line = line.split('||')

            if len(split_line) == 3:
                title, abstract, mesh = split_line
            else:
                continue

            if not abstract.strip() == 'Abstract available from the publisher.':
                text = title + abstract
                corpus.append(text)
                mesh_list.append(mesh.split('|'))

    vectorizer = TfidfVectorizer(min_df=3, max_features=50000)

    feature_matrix = vectorizer.fit_transform(corpus)

    if save:
        numpy.save('tfidf_title_abstract', feature_matrix)
        mesh_list = numpy.array(mesh_list)
        numpy.save(mesh_list, 'mesh_list')

    return feature_matrix, mesh_list


def preprocess_mesh_term(mesh_term):
    in_table = '\n'
    out_table = ' '
    translation_table = str.maketrans(in_table, out_table)
    mesh_term = mesh_term.strip()
    mesh_term = mesh_term.translate(translation_table)

    return mesh_term


def parse_mesh(xml_path, save):

    descriptor_count = 0
    mesh_descriptors = {}

    for event, element in etree.iterparse(xml_path):
        if element.tag == 'String':
            parent = element.getparent()
            if parent.tag == 'DescriptorName':
                mesh_term = preprocess_mesh_term(element.text)
                if element.text not in mesh_descriptors:
                    mesh_descriptors.update({mesh_term: descriptor_count})
                    descriptor_count += 1
        element.clear()
    if save:
        output = open('mesh_descriptors.txt', 'ab+')
        pickle.dump(mesh_descriptors, output)

    return mesh_descriptors


"""
    Helper function that parses multiple zipped XMl files.
    The medline abstract database comes in multiple zipped files with one XML file in each.
    There is also one giant ~120GB XML file with all the same abstracts but it was harder to parse
    that one XML file since it contains multiple XML structures inside of it. Thus, it was easier to
    parse separately.

    Returns the name of the output file
"""
def parse_multiple_files(xml_path, output_file_name='output.txt', save=False):
    if save:
        file = open(output_file_name, 'w')
    files = [os.path.join(xml_path,o) for o in os.listdir(xml_path) if os.path.isfile(os.path.join(xml_path,o))]
    files.pop(0)

    counts = {'ArticleTitle': 0, 'AbstractText': 0, 'MeshHeading': 0, 'Total': 0}

    for zip in files:
        archive = zipfile.ZipFile(zip, 'r')
        file_name_parts = zip.split('.')
        file_name = file_name_parts[0] + '.' + file_name_parts[1]
        file_name = '../zip/' + file_name.split('/')[-1]
        xml = archive.open(file_name)

        for out_event, out_element in etree.iterparse(xml):
            if out_event == 'end' and out_element.tag == 'MedlineCitationSet':
                for element in out_element.iterchildren():
                    if element.tag == 'MedlineCitation':
                        abstract = None
                        title = None
                        mesh_terms = None

                        for c in element.iterchildren():
                            if c.tag == 'Article':
                                for ele in c.iterchildren():
                                    if ele.tag == 'ArticleTitle':
                                        if ele.text is not None:
                                            title = ele.text
                                            count = counts['ArticleTitle'] + 1
                                            counts['ArticleTitle'] = count
                                    if ele.tag == 'Abstract':
                                        for abstract_ele in ele.iterchildren():
                                            if abstract_ele.tag == 'AbstractText':
                                                abstract = abstract_ele.text
                                                count = counts["AbstractText"] + 1
                                                counts["AbstractText"] = count

                            if c.tag == 'OtherAbstract' and abstract is not None:
                                for abstract_ele in c.iterchildren():
                                    if abstract_ele.tag == "AbstractText":
                                        if abstract_ele.text is not None:
                                            abstract = abstract_ele.text
                                            count = counts["AbstractText"] + 1
                                            counts["AbstractText"] = count
                            if c.tag == 'MeshHeadingList':
                                mesh_terms = ''
                                count = counts['MeshHeading'] + 1
                                counts['MeshHeading'] = count
                                for i, ele in enumerate(c.iterchildren()):
                                    if ele.tag == 'MeshHeading':
                                        for heading_elem in ele.iterchildren():
                                            if heading_elem.text is not None:

                                                if i == 0:
                                                    mesh_terms = heading_elem.text
                                                else:
                                                    mesh_terms = mesh_terms + ' | ' + heading_elem.text
                                if mesh_terms == '':
                                    mesh_terms = None
                        if (abstract and title and mesh_terms) is not None:
                            if save:
                                file.write(title + ' || ' + abstract + ' || ' + mesh_terms + '\n')

                    element.clear()
                    count = counts['Total'] + 1
                    counts['Total'] = count
    print(counts)


def process_save_data(limit, target_dict, path='', abstract_path='', pre=False):
    abstracts = []
    mesh_list = []

    # sample some extra numbers just in case we dont get to limit
    lines_to_sample = sample(range(total_documents), limit * 2)

    X_file_name = path + 'X_tfidf_abstracts.h5py'
    Y_file_name = path + 'Y_mesh_terms.h5py'

    if not os.path.exists(X_file_name):
        n_features = 50000
        vectorizer = TfidfVectorizer(min_df=3, max_features=n_features)

        X_train = h5py.File(X_file_name, 'w')
        Y_train = h5py.File(Y_file_name, 'w')

        assert not abstract_path == '', 'Need a path for data!'

        i = 0

        with open(abstract_path) as file:
            for n_line, line in enumerate(file):
                if i == limit:
                    break

                if n_line not in lines_to_sample:
                    continue

                split_line = line.split('||')

                if len(split_line) == 3:
                    title, abstract, mesh = split_line
                else:
                    continue

                if not abstract.strip() == 'Abstract available from the publisher.':
                    text = title + abstract
                    abstracts.append(text)
                    mesh_list.append(mesh.split('|'))
                    i += 1

        #Create h5py dataset for both X and Y

        x = vectorizer.fit_transform(abstracts).todense()
        y = get_mesh_term_matrix(target_dict, mesh_list, len(mesh_list))

        joblib.dump(vectorizer, 'vectorizer.pkl')

        X_train.create_dataset('data', (limit, x.shape[1]), dtype=numpy.float32, data=x)
        Y_train.create_dataset('data', (limit, len(target_dict)), dtype=numpy.float32, data=y)

        if not pre:
            return X_train, Y_train
    else:
        X_train = h5py.File(X_file_name, 'r')
        Y_train = h5py.File(Y_file_name, 'r')

        return X_train, Y_train

if __name__ == '__main__':
    main()