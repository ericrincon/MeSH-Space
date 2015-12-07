__author__ = 'ericrincon'

import ANNMesh
import zipfile
import os
import keras
import sys
import getopt

import matplotlib.pyplot as plt

from lxml import etree
import numpy
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from keras.callbacks import ModelCheckpoint
from sys import getsizeof

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



def main():
    #Get file names
    all_mesh_terms_path = ''
    abstracts_path = ''
    model_name = ''
    epochs = 100
    n_hidden_units = 300

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'm:d:n:e:', ['mesh_terms=', 'abstract_data',
                                                           'model_name', 'epochs', 'hidden_units'])
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
        else:
            sys.exit(2)

    print('..loading data')

    feature_matrix, mesh_list = transform_data(abstracts_path, False, limit=5000000)
    target_dict = parse_mesh(all_mesh_terms_path, False)

    target_matrix = numpy.zeros((feature_matrix.shape[0], len(target_dict)))

    for i, mesh_descriptors in enumerate(mesh_list):
        mesh_descriptor_indices = []
        target_vector = numpy.zeros((len(target_dict)))

        for mesh_descriptor in mesh_descriptors:
            mesh_descriptor = preprocess_mesh_term(mesh_descriptor)

            if mesh_descriptor in target_dict:

                #Random terms not in dict. Why is this?
                mesh_descriptor_indices.append(target_dict[mesh_descriptor])


        target_vector[mesh_descriptor_indices] = 1
        target_matrix[i, :] = target_vector

    feature_matrix = feature_matrix.todense()

    model = ANNMesh.create_model(feature_matrix.shape[1], n_hidden_units, target_matrix.shape[1])
    print('..training')

    checkpointer = ModelCheckpoint(filepath=model_name, verbose=1, save_best_only=True)
    loss_history = LossHistory()
    model.fit(feature_matrix, target_matrix, validation_split=.1,
              show_accuracy=True, callbacks=[checkpointer, loss_history], nb_epoch=epochs, batch_size=128)

    print('..saving model')
    model.save_weights("final_" + model_name)

    epochs_axis = numpy.arange(1, epochs + 1)

    train_loss, = plt.plot(epochs_axis, loss_history.losses, label='Train')
    val_loss, = plt.plot(epochs_axis, loss_history.val_losses, label='Validation')
    plt.legend(handles=[train_loss, val_loss])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    plt.savefig(model_name + '_loss_plot.png')

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
if __name__ == '__main__':
    main()