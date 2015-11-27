__author__ = 'ericrincon'

import ANNMesh
import gzip
import zipfile
import os
from lxml import etree
import numpy
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer



def main():
    print('..loading data')

    feature_matrix, mesh_list = transform_data('', False)
    target_dict = parse_mesh('', False)

    target_matrix = numpy.zeros((feature_matrix.shape[0], len(mesh_list)))

    for i, mesh_descriptors in enumerate(mesh_list):
        mesh_descriptor_indices = []
        target_vector = numpy.zeros((len(mesh_list)))

        for mesh_descriptor in mesh_descriptors:
            mesh_descriptor = mesh_descriptor.strip().lower()

            if mesh_descriptor in target_dict:
                #Random terms not in dict. Why is this?
                mesh_descriptor_indices.append(target_dict[mesh_descriptor])

        target_vector[mesh_descriptor_indices] = 1
        target_matrix[i, :] = target_vector

    feature_matrix = feature_matrix.todense()

    model = ANNMesh.create_model(feature_matrix.shape[1], 300, len(mesh_list))
    print('..training')
    model.fit(feature_matrix.todense(), target_matrix, validation_split=.1, show_accuracy=True)

    print('..saving model')
    model.save_weights()

#total docs 61898
def transform_data(path, save):
    corpus = []
    mesh_list = []
    with open(path) as file:
        for line in file:
            title, abstract, mesh = line.split('||')

            if not abstract.strip() == 'Abstract available from the publisher.':
                text = title + abstract
                corpus.append(text)
                mesh_list.append(mesh.split('|'))
    vectorizer  = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50000)
    feature_matrix = vectorizer.fit_transform(corpus)

    if save:
        numpy.save('tfidf_title_abstract', feature_matrix)
        mesh_list = numpy.array(mesh_list)
        numpy.save(mesh_list, 'mesh_list')

    return feature_matrix, mesh_list


def parse_mesh(xml_path, save):
    descriptor_count = 0
    mesh_descriptors = {}

    for event, element in etree.iterparse(xml_path):
        if element.tag == 'String':
            parent = element.getparent()
            if parent.tag == 'DescriptorName':
                if element.text not in mesh_descriptors:
                    mesh_descriptors.update({element.text.strip().lower(): descriptor_count})
                    descriptor_count+=1
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