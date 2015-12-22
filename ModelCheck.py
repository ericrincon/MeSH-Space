
import ANNMesh
import getopt
import sys
import ExperimentScript

def main():
    #Get args for running
    all_mesh_terms_path = ''
    abstracts_path = ''
    model_name = ''
    n_hidden_units = 300
    sample_size = 500000

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'm:d:n:e:h:s:b:mb:p:', ['mesh_terms=', 'abstract_data','model_name',
                                                                         'hidden_units', 'sample_size',])
    except getopt.GetoptError:
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-m', '--mesh_terms'):
            all_mesh_terms_path = arg
        elif opt in ('-d', '--abstract_data'):
            abstracts_path = arg
        elif opt in ('-n', '--model_name'):
            model_name = arg
        elif opt in ('-h', 'hidden_units'):
            n_hidden_units = int(arg)
        elif opt in ('-s', 'sample_size'):
            sample_size = int(arg)
        else:
            sys.exit(2)
    model = ANNMesh.create_model()


if __name__ == '__main__':
    main()
