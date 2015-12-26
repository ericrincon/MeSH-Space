import getopt
import sys

def main():
    folder_path = ''

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'i:', ['input_folder='])
    except getopt.GetoptError as e:
        print(e)
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-i', '--input_folder'):
            folder_path = arg
        else:
            sys.exit(2)

if __name__ == '__main__':
    main()