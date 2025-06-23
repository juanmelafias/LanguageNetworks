import argparse
from common.utils import file_generator

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Generate network files')
    parser.add_argument('--routesfile', type=str, default='csvroutes/rutascorpus.csv',
                        help='Path to the routes CSV file')
    parser.add_argument('--numlines', type=int, default=50000,
                        help='Number of lines to process')
    parser.add_argument('--lemmatized', action='store_true',
                        help='Whether to use lemmatized forms')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Call file_generator with the provided arguments
    file_generator(routesfile=args.routesfile, numlines=args.numlines, lemmatized=args.lemmatized)

if __name__ == "__main__":
    main()