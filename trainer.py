import sys
from mirex_dataset import trainer

def main(argv):
	trainer.train_songs(parse_arguments(argv))


if __name__ == '__main__':
	main(sys.argv[1:])