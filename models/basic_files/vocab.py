# Vocab() class is defined here. 
import os.path
import operator
import pickle
from nltk.tokenize import WhitespaceTokenizer 
from gensim.models import Word2Vec, KeyedVectors
from collections import defaultdict
from math import sqrt
import numpy as np 
import matplotlib.pyplot as plt 

class Vocab():

    def __init__(self):
        """ Initalize the class parameters to default values
        """

        self.word_to_index = {}
        self.index_to_word = {}
        self.unknown       = "<unk>"
        self.end_of_sym    = "<eos>"
        self.start_sym     = "<s>"
        self.padding       = "<pad>"
        self.word_freq     = {}
        self.len_vocab     = 0
        self.total_words   = 0
        self.embeddings    = None


    def get_global_embeddings(self, filenames, embedding_size, embedding_dir):
        """ Construct the Embedding Matrix for the sentences in filenames.

            Args:
                filenames: File names of the training files: Based on 
                which the vocab will be built. This is used when there
                are no pretrained embeddings present. Then instead of 
                using random embeddings, Word2Vec algorithm is used 
		to train the embeddings on the dataset avaliable.
                embedding_size: Dimensions for the embedding to be used.

            Returns
                Embedding matrix.
        """
        sentences = []

        if (os.path.exists(embedding_dir + 'vocab_len.pkl')):
                vocab_len_stored = pickle.load(open(embedding_dir + "vocab_len.pkl"))
        else:
                vocab_len_stored = 0

        if (vocab_len_stored == self.len_vocab and os.path.exists(embedding_dir + "embeddings.pkl")):
                print ("Load file")
                self.embeddings = pickle.load(open(embedding_dir +  "embeddings.pkl"))
                return None

        if (os.path.exists(embedding_dir + 'embeddings') == True):
            model = KeyedVectors.load_word2vec_format(embedding_dir + 'embeddings', binary = False)
            print ("Loading pretriained embeddings")

        else:
            for file in filenames:
                with open(file, 'rb') as f:
                    for lines in f:
                        words = [lines.split()]
                        sentences.extend(words)

            model = Word2Vec(sentences, size=embedding_size, min_count=0)
            model.save(embedding_dir + 'embeddings')

        self.embeddings_model = model
        return model

    def add_constant_tokens(self):
        """ Adds the tokens <pad> and <unk> to the vocabulary.
        """

        self.word_to_index_encode[self.padding]    = 0
        self.word_to_index_encode[self.unknown]    = 1


        self.word_to_index_decode[self.padding]    = 0
        self.word_to_index_decode[self.unknown]    = 1


    def add_word(self, word, word_to_index, word_freq):
        """ Adds the word to the dictionary encode if not already present. 

        Arguments:
             word : Word to be added.
             word_to_index: The dictionary to which the words
                            needs to be added
             word_freq: The word frequency dictionary

        Returns:
            * void
        """

        if word in word_to_index:
            word_freq[word] = word_freq[word] + 1

        else:
            index = len(word_to_index)
            word_to_index[word] = index
            word_freq[word] = 1

        return word_to_index, word_freq

    def create_reverse_dictionary(self):
        """ Creates a mapping from index to the words
            This will be helpful in decoding the predictions
            to sentences.

            This function creates the reverse dictionary for 
            both encode and decode dictionary
        """

        for key, val in self.word_to_index_encode.iteritems():
            self.index_to_word_encode[val] = key

        for key, val in self.word_to_index_decode.iteritems():
            self.index_to_word_decode[val] = key

    def fix_the_frequency(self, limit_encoder=0, limit_decoder=0):
        """ Eliminates the words from the dictionary with 
	    a frequency less than the limit provided in the 
	    argument.

	    Arguments:
            * limit_encoder: The threshold frequency for encoder
            * limit_decoder: The threshold frequency for decoder

            Returns:
            * void
        """

        temp_word_to_index_encoder = {}
        temp_index_to_word_encoder = {}

        temp_word_to_index_decoder = {}
        temp_index_to_word_decoder = {}

        word_list_encoder = []
        count_encoder = 0

        word_list_decoder = []
        count_decoder = 0
        #
        # Start from index 2 so that the constant tokens:
        # <pad> and <unk> are not eliminated
        #
        new_index_encoder = 2
        new_index_decoder = 2

        for key in self.word_to_index_encode:
            if (self.word_freq_encode[key] > limit_encoder):
                temp_word_to_index_encoder[key] = new_index_encoder
                temp_index_to_word_encoder[new_index_encoder] = key
                new_index_encoder += 1


        for key in self.word_to_index_decode:
            if (self.word_freq_decode[key] > limit_decoder):
                temp_word_to_index_decoder[key] = new_index_decoder
                temp_index_to_word_decoder[new_index_decoder] = key
                new_index_decoder += 1

        self.word_to_index_encode = temp_word_to_index_encoder
        self.word_to_index_decode = temp_word_to_index_decoder



    def construct_dictionary_single_file(self, filename, word_to_index, word_freq):
        """ Adds the words belonging to this file to the
            dictionary

            Arguments:
                * filename: The respective file from which words
                  needs to be added.
            Returns:
                * void
        """
        with open(filename, 'rb') as f:
            for lines in f:
                for words in lines.split():
                    word_to_index, word_freq = self.add_word(words, word_to_index, word_freq)

        return word_to_index, word_freq

    def construct_dictionary_multiple_files(self, filenames, word_to_index, word_freq):
        """ Dictionary is made from the words belonging to all
            the files in the set filenames

            Arguments :
                * filenames = List of the filenames 

            Returns :
                * None
        """

        for files in filenames:
            word_to_index, word_freq = self.construct_dictionary_single_file(files, word_to_index, word_freq)


        return word_to_index, word_freq

    def encode_word_encoder(self, word):
        """ Convert the word to the particular index
            based on the encoder dictionary

            Arguments :
                * word: Given word is converted to index.
    
            Returns:
                * index of the word        
        """
        if word not in self.word_to_index_encode:
            word = self.unknown
        return self.word_to_index_encode[word]


    def encode_word_decoder(self, word):
        """ Convert the word to the particular index
            based on the decoder dictionary

            Arguments :
                * word: Given word is converted to index.
    
            Returns:
                * index of the word        
        """

        if word not in self.word_to_index_decode:
            word = self.unknown
        return self.word_to_index_decode[word]


    def decode_word_encoder(self, index):
        """ Index is converted to its corresponding word
            based on the encoder dictionary

            Argument:
                * index: The index to be encoded.

            Returns:
                * returns the corresponding word
        """
        if index not in self.index_to_word_decode:
            return self.unknown
        return self.index_to_word_decode[index]


    def decode_word_decoder(self, index):
        """ Index is converted to its corresponding word
            based on the decoder dictionary

            Argument:
                * index: The index to be encoded.

            Returns:
                * returns the corresponding word
        """
        if index not in self.index_to_word_decode:
            return self.unknown
        return self.index_to_word_decode[index]


    def get_embeddings(self, embedding_size, index_to_word, embedding_dir):
        """ This function creates an embedding matrix
            of size (vocab_size * embedding_size). The embedding 
	    for each word is loaded from the embeddings learnt in the 
            function get_global_embeddings(). 

            Arguments:
		* embedding_size: Dimension size to represent the word.

            Returns:
		* embeddings: Returns the embeddings for the index_to_word
                      dictionary
        """

        sorted_list = sorted(index_to_word.items(), key = operator.itemgetter(0))
        embeddings = []

        np.random.seed(1357)



        if (os.path.exists(embedding_dir + 'vocab_len.pkl')):
                vocab_len_stored = pickle.load(open(embedding_dir + "vocab_len.pkl"))
        else:
                vocab_len_stored = 0

        if vocab_len_stored == self.len_vocab and os.path.exists(embedding_dir + "embeddings.pkl"):
                self.embeddings = pickle.load(open(embedding_dir + "embeddings.pkl"))
		#print (self.embeddings)
                return

        for index, word in sorted_list:

            if word in self.embeddings_model:
                embeddings.append(self.embeddings_model[word])
            else:
                if word in ['<pad>', '<s>', '<eos>']:
                    temp = np.zeros((embedding_size))
                else:
                    temp = np.random.uniform(-sqrt(3)/sqrt(embedding_size), sqrt(3)/sqrt(embedding_size),
                                             (embedding_size))
                embeddings.append(temp)

        embeddings = np.asarray(embeddings)
        embeddings = embeddings.astype(np.float32)

        pickle.dump(embeddings, open(embedding_dir + "embeddings.pkl", "w"))
        pickle.dump(self.len_vocab, open(embedding_dir + "vocab_len.pkl", "w"))


        return embeddings

    def construct_vocab(self, filenames_encoder, filenames_decoder,
                        embedding_size, embedding_path, limit_encoder, limit_decoder):
        """ Constructs the vocab, and initializes the embeddings 
            accordingly 

            Args:
                * filenames_encoder: List of filenames required to
                                     generate the encoder vocab
                * filenames_decoder: List of filenames required to
                                     generate the decoder vocab
                * embeddings: Dimension size for the word representation
                * embedding_path: Path for the pretrained embedding model

            Returns:
                * void
        """

        self.index_to_word_decode = {}
        self.index_to_word_encode = {}
        self.word_to_index_encode = {}
        self.word_to_index_decode = {}
        self.word_freq_encode     = {}
        self.word_freq_decode     = {}

        self.get_global_embeddings(filenames_encoder + filenames_decoder, \
                                   embedding_size, embedding_path)
        self.word_to_index_encode, self.word_freq_encode  = \
                self.construct_dictionary_multiple_files(filenames_encoder,
                                                        self.word_to_index_encode,
                                                        self.word_freq_encode)
        self.word_to_index_decode, self.word_freq_decode = \
                self.construct_dictionary_multiple_files(filenames_decoder,
                                                        self.word_to_index_decode,
                                                        self.word_freq_decode)
        
        self.fix_the_frequency(limit_encoder, limit_decoder)
        self.add_constant_tokens()
        self.create_reverse_dictionary()

        self.embeddings_encoder = self.get_embeddings(embedding_size, self.index_to_word_encode, embedding_path)
        self.embeddings_decoder = self.get_embeddings(embedding_size, self.index_to_word_decode, embedding_path)

	#print(self.word_to_index_decode)
        self.len_vocab_encode = len(self.word_to_index_encode)
        self.len_vocab_decode = len(self.word_to_index_decode)

        print ("Number of words in the encoder vocabulary is " + str(len(self.word_to_index_encode)))
        print ("Number of words in the decoder vocabulary is " + str(len(self.word_to_index_decode)))


        self.total_words_encode = float(sum(self.word_freq_encode.values()))
        self.total_words_decode = float(sum(self.word_freq_decode.values()))
