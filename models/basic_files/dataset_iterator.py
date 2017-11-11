# Dataset iterator file.
import random
import nltk
import numpy as np
import pickle
import sys
import os.path
import tensorflow as tf
import copy
from vocab import *

class Datatype:

    def __init__(self, name, title, label, content, query, num_samples, content_sequence_length, \
                 query_sequence_length,\
                 max_length_content, max_length_title, max_length_query):
        """ Defines the dataset for each category valid/train/test

        Args:
            name   : Name given to this partition. For e.g. train/valid/test
            title  : The summaries that needs to be generated.
            content: The input/source documents 
            query  : The queries given based on which the document needs to be summarized

            num_samples        :  Number of samples in this partition
            max_length_content :  Maximum length of source document across all samples
            max_length_title   :  Maximum length of summary across all samples
            
            global_count_train : pointer to retrieve the next batch during training
            global_count_test  : pointer to retrieve the next batch during testing
        """

        self.name    = name
        self.title   = title
        self.content = content
        self.labels  = label
        self.query   = query

        self.content_sequence_length = content_sequence_length
        self.query_sequence_length   = query_sequence_length

        self.number_of_samples  = num_samples
        self.max_length_content = max_length_content
        self.max_length_title   = max_length_title - 1
        self.max_length_query   = max_length_query

        self.global_count_train = 0
        self.global_count_test  = 0


class PadDataset:

    def pad_data(self, data, max_length):
        """ Pad the batch to max_length given.

            Arguments: 
                data       : Batch that needs to be padded
                max_length : Max_length to which the samples needs to be
                             padded.

            Returns:
                padded_data : Each sample in the batch is padded to 
                              make it of length max_length.
        """

        padded_data = []
        sequence_length_batch = []
        for lines in data:
            if (len(lines) < max_length):
                temp = np.lib.pad(lines, (0,max_length - len(lines)),
                    'constant', constant_values=0)

                sequence_length_batch.append(len(lines))
            else:
                temp = lines[:max_length]
                sequence_length_batch.append(max_length)

            padded_data.append(temp)

        return padded_data, sequence_length_batch


    def make_batch(self, data, batch_size, count, max_length):
        """ Make a matrix of size [batch_size * max_length]
            for given dataset

            Args:
                data: Make batch from this dataset
                batch_size : batch size
                count : pointer from where retrieval will be done
                max_length : maximum length to be padded into

            Returns
                batch: A matrix of size [batch_size * max_length]
                count: The point from where the next retrieval is done.
        """

        batch = []
        batch = data[count:count+batch_size]
        count = count + batch_size


        while (len(batch) < batch_size):
            batch.append(np.zeros(max_length, dtype = int))
            count = 0
            
        batch, sequence_length_batch = self.pad_data(batch,max_length)

        batch = np.transpose(batch)
        return batch, count, sequence_length_batch

    def make_batch_sequence(self, data, batch_size, count, max_length):
        """ Make a matrix of size [batch_size * max_length]
            for given dataset

            Args:
                data: Make batch from this dataset
                batch_size : batch size
                count : pointer from where retrieval will be done
                max_length : maximum length to be padded into

            Returns
                batch: A matrix of size [batch_size * max_length]
                count: The point from where the next retrieval is done.
        """

        batch = []
        batch = data[count:count+batch_size]
        count = count + batch_size


        while (len(batch) < batch_size):
            batch.append(np.zeros(max_length, dtype = int))
            count = 0

        return batch, count

    def next_batch(self, dt, batch_size, c=True):
        """ Creates a batch given the batch_size from
            mentioned dataset iterator.

            Arguments:
              * dt: Datatset from which the batch needs to
                    retreived
              * batch_size: Number of samples to keep in a batch

            Returns:
              * batch: Returns the batch created
        """

        if (c is True):
            count = dt.global_count_train
        
        else:
            count = dt.global_count_test


        max_length_content = max(val.max_length_content for i, val in self.datasets.iteritems())
        max_length_title   = max(val.max_length_title   for i, val in self.datasets.iteritems())
        max_length_query   = max(val.max_length_query   for i, val in self.datasets.iteritems())

        contents, count1, content_sequence_length = self.make_batch(dt.content, batch_size, count, max_length_content)
        titles,   _ ,_     = self.make_batch(dt.title,   batch_size, count,   max_length_title)
        labels,   _ ,_     = self.make_batch(dt.labels,  batch_size, count,   max_length_title)
        query,    _ , query_sequence_length     = self.make_batch(dt.query,  batch_size, count,   max_length_query)

        # Weights for the loss function for the decoder
        weights = copy.deepcopy(titles)


        # Fill the weighs matrix, based on the label parameters.
        for i in range(titles.shape[0]):
            for j in range(titles.shape[1]):
                if (weights[i][j] > 0):
                        weights[i][j] = 1
                else:
                        weights[i][j] = 0

        if (c == True): 
            dt.global_count_train = count1 % dt.number_of_samples
        else:
            dt.global_count_test  = count1 % dt.number_of_samples
    

	#print (titles)
        return contents, titles, labels, query, weights, content_sequence_length, query_sequence_length, max_length_content, max_length_title, max_length_query
   
 
    def load_data_file(self,name, title_file, content_file, query_file):
        """ Each of the (train/test/valid) is loaded separately.

        Arguments:
        * title_file   : The file containing the summaries
                * content_file : The file containing the source documents
                * query_file   : The file containing the queries


           Returns:
           * A Datatype object that contains relevant information to 
                 create batches from the given dataset
 
        """

        title   = open(title_file,'rb')
        content = open(content_file,'rb')
        query   = open(query_file, 'r')

        title_encoded   = []
        content_encoded = []
        label_encoded   = []
        query_encoded   = []

        content_sequence_length  = []
        query_sequence_length    = []
        
        max_title = 0
        for lines in title:
            temp = [self.vocab.encode_word_decoder(word) for word in lines.split()]

            if (len(temp) > max_title):
                max_title = len(temp)

            title_encoded.append(temp[:-1])
            label_encoded.append(temp[1:])

        max_content = 0

        for lines in content:
            temp = [self.vocab.encode_word_encoder(word) for word in lines.split()]

            if (len(temp) > max_content):
                max_content = len(temp)

            content_encoded.append(temp)
            content_sequence_length.append(len(temp))

        max_query = 0
        for lines in query:
            temp = [self.vocab.encode_word_encoder(word) for word in lines.split()]

            if (len(temp) > max_query):
                max_query = len(temp)

            query_encoded.append(temp)
            query_sequence_length.append(len(temp))

        return Datatype(name, title_encoded, label_encoded, content_encoded,
                        query_encoded, len(title_encoded), content_sequence_length, query_sequence_length,
                        max_content, max_title, max_query)


    def load_data(self, wd="../Data/"):
        """ Load all the datasets

            Arguments:
        * wd: Directory where all the data files are stored

            Returns:
            * void
        """
        s = wd
        self.datasets = {}

        for i in ("train", "valid", "test"):
            temp_t = s + i + "_summary"
            temp_v = s + i + "_content"
            temp_q = s + i + "_query"
            self.datasets[i] = self.load_data_file(i, temp_t, temp_v, temp_q)


    def __init__(self,  working_dir = "../Data/", embedding_size=100, global_count = 0, diff_vocab = False,\
                 embedding_path="../Data/embeddings.bin", limit_encode = 0, limit_decode=0):
        """ Create the vocabulary and load all the datasets

            Arguments:
        * working_dir   : Directory path where all the data files are stored
        * embedding_size: Dimension of vector representation for each word
        * diff_vocab    : Different vocab for encoder and decoder. 

        Returns:
        * void

        """

	#print (embedding_path, limit_encode, limit_decode)
        filenames_encode = [ working_dir + "train_content", working_dir + "train_query" ]
        filenames_decode = [ working_dir + "train_summary" ]

        self.global_count = 0
        self.vocab        = Vocab()

        if (diff_vocab == False):
            filenames_encode = filenames_encode + filenames_decode
            filenames_decode = filenames_encode

        self.vocab.construct_vocab(filenames_encode, filenames_decode, embedding_size, embedding_path,
                                   limit_encode, limit_decode)
        #print (self.vocab.word_to_index_decode)
	self.load_data(working_dir)


    def length_vocab_encode(self):
        """ Returns the encoder vocabulary size
        """
        return self.vocab.len_vocab_encode

    def length_vocab_decode(self):
        """ Returns the decoder vocabulary size
        """
        return self.vocab.len_vocab_decode

    def decode_to_sentence(self, decoder_states):
        """ Decodes the decoder_states to sentence
        """
        s = ""
        for temp in (decoder_states):
            word = self.vocab.decode_word_decoder(temp)
            s = s + " " + word

        return s

