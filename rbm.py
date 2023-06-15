# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 13:40:23 2023

@author: pgonigle
creating a recommender system based on a massive movie dataset 
"""
#importing packages
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable


#importing the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')


#preparing the training set and the test set
#and turning the sets into NP arrays
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')


#getting themax  number of users and movies
#two matrices, one for training set and one for test set

nb_users = int(max(max(training_set[:, 0], ), max(test_set[:, 0])))
nb_movies = int(max(max(training_set[:, 1], ), max(test_set[:, 1])))

#converting training set and test set into an array with users in lines and movies in columns lines
#because we will have the observations in lines and the features in columns
#so it will be using the max users and max movies number for 
#both so that data will be consistent??
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:, 1][data[:,0] == id_users]
        id_ratings = data[:, 2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data
"""
For each user (looping from 1 to nb_users + 1), it extracts all the movies that user has rated and the corresponding ratings.
Then, it creates a new list ratings of size nb_movies filled with zeros. This list will hold the ratings for each movie. If the user hasn't rated a movie, it remains zero.
It then places the user's ratings into the ratings list at the positions corresponding to the movie's ID minus one (because Python lists are 0-indexed).
Finally, the ratings list is added to new_data. At the end of the function, new_data is a list of all users' ratings.
"""
training_set = convert(training_set)
test_set = convert(test_set)
        
"""
Created on Wed Jun 14 13:40:23 2023
basically convert training set and test set
from a list of lists to a torch tensors

"""
#converting the data into Torch tensors


    