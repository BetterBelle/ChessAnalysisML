import csv
import tensorflow as tf
import autoencoder as ac
import deepchess as dc
import random
import numpy as np

def main():
    training_data = create_trainingset()
    deepchess = tf.keras.models.load_model('saved_networks/deepchess_model')
    dc.train_deepchess(deepchess, training_data)


def get_piecegames(): 
    white = []
    black = []

    with open('test.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',',quotechar='"')
        firstline = True
        for row in reader:
            if firstline:
                firstline = False
                continue
            
            if row[0][1:-1] == '1, 0':
                white.append([int(tile) for tile in row[-1][1:-1].split(', ')])
            elif row[0][1:-1] == '0, 1':
                black.append([int(tile) for tile in row[-1][1:-1].split(', ')])

    return [white, black]


def create_trainingset():
    training_left = []
    training_right = []
    training_results = []

    all_pieces = get_piecegames()
    white = all_pieces[0]
    black = all_pieces[1]

    trainingLen = 10000
    for n in range(trainingLen):
        loc = random.randint(0, 1)     
        if loc == 0:
            training_left.append(rand_game(all_pieces[0]))
            training_right.append(rand_game(all_pieces[1]))
            training_results.append([1, 0])
        else:
            training_left.append(rand_game(all_pieces[1]))
            training_right.append(rand_game(all_pieces[0]))
            training_results.append([0, 1])

    return [training_left, training_right, training_results]
    

def rand_game(game):
    loc = random.randint(0, len(game) - 1)
    return game[loc]




if __name__ == "__main__":
    main()
    #create_trainingset()
