import csv
import tensorflow as tf
import autoencoder as ac
import deepchess as dc
import random

def main():
    all_data = []
    with open('test.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',',quotechar='"')
        firstline = True
        for row in reader:
            if firstline:
                firstline = False
                continue
            all_data.append([int(tile) for tile in row[-1][1:-1].split(', ')])

    games = get_piecegames()
    print(games)

    # dc.create_deepchess(all_data)

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

    trainingLen = 1000000
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
    print(training_left[0], training_right[0])
    

def rand_game(game):
    loc = random.randint(0, len(game) - 1)
    return game[loc]




if __name__ == "__main__":
    main()
    #create_trainingset()
