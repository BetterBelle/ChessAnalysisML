import tensorflow as tf
import csv
import autoencoder as ac


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

    training_data = all_data[:len(all_data) // 2]
    testing_data = all_data[len(all_data) //2 + 1:]
    auto_encoder, encoder = ac.create_autoencoder()

    ac.train_encoder(auto_encoder, encoder, training_data, testing_data)
    auto_encoder.evaluate(testing_data, testing_data)

    
if __name__ == "__main__":
    main()