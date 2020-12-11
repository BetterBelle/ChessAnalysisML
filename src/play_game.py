import chess
import create_dataset as cd
import tensorflow as tf
import numpy as np

deepchess = tf.keras.models.load_model('saved_networks/deepchess_model')

def computermove(board, depth):
    alpha = -1
    beta = 1
    v = -1
    for move in board.legal_moves:
        cur = board.copy()
        cur.push(move)
        if v == -1:
            v = alphabeta(board, depth - 1, alpha, beta, False)
            best_move = move
            if alpha == -1:
                alpha = v
        else:
            new_v = net_predict(alphabeta(cur, depth - 1, alpha, beta, False), v)[0]
            if new_v != v:
                best_move = move
                v = new_v 
            alpha = net_predict(alpha, v)[0]

    print (best_move)
    board.push(best_move)
    return board


def alphabeta(board, depth, alpha, beta, max_player):
	if depth == 0 or board.legal_moves.count() == 0:
		return board

	if max_player:
		v = -1
		for move in board.legal_moves:
			cur = board.copy()
			cur.push(move)
			if v == -1:
				v = alphabeta(cur, depth-1, alpha, beta, False) 
			if alpha == -1:
				alpha = v
		
			v = net_predict(v, alphabeta(cur, depth-1, alpha, beta, False))[0]
			alpha = net_predict(alpha, v)[0]
			if beta != 1:
				if net_predict(alpha, beta)[0] == alpha:
					break
		return v 
	else:
		v = 1
		for move in board.legal_moves:
			cur = board.copy()
			cur.push(move)
			if v == 1:
				v = alphabeta(cur, depth-1, alpha, beta, True) 
			if beta == 1:
				beta = v
			
			v = net_predict(v, alphabeta(cur, depth-1, alpha, beta, True))[1]
			beta = net_predict(beta, v)[1] 
			if alpha != -1:
				if net_predict(alpha, beta)[0] == alpha:
					break
		return v 

def net_predict(first, second):

    result = deepchess.predict(
        {
            'left_encoder_layer_0': np.array([cd.fen_to_inputarray(first.fen())]), 
            'right_encoder_layer_0': np.array([cd.fen_to_inputarray(second.fen())])
        }
    )[0]

    if result[0] > result[1]:
        return (first, second)
    else:
        return (second, first)