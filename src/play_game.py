import chess
import create_dataset as cd
import tensorflow as tf
import numpy as np

deepchess = tf.keras.models.load_model('saved_networks/deepchess_model')

def alphabeta(board, depth, alpha_pos, beta_pos, max_player):
    # if it's the end node, return the board
    if depth == 0 or board.legal_moves.count() == 0:
        return board

    # if alpha and beta aren't set, set them each to an arbitrary position
    if alpha_pos == None and max_player:
        alpha_pos = board.copy()
        for move in board.legal_moves:
            alpha_pos.push(move)
            break
    elif beta_pos == None and not max_player:
        beta_pos = board.copy()
        for move in board.legal_moves:
            beta_pos.push(move)
            break

    if max_player:
        # value = -inf
        best_board = None
        for move in board.legal_moves:
            # set value to arbitrary move
            if best_board == None:
                best_board = board.copy()
                best_board.push(move)
                continue

            board.push(move)
            board_after_move = board.copy()
            board.pop()

            # value = max(value, alphabeta(child))
            prediction = deepchess.predict(
                {
                    'left_encoder_layer_0': np.array([cd.fen_to_inputarray(best_board.fen())]), 
                    'right_encoder_layer_0': np.array(
                        [cd.fen_to_inputarray(alphabeta(board_after_move.copy(), depth-1, alpha_pos, beta_pos, False).fen())]
                    )
                }
            )
            if prediction[0][0] < prediction[0][1]:
                best_board = board_after_move.copy()

            # alpha = max(alpha, value)
            prediction = deepchess.predict(
                {
                    'left_encoder_layer_0': np.array([cd.fen_to_inputarray(alpha_pos.fen())]), 
                    'right_encoder_layer_0': np.array([cd.fen_to_inputarray(best_board.fen())])
                }
            )
            if prediction[0][0] < prediction[0][1]:
                alpha_pos = best_board.copy()

            # pretend beta = +inf, so it's impossible for alpha to be larger
            if beta_pos == None:
                continue

            # if alpha >= beta break
            prediction = deepchess.predict(
                {
                    'left_encoder_layer_0': np.array([cd.fen_to_inputarray(alpha_pos.fen())]), 
                    'right_encoder_layer_0': np.array([cd.fen_to_inputarray(beta_pos.fen())])
                }
            )
            if prediction[0][0] >= prediction[0][1]:
                break

        return best_board
    else:
        # value = +inf
        best_board = None
        for move in board.legal_moves:
            # set value to arbitrary move
            if best_board == None:
                best_board = board.copy()
                best_board.push(move)
                continue

            board.push(move)
            board_after_move = board.copy()
            board.pop()

            # value = min(value, alphabeta(child))
            prediction = deepchess.predict(
                {
                    'left_encoder_layer_0': np.array([cd.fen_to_inputarray(best_board.fen())]), 
                    'right_encoder_layer_0': np.array(
                        [cd.fen_to_inputarray(alphabeta(board_after_move.copy(), depth-1, alpha_pos, beta_pos, True).fen())]
                    )
                }
            )
            if prediction[0][0] > prediction[0][1]:
                best_board = board_after_move.copy()

            # alpha = min(beta, value)
            prediction = deepchess.predict(
                {
                    'left_encoder_layer_0': np.array([cd.fen_to_inputarray(beta_pos.fen())]), 
                    'right_encoder_layer_0': np.array([cd.fen_to_inputarray(best_board.fen())])
                }
            )
            if prediction[0][0] > prediction[0][1]:
                beta_pos = best_board.copy()

            # pretend alpha = -inf, so it's impossible for beta to be smaller
            if alpha_pos == None:
                continue

            # if beta <= alpha break
            prediction = deepchess.predict(
                {
                    'left_encoder_layer_0': np.array([cd.fen_to_inputarray(alpha_pos.fen())]), 
                    'right_encoder_layer_0': np.array([cd.fen_to_inputarray(beta_pos.fen())])
                }
            )
            if prediction[0][0] >= prediction[0][1]:
                break

        return best_board