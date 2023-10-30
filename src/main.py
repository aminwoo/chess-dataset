import chess
import tensorflow as tf
import os
import logging
from src.models.tf_net import LeelaZeroNet
from src.features.board2planes import board2planes
from src.features.policy_index import policy_index
from src.features.board2planes import mirrorMove


def main():
    model = LeelaZeroNet(
        num_filters=128,
        num_residual_blocks=10,
        se_ratio=8,
        constrain_norms=True,
        policy_loss_weight=1.0,
        value_loss_weight=1.6,
        moves_left_loss_weight=0.5,
    )

    model.load_weights('C:/Users/benwo/PycharmProjects/blunderfish/checkpoints/training_1/cp.ckpt').expect_partial()
    board = chess.Board()

    cmd = ""
    while cmd != "quit":
        cmd = input()
        logging.warning(cmd)
        if cmd == "uci":
            print("uciok")
        if cmd == "ucinewgame":
            pass
        if cmd == "isready":
            print("readyok")
        if cmd == "stop":
            pass
        if cmd.startswith("position fen"):
            board.set_fen(cmd.split(" ", 2)[-1])
        elif cmd.startswith("position"):
            board.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            moves = cmd.split(" ")[3:]
            for move in moves:
                board.push(chess.Move.from_uci(move))
        if cmd.startswith("go"):
            planes = board2planes(board)
            policy_out, value_out, moves_left_out = model(planes)
            print(tf.nn.softmax(value_out))
            for i in tf.argsort(policy_out, direction="DESCENDING")[0]:
                move_uci = policy_index[int(i)]
                move = chess.Move.from_uci(move_uci)
                if board.turn == chess.BLACK:
                    move = mirrorMove(move)
                if board.is_legal(move):
                    print(f"bestmove {move.uci()}")
                    #break


if __name__ == "__main__":
    main()
