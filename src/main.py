import chess
import tensorflow as tf
from src.models.tf_net import LeelaZeroNet
from src.features.board2planes import board2planes
from src.features.policy_index import policy_index
from src.features.board2planes import mirror_move

INITIAL_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def main():
    model = LeelaZeroNet(
        num_filters=256,
        num_residual_blocks=10,
        se_ratio=8,
        constrain_norms=True,
        policy_loss_weight=1.0,
        value_loss_weight=1.6,
        moves_left_loss_weight=0.5,
    )

    model.load_weights("C:/Users/benwo/PycharmProjects/blunderfish/checkpoints/training_1/cp.ckpt").expect_partial()
    board = chess.Board()

    cmd = ""
    while cmd != "quit":
        cmd = input()
        if cmd == "uci":
            print("uciok")
        if cmd == "ucinewgame":
            pass
        if cmd == "isready":
            print("readyok")
        if cmd == "stop":
            pass
        if cmd.startswith("position"):
            args = cmd.split(" ")[1:]
            if args[0] == "fen":
                board = chess.Board(fen=" ".join(args[1:]))
            else:
                board = chess.Board(fen=INITIAL_FEN)
                for move in args[2:]:
                    board.push(chess.Move.from_uci(move))
        if cmd.startswith("go"):
            planes = board2planes(board)
            import sys
            import numpy
            numpy.set_printoptions(threshold=sys.maxsize)
            print(planes)
            policy_out, _, _ = model(planes)
            for i in tf.argsort(policy_out, direction="DESCENDING")[0]:
                move_uci = policy_index[int(i)]
                move = chess.Move.from_uci(move_uci)
                if board.turn == chess.BLACK:
                    move = mirror_move(move)
                #if board.is_legal(move):
                print(f"bestmove {move.uci()}")
                break


if __name__ == "__main__":
    main()
