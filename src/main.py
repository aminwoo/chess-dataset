import chess
import tensorflow as tf
from src.models.tf_net import LeelaZeroNet
from src.features.board2planes import board2planes
from src.features.policy_index import policy_index

model = LeelaZeroNet(
    num_filters=128,
    num_residual_blocks=10,
    se_ratio=8,
    constrain_norms=True,
    policy_loss_weight=1.0,
    value_loss_weight=1.6,
    moves_left_loss_weight=0.5,
)

model.load_weights('../checkpoints/my_checkpoint').expect_partial()

board = chess.Board(fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
planes = board2planes(board)

policy_out, value_out, moves_left_out = model(planes)
print(value_out)
#out = model(planes)
#print(out)

for i in tf.argsort(policy_out, 1, direction='DESCENDING')[0]:
    print(policy_index[int(i)], policy_out[0][i])
#idx = int(tf.math.argmax(out[0], 1))
#print(len(policy_index))
#print(policy_index[int(tf.math.argmax(out[0], 1))])

