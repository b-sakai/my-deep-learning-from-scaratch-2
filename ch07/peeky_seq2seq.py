import sys
sys.path.append('..')
from common.time_layers import *
from seq2seq import Seq2seq, Encoder

class PeekyDecoder:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        # Initialize weights
        embed_W = (rn(V, D) / 100).astype('f')
        lstm_Wx1 = (rn(H + D, 4 * H) / np.sqrt(H + D)).astype('f')
        lstm_Wh1 = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b1 = np.zeros(4 * H).astype('f')
        lstm_Wx2 = (rn(H + H, 4 * H) / np.sqrt(H + H)).astype('f')
        lstm_Wh2 = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b2 = np.zeros(4 * H).astype('f')
        affine_b = np.zeros(V).astype('f')

        # Create layers
        self.embed = TimeEmbedding(embed_W)
        self.lstm1 = TimeLSTM(lstm_Wx1, lstm_Wh1, lstm_b1, stateful=True)
        self.lstm2 = TimeLSTM(lstm_Wx2, lstm_Wh2, lstm_b2, stateful=True)
        self.affine = TimeAffine(embed_W.T, affine_b)

        # Put all weights and gradients in list
        self.params, self.grads = [], []
        for layer in (self.embed, self.lstm1, self.lstm2, self.affine):
            self.params += layer.params
            self.grads += layer.grads

        # Store last hidden state
        self.cache = None

    def forward(self, xs, h):
        N, T = xs.shape
        N, H = h.shape

        # Initialize hidden state
        self.lstm1.set_state(h)

        # Embedding
        out = self.embed.forward(xs)

        # Encoderから受け取るhをリピートする
        hs = np.repeat(h, T, axis=0).reshape(N, T, H)
        # hsをEmbeddingレイヤーの出力とnp.concatenateで結合する
        out = np.concatenate((hs, out), axis=2)

        # Forward propagate LSTM
        out = self.lstm1.forward(out)
        out = self.lstm2.forward(out)

        # Forward propagate affine layer
        score = self.affine.forward(out)
        self.cache = H
        return score
    
    def backward(self, dscore):
        H = self.cache

        # Backpropagate affine layer
        dout = self.affine.backward(dscore)
        dout = self.lstm2.backward(dout)
        dout = self.lstm1.backward(dout)

        # dsをdhとdxに分割する
        dh = dout[:, :, :H]
        dhs = dout[:, :, H:]

        # dhの合計を求める
        dhs = np.sum(dhs, axis=1)

        return dh, dhs
    
    def generate(self, h, start_id, sample_size):
        sampled = []
        char_id = start_id
        self.lstm1.set_state(h)
        self.lstm2.set_state(h)

        # Start with start_id
        x = np.array([char_id]).reshape((1, 1))

        for _ in range(sample_size):
            # Forward propagate RNN
            out = self.embed.forward(x)
            out = np.concatenate((h, out), axis=1)
            out = self.lstm1.forward(out)
            out = self.lstm2.forward(out)
            score = self.affine.forward(out)

            # Get the most probable word ID
            char_id = np.argmax(score.flatten())

            # Append to output
            sampled.append(char_id)

            # Forward propagate RNN
            x = np.array([char_id]).reshape((1, 1))

        return sampled
    
class PeekySeq2seq(Seq2seq):
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        
        # Initialize weights
        self.encoder = Encoder(V, D, H)
        self.decoder = PeekyDecoder(V, D, H) # PeekyDecoderを使用
        self.softmax = TimeSoftmaxWithLoss()
        
        # Put all weights and gradients in list
        self.params, self.grads = [], []
        for layer in (self.encoder, self.decoder):
            self.params += layer.params
            self.grads += layer.grads
