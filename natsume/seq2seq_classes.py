from chainer import Chain, Variable
import chainer.links as L
import chainer.functions as F

#####使うクラスの定義#####
class Encoder(Chain):
    def __init__(self, vocab_size, embed_size, hidden_size):
        """
        vacab_size = 使われる単語の語彙数
        embed_size = 単語をベクトル表現したときのサイズ
        hidden_size = 中間層のサイズ
        """
        super(Encoder, self).__init__(#パラメータを持つ層の定義
            #単語をベクトルに変換する層
            xe = L.EmbedID(vocab_size, embed_size, ignore_label=-1),
            #単語ベクトルを隠れ層の四倍のサイズに変換する層
            eh = L.Linear(embed_size, 4 * hidden_size),
            #出力された中間層を隠れ層の四倍のサイズに変換する層
            hh = L.Linear(hidden_size, 4 * hidden_size)
            )

    def __call__(self, x, c, h):
        """
        param:x = 入力された単語
        param:c = LSTMの内部メモリ
        param:h = 隠れ層
        """
        #xeで単語ベクトルに変換して、そのベクトルをtanhにかける
        e = F.tanh(self.xe(x))
        #前の内部メモリの値と、単語ベクトルを四倍したもの、中間層を四倍したものを足してLSTMに入力
        c, h = F.lstm(c, self.eh(e) + self.hh(h))
        return c, h

class Decoder(Chain):
    def __init__(self, vocab_size, embed_size, hidden_size):
        """
        vacab_size = 使われる単語の語彙数
        embed_size = 単語をベクトル表現したときのサイズ
        hidden_size = 中間層のサイズ
        """
        super(Decoder, self).__init__(#パラメータを持つ層の登録
            #単語をベクトルに変換する層
            ye = L.EmbedID(vocab_size, embed_size, ignore_label = -1),
            #単語ベクトルを隠れ層の四倍のサイズに変換する層
            eh = L.Linear(embed_size, 4 * hidden_size),
            #出力された中間層を四倍のサイズに変換する層
            hh = L.Linear(hidden_size, 4 * hidden_size),
            #出力されたベクトルを単語ベクトルのサイズに変換する
            he = L.Linear(hidden_size, embed_size),
            #単語ベクトルを語彙サイズのベクトル(one-hot)に変換する層
            ey = L.Linear(embed_size, vocab_size),
            )
    def __call__(self, y, c, h):
        """
        y = one-hotのベクトル
        c = LSTMの内部メモリ
        h = 隠れ層
        """
        #入力された単語を、単語ベクトルに変換して、そのベクトルをtanhにかける
        e = F.tanh(self.ye(y))
        #前の内部メモリの値と、単語ベクトルを四倍したもの、中間層を四倍したものを足してLSTMに入力
        c, h = F.lstm(c, self.eh(e) + self.hh(h))
        #出力された中間ベクトルを単語ベクトルに、単語ベクトルを語彙サイズのベクトルに、変換
        t = self.ey(F.tanh(self.he(h)))
        return t, c, h
