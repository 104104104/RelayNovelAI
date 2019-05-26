import numpy as np
from chainer import Chain, Variable, optimizers, serializers
import chainer.links as L
import chainer.functions as F
from seq2seq_classes import *

#####モデルの記述#####
class Seq2Seq(Chain):
    def __init__(self, vocab_size, embed_size, hidden_size, backet_size):
        """
         vocab_size = 語彙サイズ
         embed_size = 単語ベクトルのサイズ
         hidden_size = 中間ベクトルのサイズ
         backet_size = バケットのサイズ
        """
        super(Seq2Seq, self).__init__(#パラメータを持つ層
            #Encoderのインスタンス化
            encoder = Encoder(vocab_size, embed_size, hidden_size),
            #Decoderのインスタンス化
            decoder = Decoder(vocab_size, embed_size, hidden_size),
            )
        #各種変数の設定
        self.hidden_size = hidden_size
        self.backet_size = backet_size
        self.decode_max_size = 50 # デコードはEOSが出力されれば終了する、出力されない場合の最大出力語彙数

    def encode(self, words, backet_size):
        """
        Encodeの計算
        words = 入力で使用する単語が記録されたリスト
        backet_size = バケットサイズ
        """
        # 内部メモリ、中間ベクトルの初期化
        c = Variable(np.zeros((backet_size, self.hidden_size),dtype='float32'))
        h = Variable(np.zeros((backet_size, self.hidden_size),dtype='float32'))
        # エンコーダーに単語を順番に読み込ませる
        for w in words:
            c, h = self.encoder(w, c, h)
        # 計算された中間ベクトル引き継ぎのために記録
        self.h = h
        # 内部メモリの初期化
        self.c = Variable(np.zeros((backet_size, self.hidden_size),dtype='float32'))

    def decode(self, word):
        """
        Decodeの計算
        word = 単語
        return: 予測単語
        """
        t, self.c, self.h = self.decoder(word, self.c, self.h)
        return t

    def reset(self):
        """
        インスタンス変数の初期化
        """
        self.h = Variable(np.zeros((self.backet_size, self.hidden_size),dtype='float32'))
        self.c = Variable(np.zeros((self.backet_size, self.hidden_size),dtype='float32'))
        self.zerograds()

    def __call__(self, enc_words, dec_words=None, train=True):
        """順伝播の計算を行う関数
        enc_words = 発話文の単語を記録したリスト
        dec_words = 応答文の単語を記録したリスト
        train = 学習か予測か
        return: 計算した損失の合計 or 予測したデコード文字列
        """
        enc_words = enc_words.T
        #行=文章番号、列=単語id -> 逆にする
        backet_size = len(enc_words[0])#バケットサイズ(一文章の長さ)の記録
        if train:
            dec_words = dec_words.T
        self.reset()
        # エンコーダーの計算
        self.encode(enc_words, backet_size)
        # デコーダーに読み込ませる用の<eos>、今回は"。"を<eos>とする
        t = Variable(np.array([0 for _ in range(backet_size)],dtype="int32")) #eos
        loss = Variable(np.zeros((), dtype="float32")) #損失の初期化
        # デコーダーの計算
        if train: # 学習の場合は損失を計算する
            for w in dec_words:
                y = self.decode(t)# 1単語ずつデコードする
                t = w
                loss += F.softmax_cross_entropy(y, w)# 正解単語(次の単語)と予測単語を照らし合わせて損失を計算
            return loss
        else:
            ys = []
            for i in range(self.decode_max_size):
                y = self.decode(t)
                y = np.argmax(y.data

                ) # 確率で出力されたままなので、確率が高い予測単語を取得する
                ys.append(y)
                t = Variable(np.array([y], dtype='int32'))
                if y == 0: # eos("。")を出力したならばデコードを終了する
                    break
            return ys
