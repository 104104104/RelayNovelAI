import sys, pickle, MeCab, re
import numpy as np
from chainer import Variable,serializers
from seq2seq_model import Seq2Seq

#使いかた
#第一引数 = 使う学習ファイル、第二引数 = 文

#単語辞書の読み込み
with open('dic.pkl', 'rb') as f:
    dic = pickle.load(f)

# まず同じネットワークのオブジェクトを作る
relay_net = Seq2Seq(vocab_size=len(dic),
                embed_size=200,
                hidden_size=100,
                backet_size = 50)
# そのオブジェクトに保存済みパラメータをロードする
serializers.load_npz(sys.argv[1], relay_net)

try:
    sentence = sys.argv[2]
    mecab = MeCab.Tagger("-Owakati")
    words=re.sub("\n", "", mecab.parse(sentence))
except:
    print("miss!　本文中にない単語 or 引数エラーでは？")

w_li=words.split()
ids = Variable(np.array([[dic[i] for i in w_li]], dtype="int32"))
dec_response = relay_net(enc_words=ids, train=False)
d=dic.keys()
print("relay:",end="")
for i in dec_response:
    print(list(d)[i], end="")
print("\n")
