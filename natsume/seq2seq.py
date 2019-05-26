import numpy as np
from chainer import Variable, optimizers, serializers
import chainer.links as L
import chainer.functions as F
import pickle, datetime, MeCab, re, sys
from seq2seq_model import Seq2Seq
from seq2seq_classes import *

#第一引数はembed_size第二引数はhidden_size第三引数はbs第四引数はepoch

def make_data(backet):
    """
    データをx,yに仕分ける関数
    backet = バケットデータ
    return : x, yに分けたデータ
    """
    #x_data, y_dataを作り、それぞれlenの最大値を求める
    max_x_len = 0
    max_y_len = 0
    x_data=[]
    y_data=[]
    for data in backet:
        x_data.append(data[0])
        if max_x_len < len(data[0]):
            max_x_len = len(data[0])
        y_data.append(data[1])
        if max_y_len < len(data[1]):
            max_y_len = len(data[1])
    #-1を適宜挿入
    for i, d in enumerate(x_data):
        for _ in range(len(d), max_x_len):
            x_data[i].append(-1)
    for i, d in enumerate(y_data):
        for _ in range(len(d), max_y_len):
            y_data[i].append(-1)
    return x_data, y_data

#####データの準備#####
with open('data_set.pkl', 'rb') as f:
    pair = pickle.load(f)
with open('dic.pkl', 'rb') as f:
    dic = pickle.load(f)

#####各種設定######
vocab_size = len(dic)
#パラメータの数指定、以下三つはrelay.pyと合わせる必要がある
embed_size = int(sys.argv[1])
hidden_size = int(sys.argv[2])
bs = int(sys.argv[3])   #バッチのサイズ
epoch = int(sys.argv[4])
with open("param_count.pkl", "wb") as f:
    pickle.dump([embed_size, hidden_size, bs],f)
model=Seq2Seq(vocab_size=vocab_size,#モデルのインスタンス化
                embed_size=embed_size,
                hidden_size=hidden_size,
                backet_size = bs)
opt = optimizers.Adam()#とりあえずAdamにしておけば、だいたい大丈夫
opt.setup(model)

#プリント用にエポック数の合計を得る（バケットで処理なんてするからめんどうなことに……）
sum_epoch = 0
for p, data in enumerate(pair):
    sum_epoch +=  int(epoch/len(pair)) * (int(len(data)/bs)+1)
#####学習部分######
sum_st = datetime.datetime.now()
st = datetime.datetime.now()
count = 1
for p, data in enumerate(pair):
    x, y = make_data(data)
    print("バケット",p+1,"の要素数",len(y))
    #各バケットごとに学習
    for _ in range(int(epoch / len(pair))):#エポック
        for i in range(0, len(data), bs):#バッチごとの学習
            batch_x = Variable(np.array(x[i:(i+bs) if (i+bs) < len(data) else len(data)], dtype="int32"))
            batch_y = Variable(np.array(y[i:(i+bs) if (i+bs) < len(data) else len(data)], dtype="int32"))
            model.cleargrads()   #勾配初期化
            loss=model(batch_x, batch_y)     #順方向の計算
            loss.backward()     #誤差逆伝搬
            opt.update()  #パラメータ更新
            #10回に一回保存, 最初の一回も保存
            if count % 500 == 0 or count == 1:
                save_f_name = str(count) + "_seq2seq_model.npz"
                serializers.save_npz(save_f_name, model)
            ed = datetime.datetime.now()
            print("epoch:\t{} / {}\tloss:\t{}\ttime:\t{}\ttotal_time:\t{}".format(count, sum_epoch, loss, ed-st, ed-sum_st))#学習状況
            count+=1
            st = datetime.datetime.now()
    sum_ed = datetime.datetime.now()
    print(p+1, "/", len(pair), "　終わり\tここまでの合計時間は",sum_ed-sum_st)

#####学習済みデータの保存#####
serializers.save_npz('finished_seq2seq.npz', model)
