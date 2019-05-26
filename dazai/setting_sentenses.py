import MeCab, re, collections, pickle, random, sys

def id(t):
    """
    単語をIDにする関数
    return: dict型のdic{"文字":id}, リスト型のid化した文章
    """
    #"「","『", "』", "\n", "　"は消去
    #"」"は"。"に
    #"。"が連続して存在すれば、一個に
    t = re.sub("「", "", t)
    t = re.sub("『", "", t)
    t = re.sub("』", "", t)
    t = re.sub(r"　+", "", t)
    t = re.sub(r"\n+", "", t)
    t = re.sub("〇", "", t)
    t = re.sub("」", "。", t)
    #"。"を"\t"に置き換え、"\t"でsplit
    t = re.sub("。", "。\t", t)
    t_li = t.split("\t")

    #リストの各要素ごとに分かち書きする、メモリを食いすぎないために一文ずつ
    mecab = MeCab.Tagger("-Owakati")
    wakati_li = []#分かち書きされた文章からなるリスト
    for i in t_li:
        wakati_li.append(re.sub("\n", "", mecab.parse(i)))

    #単語のID化
    #まずは、文章全体をID化
    all_wakati_sentence = "　".join(wakati_li).split()
    dic = {}
    dic["。"]=0#。のIDは0
    for word in all_wakati_sentence:
        if word not in dic:
            dic[word] = len(dic)
    print("単語数 : ", len(dic))
    word2id_li = [dic[word] for word in all_wakati_sentence]
    #１文章を表すリスト、からなるリストにする
    m = []
    n = []
    e = dic["。"]
    for i in word2id_li:
        if i != e:
            m.append(i)
        else:#"。"に対応するIDでリストを区切る
            m.append(i)
            n.append(m)
            m=[]
    return dic, n

def pair_sentence(t):
    """
    文章のペアを作る
    return: 文章のペア
    """
    pair_x = []
    pair_y = []
    for i in range(len(t)-1):
        pair_x.append(t[i])
        pair_y.append(t[i+1])
    return pair_x, pair_y

def make_backet(x, y, backet_size):
    """
    文章のペアをバケット長ごとに仕分ける
    バケット長より長い文章は無視する
    return: バケット長ごとに分けられた文章のリスト　(四次元のリスト)
    """
    backet = [[] for _ in backet_size]
    #バケットに追加
    for x_id, y_id in zip(x, y):
        for bucket_id, (x_size, y_size) in enumerate(backet_size):
            #print(len(x_id))
            if len(x_id) <= x_size and len(y_id) <= y_size:
                backet[bucket_id].append([x_id, y_id])
                break
    #確認用の出力
    for i, (a,b) in enumerate(backet_size):
        print("バケット(", a,b,"):",len(backet[i]), end="\t\t")
    print("\nバケットに入った文章数:",len(backet[0])+len(backet[1])+len(backet[2]), "/", len(x))

    #ランダムに並び替え
    rand_backet = []
    for rand_ba in backet:
        rand_backet.append(random.sample(rand_ba, len(rand_ba)))
    return rand_backet

fname = sys.argv[1]#テキストファイル
#ファイルのオープン
with open(fname, "r") as f:
    t = f.read()

#関数を使う
id_dic, id_li = id(t)
x, y = pair_sentence(id_li)
backet_size = [(20, 20), (30,30), (50,50)]
b_pair = make_backet(x, y, backet_size)

#pickleにして、バケットのリストと、dicを保存
with open('data_set.pkl', 'wb') as f:
    pickle.dump(b_pair, f)
with open('dic.pkl', 'wb') as f:
    pickle.dump(id_dic, f)
