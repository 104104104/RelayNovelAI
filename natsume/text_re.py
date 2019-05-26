# encoding : utf-8
import sys, os, re
from chardet.universaldetector import UniversalDetector

try:
    fname=sys.argv[1]
    #エンコードの判定
    detector = UniversalDetector()
    with open(fname, "rb") as t:
        for l in t:
            detector.feed(l)
            if detector.done:
                break
    detector.close()
    fencoding = detector.result['encoding']
    print(fencoding)

    #読み込み
    bindata = open(fname, "rb").read()
    t = bindata.decode(fencoding)

    if re.search(r"\-{5,}", t)!=None and re.search(r"底本：", t)!=None:
        print("これは青空文庫のテキスト")
        #ヘッダーとフッターの除去
        t = re.split(r"\-{5,}", t)[2]
        t = re.split(r"底本：", t)[0]
        t = t.strip()
        #ルビとかの削除
        t = t.replace("|", "")
        t = re.sub(r"《.+?》", "", t)
        t = re.sub(r"［＃.+?］", "", t)
    else:
        print("これは普通のテキスト")#処理の必要がない
    with open(fname[:-4]+"_utf-8.txt", "w", encoding="utf-8") as f:
        f.write(t)
        print(fname[:-4]+"_utf-8.txt"+"として保存しました")

except Exception as e:
    print("error! ", e)
