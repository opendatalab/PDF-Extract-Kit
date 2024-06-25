import os
import re
from tqdm import tqdm
from collections import Counter

log_str = """
[2024/03/22 13:26:22] ppocr WARNING: Since the angle classifier is not initialized, it will not be used during the forward process
[2024/03/22 13:26:22] ppocr WARNING: Since the angle classifier is not initialized, it will not be used during the forward process
downloading s3://private-dataset-pnorm/namath/pdf/0ff8cc3070554761bd4fd8c04bb859bc.pdf ...
got exception: cannot open broken document
downloading s3://private-dataset-pnorm/namath/pdf/10061c85ed2da5b7ce0a46005134a4ca.pdf ...
pdf pages: 12
[2024/03/22 13:27:20] ppocr WARNING: Since the angle classifier is not initialized, it will not be used during the forward process
downloading s3://private-dataset-pnorm/kmath/kmath_textbook/files/更多按丛书2（按整理的书籍）/01计算机科学丛书/04计算机软件61册/01软件工程32册/面向对象与传统软件工程（第5版）[（美）Stephen.R.Schach].pdf ...
got exception: [Errno 36] File name too long: '/tmp/asset__private-dataset-pnorm__kmath__kmath_textbook__files__更多按丛书2（按整理的书籍）__01计算机科学丛书__04计算机软件61册__01软件工程32册__面向对象与传统软件工程（第5版）[（美）Stephen.R.Schach].pdf.CeCc8aA7'
downloading s3://private-dataset-pnorm/kmath/kmath_textbook/files/更多按丛书2（按整理的书籍）/01计算机科学丛书/04计算机软件61册/01软件工程32册/面向对象软件工程[（美）Stephen.R.Schach].pdf ...
pdf pages: 365
[2024/03/22 20:49:35] ppocr WARNING: Since the angle classifier is not initialized, it will not be used during the forward process
[2024/03/22 20:49:35] ppocr WARNING: Since the angle classifier is not initialized, it will not be used during the forward process
[2024/03/22 20:49:36] ppocr WARNING: Since the angle classifier is not initialized, it will not be used during the forward process
[2024/03/22 20:49:36] ppocr WARNING: Since the angle classifier is not initialized, it will not be used during the forward process
[2024/03/22 20:49:36] ppocr WARNING: Since the angle classifier is not initialized, it will not be used during the forward process
got exception: [Errno 36] File name too long: '/tmp/asset__private-dataset-pnorm__kmath__kmath_textbook__files__更多按丛书2（按整理的书籍）__01计算机科学丛书__04计算机软件61册__02编译原理6册__编译原理（第2版）[（美）Alfred.V.Aho Monica.S.Lam Ravi.Sethi Jeffrey.D.Ullman].pdf.lock'
downloading s3://private-dataset-pnorm/kmath/kmath_textbook/files/更多按丛书2（按整理的书籍）/01计算机科学丛书/04计算机软件61册/02编译原理6册/编译器工程[（美）Keith.D.Cooper Linda.Torczon].pdf ...
pdf pages: 512
[2024/03/22 21:29:09] ppocr WARNING: Since the angle classifier is not initialized, it will not be used during the forward process
[2024/03/22 21:29:09] ppocr WARNING: Since the angle classifier is not initialized, it will not be used during the forward process
[2024/03/22 21:32:30] ppocr WARNING: Since the angle classifier is not initialized, it will not be used during the forward process
got exception: [Errno 36] File name too long: '/tmp/asset__private-dataset-pnorm__kmath__kmath_textbook__files__更多按丛书2（按整理的书籍）__01计算机科学丛书__04计算机软件61册__02编译原理6册__编译器构造：C语言描述[（美）Charles.N.Fischer Richard.J.LeBlanc.Jr].pdf.lock'
got exception: [Errno 36] File name too long: '/tmp/asset__private-dataset-pnorm__kmath__kmath_textbook__files__更多按丛书2（按整理的书籍）__01计算机科学丛书__04计算机软件61册__02编译原理6册__编译程序设计艺术：理论与实践[（美）Thomas.Pittman James.Peters].pdf.lock'
downloading s3://private-dataset-pnorm/kmath/kmath_textbook/files/更多按丛书2（按整理的书籍）/01计算机科学丛书/04计算机软件61册/02编译原理6册/高级编译器设计与实现[（美）Steven.S.Muchnick].pdf ...
pdf pages: 646
[2024/03/22 22:16:41] ppocr WARNING: Since the angle classifier is not initialized, it will not be used during the forward process
"""

log_files = [item for item in os.listdir("./") if item.endswith(".out")]
pattern = r'got exception:(.*?)\n'

key_words = []
processed_num = 0
for log_f in tqdm(log_files):
    with open(log_f, 'r') as f:
        log_str = f.read()
    processed = re.findall("pdf already processed", log_str, re.DOTALL)
    processed_num += len(processed)
    res = re.findall(pattern, log_str, re.DOTALL)
    res = [item.strip() for item in res]
    clean_res = []
    for item in res:
        if ":" in item:
            temp = item.split(":")[0]
            if "pid" in temp:
                l = temp.index("(")
                r = len(temp) - temp[::-1].index(")")
                temp = temp.replace(temp[l:r], "(pid xxxxx)")
            clean_res.append(temp)
        else:
            temp = item
            if "pid" in temp:
                l = temp.index("(")
                r = len(temp) - temp[::-1].index(")")
                temp = temp.replace(temp[l:r], "(pid xxxxx)")
            clean_res.append(temp)
    assert len(clean_res) == len(res)
    key_words.extend(clean_res)

print(f"=> total log num: `{len(log_files)}`, exception num: `{len(key_words)}`" )
key_words = ['pdf already processed']*processed_num + key_words
c = Counter(key_words)
for key, val in c.items():
    print(f"  => exception type: `{key}`, num: `{val}`.")
    
    
