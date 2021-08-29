import string
import re
import numpy as np
import argparse
import io
import os
import codecs

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-i', '--input', type=str)
parser.add_argument('-o', '--output', type=str)

args = parser.parse_args()

def split_sents_lo(text, debug=False):
    
    def clean_text(s):
        s = re.sub('\ufeff', '', s)
        s = re.sub('\u200b', '', s)
        s = re.sub(" +", ' ', s)
        s = s.strip()
        return s

    def isLao(s):
        punct = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "
        count = 0
        for c in s:
            if c in punct:
                count += 1
        if count == len(s):
            return True
        return False
    
    
    paras = [x.strip() for x in text.strip().split('\n')]
    content = []
    num_latin = "0,O,1,2,3,4,5,6,7,8,9".split(',')
    for pr in paras:
        prsents = []
        prs = pr
        start = 0
        punc = ['.','!','?']
        while prs:
            #tìm vị trí 3 dấu câu ở trên
            idxs = np.array([prs.find(punc[0],start),prs.find(punc[1],start),prs.find(punc[2],start)])
            idxs[idxs<0] = 1000000
            #lấy dấu có vị trí nhỏ nhất
            minidx = np.argmin(idxs)
            #nếu không tìm thấy dấu nào
            if idxs[minidx]==1000000:
                prsents.append(prs)
                break
            #nếu là dấu ở đầu câu thì bỏ đi
            if idxs[minidx]==0:
                prs = prs[1:]
                continue

            if minidx==0 and idxs[minidx]+1<len(prs): #là dấu chấm thì xét xem có phải là số không
                if (prs[idxs[minidx]-1].isnumeric() and prs[idxs[minidx]+1].isnumeric()) or prs[idxs[minidx]-2:idxs[minidx]+1].lower()=="tp.": #là số vd: 2.3
                    start = idxs[minidx]+1
                    continue
            sent = prs[:idxs[minidx]+1].strip() #+1 để lấy dấu câu
            if sent.strip(string.punctuation+'…“” ').isnumeric(): #là dạng số thứ tự đứng đầu câu như "2. ..."
                start = idxs[minidx]+1
                continue
            prsents.append(sent)
            prs = prs[idxs[minidx]+1:].strip()
            start = 0

        if prsents:
            if debug:
                for ss in prsents:
                    print('**> ',ss)
                print('----')
#             content.append("\n".join(prsents)) #mỗi câu 1 dòng
            i = 0
            mask = [False]*len(prsents)
            while i < len(prsents):
                new = ''
                if (prsents[i].strip()[-2] in num_latin and i <= len(prsents) - 2 and prsents[i+1].strip()[0] in num_latin):
                    new_line = prsents[i] + prsents[i+1]
                    mask[i] = True
                    mask[i+1] = True
                    i += 1
                elif isLao(prsents[i]) is True and i <= len(prsents) - 2:
                    new_line = prsents[i] + prsents[i+1]
                    mask[i] = True
                    mask[i+1] = True
                    i += 1
                elif i <= len(prsents) - 2 and isLao(prsents[i+1]) is True:
                    new_line = prsents[i] + prsents[i+1]
                    mask[i] = True
                    mask[i+1] = True
                    i += 1
                elif (len(prsents[i].strip()) < 10 and i <= len(prsents) - 2) or (i <= len(prsents) - 2 and len(prsents[i+1].strip()) < 10):
                    new_line = prsents[i] + prsents[i+1]
                    mask[i] = True
                    mask[i+1] = True
                    i += 1
                else:
                    new_line = prsents[i]
                content.append(new_line)
                i += 1
                
                if i == len(prsents) - 1 and mask[i] is False:
                    content.append(prsents[i])
                    i += 1
                elif i == len(prsents) - 1 and mask[i] is True:
                    break
    result = []
    for line in content:
        line = clean_text(line)
        if line != '':
            if line not in result:
                result.append(line)
    return result

if __name__ == '__main__':

    input_file = args.input
    output_file = args.output

    try:
        f = codecs.open(input_file, encoding='utf-8', errors='strict')
        for line in f:
            pass
    except UnicodeDecodeError:
        try:
            f = codecs.open(input_file, encoding='utf-16', errors='strict')
            for line in f:
                pass
            new_file = input_file + '.utf8'
            script = 'iconv -f UTF-16LE -t UTF-8 ' + input_file + ' > ' + new_file
            os.system(script)
            os.remove(input_file)
            input_file = new_file
        except UnicodeDecodeError:
            new_file = input_file + '.utf8'
            script = 'iconv -f UTF-32LE -t UTF-8 ' + input_file + ' > ' + new_file
            os.system(script)
            os.remove(input_file)
            input_file = new_file

    lines = io.open(input_file, encoding='utf-8').read().split('\n')

    f = open(output_file, 'w')
    for line in lines:
    	for l in split_sents_lo(line):
    		f.write(l)
    		f.write('\n')
    f.close()