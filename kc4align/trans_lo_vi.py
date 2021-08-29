# -*- coding: utf-8 -*-
import argparse
import codecs
import os
import io
from underthesea import sent_tokenize
import re
from underthesea import word_tokenize
from deep_translator import GoogleTranslator

def clean_text(s):
    s = re.sub('\ufeff', '', s)
    s = re.sub('\u200b', '', s)
    s = re.sub('"', '', s)
    s = re.sub("'", '', s)
    s = re.sub(" +", ' ', s)
    s = s.strip()
    return s

def clear_text(s):
    letters = list(
        "abcdefghijklmnopqrstuvwxyzáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩị"
        "óòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ "
    )
    telex_words = list("áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựđ")
    number = list("0123456789")
    alphabel = telex_words + letters + number
    s = s.lower()
    for c in s:
        if c not in alphabel:
            s = s.replace(c, '')
    s = re.sub(" +", ' ', s)
    s = s.strip()
    return s

if __name__ == '__main__':

    parser = argparse.ArgumentParser('',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-i', '--input', type=str, required=True,
                        help='folder is needed align to get bilanguage sentences')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='folder is file output aligned')

    args = parser.parse_args()


    file_lo = args.input
    try:
        f = codecs.open(file_lo, encoding='utf-8', errors='strict')
        for line in f:
            pass
    except UnicodeDecodeError:
        try:
            f = codecs.open(file_lo, encoding='utf-16', errors='strict')
            for line in f:
                pass
            new_file = file_lo + '.utf8'
            script = 'iconv -f UTF-16LE -t UTF-8 ' + file_lo + ' > ' + new_file
            os.system(script)
            os.remove(file_lo)
            file_lo = new_file
        except UnicodeDecodeError:
            new_file = file_lo + '.utf8'
            script = 'iconv -f UTF-32LE -t UTF-8 ' + file_lo + ' > ' + new_file
            os.system(script)
            os.remove(file_lo)
            file_lo = new_file

    los = io.open(file_lo, encoding='utf-8').read().split('\n')

    # translator = google_translator()

    f = open(args.output, 'w')

    for line in los:
        new_line = clean_text(line)
        if new_line.strip() != '':
            f.write(GoogleTranslator(source='lo', target='vi').translate(text=new_line))
            f.write('\n')
    f.close()

    # for i in range(len(los)):
        # vi = translator.translate(clean_text(los[i]),lang_tgt='vi')
        # if abs(len(vi) - len(los[i])) > 50 and len(vi) < len(los[i]):
        #     new_s = word_tokenize(vi)
        #     new_s = ' '.join(new_s[:round(len(new_s) / 2)])
        #     new_lo = los[i][len(new_s):]
        #     vi = new_s + ' ' + translator.translate(new_lo,lang_tgt='vi')

        # f.write(clear_text(vi))
        # f.write('\n')
    # f.close()