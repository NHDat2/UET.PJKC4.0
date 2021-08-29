import io
import os
import shutil
import glob
import numpy as np
from numpy import dot
from numpy.linalg import norm
import argparse
from random import seed as seed
import re
from dp_utils import clean_text_v2, split_sents_lo
from shutil import copyfile
from align_paragraphs import align_paragraphs
import codecs
import random
from underthesea import sent_tokenize

parser = argparse.ArgumentParser('Sentence alignment using sentence embeddings and FastDTW',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--src_file', type=str, required=True,
                    help='')

parser.add_argument('--tgt_file', type=str, required=True,
                    help='')

parser.add_argument('-o', '--output', type=str, required=True,
                    help='folder is file output aligned')

args = parser.parse_args()

def split_sents(path_temp_vi, path_temp_lo, embdir):
    tmp_vi = io.open(path_temp_vi, encoding='utf-8').read().split('\n')
    tmp_lo = io.open(path_temp_lo, encoding='utf-8').read().split('\n')

    path_vi = embdir + '/' + path_temp_vi.split('/')[-1]
    path_lo = embdir + '/' + path_temp_lo.split('/')[-1]

    f_vi = open(path_vi, 'w')
    for line in tmp_vi:
        if sent_tokenize(line) != []:
            for l in sent_tokenize(line):
                if l.strip() != '':
                    l = l.strip()
                    if l[-1] != '.':
                        l += '.'
                    f_vi.write(l.strip())
                    f_vi.write('\n')
    f_vi.close()

    f_lo = open(path_lo, 'w')
    for line in tmp_lo:
        for l in split_sents_lo(line):
            if l.strip() != '':
                l = l.strip()
                if l[-1] != '.':
                    l += '.'
                f_lo.write(l.strip())
                f_lo.write('\n')
    f_lo.close()
    return path_vi, path_lo

def preprocess_line(line):
    line = line.strip()
    if len(line) == 0:
        line = 'BLANK_LINE'
    return line

def layer(lines, num_overlaps, comb=' '):
    """
    make front-padded overlapping sentences
    """
    if num_overlaps < 1:
        raise Exception('num_overlaps must be >= 1')
    out = ['PAD', ] * min(num_overlaps - 1, len(lines))
    for ii in range(len(lines) - num_overlaps + 1):
        out.append(comb.join(lines[ii:ii + num_overlaps]))
    return out

def read_in_embeddings(text_file, embed_file):
    """
    Given a text file with candidate sentences and a corresponing embedding file,
       make a maping from candidate sentence to embedding index, 
       and a numpy array of the embeddings
    """
    sent2line = dict()
    with open(text_file, 'rt', encoding="utf-8") as fin:
        for ii, line in enumerate(fin):
#             if line.strip() in sent2line:
#                 raise Exception('got multiple embeddings for the same line')
            sent2line[line.strip()] = ii

    line_embeddings = np.fromfile(embed_file, dtype=np.float32, count=-1)
    if line_embeddings.size == 0:
        raise Exception('Got empty embedding file')

    laser_embedding_size = line_embeddings.size // len(sent2line)  # currently hardcoded to 1024
    if laser_embedding_size != 1024:
        laser_embedding_size = 1024
#     logger.info('laser_embedding_size determined to be %d', laser_embedding_size)
    line_embeddings.resize(line_embeddings.shape[0] // laser_embedding_size, laser_embedding_size)
    return sent2line, line_embeddings

def make_doc_embedding(sent2line, line_embeddings, lines, num_overlaps):
    """
    lines: sentences in input document to embed
    sent2line, line_embeddings: precomputed embeddings for lines (and overlaps of lines)
    """

    lines = [preprocess_line(line) for line in lines]

    vecsize = line_embeddings.shape[1]

    vecs0 = np.empty((num_overlaps, len(lines), vecsize), dtype=np.float32)

    for ii, overlap in enumerate(range(1, num_overlaps + 1)):
        for jj, out_line in enumerate(layer(lines, overlap)):
            try:
                line_id = sent2line[out_line]
            except KeyError:
                # logger.warning('Failed to find overlap=%d line "%s". Will use random vector.', overlap, out_line)
                line_id = None

            if line_id is not None:
                vec = line_embeddings[line_id]
            else:
                vec = np.random.random(vecsize) - 0.5
                vec = vec / np.linalg.norm(vec)

            vecs0[ii, jj, :] = vec

    return vecs0

def make_norm1(vecs0):
    """
    make vectors norm==1 so that cosine distance can be computed via dot product
    """
    for ii in range(vecs0.shape[0]):
        for jj in range(vecs0.shape[1]):
            norm = np.sqrt(np.square(vecs0[ii, jj, :]).sum())
            vecs0[ii, jj, :] = vecs0[ii, jj, :] / (norm + 1e-5)

def embed_paragraph(path_vi, path_lo_trans2vi, embdir):
    path_emb_vi = embdir + '/embed.vi'
    path_emb_lo_trans2vi = embdir + '/embed.lo_trans2vi'
    
    script_vi = 'bash $LASER/tasks/embed/embed.sh ' + path_vi + ' vi ' + path_emb_vi
    script_lo_trans2vi = 'bash $LASER/tasks/embed/embed.sh ' + path_lo_trans2vi + ' vi ' + path_emb_lo_trans2vi

    os.system(script_vi)
    os.system(script_lo_trans2vi)
    
    return path_emb_vi, path_emb_lo_trans2vi

def cosim_two_embed(path_temp_vi, path_temp_lo_trans2vi, embdir):
    path_vi = path_temp_vi
    path_lo_trans2vi = path_temp_lo_trans2vi
    
    path_emb_vi, path_emb_lo_trans2vi = embed_paragraph(path_vi, path_lo_trans2vi, embdir)
    
    src_sent2line, src_line_embeddings = read_in_embeddings(path_vi, path_emb_vi)
    tgt_sent2line, tgt_line_embeddings = read_in_embeddings(path_lo_trans2vi, path_emb_lo_trans2vi)
    
    vi = io.open(path_vi, encoding='utf-8').read().split('\n')
    lo_trans2vi = io.open(path_lo_trans2vi, encoding='utf-8').read().split('\n')
    tmp_vi = io.open(path_temp_vi, encoding='utf-8').read().split('\n')
    tmp_lo_trans2vi = io.open(path_temp_lo_trans2vi, encoding='utf-8').read().split('\n')
    
    vecs_vis = make_doc_embedding(src_sent2line, src_line_embeddings, vi, 1)
    vecs_lo_trans2vis = make_doc_embedding(tgt_sent2line, tgt_line_embeddings, lo_trans2vi, 1)


    make_norm1(vecs_vis)
    make_norm1(vecs_lo_trans2vis)

    vecs_vi = vecs_vis[0]
    vecs_lo_trans2vi = vecs_lo_trans2vis[0]
    
    all_max_cos_sim = []
    vecs_emb_vi = []
    vecs_emb_lo_trans2vi = []

    id_sent_vi = 0
    id_vec_vi = 0
    while id_sent_vi < len(tmp_vi):

        vec_visted_vi = vecs_vi[id_vec_vi:id_vec_vi+len(sent_tokenize(tmp_vi[id_sent_vi]))]

        if vec_visted_vi.shape[0] > 0:
            nSentx = vec_visted_vi.shape[0]
            vec_x = np.average(vec_visted_vi, axis=0)
            vec_x = np.expand_dims(vec_x, axis=0)


            id_sent_lo_trans2vi = 0
            id_vec_lo_trans2vi = 0
            max_cos_similarity = 0.0
            while id_sent_lo_trans2vi < len(tmp_lo_trans2vi):
                split_lo_trans2vi = tmp_lo_trans2vi[id_sent_lo_trans2vi].split('。')
                while '' in split_lo_trans2vi:
                    split_lo_trans2vi.remove('')
                vec_visted_lo_trans2vi = vecs_lo_trans2vi[id_vec_lo_trans2vi:id_vec_lo_trans2vi+len(split_lo_trans2vi)]

                if vec_visted_lo_trans2vi.shape[0] > 0:
                    nSenty = vec_visted_lo_trans2vi.shape[0]
                    vec_y = np.average(vec_visted_lo_trans2vi, axis=0)
                    vec_y = np.expand_dims(vec_y, axis=0)

                    cos_x_y = dot(vec_x, vec_y.transpose())/(norm(vec_x)*norm(vec_y))

                    sum_x_ys = 0
                    for vec_ys in vec_visted_lo_trans2vi:
                        vec_ys = np.expand_dims(vec_ys, axis=0)
                        cos_x_ys = dot(vec_x, vec_ys.transpose())/(norm(vec_x)*norm(vec_ys))
                        sum_x_ys += 1 - cos_x_ys

                    sum_xs_y = 0
                    for vec_xs in vec_visted_vi:
                        vec_xs = np.expand_dims(vec_xs, axis=0)
                        cos_xs_y = dot(vec_xs, vec_y.transpose())/(norm(vec_xs)*norm(vec_y))
                        sum_xs_y += 1 - cos_xs_y

                    cos_similarity = (1 - cos_x_y)*nSentx*nSenty / (sum_x_ys + sum_xs_y)
                    cos_similarity = cos_similarity[0][0]

                    if cos_similarity == 0.5:
                        cos_similarity = cos_x_y[0][0]

                    if max_cos_similarity < cos_similarity:
                        max_cos_similarity = cos_similarity

                if len(split_lo_trans2vi) == 0:
                    count = 0
                else:
                    count = len(split_lo_trans2vi)
                id_sent_lo_trans2vi += 1
                id_vec_lo_trans2vi += count

            all_max_cos_sim.append(max_cos_similarity)



        if len(sent_tokenize(tmp_vi[id_sent_vi])) == 0:
            count = 0
        else:
            count = len(sent_tokenize(tmp_vi[id_sent_vi]))
        id_sent_vi += 1
        id_vec_vi += count
    return sum(all_max_cos_sim) / len(all_max_cos_sim)

def _main():

    embed_dir = '/tmp/embedd' + str(random.randint(0, 100))
    while os.path.isdir(embed_dir):
        embed_dir = '/tmp/embedd' + str(random.randint(0, 100))
    print ("path_to_embed_folder\t: ", embed_dir)
    os.mkdir(embed_dir)

    file_vi = args.src_file
    file_lo = args.tgt_file

    path_to_output = args.output + '/vi-lo'
    path_to_log_output = args.output + '/log_alignDocs'
    os.mkdir(path_to_output)
    os.mkdir(path_to_log_output)

    try:
        f = codecs.open(file_vi, encoding='utf-8', errors='strict')
        for line in f:
            pass
    except UnicodeDecodeError:
        try:
            f = codecs.open(file_vi, encoding='utf-16', errors='strict')
            for line in f:
                pass
            new_file = file_vi + '.utf8'
            script = 'iconv -f UTF-16LE -t UTF-8 ' + file_vi + ' > ' + new_file
            os.system(script)
            os.remove(file_vi)
            file_vi = new_file
        except UnicodeDecodeError:
            new_file = file_vi + '.utf8'
            script = 'iconv -f UTF-32LE -t UTF-8 ' + file_vi + ' > ' + new_file
            os.system(script)
            os.remove(file_vi)
            file_vi = new_file

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

    file_vi, file_lo = split_sents(file_vi, file_lo, embed_dir)

    print ("vietname file is visting: ", file_vi)
    print ("laos file is visting : ", file_lo)
    print ('='*60)
    f_log_pass = open(path_to_log_output + '/log_pass.txt', 'a')
    f_log_visted = open(path_to_log_output + '/log_visted.txt', 'a')

    if os.stat(file_vi).st_size != 0 and os.stat(file_lo).st_size != 0:
        path_temp_file_vi = embed_dir + '/' + clean_text_v2(file_vi).split('/')[-1] + '.vi'
        path_temp_file_lo_trans2vi = embed_dir + '/' + clean_text_v2(file_lo).split('/')[-1] + '.lo'

        #*************************************************************************************************
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
                "ABCDEFGHIJKLMNOPQRSTUVWXYZÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊ"
                "óòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ "
                "ÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ "
            )
            telex_words = list("áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựđÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰĐ")
            number = list("0123456789")
            alphabel = telex_words + letters + number
            s = s.lower()
            for c in s:
                if c not in alphabel:
                    s = s.replace(c, '')
            s = re.sub(" +", ' ', s)
            s = s.strip()
            return s

        f_path_temp_file_vi = open(path_temp_file_vi, 'w')
        lines_file_vi = io.open(file_vi, encoding='utf-8').read().lower().split('\n')
        lines_file_vi = io.open(file_vi, encoding='utf-8').read().split('\n')
        for line in lines_file_vi:
            f_path_temp_file_vi.write(clear_text(line))
            f_path_temp_file_vi.write('\n')
        f_path_temp_file_vi.close()
        #*************************************************************************************************

        script_trans2vi = 'python ' + os.environ['KC4ALIGN'] + '/trans_lo_vi.py -i ' + file_lo + ' -o ' + path_temp_file_lo_trans2vi
        os.system(script_trans2vi)

        cos_sim = cosim_two_embed(path_temp_file_vi, path_temp_file_lo_trans2vi, embed_dir)

        os.remove(embed_dir + '/embed.vi')
        os.remove(embed_dir + '/embed.lo_trans2vi')
        f_log_visted.write('file_vi: ' + file_vi.split('/')[-1] + '\n')
        f_log_visted.write('file_lo: ' + file_lo.split('/')[-1] + '\n')
        f_log_visted.write('score  : ' + str(cos_sim) + '\n')
        f_log_visted.write('='*60 + '\n')
        f_log_visted.close()
        if cos_sim > 0.7:
            f_log_pass.write('file_vi: ' + file_vi.split('/')[-1] + '\n')
            f_log_pass.write('file_lo: ' + file_lo.split('/')[-1] + '\n')
            f_log_pass.write('score  : ' + str(cos_sim) + '\n')
            f_log_pass.write('='*60 + '\n')
            f_log_pass.close()

            output_file = path_to_output + '/' + 'aligned.txt'

            path_temp_file_vi, path_temp_file_lo_trans2vi, path_temp_file_vi_raw, path_temp_file_lo = align_paragraphs(path_temp_file_vi, path_temp_file_lo_trans2vi, file_vi, file_lo, embed_dir)

            script = 'bash ' + os.environ['KC4ALIGN'] + '/align_sentences.sh ' + path_temp_file_vi + ' ' + path_temp_file_lo_trans2vi + ' ' + output_file + ' ' + path_temp_file_vi_raw + ' ' + path_temp_file_lo
            print (os.system(script))
            
            if os.stat(output_file).st_size == 0:
                os.remove(output_file)

            if os.path.isfile(path_temp_file_vi) and os.path.isfile(path_temp_file_lo_trans2vi):
                os.remove(path_temp_file_vi)
                os.remove(path_temp_file_lo_trans2vi)

            for f in os.listdir(embed_dir):
                os.remove(os.path.join(embed_dir, f))

            print ("="*30 + "Done" + "="*30)
        if os.path.isfile(path_temp_file_vi) and os.path.isfile(path_temp_file_lo_trans2vi):
            os.remove(path_temp_file_vi)
            os.remove(path_temp_file_lo_trans2vi)
    for f in os.listdir(embed_dir):
        os.remove(os.path.join(embed_dir, f))
    os.rmdir(embed_dir)

if __name__ == '__main__':
    if os.path.isdir(args.output):
        for f in os.listdir(args.output):
            shutil.rmtree(os.path.join(args.output, f))
    else:
        os.mkdir(args.output)
    _main()
