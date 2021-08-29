import numpy as np
import logging
import re
import sys
from ast import literal_eval
from collections import OrderedDict
from math import ceil
from time import time
from numpy import dot
from numpy.linalg import norm
import argparse
import os
import io
import re
from underthesea import sent_tokenize

from dp_utils import sentence_segment_vi, split_sents_lo

def overlap(path, lang, isSegmentSentence):
    
    lines = io.open(path, encoding='utf-8').read().split('\n')
    # lines = clean_text(lines)
    
    if isSegmentSentence is False:
        if lang == 'vi':
            lines = sentence_segment_vi(lines)
        elif lang == 'lo':
            lines = split_sents_lo(lines)

    result = []
    id_visted = 0
    n = 2
    
    if lines[-1].strip() == '':
        lines = lines[:-1]
    while id_visted < len(lines):
        sub_sent_n_len = ''
        id_sub_visted = id_visted
        if (id_sub_visted + n) >= len(lines):
            n = len(lines) - id_sub_visted
        result.append(lines[id_sub_visted].strip())
        sub_sent_n_len += str(lines[id_sub_visted]) + ' '
        for ii in range(id_sub_visted+1, id_sub_visted + n, 1):
            sub_sent_n_len += str(lines[ii]) + ' '
            result.append(str(lines[id_sub_visted]).strip() + ' ' + str(lines[ii]).strip())
        if sub_sent_n_len.strip() not in result:
            result.append(sub_sent_n_len.strip())
        id_visted += 1
    return result

def read_in_embeddings(text_file, embed_file):

    line_embeddings = np.fromfile(embed_file, dtype=np.float32, count=-1)
    if line_embeddings.size == 0:
        raise Exception('Got empty embedding file')

    laser_embedding_size = 1024
    line_embeddings.resize(line_embeddings.shape[0] // laser_embedding_size, laser_embedding_size)
    return line_embeddings

def create_mark_id_list(length):
    idx = 0
    mark_id = []
    for i in range(length):
        if i % 2 == 0:
            idx = 0
        mark_id.append(idx)
        idx += 1
    return mark_id

def align(vis, trans2vis, raw_vis, los, emb_vi, emb_trans2vi):
    mark_vi = [False] * len(vis)
    mark_trans2vi = [False] * len(trans2vis)
    mark_id_vis = create_mark_id_list(emb_vi.shape[0])
    mark_id_trans2vis = create_mark_id_list(emb_trans2vi.shape[0])

    result_scores = []
    result_vis = []
    result_trans2vis = []
    result_len = []
    result_los = []
    result_raw_vis = []

    for id_visted_vi in range(0, emb_vi.shape[0], 2):
        if mark_vi[id_visted_vi] is True:
            continue
        
        for id_visted_trans2vi in range(0, emb_trans2vi.shape[0], 2):
            if mark_trans2vi[id_visted_trans2vi] is True:
                continue
                
            cos_sim = dot(emb_vi[id_visted_vi], emb_trans2vi[id_visted_trans2vi].transpose())/(norm(emb_vi[id_visted_vi])*norm(emb_trans2vi[id_visted_trans2vi]))

            if cos_sim > 0.9:

                result_scores.append(cos_sim)
                result_vis.append(vis[id_visted_vi])
                result_raw_vis.append(raw_vis[id_visted_vi])

                result_trans2vis.append(trans2vis[id_visted_trans2vi])
                result_los.append(los[id_visted_trans2vi])
                result_len.append(str(len(vis[id_visted_vi].split(' '))) + '  ' + str(len(trans2vis[id_visted_trans2vi])))

                for i in range(1, mark_id_vis[id_visted_vi]+1):
                    left = id_visted_vi - i
                    if left < 0:
                        left = 0
                    mark_vi[left] = True

                for i in range(1, 2 - mark_id_vis[id_visted_vi], 1):
                    right = id_visted_vi + i
                    if right >= emb_vi.shape[0]:
                        right = emb_vi.shape[0]-1
                    mark_vi[right] = True


                for i in range(1, mark_id_trans2vis[id_visted_trans2vi]+1):
                    left = id_visted_trans2vi - i
                    if left < 0:
                        left = 0
                    mark_trans2vi[left] = True

                for i in range(1, 2 - mark_id_trans2vis[id_visted_trans2vi], 1):
                    right = id_visted_trans2vi + i
                    if right >= emb_trans2vi.shape[0]:
                        right = emb_trans2vi.shape[0]-1
                    mark_trans2vi[right] = True

                mark_vi[id_visted_vi] = True
                mark_trans2vi[id_visted_trans2vi] = True

                break
        

    id_visted_vi = 0
    id_visted_trans2vi = 0
    for id_visted_vi in range(0, emb_vi.shape[0], 1):
        if mark_vi[id_visted_vi] is True:
            continue
        for id_visted_trans2vi in range(0, emb_trans2vi.shape[0], 1):
            if mark_trans2vi[id_visted_trans2vi] is True:
                continue
            cos_sim = dot(emb_vi[id_visted_vi], emb_trans2vi[id_visted_trans2vi].transpose())/(norm(emb_vi[id_visted_vi])*norm(emb_trans2vi[id_visted_trans2vi]))

            if cos_sim >= 0.85:

                
                if id_visted_vi % 2 != 0:
                    result_scores.append(cos_sim)
                    result_scores.append(cos_sim)
                    
                    result_vis.append(vis[id_visted_vi - 1])
                    result_vis.append(vis[id_visted_vi])

                    result_raw_vis.append(raw_vis[id_visted_vi - 1])
                    result_raw_vis.append(raw_vis[id_visted_vi])
                else:
                    result_scores.append(cos_sim)
                    result_vis.append(vis[id_visted_vi])
                    result_raw_vis.append(raw_vis[id_visted_vi])
                if id_visted_trans2vi % 2 != 0:
                    result_trans2vis.append(trans2vis[id_visted_trans2vi - 1])
                    result_trans2vis.append(trans2vis[id_visted_trans2vi])

                    result_los.append(los[id_visted_trans2vi -1])
                    result_los.append(los[id_visted_trans2vi])
                else:
                    result_trans2vis.append(trans2vis[id_visted_trans2vi])
                    result_los.append(los[id_visted_trans2vi])
                
                result_len.append(str(len(vis[id_visted_vi].split(' '))) + '  ' + str(len(trans2vis[id_visted_trans2vi])))

                for i in range(1, mark_id_vis[id_visted_vi]+1):
                    left = id_visted_vi - i
                    if left < 0:
                        left = 0
                    mark_vi[left] = True

                for i in range(1, 2 - mark_id_vis[id_visted_vi], 1):
                    right = id_visted_vi + i
                    if right >= emb_vi.shape[0]:
                        right = emb_vi.shape[0]-1
                    mark_vi[right] = True


                for i in range(1, mark_id_trans2vis[id_visted_trans2vi]+1):
                    left = id_visted_trans2vi - i
                    if left < 0:
                        left = 0
                    mark_trans2vi[left] = True

                for i in range(1, 2 - mark_id_trans2vis[id_visted_trans2vi], 1):
                    right = id_visted_trans2vi + i
                    if right >= emb_trans2vi.shape[0]:
                        right = emb_trans2vi.shape[0]-1
                    mark_trans2vi[right] = True

                mark_vi[id_visted_vi] = True
                mark_trans2vi[id_visted_trans2vi] = True

                break
                
    return result_vis, result_trans2vis, result_raw_vis, result_los

def align_paragraphs(file_vi, file_lo_trans2vi, file_vi_raw, file_lo, embed_tmp_dir, return_new_file=True, isSegmentSentence=True):
    new_file_vi = embed_tmp_dir + '/' + file_vi.split('/')[-1] + '.align_paragraphs'
    new_file_vi_raw = embed_tmp_dir + '/' + file_vi_raw.split('/')[-1] + '.align_paragraphs'
    new_file_lo_trans2vi = embed_tmp_dir + '/' + file_lo_trans2vi.split('/')[-1] + '.align_paragraphs'
    new_file_lo = embed_tmp_dir + '/' + file_lo.split('/')[-1] + '.align_paragraphs'

    file_vi_overlap = new_file_vi + '.overlap'
    file_vi_overlap_raw = new_file_vi_raw + '.overlap'
    file_lo_trans2vi_overlap = new_file_lo_trans2vi + '.overlap'
    file_lo_overlap = new_file_lo + '.overlap'

    file_vi_overlap_emb = file_vi_overlap + '.emb'
    file_lo_trans2vi_overlap_emb = file_lo_trans2vi_overlap + '.emb'

    vis = overlap(file_vi, 'vi', isSegmentSentence)
    vis_raw = overlap(file_vi_raw, 'vi', isSegmentSentence)
    lo_trans2vis = overlap(file_lo_trans2vi, 'vi', isSegmentSentence)
    los = overlap(file_lo, 'vi', isSegmentSentence)

    f_overlap_vi = open(file_vi_overlap, 'w')
    f_overlap_vi_raw = open(file_vi_overlap_raw, 'w')
    f_overlap_lo_trans2vi = open(file_lo_trans2vi_overlap, 'w')
    f_overlap_lo = open(file_lo_overlap, 'w')

    for line in vis:
        # if len(line.strip()) > 2:
        if line.strip() != '':
            f_overlap_vi.write(line.strip() + '\n')
    f_overlap_vi.close()

    for line in vis_raw:
        # if len(line.strip()) > 2:
        if line.strip() != '':
            f_overlap_vi_raw.write(line.strip() + '\n')
    f_overlap_vi_raw.close()

    for line in lo_trans2vis:
        if line.strip() != '':
            f_overlap_lo_trans2vi.write(line.strip() + '\n')
    f_overlap_lo_trans2vi.close()

    for line in los:
        if line.strip() != '':
            f_overlap_lo.write(line.strip() + '\n')
    f_overlap_lo.close()

    script_emb_vi = 'bash $LASER/tasks/embed/embed.sh ' + file_vi_overlap + ' vi ' + file_vi_overlap_emb
    script_emb_lo_trans2vi = 'bash $LASER/tasks/embed/embed.sh ' + file_lo_trans2vi_overlap + ' vi ' + file_lo_trans2vi_overlap_emb

    os.system(script_emb_vi)
    os.system(script_emb_lo_trans2vi)

    emb_vi = read_in_embeddings(file_vi_overlap, file_vi_overlap_emb)
    emb_lo_trans2vi = read_in_embeddings(file_lo_trans2vi_overlap, file_lo_trans2vi_overlap_emb)

    result_vis, result_lo_trans2vis, result_vis_raw, result_los = align(vis, lo_trans2vis, vis_raw, los, emb_vi, emb_lo_trans2vi)

    raw_vis = io.open(file_vi, encoding='utf-8').read().split('\n')
    raw_vis_raw = io.open(file_vi_raw, encoding='utf-8').read().split('\n')
    raw_lo_trans2vis = io.open(file_lo_trans2vi, encoding='utf-8').read().split('\n')
    raw_los = io.open(file_lo, encoding='utf-8').read().split('\n')

    for line in raw_vis:
        if line not in result_vis:
            result_vis.append(line)
    for line in raw_lo_trans2vis:
        if line not in result_lo_trans2vis:
            result_lo_trans2vis.append(line)

    for line in raw_vis_raw:
        if line not in result_vis_raw:
            result_vis_raw.append(line)
    for line in raw_los:
        if line not in result_los:
            result_los.append(line)

    f_new_vi = open(new_file_vi, 'w')
    f_new_vi_raw = open(new_file_vi_raw, 'w')
    f_new_lo_trans2vi = open(new_file_lo_trans2vi, 'w')
    f_new_lo = open(new_file_lo, 'w')
    for line in result_vis:
        if line.strip() != '':
            f_new_vi.write(line + '\n')
    for line in result_vis_raw:
        if line.strip() != '':
            f_new_vi_raw.write(line + '\n')

    for line in result_lo_trans2vis:
        if line.strip() != '':
            f_new_lo_trans2vi.write(line + '\n')
    for line in result_los:
        if line.strip() != '':
            f_new_lo.write(line + '\n')

    f_new_vi.close()
    f_new_lo_trans2vi.close()
    f_new_lo.close()
    f_new_vi_raw.close()

    # print (result_los)
    # print ('='*60)
    # print (result_vis_raw)

    if return_new_file is True:
        return new_file_vi, new_file_lo_trans2vi, new_file_vi_raw, new_file_lo

# if __name__ == '__main__':
#     file_vi = args.file_vi
#     file_trans2vi = args.file_trans2vi
#     dir_embed = args.dir_emb
#     align_paragraphs(file_vi, file_trans2vi, dir_embed, return_new_file=False, isSegmentSentence=False)