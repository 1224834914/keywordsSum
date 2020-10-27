#coding:utf-8
import re
import jieba
from gensim.models import word2vec
import sys
import numpy as np
import random
import sys
import os
import hashlib
import struct
import subprocess
import collections
import tensorflow as tf
from tensorflow.core.example import example_pb2
from textrank4zh import TextRank4Keyword

SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

VOCAB_SIZE = 200000
CHUNK_SIZE = 1000

def get_abstrct_keywords(str):
    keywords = ''
    word = TextRank4Keyword()
    word.analyze(str,window = 2)
    w_list = word.get_keywords(num = 8,word_min_len = 1)
    for w in w_list:
        keywords = keywords + ' ' +w.word
    return keywords[1:],w_list

def readTrain(inFile,outFile):
    reader = open(inFile, 'r',encoding = 'UTF-8')
    contentlines = reader.readlines()
    writer = open(outFile, 'w',encoding = 'UTF-8')
    for str in contentlines:
        str = str.encode('utf-8').decode('unicode_escape')#将Unicode转为汉字
        str = re.sub("\n","",str)#去掉得到字符串的换行符
        str = re.sub("\r","",str)#去掉得到字符串的换行符
        strlist = re.findall('{"summarization": "(.+)", "article": "(.+)"}',str)
        t_utf1 = re.sub("<","。",strlist[0][1])
        t_utf = re.sub("[A-Z|a-z>]","",t_utf1)
        input_string = t_utf + '@abstract' + strlist[0][0] + '\n'
        writer.write(input_string)
    reader.close()
    writer.close()

def readVal(inFile,outFile):
    reader = open(inFile, 'r',encoding = 'UTF-8')
    contentlines = reader.readlines()
    writer = open(outFile, 'w',encoding = 'UTF-8')
    for str in contentlines:
        str = re.sub("\n","",str)#去掉得到字符串的换行符
        str = re.sub("\r","",str)#去掉得到字符串的换行符
        strlist = re.findall('{"summarization": "(.+)", "article": "(.+)", "index": (.+)}',str)           
        t_utf1 = re.sub("<","。",strlist[0][1])
        t_utf = re.sub("[A-Z|a-z>]","",t_utf1)
        input_string = t_utf + '@abstract' + strlist[0][0] + '\n'
        writer.write(input_string)
    reader.close()
    writer.close()

def readTest(inFile,outFile):
    reader = open(inFile, 'r',encoding = 'utf-8')
    contentlines = reader.readlines()
    writer = open(outFile, 'w',encoding = 'UTF-8')
    for str in contentlines:
        str = re.sub("\n","",str)#去掉得到字符串的换行符
        str = re.sub("\r","",str)#去掉得到字符串的换行符
        #strlist = re.findall('"summarization": "", "article": "(.+)"', str)
        strlist = re.findall('{"summarization": "", "article": "(.+)", "index": (.+)}',str)
        t_utf1 = re.sub("<","。",strlist[0][0])
        t_utf = re.sub("[A-Z|a-z>]","",t_utf1)
        input_string = t_utf + '@abstract' +  '\n'
        writer.write(input_string)
    reader.close()
    writer.close()

def wordSeg(file_name,segFile):
    reader = open(file_name, 'r',encoding = 'UTF-8')
    Writer = open(segFile, 'w',encoding = 'UTF-8')
    content = reader.readlines()
    for i in range(len(content)):
        terms = jieba.cut(content[i],cut_all=False)
        #print(terms)
        Writer.write(" ".join(terms))
        #Writer.write('\n')
    reader.close()
    Writer.close()
    print("-----wordSeg--end--------") 

def write_to_bin(seg_file, out_file, makevocab=False):
    print("Making bin file..." )
    reader = open(seg_file, 'r',encoding = 'UTF-8')
    dataLines = reader.readlines()    
    dataNum = len(dataLines)
    print(dataNum)#50000
    if makevocab:
        vocab_counter = collections.Counter()

    with open(out_file, 'wb') as writer:
        for idx,s in enumerate(dataLines):
            if idx % 100 == 0:
                print("Writing story %i of %i; %.2f percent done" % (idx, dataNum, float(idx)*100.0/float(dataNum)))

            # Get the strings to write to .bin file
            article_abstract_List = s.split('@ abstract')
            article = article_abstract_List[0]
            abstract = SENTENCE_START + ' '+ article_abstract_List[1][:-1] + SENTENCE_END + ' '+ '\n'

            art_temp = article.replace(' ','')
            keywords,keywords_list = get_abstrct_keywords(art_temp)

            if(len(keywords_list)) < 8:
                print(len(keywords_list),idx)

            # Write to tf.Example
            tf_example = example_pb2.Example()
            tf_example.features.feature['article'].bytes_list.value.extend([article.encode()])
            tf_example.features.feature['abstract'].bytes_list.value.extend([abstract.encode()])
            tf_example.features.feature['keywords'].bytes_list.value.extend([keywords.encode()])
            tf_example_str = tf_example.SerializeToString()
            str_len = len(tf_example_str)
            writer.write(struct.pack('q', str_len))
            writer.write(struct.pack('%ds' % str_len, tf_example_str))

            # Write the vocab to file, if applicable
            if makevocab:
                art_tokens = article.split(' ')
                abs_tokens = abstract.split(' ')
                abs_tokens = [t for t in abs_tokens if t not in [SENTENCE_START, SENTENCE_END]] # remove these tags from vocab
                tokens = art_tokens + abs_tokens
                tokens = [t.strip() for t in tokens] # strip
                tokens = [t for t in tokens if t!=""] # remove empty
                vocab_counter.update(tokens)
    print("Finished writing file %s\n" % out_file)

    #write vocab to file
    if makevocab:
        print("Writing vocab file...")
        with open(os.path.join(finished_files_dir, "vocab"), 'w',encoding= 'utf-8') as writer:
            for word, count in vocab_counter.most_common(VOCAB_SIZE):
                writer.write(word + ' ' + str(count) + '\n')
        print("Finished writing vocab file")

def chunk_file(set_name):
  in_file = 'data/finished_files/%s.bin' % set_name
  reader = open(in_file, "rb")
  chunk = 0
  finished = False
  while not finished:
    chunk_fname = os.path.join(chunks_dir, '%s_%03d.bin' % (set_name, chunk)) # new chunk
    with open(chunk_fname, 'wb') as writer:
      for _ in range(CHUNK_SIZE):
        len_bytes = reader.read(8)
        if not len_bytes:
          finished = True
          break
        str_len = struct.unpack('q', len_bytes)[0]
        example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
        writer.write(struct.pack('q', str_len))
        writer.write(struct.pack('%ds' % str_len, example_str))
      chunk += 1

def chunk_train():
  # Make a dir to hold the chunks
  if not os.path.isdir(chunks_dir):
    os.mkdir(chunks_dir)
  # Chunk the data
  for set_name in ['train']:
    print("Splitting %s data into chunks..." % set_name)
    chunk_file(set_name)
  print("Saved chunked data in %s" % chunks_dir)

def chunk_val():
  # Make a dir to hold the chunks
  if not os.path.isdir(chunks_dir):
    os.mkdir(chunks_dir)
  # Chunk the data
  for set_name in ['val']:
    print("Splitting %s data into chunks..." % set_name)
    chunk_file(set_name)
  print("Saved chunked data in %s" % chunks_dir)

def chunk_test():
  # Make a dir to hold the chunks
  if not os.path.isdir(chunks_dir):
    os.mkdir(chunks_dir)
  # Chunk the data
  for set_name in ['test']:
    print("Splitting %s data into chunks..." % set_name)
    chunk_file(set_name)
  print("Saved chunked data in %s" % chunks_dir)

#原始文件
train_file = 'data/train_with_summ.txt'
evaluation_file = 'data/evaluation_with_ground_truth.txt'
test_file = 'data/tasktestdata03.txt'
new_test_file = 'data/new_test.txt'
#数据清理后的文件存放的目录
finished_files_dir = "data/finished_files"
#分区文件夹
chunks_dir = os.path.join(finished_files_dir, "chunked")

#临时文件
train50000_file = 'data/train50000.txt'
train50000_seg_file = 'data/train50000_seg.txt'
evaluation2000_file = 'data/evaluation2000.txt'
evaluation2000_seg_file = 'data/evaluation2000_seg.txt'
test2000_file = 'data/test2000.txt'
test2000_seg_file = 'data/test2000_seg.txt'

new_test2000_file = 'data/new_test2000.txt'

if __name__ == '__main__':
    # ==========处理训练集=======
    print('begin to wash train data...')
    readTrain(train_file,train50000_file)
    wordSeg(train50000_file,train50000_seg_file)
    write_to_bin(train50000_seg_file,os.path.join(finished_files_dir, "train.bin"), makevocab=True)
    chunk_train()
    print('finished wash train data !')
    # ==========处理验证集=======
    print('begin to wash evaluation data...')
    readVal(evaluation_file,evaluation2000_file)
    wordSeg(evaluation2000_file,evaluation2000_seg_file)
    write_to_bin(evaluation2000_seg_file,os.path.join(finished_files_dir, "val.bin"))
    chunk_val()
    print('finished wash evaluation data !')
    # ==========处理测试集=======
    print('begin to wash test data...')
    readTest(new_test_file,new_test2000_file)
    wordSeg(test2000_file,test2000_seg_file)
    write_to_bin(test2000_seg_file,os.path.join(finished_files_dir, "test.bin"), makevocab=False)
    chunk_test()
    print('finished wash test data !')


