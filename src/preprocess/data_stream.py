from http.client import CONTINUE
import json
import codecs
import os
import penman
import torch
import codecs
import sys
import re
from collections import Counter
from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm
from pathlib import Path
import string
import nltk
nltk.download('averaged_perceptron_tagger')

json_load = lambda x: json.load(codecs.open(x, 'r', encoding='utf-8'))
#json_load = lambda x: json.load(codecs.open(x, 'r', encoding='utf-8'),strict=False)
json_dump = lambda d, p: json.dump(d, codecs.open(p, 'w', 'utf-8'), indent=2, ensure_ascii=False)

def check_sim(word1, word2):
  if(word1 == word2):
    return True
  return False

class AMRSentence :
  node = []
  edge = []
  text = []
  length = 0
  def __init__(self, node, edge, text):
    self.node = node
    self.edge = edge
    self.text = text
    self.length = len(self.node)
  
  def connect(self,s):
    #######################
    #if self.length != 0:
    #  id_st = self.length - 1
    #  edge = s.edge[0] 
    #  node_s = s.node[edge[0]] 
    #  if node_s in self.node:
    #    k = self.node.index(node_s)
    #    ed = k
    #  else:
    #    self.node.append(node_s)
    #    self.length += 1
    #    ed = s.node.index(node_s)  
    #  new_edge = (id_st, ed, ':nei')
    #  self.edge.append(new_edge)
    #else:
    #  edge = s.edge[0] 
    #  node_s = s.node[edge[0]]
    #  self.node.append(node_s)
    #  self.length += 1   

    for i in range(len(s.edge)):
      edge = s.edge[i] 
      node_s = s.node[edge[0]] 
      if node_s in self.node:
        k = self.node.index(node_s)
        st = k
      else:  
        self.node.append(node_s)
        self.length += 1
        st = self.node.index(node_s)
        
      node_e = s.node[edge[1]] 
      if node_e in self.node:
        k = self.node.index(node_e)
        ed = k
      else: 
        self.node.append(node_e)
        self.length += 1
        ed = self.node.index(node_e)
              
      new_edge = (st, ed, edge[2])
      self.edge.append(new_edge)

    for i in range(len(s.text)):
      self.text.append(s.text[i])
  
def read_anonymized(amr_lst, amr_node, amr_edge):

    #assert sum(x=='(' for x in amr_lst) == sum(x==')' for x in amr_lst)  #判断（）是否匹配
    cur_str = amr_lst[0]
    cur_id = len(amr_node)
    amr_node.append(cur_str)

    i = 1
    while i < len(amr_lst):
        if amr_lst[i].startswith(':') == False: ## cur cur-num_0
            nxt_str = amr_lst[i]
            nxt_id = len(amr_node)
            amr_node.append(nxt_str)
            amr_edge.append((cur_id, nxt_id, ':value'))
            i = i + 1
        elif amr_lst[i].startswith(':') and len(amr_lst) == 2: ## cur :edge
            nxt_str = 'num_unk'
            nxt_id = len(amr_node)
            amr_node.append(nxt_str)
            amr_edge.append((cur_id, nxt_id, amr_lst[i]))
            i = i + 1
        elif amr_lst[i].startswith(':') and amr_lst[i+1] != '(': ## cur :edge nxt
            nxt_str = amr_lst[i+1]
            nxt_id = len(amr_node)
            amr_node.append(nxt_str)
            amr_edge.append((cur_id, nxt_id, amr_lst[i]))
            i = i + 2
        elif amr_lst[i].startswith(':') and amr_lst[i+1] == '(': ## cur :edge ( ... )
            number = 1
            j = i+2
            while j < len(amr_lst):
                number += (amr_lst[j] == '(')
                number -= (amr_lst[j] == ')')
                if number == 0:
                    break
                j += 1
            assert number == 0 and amr_lst[j] == ')', ' '.join(amr_lst[i+2:j])
            nxt_id = read_anonymized(amr_lst[i+2:j], amr_node, amr_edge)
            amr_edge.append((cur_id, nxt_id, amr_lst[i]))
            i = j + 1
        else:
            assert False, ' '.join(amr_lst)
    return cur_id

def is_match(amr_lst):
    left = 0
    right = 0
    for x in amr_lst:
      if(x == '('):
        left += 1
      if(x == ')'):
        right += 1
    #print(left,right)
    if(left != right):
      return False 
    return True

def read_amr_file(inpath):
    punctuation_string = string.punctuation
    data = json_load(inpath)
    nodes = [] # [batch, node_num,]
    in_neigh_indices = [] # [batch, node_num, neighbor_num,]
    in_neigh_edges = []
    out_neigh_indices = [] # [batch, node_num, neighbor_num,]
    out_neigh_edges = []
    sources = [] # [batch, sent_length,]
    sentences = [] # [batch, tgt_length]
    pos_tags = []
    answer = []
    max_in_neigh = 0
    max_out_neigh = 0
    max_node = 0
    max_sent = 0

    samples = tqdm(data, desc='  - (GENERATING NODE AND EDGE) -  ')
    
    for sample in samples:
      #print(sample)
      amr_node = []
      amr_edge = []
      src = []
      pos_tag = []
      #if(config.ADD_TITLE):  ###添加title信息
        #title_node = sample["title"].strip().split()
        #title_edge = 
      #ss = AMRSentence(title_node, title_edge, [])

      ss = AMRSentence([], [], [])
      flag = True
      sent = sample["question"]
      ans = sample["answer"].strip().split()
      for evi in sample["evidence"]:
        amr = evi["amr"]
        #print(punctuation_string)

        txt = evi["text"][0]
        for i in punctuation_string:
          txt = str(txt).replace(i, '')
        text = txt.split() 

        title = evi["title"]
        node = []
        edge = []
        amr_lst = amr.strip().split()
        #print('amr:', amr_lst)
        #print('text:', text)
        if(amr_lst[0] == 'FAILED_TO_PARSE'):
          continue
        else:
          if(is_match(amr_lst)):
            read_anonymized(amr_lst, node, edge)
            #print("node:", node)
            #amr_node.append(node)
            #amr_edge.append(edge)          
            src += text
            #print(text)
            #print(node)
            #print(edge)
            if(len(edge) > 0):
              s = AMRSentence(node, edge, text)
              ss.connect(s)
          else:
          #解析的AMR不合标准
            #add_node(text, node, edge)  
              continue
          #flag = False
        #print(ss.length)
      if(len(ss.node)!=0):
        nodes.append(ss.node)
        #print('node:', ss.node)
        #print("edge:", ss.edge)

        in_indices = [[i,] for i, x in enumerate(ss.node)]
        in_edges = [[':self',] for i, x in enumerate(ss.node)]
        out_indices = [[i,] for i, x in enumerate(ss.node)]
        out_edges = [[':self',] for i, x in enumerate(ss.node)]
        for (i,j,lb) in ss.edge:
          in_indices[j].append(i)
          in_edges[j].append(lb)
          out_indices[i].append(j)
          out_edges[i].append(lb)

        #print(ss.node)
        #print(ss.edge)
        #print(in_indices)
        #print(in_edges)
        #print("================")
        in_neigh_indices.append(in_indices)
        in_neigh_edges.append(in_edges)
        out_neigh_indices.append(out_indices)
        out_neigh_edges.append(out_edges)

        for i in punctuation_string:
          sent = str(sent).replace(i, '')
        #print('src:', src)
        #print(sent.split())
        pos = nltk.pos_tag(src)
        for data  in pos:
          pos_tag.append(data[1])
        #print(pos_tag)
        pos_tags.append(pos_tag)
        sources.append(src)
        sentences.append(sent.split())
        answer.append(ans)

        max_in_neigh = max(max_in_neigh, max(len(x) for x in in_indices))
        max_out_neigh = max(max_out_neigh, max(len(x) for x in out_indices))
        max_node = max(max_node, len(ss.node))
        max_sent = max(max_sent, len(sent))
    return zip(nodes, in_neigh_indices, in_neigh_edges, out_neigh_indices, out_neigh_edges, sources, sentences, answer, pos_tags), \
            max_node, max_in_neigh, max_out_neigh, max_sent 

if __name__ == '__main__':
  read_amr_file(r'/data1/lkx/cs/qg/data/train_data/mini/valid_with_simple_amr.json')