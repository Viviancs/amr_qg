from onqg.dataset.Vocab import Vocab
from onqg.dataset.Dataset import Dataset
from preprocess import data_stream
from onqg.dataset import Constants as Constants

from preprocess import opt
import torch
import sys

def get_graph_node_index(dataset, vocab):
  node_index = []
  return node_index

def load_vocab(filename):
  vocab_dict = {}
  with open(filename, 'r', encoding='utf-8') as f:
      text = f.read().strip().split('\n')
  text = [word.split(' ') for word in text]
  vocab_dict = {word[0]:word[1:] for word in text}
  vocab_dict = {k:[float(d) for d in v] for k,v in vocab_dict.items()}
  #print(vocab_dict)
  return vocab_dict

def convert_word_to_idx(text, vocab, lower=True, sep=False, pretrained=''):

  def lower_sent(sent):
      for idx, w in enumerate(sent):
          if w not in [Constants.BOS_WORD, Constants.EOS_WORD, Constants.PAD_WORD, Constants.UNK_WORD, Constants.SEP_WORD]:
              sent[idx] = w.lower()
      return sent
    
  def get_dict(sent, length, raw, separate=False):
      raw = raw.split(' ')
      bert_sent = [w for w in sent]
      ans_indexes = []
      if separate:
          sep_id = sent.index(Constants.SEP_WORD)
          sent = sent[:sep_id]
          ans_indexes = [i for i in range(sep_id + 1, len(bert_sent) - 1)]

      indexes = [[i for i in ans_indexes] for _ in raw]
      word, idraw = '', 0
      for idx, w in enumerate(sent):
          if word == raw[idraw] and idx != 0:
              idraw += 1
              while len(raw[idraw]) < len(w):
                  idraw += 1
              word = w
          else:
              word = word + w.lstrip('##')
              while len(raw[idraw]) < len(word):
                  idraw += 1
          indexes[idraw].append(idx)
            
      flags = [len(idx) > 0 for idx in indexes]
      return indexes

  if pretrained.count('bert'):
      lengths = [len(sent) for sent in text]
      text = [' '.join(sent) for sent in text]
      if sep:
          text = [sent.split(' ' + Constants.SEP_WORD + ' ') for sent in text]
          tokens = [vocab.tokenizer.tokenize(sent[0]) + [Constants.SEP_WORD] + vocab.tokenizer.tokenize(sent[1]) + [Constants.SEP_WORD] for sent in text]
          text = [sent[0] for sent in text]
          index_dict = [get_dict(sent, length, raw.lower(), separate=sep) for sent, length, raw in zip(tokens, lengths, text)]
      else:
          tokens = [vocab.tokenizer.tokenize(sent) for sent in text]
          index_dict = [get_dict(sent, length, raw.lower()) for sent, length, raw in zip(tokens, lengths, text)]
  else:
      index_dict = None
      tokens = [lower_sent(sent) for sent in text] if lower else text    

  indexes = [vocab.convertToIdx(sent) for sent in tokens]

  return indexes, tokens, index_dict

def get_embedding(vocab_dict, vocab):

  def get_vector(idx):
      word = vocab.idxToLabel[idx]
      if idx in vocab.special or word not in vocab_dict:
          vector = torch.tensor([])
          vector = vector.new_full((opt.word_vec_size,), 1.0)
          vector.normal_(0, math.sqrt(6 / (1 + vector.size(0))))
      else:
          vector = torch.Tensor(vocab_dict[word])
      return vector
    
  embedding = [get_vector(idx) for idx in range(vocab.size)]
  embedding = torch.stack(embedding)
    
  print(embedding.size())

  return embedding

def list_to_matrix(indices, edges):
  node_num = len(edges)
  
  fi_edge = []
  for s in range(node_num):
    a=[]
    for j in range(node_num): 
      a.append("[PAD]")
    fi_edge.append(a)
  
  for i in range(node_num):
    node = indices[i]
    edge = edges[i]
    for idx, e in zip(node, edge):  
      fi_edge[i][idx] = e
  return fi_edge

def get_data(dataset):
  final_indexes = []
  i = 0
  graph_dataset = {'indexes': [], 'is_ans': [], 'edge_in':[], 'edge_out':[]}
  rst = {'src':[], 'tgt':[], 'amr_node': [],'ans':[], 'is_ans':[], 'feature':[]}
  for nodes, in_neigh_indices, in_neigh_edges, out_neigh_indices, out_neigh_edges, sources, sentences, answer in dataset: 
    src_len = len(sources)
    tgt_len = len(sentences)
    if src_len <= opt.src_seq_length and tgt_len - 1 <= opt.tgt_seq_length:
      if src_len * tgt_len > 0 and src_len >= 10:
        rst['src'].append(sources)
        rst['tgt'].append([Constants.BOS_WORD] + sentences + [Constants.EOS_WORD])
        #if answer:
        rst['ans'].append(answer)
        rst['amr_node'].append(nodes)
        #print(sentences)
        sr_tags = []
        for w in enumerate(sources):
          if w in answer:
            sr_tag = 1
          else:
            sr_tag = 0
          sr_tags.append(sr_tag)

        node_indexes, ans_tags = [], []
        for idx, n in enumerate(nodes):
          node_indexes.append([idx])
          if n in answer:
            tag = 1
          else:
            tag = 0
          ans_tags.append(tag)
           

        final_indexes.append(i)
        i += 1

        #print(in_neigh_indices)
        #print(in_neigh_edges)
        edge_in = list_to_matrix(in_neigh_indices, in_neigh_edges)
        edge_out = list_to_matrix(out_neigh_indices, out_neigh_edges)

        graph_dataset['indexes'].append(node_indexes)
        graph_dataset['is_ans'].append(ans_tags)
        graph_dataset['edge_in'].append(edge_in)
        graph_dataset['edge_out'].append(edge_out)
        rst['is_ans'].append(sr_tags)
    #print(edge_in)

  graph_dataset['features'] = [graph_dataset['is_ans']]
  #graph_dataset['features'] = [graph_dataset['is_ans']]
  rst['features'] = [rst['is_ans']]

  return rst, graph_dataset, final_indexes

def wrap_copy_idx(splited, tgt, tgt_vocab, bert, vocab_dict):
    
  def map_src(sp):
      sp_split = {}
      if bert:
          tmp_idx = 0
          tmp_word = ''
          for i, w in enumerate(sp):
              if not w.startswith('##'):
                  if tmp_word:
                      sp_split[tmp_word] = tmp_idx
                  tmp_word = w
                  tmp_idx = i
              else:
                  tmp_word += w.lstrip('##')
          sp_split[tmp_word] = tmp_idx
      else:
          sp_split = {w:idx for idx, w in enumerate(sp)}
      return sp_split

  def wrap_sent(sp, t):
      sp_dict = map_src(sp)
      swt, cpt = [0 for w in t], [0 for w in t]
      for i, w in enumerate(t):
          if w not in tgt_vocab.labelToIdx or tgt_vocab.frequencies[tgt_vocab.labelToIdx[w]] <= 1:
          #if w not in tgt_vocab.labelToIdx or w not in vocab_dict or tgt_vocab.frequencies[tgt_vocab.labelToIdx[w]] <= 1:
              if w in sp_dict:
                  swt[i] = 1
                  cpt[i] = sp_dict[w]
      return torch.Tensor(swt), torch.LongTensor(cpt)
    
  copy = [wrap_sent(sp, t) for sp, t in zip(splited, tgt)]
  switch, cp_tgt = [c[0] for c in copy], [c[1] for c in copy]
  return [switch, cp_tgt]

def sequence_data(tr_dataset, va_dataset):
  train_data, train_graph, train_final_indexes = get_data(tr_dataset)
  valid_data, valid_graph, valid_final_indexes = get_data(va_dataset)

  train_src, train_tgt = train_data['src'], train_data['tgt']
  train_ans = train_data['ans'] 
  train_node = train_data['amr_node']

  valid_src, valid_tgt = valid_data['src'], valid_data['tgt']
  valid_ans = valid_data['ans']   
  valid_node = valid_data['amr_node']

  #========== build vocabulary ==========#
  print('Loading pretrained word embeddings ...')
  #pre_trained_vocab = None
  pre_trained_vocab = load_vocab(opt.pre_trained_vocab) if opt.pre_trained_vocab else None
  print('Done .')

  print("building src vocabulary")

  train_corpus = train_src + train_ans + train_node  #包含source/answer/graph中所有的word
  ans_corpus = train_src + train_ans
  options = {'lower':True, 'mode':'frequency', 'tgt':False, 'size':opt.src_vocab_size, 'frequency':opt.src_words_min_frequency}
  
  src_vocab = Vocab.from_opt(corpus=train_corpus, opt=options)
  ans_vocab = Vocab.from_opt(corpus=ans_corpus, opt=options)

  train_corpus_1 = train_src
  src_vocab_1 = Vocab.from_opt(corpus=train_corpus_1, opt=options)

  print("building tgt vocabulary")
  options = {'lower':True, 'mode':'frequency', 'tgt':True, 'size':opt.tgt_vocab_size, 'frequency':opt.tgt_words_min_frequency}
  tgt_vocab = Vocab.from_opt(corpus=train_tgt, opt=options)
  
  print('build word feature vocabularies')
  options = {'lower':False, 'mode':'size', 'tgt':False, 'size':opt.feat_vocab_size, 'frequency':opt.feat_words_min_frequency}
  feats_vocab = [Vocab.from_opt(corpus=ft + fv, opt=options) for ft, fv in zip(train_data['features'], valid_data['features'])] if opt.feature else None

  print('build node feature vocabularies')
  options = {'lower':False, 'mode':'size', 'tgt':False, 'size':opt.feat_vocab_size, 'frequency':opt.feat_words_min_frequency}
  graph_feats_vocab = [Vocab.from_opt(corpus=ft + fv, opt=options) for ft, fv in zip(train_graph['features'], valid_graph['features'])] if opt.node_feature else None

  print('build edge vocabularies')
  options = {'lower':True, 'mode':'size', 'tgt':False, 'size':opt.feat_vocab_size, 'frequency':opt.feat_words_min_frequency}
  edge_in_corpus = [edge_set for edges in train_graph['edge_in'] + valid_graph['edge_in'] for edge_set in edges]
  edge_out_corpus = [edge_set for edges in train_graph['edge_out'] + valid_graph['edge_out'] for edge_set in edges]
  edge_in_vocab = Vocab.from_opt(corpus=edge_in_corpus, opt=options)
  edge_out_vocab = Vocab.from_opt(corpus=edge_out_corpus, opt=options)

  #========get node index=========
  #train_graph_node_index = [get_graph_node_index(nodes, src_vocab) for nodes in train_node]
  #valid_graph_node_index = get_graph_node_index(va_dataset, src_vocab)

  #========word to index==========
  train_src_idx, train_src_tokens, train_src_indexes = convert_word_to_idx(train_src, src_vocab, sep=opt.answer == 'sep', pretrained = '')
  valid_src_idx, valid_src_tokens, valid_src_indexes = convert_word_to_idx(valid_src, src_vocab, sep=opt.answer == 'sep', pretrained = '')

  train_graph_idx, train_graph_tokens, train_graph_indexes = convert_word_to_idx(train_node, src_vocab, sep=opt.answer == 'sep', pretrained = '')
  valid_graph_idx, valid_graph_tokens, valid_graph_indexes = convert_word_to_idx(valid_node, src_vocab, sep=opt.answer == 'sep', pretrained = '')

  train_tgt_idx, train_tgt_tokens, _ = convert_word_to_idx(train_tgt, tgt_vocab)    
  valid_tgt_idx, valid_tgt_tokens, _ = convert_word_to_idx(valid_tgt, tgt_vocab)


  
  train_copy = wrap_copy_idx(train_src_tokens, train_tgt_tokens, tgt_vocab, '', pre_trained_vocab)
  valid_copy = wrap_copy_idx(valid_src_tokens, valid_tgt_tokens, tgt_vocab, '', pre_trained_vocab)
  train_copy_switch, train_copy_tgt = train_copy[0], train_copy[1]
  valid_copy_switch, valid_copy_tgt = valid_copy[0], valid_copy[1]

  train_ans_idx = convert_word_to_idx(train_ans, ans_vocab)[0] if opt.answer else None
  valid_ans_idx = convert_word_to_idx(valid_ans, ans_vocab)[0] if opt.answer else None

  train_feat_idxs = [convert_word_to_idx(feat, vocab, lower=False)[0] for feat, vocab in zip(train_data['features'], feats_vocab)] if opt.feature else None
  valid_feat_idxs = [convert_word_to_idx(feat, vocab, lower=False)[0] for feat, vocab in zip(valid_data['features'], feats_vocab)] if opt.feature else None

  #print(len(train_graph['features']))
  train_graph_feat_idxs = [convert_word_to_idx(feat, vocab, lower=False)[0] for feat, vocab
                        in zip(train_graph['features'], graph_feats_vocab)] if opt.node_feature else None
  valid_graph_feat_idxs = [convert_word_to_idx(feat, vocab, lower=False)[0] for feat, vocab
                        in zip(valid_graph['features'], graph_feats_vocab)] if opt.node_feature else None
  train_edge_in_idxs = [convert_word_to_idx(sample, edge_in_vocab, lower=False)[0] for sample in train_graph['edge_in']]
  valid_edge_in_idxs = [convert_word_to_idx(sample, edge_in_vocab, lower=False)[0] for sample in valid_graph['edge_in']]
  train_edge_out_idxs = [convert_word_to_idx(sample, edge_out_vocab, lower=False)[0] for sample in train_graph['edge_out']]
  valid_edge_out_idxs = [convert_word_to_idx(sample, edge_out_vocab, lower=False)[0] for sample in valid_graph['edge_out']]
    
  #========== prepare pretrained vetors ==========#
  if pre_trained_vocab:
      pre_trained_src_vocab = None if opt.pretrained else get_embedding(pre_trained_vocab, src_vocab)
      pre_trained_ans_vocab = get_embedding(pre_trained_vocab, ans_vocab) if opt.answer else None
      pre_trained_tgt_vocab = get_embedding(pre_trained_vocab, tgt_vocab)
      pre_trained_vocab = {'src':pre_trained_src_vocab, 'tgt':pre_trained_tgt_vocab}
  #print(pre_trained_vocab)
  #========== save data ===========#
  seq_data = {
          'dict': {'src': src_vocab,
               'tgt': tgt_vocab,
               'ans': ans_vocab if opt.answer else None,
               'feature': feats_vocab, 
               'pre-trained': pre_trained_vocab
          },
          'train': {'src': train_src_idx,
                'tgt': train_tgt_idx,
                'ans': train_ans_idx,
                'feature': train_feat_idxs,
                'copy':{'switch':train_copy_switch,
                    'tgt':train_copy_tgt}
          },
          'valid': {'src': valid_src_idx,
                'tgt': valid_tgt_idx,
                'ans': valid_ans_idx,
                'feature': valid_feat_idxs,
                'copy':{'switch':valid_copy_switch,
                    'tgt':valid_copy_tgt},
                'tokens':{'src': valid_src_tokens,
                      'tgt': valid_tgt_tokens}
          }
      }
  
  graph_data = {
          'dict': {'feature': graph_feats_vocab, 
               'edge': {
                   'in': edge_in_vocab,
                   'out': edge_out_vocab
               }
          },
          'train': {'src_index':train_graph_idx,
                'index': train_graph['indexes'],
                'feature': train_graph_feat_idxs,
                'edge': {
                    'in': train_edge_in_idxs,
                    'out': train_edge_out_idxs
                }
          },
          'valid': {'src_index':valid_graph_idx,
                'index': valid_graph['indexes'],
                'feature': valid_graph_feat_idxs,
                'edge': {
                    'in': valid_edge_in_idxs,
                    'out': valid_edge_out_idxs
                }
          }
      }
    
  return seq_data, graph_data, (train_final_indexes, valid_final_indexes), (train_src_indexes, valid_src_indexes)



def main():
    trainset, max_node, max_in_neigh, max_out_neigh, max_sent  = data_stream.read_amr_file(sys.argv[1])
    validset, max_node, max_in_neigh, max_out_neigh, max_sent  = data_stream.read_amr_file(sys.argv[2])
    sequences, graphs, final_indexes, bert_indexes = sequence_data(trainset, validset)
    #sequence_data(trainset, validset)
    print("Saving data ......")
    torch.save(sequences, sys.argv[3])
    torch.save(graphs, sys.argv[4])
    print('Saving Datasets ......')
    trainData = Dataset(sequences['train'], graphs['train'], opt.batch_size, answer=opt.answer, node_feature=opt.node_feature, copy=opt.copy)
    validData = Dataset(sequences['valid'], graphs['valid'], opt.batch_size, answer=opt.answer, node_feature=opt.node_feature, copy=opt.copy)
    torch.save(trainData, sys.argv[5])
    torch.save(validData, sys.argv[6])
    print("Done.")

if __name__ == '__main__':
    main()