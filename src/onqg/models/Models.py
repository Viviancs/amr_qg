import torch
import torch.nn as nn
from torch.autograd import Variable

import onqg.dataset.Constants as Constants
from onqg.models.modules.Attention import ConcatAttention


class UnifiedModel(nn.Module):
    ''' Unify Sequence-Encoder and Graph-Encoder

    Input:  seq-encoder: src_seq, lengths, feat_seqs
            graph-encoder: edges
            encoder-transform: index, lengths, root
            decoder: tgt_seq, src_seq, feat_seqs
            answer-encoder: src_seq, lengths, feat_seqs

    Output: results output from the Decoder (type: dict)
    '''
    def __init__(self, model_type, seq_encoder, node_encoder, graph_encoder, encoder_transformer, aligner,
                 decoder, decoder_transformer):
        super(UnifiedModel, self).__init__()

        self.model_type = model_type

        self.seq_encoder = seq_encoder
        self.node_encoder = node_encoder

        self.encoder_transformer = encoder_transformer
        self.graph_encoder = graph_encoder

        self.aligner = aligner
        
        self.decoder_transformer = decoder_transformer
        self.decoder = decoder
    
    def forward(self, inputs, max_length=None):
        #========== forward ==========#
        ## RNN encode ## 
        seq_output, hidden = self.seq_encoder(inputs['seq-encoder'])
        ## Node encode ##
        node_output, hidden_node = self.node_encoder(inputs['node-encoder'])
        ## encoder transform ##
        inputs['encoder-transform']['seq_output'] = node_output
        inputs['encoder-transform']['hidden'] = hidden_node
        node_input, graph_hidden = self.encoder_transformer(inputs['encoder-transform'], max_length)
        ## graph encode ##
        #print("=======================")
        #print(hidden_node.size())
        #print(graph_hidden.size())
        inputs['graph-encoder']['nodes'] = node_input
        node_output, _ = self.graph_encoder(inputs['graph-encoder'])
        #将图中结点信息与词嵌入进行融合
        #node_output, hidden = self.aligner()

        outputs = {}

        #========== classify =========#
        if self.model_type != 'generate':
            #预测每个单词是否在问题中
            scores = self.classifier(seq_output) if not self.decoder.layer_attn else self.classifier(seq_output[-1])
            #print(scores.size())
            inputs['decoder-transform']['scores'] = scores
            outputs['classification'] = scores   
        #========== generate =========#
        inputs['decoder-transform']['graph_output'] = node_output
        inputs['decoder-transform']['seq_output'] = seq_output
        inputs['decoder-transform']['hidden'] = hidden

        inputs['decoder-transform']['seq_index'] = inputs['seq-encoder']['index']
        inputs['decoder-transform']['node_index'] = inputs['graph-encoder']['index']
        
        inputs['decoder']['enc_output'], inputs['decoder']['scores'], hidden, graph_hidden = self.decoder_transformer(inputs['decoder-transform'])
        inputs['decoder']['hidden'] = hidden
        inputs['decoder']['graph_hidden'] = graph_hidden
        dec_output = self.decoder(inputs['decoder'])
        outputs['generation'] = dec_output
        #========== generate =========#
        if self.model_type != 'classify':
            outputs['generation']['pred'] = self.generator(dec_output['pred'])
        
        return outputs        
