import torch
import torch.nn as nn
from opt_einsum import contract
import torch.nn.functional as F
from long_seq import process_long_input
from losses import ATLoss,AFLoss
from transformers import AutoImageProcessor, AutoModel
from torch.nn.parameter import Parameter
class Expert(nn.Module):
    def __init__(self, input_features, output_features,num_heads=4):
        super(Expert, self).__init__()
        self.linear = nn.Linear(input_features, output_features)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(normalized_shape=output_features)

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        x = self.layer_norm(x)
        return x
class MoE(nn.Module):
    def __init__(self, num_experts=4, input_size=1024, output_size=97, k=2):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([Expert(input_size, output_size) for _ in range(num_experts)])
        self.gate= nn.Sequential(
            nn.Linear(input_size, num_experts),
            nn.Tanh(),
        )
        self.k = k  # Number of experts to select

    def forward(self, x):
        gate_outputs = self.gate(x)
        top_k_gates, top_k_indices = torch.topk(gate_outputs, self.k, dim=1)
        top_k_gates = F.softmax(top_k_gates, dim=1)
        top_k_gates = top_k_gates.unsqueeze(1)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=2)
        top_k_indices = top_k_indices.unsqueeze(1).expand(-1, expert_outputs.size(1), -1)
        top_k_outputs = expert_outputs.gather(2, top_k_indices)
        
        output = (top_k_outputs * top_k_gates).sum(dim=2)
        return output
class DocREModel(nn.Module):

    def __init__(self, config, model, tokenizer,
                emb_size=1024, block_size=64, num_labels=-1,
                max_sent_num=25, evi_thresh=0.2):
        '''
        Initialize the model.
        :model: Pretrained langage model encoder;
        :tokenizer: Tokenzier corresponding to the pretrained language model encoder;
        :emb_size: Dimension of embeddings for subject/object (head/tail) representations;
        :block_size: Number of blocks for grouped bilinear classification;
        :num_labels: Maximum number of relation labels for each entity pair;
        :max_sent_num: Maximum number of sentences for each document;
        :evi_thresh: Threshold for selecting evidence sentences.
        '''
        
        super().__init__()
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.hidden_size = config.hidden_size

        # self.loss_fnt = ATLoss()
        self.loss_fnt = AFLoss()
        self.loss_fnt_evi = nn.KLDivLoss(reduction="batchmean")

        self.head_extractor = nn.Linear(self.hidden_size * 2, emb_size)
        self.tail_extractor = nn.Linear(self.hidden_size * 2, emb_size)     
        
        self.bilinear = nn.Linear(emb_size * block_size, config.num_labels)
        """HTR"""
        if emb_size==1024:
            self.new_bil = nn.Linear(int(emb_size *1.875), config.num_labels) #small
            # self.new_bil = nn.Linear(int(emb_size *8.25), config.num_labels) #small-upsample
            # self.new_bil = nn.Linear(int(emb_size *5), config.num_labels)       #large
            # self.new_bil = nn.Linear(int(emb_size *3.75), config.num_labels)       #base
        else:
            self.new_bil = nn.Linear(int(emb_size / 2 * 3), config.num_labels)
        self.image_model = AutoModel.from_pretrained('facebook/dinov2-small',local_files_only=True)
        """对比"""
        # self.contra=ContrastiveCombiner()
        self.moe_1=MoE(num_experts=4,input_size=emb_size,output_size=emb_size)
        self.moe_2=MoE(num_experts=4,input_size=emb_size,output_size=emb_size)
        self.conbine=nn.Linear(2*config.num_labels, config.num_labels)
        self.emb_size = emb_size
        self.block_size = block_size
        self.num_labels = num_labels
        self.total_labels = config.num_labels
        self.max_sent_num = max_sent_num
        self.evi_thresh = evi_thresh

        self.a = nn.Parameter(torch.tensor(1.0))  # 初始化 self.a 为 1
        self.b = nn.Parameter(torch.tensor(1.0))  # 初始化 self.b 为 1
        self.c = nn.Parameter(torch.tensor(1.0))  # 初始化 self.c 为 1

    def encode(self, input_ids, attention_mask):
        
        '''
        Get the embedding of each token. For long document that has more than 512 tokens, split it into two overlapping chunks.
        Inputs:
            :input_ids: (batch_size, doc_len)
            :attention_mask: (batch_size, doc_len)
        Outputs:
            :sequence_output: (batch_size, doc_len, hidden_dim)
            :attention: (batch_size, num_attn_heads, doc_len, doc_len)
        '''
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
        # process long documents.
        sequence_output, attention = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens)
        
        return sequence_output, attention

    def get_hrt(self, sequence_output, attention, entity_pos, hts, offset):

        '''
        Get head, tail, context embeddings from token embeddings.
        Inputs:
            :sequence_output: (batch_size, doc_len, hidden_dim)
            :attention: (batch_size, num_attn_heads, doc_len, doc_len)
            :entity_pos: list of list. Outer length = batch size, inner length = number of entities each batch.
            :hts: list of list. Outer length = batch size, inner length = number of combination of entity pairs each batch.
            :offset: 1 for bert and roberta. Offset caused by [CLS] token.
        Outputs:
            :hss: (num_ent_pairs_all_batches, emb_size)
            :tss: (num_ent_pairs_all_batches, emb_size)
            :rss: (num_ent_pairs_all_batches, emb_size)
            :ht_atts: (num_ent_pairs_all_batches, doc_len)
            :rels_per_batch: list of length = batch size. Each entry represents the number of entity pairs of the batch.
        '''
        
        n, h, _, c = attention.size()
        hss, tss, rss = [], [], []
        ht_atts = []

        for i in range(len(entity_pos)): # for each batch
            entity_embs, entity_atts = [], []
            
            # obtain entity embedding from mention embeddings.
            for eid, e in enumerate(entity_pos[i]): # for each entity
                if len(e) > 1:
                    e_emb, e_att = [], []
                    for mid, (start, end) in enumerate(e): # for every mention
                        if start + offset < c:
                            # In case the entity mention is truncated due to limited max seq length.
                            e_emb.append(sequence_output[i, start + offset])
                            e_att.append(attention[i, :, start + offset])

                    if len(e_emb) > 0:
                        e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                        e_att = torch.stack(e_att, dim=0).mean(0)
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                else:
                    start, end = e[0]
                    if start + offset < c:
                        e_emb = sequence_output[i, start + offset]
                        e_att = attention[i, :, start + offset]
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)

                entity_embs.append(e_emb)
                entity_atts.append(e_att)
                
            entity_embs = torch.stack(entity_embs, dim=0)  # [n_e, d]
            entity_atts = torch.stack(entity_atts, dim=0)  # [n_e, h, seq_len]

            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)

            # obtain subject/object (head/tail) embeddings from entity embeddings.
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])
                
            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])

            ht_att = (h_att * t_att).mean(1) # average over all heads        
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-30) 
            ht_atts.append(ht_att)
            # obtain local context embeddings.
            rs = contract("ld,rl->rd", sequence_output[i], ht_att)

            hss.append(hs)
            tss.append(ts)
            rss.append(rs)
        
        rels_per_batch = [len(b) for b in hss]
        hss = torch.cat(hss, dim=0) # (num_ent_pairs_all_batches, emb_size)
        tss = torch.cat(tss, dim=0) # (num_ent_pairs_all_batches, emb_size)
        rss = torch.cat(rss, dim=0) # (num_ent_pairs_all_batches, emb_size)
        ht_atts = torch.cat(ht_atts, dim=0) # (num_ent_pairs_all_batches, max_doc_len)

        return hss, rss, tss, ht_atts, rels_per_batch


    def forward_rel(self, hs, ts, rs):
        '''
        Forward computation for RE.
        Inputs:
            :hs: (num_ent_pairs_all_batches, emb_size)
            :ts: (num_ent_pairs_all_batches, emb_size)
            :rs: (num_ent_pairs_all_batches, emb_size)
        Outputs:
            :logits: (num_ent_pairs_all_batches, num_rel_labels)
        '''
        #HTRI
        h_s_img = hs.view(-1, 1, self.emb_size // self.block_size*2, self.block_size//2)
        r_s_img = rs.view(-1, 1, self.emb_size // self.block_size*2, self.block_size//2)
        t_s_img = ts.view(-1, 1, self.emb_size // self.block_size*2, self.block_size//2)
        img = torch.cat([self.a*h_s_img, self.b*r_s_img, self.c*t_s_img], dim=1) #[num_ent_pairs_all_batches,3,32,32]
        # #upsample
        # # img_64= F.interpolate(img, size=(64,64), mode='bilinear', align_corners=False)
        # # img_res_32=self.image_model(img).last_hidden_state
        # # img_res_64=self.image_model(img_64).last_hidden_state
        # # img_res=torch.cat([img_res_32,img_res_64], dim=1).reshape(hs.shape[0], -1)
        # # img_fea = self.new_bil(img_res)
        img_res = self.image_model(img)
        img_res = img_res.last_hidden_state.reshape(hs.shape[0], -1)
        img_fea = self.new_bil(img_res)
        #MoE
        hs = torch.tanh(self.head_extractor(torch.cat([hs, rs], dim=-1)))
        ts = torch.tanh(self.tail_extractor(torch.cat([ts, rs], dim=-1)))
        hs = self.moe_1(hs)
        ts = self.moe_2(ts)
        # split into several groups.
        b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
        moe_fea = self.bilinear(bl)
        logits=self.conbine(torch.cat([img_fea, moe_fea], dim=-1))
        return logits


    def forward_evi(self, doc_attn, sent_pos, batch_rel, offset):
        '''
        Forward computation for ER.
        Inputs:
            :doc_attn: (num_ent_pairs_all_batches, doc_len), attention weight of each token for computing localized context pooling.
            :sent_pos: list of list. The outer length = batch size. The inner list contains (start, end) position of each sentence in each batch.
            :batch_rel: list of length = batch size. Each entry represents the number of entity pairs of the batch.
            :offset: 1 for bert and roberta. Offset caused by [CLS] token.
        Outputs:
            :s_attn:  (num_ent_pairs_all_batches, max_sent_all_batch), sentence-level evidence distribution of each entity pair.
        '''
        
        max_sent_num = max([len(sent) for sent in sent_pos])
        rel_sent_attn = []
        for i in range(len(sent_pos)): # for each batch
            # the relation ids corresponds to document in batch i is [sum(batch_rel[:i]), sum(batch_rel[:i+1]))
            curr_attn = doc_attn[sum(batch_rel[:i]):sum(batch_rel[:i+1])]
            curr_sent_pos = [torch.arange(s[0], s[1]).to(curr_attn.device) + offset for s in sent_pos[i]] # + offset

            curr_attn_per_sent = [curr_attn.index_select(-1, sent) for sent in curr_sent_pos]
            curr_attn_per_sent += [torch.zeros_like(curr_attn_per_sent[0])] * (max_sent_num - len(curr_attn_per_sent))
            sum_attn = torch.stack([attn.sum(dim=-1) for attn in curr_attn_per_sent], dim=-1) # sum across those attentions
            rel_sent_attn.append(sum_attn)

        s_attn = torch.cat(rel_sent_attn, dim=0)
        return s_attn


    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None, # relation labels
                entity_pos=None,
                hts=None, # entity pairs
                sent_pos=None, 
                sent_labels=None, # evidence labels (0/1)
                teacher_attns=None, # evidence distribution from teacher model
                tag="train",
                ):

        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        output = {}
        sequence_output, attention = self.encode(input_ids, attention_mask)

        hs, rs, ts, doc_attn, batch_rel = self.get_hrt(sequence_output, attention, entity_pos, hts, offset)
        logits = self.forward_rel(hs, ts, rs)
        # contra_loss=self.contra(hs, ts, rs)
        output["rel_pred"] = self.loss_fnt.get_label(logits, num_labels=self.num_labels)

        if sent_labels != None: # human-annotated evidence available

            s_attn = self.forward_evi(doc_attn, sent_pos, batch_rel, offset)
            output["evi_pred"] = F.pad(s_attn > self.evi_thresh, (0, self.max_sent_num - s_attn.shape[-1]))

        if tag in ["test", "dev"]: # testing
            scores_topk = self.loss_fnt.get_score(logits, self.num_labels)
            output["scores"] = scores_topk[0]
            output["topks"] = scores_topk[1]
        
        if tag == "infer": # teacher model inference
            output["attns"] = doc_attn.split(batch_rel)

        else: # training
            # relation extraction loss
            loss = self.loss_fnt(logits.float(), labels.float())
            output["loss"] = {"rel_loss": loss.to(sequence_output)}
                
            if sent_labels != None: # supervised training with human evidence

                idx_used = torch.nonzero(labels[:,1:].sum(dim=-1)).view(-1)
                # evidence retrieval loss (kldiv loss)
                s_attn = s_attn[idx_used]
                sent_labels = sent_labels[idx_used]
                norm_s_labels = sent_labels/(sent_labels.sum(dim=-1, keepdim=True) + 1e-30)
                norm_s_labels[norm_s_labels == 0] = 1e-30
                s_attn[s_attn == 0] = 1e-30
                evi_loss = self.loss_fnt_evi(s_attn.log(), norm_s_labels)
                output["loss"]["evi_loss"] = evi_loss.to(sequence_output)
            
            elif teacher_attns != None: # self training with teacher attention
                
                doc_attn[doc_attn == 0] = 1e-30
                teacher_attns[teacher_attns == 0] = 1e-30
                attn_loss = self.loss_fnt_evi(doc_attn.log(), teacher_attns)
                output["loss"]["attn_loss"] = attn_loss.to(sequence_output)
        
        return output
