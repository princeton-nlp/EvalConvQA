import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.generic_utils import *
from ..layers.common import *
from ..layers.attention import *
from ..layers.graphs import *


INF = 1e20
class GraphFlow(nn.Module):
    def __init__(self, config, w_embedding):
        super(GraphFlow, self).__init__()
        self.config = config
        self.device = config['device']
        vocab_embed_size = config['vocab_embed_size']
        hidden_size = config['hidden_size']

        # Extra features
        self.n_history = config['n_history']
        self.f_qem = config['f_qem']
        self.f_pos = config['f_pos']
        self.f_ner = config['f_ner']
        self.f_tf = config['f_tf']
        ctx_exact_match_embed_dim = config['ctx_exact_match_embed_dim']
        ctx_pos_embed_dim = config['ctx_pos_embed_dim']
        ctx_ner_embed_dim = config['ctx_ner_embed_dim']
        answer_marker_embed_dim = config['answer_marker_embed_dim']
        ques_marker_embed_dim = config['ques_marker_embed_dim']
        ques_turn_marker_embed_dim = config['ques_turn_marker_embed_dim']

        self.use_bert = config['use_bert']
        self.finetune_bert = config.get('finetune_bert', None)
        self.use_bert_weight = config['use_bert_weight']
        bert_dim = config['bert_dim']

        self.use_spatial_kernels = config['use_spatial_kernels']
        n_spatial_kernels = config['n_spatial_kernels'] if self.use_spatial_kernels else 1

        self.word_dropout = config['word_dropout']
        self.bert_dropout = config['bert_dropout']
        self.rnn_dropout = config['rnn_dropout']
        self.rnn_input_dropout = config.get('rnn_input_dropout', 0)
        self.ctx_graph_topk = config.get('ctx_graph_topk', None)
        self.ctx_graph_epsilon = config.get('ctx_graph_epsilon', None)
        self.static_graph = config.get('static_graph', None)
        self.word_embed = w_embedding
        if config['fix_vocab_embed']:
            print('[ Fix word embeddings ]')
            for param in self.word_embed.parameters():
                param.requires_grad = False

        ctx_feature_dim = 0
        if self.f_qem:
            self.ctx_exact_match_embed = nn.Embedding(config['num_features_f_qem'], ctx_exact_match_embed_dim)
            ctx_feature_dim += ctx_exact_match_embed_dim
        if self.f_pos:
            self.ctx_pos_embed = nn.Embedding(config['num_features_f_pos'], ctx_pos_embed_dim)
            ctx_feature_dim += ctx_pos_embed_dim
        if self.f_ner:
            self.ctx_ner_embed = nn.Embedding(config['num_features_f_ner'], ctx_ner_embed_dim)
            ctx_feature_dim += ctx_ner_embed_dim
        if self.f_tf:
            ctx_feature_dim += 1

        if self.n_history > 0:
            if answer_marker_embed_dim != 0:
                self.ctx_ans_marker_embed = nn.Embedding((self.n_history * 4) + 1, answer_marker_embed_dim)
            if config['use_ques_marker']:
                self.ques_marker_embed = nn.Embedding(sum([not config['no_pre_question'], not config['no_pre_answer']]) * self.n_history + 1, ques_marker_embed_dim)
                ques_turn_marker_embed_dim = 0
            else:
                self.ques_num_marker_embed = nn.Embedding(config['max_turn_num'], ques_turn_marker_embed_dim)

        if self.use_bert and self.use_bert_weight:
            bert_layer_start = config['bert_layer_indexes'][0]
            bert_layer_end = config['bert_layer_indexes'][1]
            num_bert_layers = int(bert_layer_end) - int(bert_layer_start)
            self.logits_bert_layers = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(1, num_bert_layers)))
            if config['use_bert_gamma']:
                self.gamma_bert_layers = nn.Parameter(nn.init.constant_(torch.Tensor(1, 1), 1.))


        self.ques_enc = EncoderRNN(vocab_embed_size + (ques_turn_marker_embed_dim if self.n_history > 0 else 0) + (ques_marker_embed_dim if self.n_history > 0 and config['use_ques_marker'] else 0) + (bert_dim if self.use_bert else 0), hidden_size, \
                        bidirectional=True, \
                        rnn_type='lstm', \
                        rnn_dropout=self.rnn_dropout, \
                        rnn_input_dropout=self.rnn_input_dropout, \
                        device=self.device)

        self.ctx_enc_l1 = EncoderRNN(2 * vocab_embed_size + ctx_feature_dim + self.n_history * answer_marker_embed_dim + (bert_dim if self.use_bert else 0), hidden_size, \
                        bidirectional=True, \
                        rnn_type='lstm', \
                        rnn_dropout=self.rnn_dropout, \
                        rnn_input_dropout=self.rnn_input_dropout, \
                        device=self.device)
        self.ctx_enc_l2 = EncoderRNN(2 * hidden_size, hidden_size, \
                        bidirectional=True, \
                        rnn_type='lstm', \
                        rnn_dropout=self.rnn_dropout, \
                        rnn_input_dropout=self.rnn_input_dropout, \
                        device=self.device)
        self.rnn_ques_over_time = EncoderRNN(1 * hidden_size, hidden_size, \
                        bidirectional=False, \
                        rnn_type='lstm', \
                        rnn_dropout=self.rnn_dropout, \
                        rnn_input_dropout=self.rnn_input_dropout, \
                        device=self.device)

        # Question attention
        self.ctx2ques_attn = Context2QuestionAttention(vocab_embed_size, hidden_size)
        self.ctx2ques_attn_l2 = Context2QuestionAttention(vocab_embed_size + hidden_size + (bert_dim if self.use_bert else 0), hidden_size)
        self.ques_self_atten = SelfAttention(hidden_size, hidden_size)

        if config['use_gnn']:
            # Graph computation
            if self.static_graph:
                self.ctx_gnn = StaticContextGraphNN(hidden_size,  graph_hops=config['ctx_graph_hops'], device=self.device)
                self.ctx_gnn_l2 = StaticContextGraphNN(hidden_size, graph_hops=config['ctx_graph_hops'], device=self.device)
            else:
                self.graph_learner = GraphLearner(2 * vocab_embed_size + ctx_feature_dim + self.n_history * answer_marker_embed_dim + (bert_dim if self.use_bert else 0), \
                            hidden_size, self.ctx_graph_topk, self.ctx_graph_epsilon, n_spatial_kernels, use_spatial_kernels=self.use_spatial_kernels, use_position_enc=config['use_position_enc'], \
                            position_emb_size=config['position_emb_size'], max_position_distance=config['max_position_distance'], num_pers=config['graph_learner_num_pers'], device=self.device)
                self.ctx_gnn = ContextGraphNN(hidden_size, n_spatial_kernels, use_spatial_kernels=self.use_spatial_kernels, graph_hops=config['ctx_graph_hops'], bignn=config['bignn'], device=self.device)
                self.ctx_gnn_l2 = ContextGraphNN(hidden_size, n_spatial_kernels, use_spatial_kernels=self.use_spatial_kernels, graph_hops=config['ctx_graph_hops'], bignn=config['bignn'], device=self.device)

            if config['temporal_gnn']:
                self.graph_gru_step = GatedFusion(hidden_size)
                self.graph_gru_step_l2 = GatedFusion(hidden_size)


        # Prediction
        self.gru_step = GRUStep(hidden_size, hidden_size)
        self.linear_start = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear_end = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear_unk_answer = nn.Linear(hidden_size, 2 * hidden_size, bias=False)
        self.fc_followup = nn.Linear(
            hidden_size, 2 * hidden_size * self.config.get('quac_followup_class_num', 3), bias=True)
        self.fc_yesno = nn.Linear(hidden_size, 2 * hidden_size * self.config.get('quac_yesno_class_num',3), bias=True)


    def forward(self, ex):
        """
        Parameters
        :questions, (batch_size, turn_size, ques_size)
        :bert_questions_f, (batch_size, turn_size, ques_size, bert_dim)
        :ques_len, (batch_size, turn_size)
        :context, (batch_size, ctx_size)
        :context_f, dict, val shape: (batch_size, turn_size, ctx_size)
        :bert_context_f, (batch_size, ctx_size, bert_dim)
        :context_ans_marker, (batch_size, turn_size, ctx_size, n_history)
        :ctx_len, (batch_size,)
        :num_turn, (batch_size,)

        Returns
        :start_logits, (batch_size, turn_size, ctx_size)
        :end_logits, (batch_size, turn_size, ctx_size)
        :unk_answer_logits, (batch_size, turn_size)
        :yes_answer_logits, (batch_size, turn_size)
        :no_answer_logits, (batch_size, turn_size)
        """
        questions = ex['xq']
        ques_len = ex['xq_len']
        context = ex['xd']
        ctx_len = ex['xd_len']
        num_turn = ex['num_turn']

        if self.use_bert:
            bert_questions_f = ex['bert_xq_f']
            bert_context_f = ex['bert_xd_f']
            if not self.finetune_bert:
                assert bert_questions_f.requires_grad == False
                assert bert_context_f.requires_grad == False
            if self.use_bert_weight:
                weights_bert_layers = torch.softmax(self.logits_bert_layers, dim=-1)
                if self.config['use_bert_gamma']:
                    weights_bert_layers = weights_bert_layers * self.gamma_bert_layers

                bert_questions_f = torch.mm(weights_bert_layers, bert_questions_f.view(bert_questions_f.size(0), -1)).view(bert_questions_f.shape[1:])
                bert_questions_f = dropout(bert_questions_f, self.bert_dropout, shared_axes=[-2], training=self.training)

                bert_context_f = torch.mm(weights_bert_layers, bert_context_f.view(bert_context_f.size(0), -1)).view(bert_context_f.shape[1:])
                bert_context_f = dropout(bert_context_f, self.bert_dropout, shared_axes=[-2], training=self.training)

        ques_mask = create_mask(ques_len.view(-1), questions.size(-1), self.device)
        ctx_mask = create_mask(ctx_len, context.size(-1), self.device)
        expand_ctx_len = ctx_len.unsqueeze(1).expand(-1, questions.size(1)).contiguous()
        turn_mask = create_mask(num_turn, questions.size(1), self.device)

        # Encoding module
        # Encode questions & context
        # shape: (batch_size * turn_size, ques_size, emb_dim)
        ques_emb = self.word_embed(questions.view(-1, questions.size(-1)))
        ques_emb = dropout(ques_emb, self.word_dropout, shared_axes=[-2], training=self.training)
        # shape: (batch_size, ctx_size, emb_dim)
        ctx_emb = self.word_embed(context)
        ctx_emb = dropout(ctx_emb, self.word_dropout, shared_axes=[-2], training=self.training)
        # print("ctx_emb:", ctx_emb.size())

        # shape: (batch_size, turn_size, ctx_size, emb_dim)
        ctx_aware_ques_emb = self.ctx2ques_attn(ctx_emb.unsqueeze(1), ques_emb.view(questions.size() + (-1,)), \
                                ques_emb.view(questions.size() + (-1,)), ques_mask.view(ques_len.size() + (-1,)))


        ques_cat_0 = [ques_emb]
        # print("ques_cat_0:", len(ques_cat_0), ques_cat_0[0].size())
        # print("questions[1]:", questions.size(1))
        ctx_cat_0 = [ctx_emb.unsqueeze(1).expand(-1, questions.size(1), -1, -1), ctx_aware_ques_emb]
        # print("ctx_cat_0:", len(ctx_cat_0), ctx_cat_0[0].size())
        # Add extra context features
        if self.f_qem:
            context_f_qem = self.ctx_exact_match_embed(ex['xd_f']['f_qem'].view(-1, ex['xd_f']['f_qem'].size(-1))).view(ex['xd_f']['f_qem'].shape[:3] + (-1,))
            ctx_cat_0.append(context_f_qem)
            # print("ctx_cat_0:", len(ctx_cat_0), ctx_cat_0[0].size())
        if self.f_pos:
            context_f_pos = self.ctx_pos_embed(ex['xd_f']['f_pos'].view(-1, ex['xd_f']['f_pos'].size(-1))).view(ex['xd_f']['f_pos'].shape[:3] + (-1,))
            ctx_cat_0.append(context_f_pos)
            # print("ctx_cat_0:", len(ctx_cat_0), ctx_cat_0[0].size())
        if self.f_ner:
            context_f_ner = self.ctx_ner_embed(ex['xd_f']['f_ner'].view(-1, ex['xd_f']['f_ner'].size(-1))).view(ex['xd_f']['f_ner'].shape[:3] + (-1,))
            ctx_cat_0.append(context_f_ner)
            # print("ctx_cat_0:", len(ctx_cat_0), ctx_cat_0[0].size())
        if self.f_tf:
            context_f_tf = ex['xd_tf'].unsqueeze(1).unsqueeze(-1).expand(-1, questions.size(1), -1, -1)
            ctx_cat_0.append(context_f_tf)
            # print("ctx_cat_0:", len(ctx_cat_0), ctx_cat_0[0].size())

        if self.n_history > 0:
            # Encode previous N answer locations to the context embeddings
            # shape: (batch_size, turn_size, ctx_size, n_history * answer_marker_embed_dim)
            if self.config['answer_marker_embed_dim'] != 0:
                context_ans_marker = ex['xd_answer_marker']
                ctx_ans_marker_emb = self.ctx_ans_marker_embed(context_ans_marker.view(-1, context_ans_marker.size(-1))) \
                                        .view(context_ans_marker.shape[:3] + (-1,))
                ctx_cat_0.append(ctx_ans_marker_emb)
                # print("ctx_cat_0:", len(ctx_cat_0), ctx_cat_0[0].size())

            if self.config['use_ques_marker']:
                question_f = ex['xq_f']
                question_f = self.ques_marker_embed(question_f.view(-1, question_f.size(-1)))
                ques_cat_0.append(question_f)
                # print("ques_cat_0:", len(ques_cat_0), ques_cat_0[0].size())
            else:
                # Encode question turn number inside the dialog into question embeddings
                question_num_ind = get_range_vector(questions.size(1), self.device)
                question_num_ind = question_num_ind.unsqueeze(-1).expand(-1, questions.size(-1))
                question_num_ind = question_num_ind.unsqueeze(0).expand(questions.size(0), -1, -1)
                question_num_ind = question_num_ind.reshape(-1, questions.size(-1))
                question_num_marker_emb = self.ques_num_marker_embed(question_num_ind)
                ques_cat_0.append(question_num_marker_emb)
                # print("ques_cat_0:", len(ques_cat_0), ques_cat_0[0].size())

        if self.use_bert:
            ques_cat_0.append(bert_questions_f.view((-1,) + bert_questions_f.shape[-2:]))
            # print("ques_cat_0:", len(ques_cat_0), ques_cat_0[0].size())
            ctx_cat_0.append(bert_context_f.unsqueeze(1).expand(-1, questions.size(1), -1, -1))
            # print("ctx_cat_0:", len(ctx_cat_0), ctx_cat_0[0].size())
        # print("ctx_cat_0:", len(ctx_cat_0), ctx_cat_0[0].size())
        ques_hidden_state_0 = torch.cat(ques_cat_0, dim=-1)
        # print("ques_hidden_state_0:", ques_hidden_state_0.size())
        ctx_hidden_state_0 = torch.cat(ctx_cat_0, dim=-1)
        # print("ctx_hidden_state_0:", ctx_hidden_state_0.size())

        # Run RNN on questions
        # shape: (batch_size * turn_size, ques_size, hidden_size)
        ques_hidden_state = self.ques_enc(ques_hidden_state_0, ques_len.view(-1))[0]

        # shape: (batch_size, turn_size, hidden_size)
        # print("ques_hidden_state:", ques_hidden_state.size())
        # print("ques_mask:", ques_mask.size())
        # print("ques_len:", ques_len.size())
        ques_state = self.ques_self_atten(ques_hidden_state, ques_mask).view(ques_len.size() + (-1,))
        # print("ques_state:", ques_state.size())


        # Reasoning module

        # Run GNN on context graphs Layer 1
        # shape: (batch_size, turn_size, ctx_size, hidden_size)
        ctx_node_state_l1 = self.ctx_enc_l1(ctx_hidden_state_0.view(-1, ctx_hidden_state_0.size(-2), ctx_hidden_state_0.size(-1)), expand_ctx_len.view(-1))[0]\
                    .view(ctx_hidden_state_0.shape[:3] + (-1,))
        # print("ctx_node_state_l1", ctx_node_state_l1.size())

        if self.config['use_gnn']:
            if self.static_graph:
                input_graphs = ex['xd_graphs']
                ctx_adjacency_matrix = (input_graphs['node2edge'], input_graphs['edge2node'])
            else:
                # Construct context graphs
                # shape: (batch_size, turn_size, n_spatial_kernels, ctx_size, ctx_size)
                # or (batch_size, turn_size, ctx_size, ctx_size)
                ctx_adjacency_matrix = self.graph_learner(ctx_hidden_state_0, ctx_mask)


            for turn_id in range(questions.size(1)):
                # shape: (batch_size, ctx_size, hidden_size)
                if self.static_graph:
                    ctx_turn_node_state = self.ctx_gnn(ctx_node_state_l1[:, turn_id].clone(), ctx_adjacency_matrix)
                else:
                    ctx_turn_node_state = self.ctx_gnn(ctx_node_state_l1[:, turn_id].clone(), ctx_adjacency_matrix[:, turn_id])
                ctx_node_state_l1[:, turn_id] = ctx_turn_node_state
                if self.config['temporal_gnn']:
                    next_turn_id = turn_id + 1
                    if next_turn_id < questions.size(1):
                        ctx_node_state_l1[:, next_turn_id] = self.graph_gru_step(ctx_turn_node_state, ctx_node_state_l1[:, next_turn_id].clone())



        if self.config.get('stacked_layer', True):
            # Run GNN on context graphs Layer 2
            ctx_cat_l2 = torch.cat([ctx_node_state_l1, ctx_emb.unsqueeze(1).expand(-1, questions.size(1), -1, -1)], -1)
            ques_cat_l2 = torch.cat([ques_hidden_state.view(questions.size() + (-1,)), ques_emb.view(questions.size() + (-1,))], -1)
            if self.use_bert:
                ctx_cat_l2 = torch.cat([ctx_cat_l2, bert_context_f.unsqueeze(1).expand(-1, questions.size(1), -1, -1)], -1)
                ques_cat_l2 = torch.cat([ques_cat_l2, bert_questions_f], -1)
            ctx_aware_ques_emb_l2 = self.ctx2ques_attn_l2(ctx_cat_l2, ques_cat_l2, ques_hidden_state.view(questions.size() + (-1,)), \
                ques_mask.view(ques_len.size() + (-1,)))
            # Shape: (batch_size, turn_size, ctx_size, 2 * hidden_size)
            ctx_hidden_state_1 = torch.cat([ctx_node_state_l1, ctx_aware_ques_emb_l2], -1)


            ctx_node_state_l2 = self.ctx_enc_l2(ctx_hidden_state_1.view(-1, ctx_hidden_state_1.size(-2), ctx_hidden_state_1.size(-1)), expand_ctx_len.view(-1))[0]\
                        .view(ctx_hidden_state_1.shape[:3] + (-1,))


            if self.config['use_gnn']:
                for turn_id in range(questions.size(1)):
                    # shape: (batch_size, ctx_size, hidden_size)
                    if self.static_graph:
                        ctx_turn_node_state = self.ctx_gnn_l2(ctx_node_state_l2[:, turn_id].clone(), ctx_adjacency_matrix)
                    else:
                        ctx_turn_node_state = self.ctx_gnn_l2(ctx_node_state_l2[:, turn_id].clone(), ctx_adjacency_matrix[:, turn_id])
                    ctx_node_state_l2[:, turn_id] = ctx_turn_node_state
                    if self.config['temporal_gnn']:
                        next_turn_id = turn_id + 1
                        if next_turn_id < questions.size(1):
                            ctx_node_state_l2[:, next_turn_id] = self.graph_gru_step_l2(ctx_turn_node_state, ctx_node_state_l2[:, next_turn_id].clone())

            ctx_node_state_final = ctx_node_state_l2

        else:
            ctx_node_state_final = ctx_node_state_l1



        # Prediction module
        # print("num_turn:", num_turn.size())
        # print("num_turn.view(-1):", num_turn.view(-1).size())
        # print("ques_state:", ques_state.size())
        ques_state = self.rnn_ques_over_time(ques_state, num_turn.view(-1))[0]


        # Answer span prediction
        p_start = ques_state
        
        # shape: (batch_size, turn_size, ctx_size)
        start_ = torch.matmul(ctx_node_state_final, self.linear_start(p_start).unsqueeze(-1)).squeeze(-1)
        mask1 = (1 - ctx_mask.byte().unsqueeze(1)).to(torch.bool)
        start_ = start_.masked_fill_(mask1, -INF)
        start_logits = F.log_softmax(start_, dim=-1)
        start_probs = torch.exp(start_logits).unsqueeze(2)
        
        # print("start_probs:", start_probs.size())
        # print("ctx_node_state_final:", ctx_node_state_final.size())
        # interm = torch.matmul(start_probs, ctx_node_state_final).squeeze(2)
        # print("ques_state:", ques_state.size())
        p_end = self.gru_step(p_start, torch.matmul(
            start_probs, ctx_node_state_final).squeeze(2))
        end_ = torch.matmul(ctx_node_state_final, self.linear_end(p_end).unsqueeze(-1)).squeeze(-1)
        mask2 = (1 - ctx_mask.byte().unsqueeze(1)).to(torch.bool)
        end_ = end_.masked_fill_(mask2, -INF)
        end_logits = F.log_softmax(end_, dim=-1)
        # end_probs = torch.exp(end_logits).unsqueeze(2)

        # ctx_sum = torch.matmul(start_probs * end_probs, ctx_node_state_final).squeeze(2)
        # UNK/Yes/No prediction
        ctx_mean = torch.mean(ctx_node_state_final, dim=2)
        ctx_max = torch.max(ctx_node_state_final, dim=2)[0]
        # shape: (batch_size, turn_size, 2 * hidden_size, 1)
        ctx_cat = torch.cat([ctx_mean, ctx_max], -1).unsqueeze(-1)

        # Answer type prediction
        
        unk_answer_probs = torch.sigmoid(torch.matmul(self.linear_unk_answer(p_start).unsqueeze(2), ctx_cat).squeeze(-1).squeeze(-1))

        p_yesno = self.fc_yesno(p_start).view(p_start.shape[:2] + (self.config.get('quac_yesno_class_num',3), -1))
        score_yesno = torch.matmul(p_yesno, ctx_cat).squeeze(-1)

        p_followup = self.fc_followup(p_start).view(p_start.shape[:2] + (self.config.get('quac_followup_class_num',3), -1))
        score_followup = torch.matmul(p_followup, ctx_cat).squeeze(-1)
        return {'start_logits': start_logits,
                'end_logits': end_logits,
                'unk_probs': unk_answer_probs,
                'score_yesno': score_yesno,
                'score_followup': score_followup,
                'turn_mask': turn_mask}
