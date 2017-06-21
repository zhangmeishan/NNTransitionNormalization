/*
* Feature.h
*
*  Created on: Aug 25, 2016
*      Author: mszhang
*/

#ifndef SRC_ActionedNodes_H_
#define SRC_ActionedNodes_H_

#include "ModelParams.h"
#include "AtomFeatures.h"
#include "Action.h"

struct ActionedNodes {
    LookupNode last_word_input;
    IncLSTM1Builder word_lstm;
    ConcatNode word_state_concat;
    UniNode word_state_hidden;

    PSubNode char_span_repsent_left;
    PSubNode char_span_repsent_right;
    ConcatNode char_state_concat;
    UniNode char_state_hidden;

    UniNode app_state_represent;
    BiNode sep_state_represent;

    LinearNode app_score;
    LinearNode sep_score;


    vector<PAddNode> outputs;

    BucketNode bucket_char, bucket_word;


public:
    ~ActionedNodes() {
        //m_outf.close();
    }
public:
    inline void initial(ModelParams& params, HyperParams& hyparams, AlignedMemoryPool* mem) {
        last_word_input.setParam(&(params.word_table));
        last_word_input.init(hyparams.word_dim, hyparams.dropProb, mem);
        word_lstm.init(&(params.word_lstm), hyparams.dropProb, mem); //already allocated here
        word_state_concat.init(hyparams.word_feat_dim, -1, mem);
        word_state_hidden.setParam(&(params.word_state_hidden));
        word_state_hidden.init(hyparams.word_state_dim, hyparams.dropProb, mem);

        char_span_repsent_left.init(hyparams.char_lstm_dim, -1, mem);
        char_span_repsent_right.init(hyparams.char_lstm_dim, -1, mem);
        char_state_concat.init(hyparams.char_feat_dim, -1, mem);
        char_state_hidden.setParam(&params.char_state_hidden);
        char_state_hidden.init(hyparams.char_state_dim, hyparams.dropProb, mem);

        app_state_represent.setParam(&params.app_state_represent);
        app_state_represent.init(hyparams.app_dim, hyparams.dropProb, mem);
        sep_state_represent.setParam(&params.sep_state_represent);
        sep_state_represent.init(hyparams.sep_dim, hyparams.dropProb, mem);


        app_score.setParam(&(params.app_score));
        app_score.init(1, -1, mem);
        sep_score.setParam(&(params.sep_score));
        sep_score.init(1, -1, mem);
        outputs.resize(hyparams.action_num);

        //neural features
        for (int idx = 0; idx < hyparams.action_num; idx++) {
            outputs[idx].init(1, -1, mem);
        }

        bucket_char.init(hyparams.char_lstm_dim, -1, mem);
        bucket_word.init(hyparams.word_lstm_dim, -1, mem);
    }


public:
    inline void forward(Graph* cg, const vector<CAction>& actions, AtomFeatures& atomFeat, PNode prevStateNode) {
        vector<PNode> sumNodes;
        CAction ac;
        int ac_num;
        ac_num = actions.size();

        bucket_char.forward(cg, 0);
        bucket_word.forward(cg, 0);
        PNode pseudo_char = &(bucket_char);
        PNode pseudo_word = &(bucket_word);



        vector<PNode> wordNodes;
        last_word_input.forward(cg, atomFeat.str_1W);
        word_lstm.forward(cg, &last_word_input, atomFeat.p_word_lstm);
        wordNodes.push_back(&word_lstm._hidden);
        if (word_lstm._nSize > 1) {
            wordNodes.push_back(&word_lstm._pPrev->_hidden);
        }
        else {
            wordNodes.push_back(pseudo_word);
        }
        word_state_concat.forward(cg, wordNodes);
        word_state_hidden.forward(cg, &word_state_concat);


        //
        vector<PNode> charNodes;
        int char_posi = atomFeat.next_position;
        PNode char_node_left_curr = (char_posi  < atomFeat.char_size) ? &(atomFeat.p_char_left_lstm->_hiddens[char_posi]) : pseudo_char;
        PNode char_node_left_next1 = (char_posi + 1  < atomFeat.char_size) ? &(atomFeat.p_char_left_lstm->_hiddens[char_posi + 1]) : pseudo_char;
        PNode char_node_left_next2 = (char_posi + 2  < atomFeat.char_size) ? &(atomFeat.p_char_left_lstm->_hiddens[char_posi + 2]) : pseudo_char;
        PNode char_node_left_prev1 = (char_posi > 0) ? &(atomFeat.p_char_left_lstm->_hiddens[char_posi - 1]) : pseudo_char;
        PNode char_node_left_prev2 = (char_posi > 1) ? &(atomFeat.p_char_left_lstm->_hiddens[char_posi - 2]) : pseudo_char;

        PNode char_node_right_curr = (char_posi  < atomFeat.char_size) ? &(atomFeat.p_char_right_lstm->_hiddens[char_posi]) : pseudo_char;
        PNode char_node_right_next1 = (char_posi + 1  < atomFeat.char_size) ? &(atomFeat.p_char_right_lstm->_hiddens[char_posi + 1]) : pseudo_char;
        PNode char_node_right_next2 = (char_posi + 2  < atomFeat.char_size) ? &(atomFeat.p_char_right_lstm->_hiddens[char_posi + 2]) : pseudo_char;
        PNode char_node_right_prev1 = (char_posi > 0) ? &(atomFeat.p_char_right_lstm->_hiddens[char_posi - 1]) : pseudo_char;
        PNode char_node_right_prev2 = (char_posi > 1) ? &(atomFeat.p_char_right_lstm->_hiddens[char_posi - 2]) : pseudo_char;

        charNodes.push_back(char_node_left_curr);
        charNodes.push_back(char_node_left_next1);
        charNodes.push_back(char_node_left_next2);
        charNodes.push_back(char_node_left_prev1);
        charNodes.push_back(char_node_left_prev2);

        charNodes.push_back(char_node_right_curr);
        charNodes.push_back(char_node_right_next1);
        charNodes.push_back(char_node_right_next2);
        charNodes.push_back(char_node_right_prev1);
        charNodes.push_back(char_node_right_prev2);

        PNode char_lstm_left_start = atomFeat.word_start > 0 ? &(atomFeat.p_char_left_lstm->_hiddens[atomFeat.word_start - 1]) : pseudo_char;
        PNode char_lstm_left_end = char_node_left_prev1;

        PNode char_lstm_right_start = atomFeat.word_start >= 0 ? &(atomFeat.p_char_right_lstm->_hiddens[atomFeat.word_start]) : pseudo_char;
        PNode char_lstm_right_end = char_node_right_curr;

        char_span_repsent_left.forward(cg, char_lstm_left_end, char_lstm_left_start);
        charNodes.push_back(&char_span_repsent_left);
        char_span_repsent_right.forward(cg, char_lstm_right_start, char_lstm_right_end);
        charNodes.push_back(&char_span_repsent_right);

        char_state_concat.forward(cg, charNodes);
        char_state_hidden.forward(cg, &char_state_concat);

        app_state_represent.forward(cg, &char_state_hidden);
        sep_state_represent.forward(cg, &char_state_hidden, &word_state_hidden);

        for (int idx = 0; idx < ac_num; idx++) {
            ac.set(actions[idx]);
            sumNodes.clear();

            if (prevStateNode != NULL) {
                sumNodes.push_back(prevStateNode);
            }

            if (ac.isAppend()) {
                app_score.forward(cg, &app_state_represent);
                sumNodes.push_back(&app_score);
            }
            else if (ac.isSeparate() || ac.isFinish()) {
                sep_score.forward(cg, &sep_state_represent);
                sumNodes.push_back(&sep_score);
            }
            else {
                std::cout << "error action here" << std::endl;
            }

            outputs[idx].forward(cg, sumNodes);
        }
    }

};

#endif /* SRC_ActionedNodes_H_ */
