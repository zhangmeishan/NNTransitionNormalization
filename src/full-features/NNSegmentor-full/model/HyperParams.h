#ifndef SRC_HyperParams_H_
#define SRC_HyperParams_H_

#include "N3LDG.h"
#include "Options.h"

using namespace nr;
using namespace std;

struct HyperParams {
  //required
  int beam;
  int maxlength;
  int action_num;
  dtype delta;
  int batch;


  dtype nnRegular; // for optimization
  dtype adaAlpha;  // for optimization
  dtype adaEps; // for optimization
  dtype dropProb;

  int char_dim;
  int chartype_dim;
  int bichar_dim;
  int char_represent_dim;
  int char_hidden_dim;
  int char_lstm_dim;
  int char_feat_dim;
  int char_state_dim;
  bool char_tune;
  bool bichar_tune;

  int word_dim;
  int word_lstm_dim;
  int word_feat_dim;
  int word_state_dim;
  bool word_tune;

  int action_dim;
  int action_lstm_dim;
  int action_feat_dim;
  int action_state_dim;

  int app_dim;
  int sep_dim;

public:
  HyperParams() {
    maxlength = max_sentence_clength + 1;
    bAssigned = false;
  }

public:
  void setRequared(Options& opt) {
    //please specify dictionary outside
    //please sepcify char_dim, word_dim and action_dim outside.
    beam = opt.beam;
    delta = opt.delta;
    bAssigned = true;

    nnRegular = opt.regParameter;
    adaAlpha = opt.adaAlpha;
    adaEps = opt.adaEps;
    dropProb = opt.dropProb;
    batch = opt.batchSize;

    char_dim = opt.charEmbSize;
    bichar_dim = opt.bicharEmbSize;
    chartype_dim = opt.charTypeEmbSize;
    char_represent_dim = char_dim + bichar_dim + chartype_dim;
    char_hidden_dim = opt.charHiddenSize;
    char_lstm_dim = opt.charRNNHiddenSize;
    char_feat_dim = 12 * char_lstm_dim;
    char_state_dim = opt.charStateSize;
    char_tune = opt.charEmbFineTune;
    bichar_tune = opt.bicharEmbFineTune;

    word_dim = opt.wordEmbSize;
    word_lstm_dim = opt.wordRNNHiddenSize;
    word_feat_dim = 2 * word_lstm_dim;
    word_state_dim = opt.wordStateSize;
    word_tune = opt.wordEmbFineTune;

    action_dim = opt.actionEmbSize;
    action_lstm_dim = opt.actionRNNHiddenSize;
    action_feat_dim = 2 * action_lstm_dim;
    action_state_dim = opt.actionStateSize;

    app_dim = opt.appHiddenSize;
    sep_dim = opt.sepHiddenSize;
  }

  void clear() {
    bAssigned = false;
  }

  bool bValid() {
    return bAssigned;
  }


public:

  void print() {

  }

private:
  bool bAssigned;
};


#endif /* SRC_HyperParams_H_ */