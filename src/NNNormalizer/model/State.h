/*
 * State.h
 *
 *  Created on: Oct 1, 2015
 *      Author: mszhang
 */

#ifndef SEG_STATE_H_
#define SEG_STATE_H_

#include "ModelParams.h"
#include "Action.h"
#include "ActionedNodes.h"
#include "AtomFeatures.h"
#include "Utf.h"
#include "GlobalNodes.h"

class CStateItem {
public:
  std::string _word;
  std::string _tag;
  int _wstart;
  int _wend;
  CStateItem *_prevStackState;
  CStateItem *_prevState;
  int _next_index;

  const std::vector<std::string> *_chars;
  int _char_size;
  int _word_count;

  CAction _lastAction;
  PNode _score;
  int _inform_count;
  int _max_inform_count;

  // features
  ActionedNodes _nextscores;  // features current used
  AtomFeatures _atomFeat;  //features will be used for future

public:
  bool _bStart; // whether it is a start state
  bool _bGold; // for train

public:
  CStateItem() {
    clear();
  }


  virtual ~CStateItem() {
    clear();
  }

  void initial(ModelParams& params, HyperParams& hyparams, AlignedMemoryPool* mem) {
    _nextscores.initial(params, hyparams, mem);
  }

  void setInput(const std::vector<std::string>* pCharacters, dtype ratio) {
    _chars = pCharacters;
    _char_size = pCharacters->size();
    _max_inform_count = (int)(_char_size * ratio);
  }

  void clear() {
    _word = "</s>";
    _tag = "1";
    _wstart = -1;
    _wend = -1;
    _prevStackState = 0;
    _prevState = 0;
    _next_index = 0;
    _chars = 0;
    _char_size = 0;
    _lastAction.clear();
    _word_count = 0;
    _bStart = true;
    _bGold = true;
    _score = NULL;
    _inform_count = 0;
    _max_inform_count = 0;
  }


  const CStateItem* getPrevStackState() const {
    return _prevStackState;
  }

  const CStateItem* getPrevState() const {
    return _prevState;
  }

  std::string getLastWord() {
    return _word;
  }


public:
  //only assign context
  void separate(CStateItem* next) {
    if (_next_index >= _char_size) {
      std::cout << "separate error" << std::endl;
      return;
    }
    next->_word = (*_chars)[_next_index];
    next->_tag = "1";
    next->_wstart = _next_index;
    next->_wend = _next_index;
    next->_prevStackState = this;
    next->_prevState = this;
    next->_next_index = _next_index + 1;
    next->_chars = _chars;
    next->_char_size = _char_size;
    next->_word_count = _word_count + 1;
    next->_lastAction.set(CAction::SEP);
    next->_inform_count = _inform_count;
    next->_max_inform_count = _max_inform_count;
  }

  //only assign context
  void finish(CStateItem* next) {
    if (_next_index != _char_size) {
      std::cout << "finish error" << std::endl;
      return;
    }
    next->_word = "</s>";
    next->_tag = "1";
    next->_wstart = -1;
    next->_wend = -1;
    next->_prevStackState = this;
    next->_prevState = this;
    next->_next_index = _next_index + 1;
    next->_chars = _chars;
    next->_char_size = _char_size;
    next->_word_count = _word_count;
    next->_lastAction.set(CAction::FIN);
    next->_inform_count = _inform_count;
    next->_max_inform_count = _max_inform_count;
  }

  //only assign context
  void append(CStateItem* next) {
    if (_next_index >= _char_size) {
      std::cout << "append error" << std::endl;
      return;
    }
    next->_word = _word + (*_chars)[_next_index];
    next->_tag = _tag;
    next->_wstart = _wstart;
    next->_wend = _next_index;
    next->_prevStackState = _prevStackState;
    next->_prevState = this;
    next->_next_index = _next_index + 1;
    next->_chars = _chars;
    next->_char_size = _char_size;
    next->_word_count = _word_count;
    next->_lastAction.set(CAction::APP);
    next->_inform_count = _inform_count;
    next->_max_inform_count = _max_inform_count;
  }

  void substitute(CStateItem* next) {
    if (_next_index >= _char_size) {
      std::cout << "separate error" << std::endl;
      return;
    }
    next->_word = (*_chars)[_next_index];
    next->_tag = "0";
    next->_wstart = _next_index;
    next->_wend = _next_index;
    next->_prevStackState = this;
    next->_prevState = this;
    next->_next_index = _next_index + 1;
    next->_chars = _chars;
    next->_char_size = _char_size;
    next->_word_count = _word_count + 1;
    next->_lastAction.set(CAction::SUT);
    next->_inform_count = _inform_count + 1;
    next->_max_inform_count = _max_inform_count;
  }

  void move(CStateItem* next, const CAction& ac) {
    if (ac.isAppend()) {
      append(next);
    }
    else if (ac.isSeparate()) {
      separate(next);
    }
    else if (ac.isSubstitute()) {
      substitute(next);
    }
    else if (ac.isFinish()) {
      finish(next);
    }
    else {
      std::cout << "error action" << std::endl;
    }

    next->_bStart = false;
    next->_bGold = false;
  }

  bool IsTerminated() const {
    if (_lastAction.isFinish())
      return true;
    return false;
  }

  //partial results
  void getSegResults(std::vector<std::string>& words, std::vector<std::string>& tags) const {
    words.clear();
    tags.clear();
    static vector<const CStateItem *> preSepStates;
    preSepStates.clear();
    if (!IsTerminated()) {
      preSepStates.insert(preSepStates.begin(), this);
    }
    const CStateItem *prevStackState = _prevStackState;
    while (prevStackState != 0 && !prevStackState->_bStart) {
      preSepStates.insert(preSepStates.begin(), prevStackState);
      prevStackState = prevStackState->_prevStackState;
    }
    //will add results
    static int state_num;
    state_num = preSepStates.size();
    if (state_num != _word_count) {
      std::cout << "bug exists: " << state_num << " " << _word_count << std::endl;
    }
    for (int idx = 0; idx < state_num; idx++) {
      words.push_back(preSepStates[idx]->_word);
      tags.push_back(preSepStates[idx]->_tag);
    }
  }


  void getGoldAction(const std::vector<std::string>& segments, const std::vector<std::string>& tags, CAction& ac) const {
    if (_next_index == _char_size) {
      ac.set(CAction::FIN);
      return;
    }
    if (_next_index == 0) {
      if (tags[0] == "1") {
        ac.set(CAction::SEP);
        return;
      }
      else {
        ac.set(CAction::SUT);
        return;
      }
    }

    if (_next_index > 0 && _next_index < _char_size) {
      // should have a check here to see whether the words are match, but I did not do it here
      if (_word.length() == segments[_word_count - 1].length()) {
        if (tags[_word_count] == "1") {
          ac.set(CAction::SEP);
          return;
        }
        else {
          ac.set(CAction::SUT);
          return;
        }
      }
      else {
        ac.set(CAction::APP);
        return;
      }
    }

    ac.set(CAction::NO_ACTION);
    return;
  }

  // we did not judge whether history actions are match with current state.
  void getGoldAction(const CStateItem* goldState, CAction& ac) const {
    if (_next_index > goldState->_next_index || _next_index < 0) {
      ac.set(CAction::NO_ACTION);
      return;
    }
    const CStateItem *prevState = goldState->_prevState;
    CAction curAction = goldState->_lastAction;
    while (_next_index < prevState->_next_index) {
      curAction = prevState->_lastAction;
      prevState = prevState->_prevState;
    }
    return ac.set(curAction._code);
  }

  void getCandidateActions(vector<CAction> & actions) const {
    actions.clear();
    static CAction ac;
    if (_next_index == 0) {
      ac.set(CAction::SEP);
      actions.push_back(ac);
      if (_inform_count < _max_inform_count) {
        ac.set(CAction::SUT);
        actions.push_back(ac);
      }
    }
    else if (_next_index == _char_size) {
      ac.set(CAction::FIN);
      actions.push_back(ac);
    }
    else if (_next_index > 0 && _next_index < _char_size) {
      ac.set(CAction::SEP);
      actions.push_back(ac);
      ac.set(CAction::APP);
      actions.push_back(ac);
      if (_inform_count < _max_inform_count) {
        ac.set(CAction::SUT);
        actions.push_back(ac);
      }
    }
    else {

    }

  }


  inline std::string str() const {
    stringstream curoutstr;
    curoutstr << "score: " << _score->val[0] << " ";
    curoutstr << "seg:";
    std::vector<std::string> words, tags;
    getSegResults(words, tags);
    for (int idx = 0; idx < words.size(); idx++) {
      curoutstr << " " << words[idx] + "_" + tags[idx];
    }

    return curoutstr.str();
  }


public:
  inline void computeNextScore(Graph *cg, const vector<CAction>& acs) {
    if (_bStart) {
      _nextscores.forward(cg, acs, _atomFeat, NULL);
    }
    else {
      _nextscores.forward(cg, acs, _atomFeat, _score);
    }
  }

  inline void prepare(HyperParams* hyper_params, ModelParams* model_params, GlobalNodes* global_nodes) {
    _atomFeat.str_1W = _word;
    _atomFeat.str_1T = _tag;
    _atomFeat.str_3T = getStateDictTag(hyper_params, this);
    if (_prevStackState != 0) {
      _atomFeat.str_3T = getStateDictTag(hyper_params, _prevStackState) + _atomFeat.str_3T;
      if (_prevStackState->_prevStackState != 0) {
        _atomFeat.str_3T = getStateDictTag(hyper_params, _prevStackState->_prevStackState) + _atomFeat.str_3T;
      }
      else {
        _atomFeat.str_3T = "15" + _atomFeat.str_3T;
      }
    }
    else {
      _atomFeat.str_3T = "1515" + _atomFeat.str_3T;
    }
    _atomFeat.str_AC = _lastAction.str();
    _atomFeat.word_num = _word_count;
    _atomFeat.word_start = _wstart;
    _atomFeat.next_position = _next_index;
    _atomFeat.char_size = _char_size;
    _atomFeat.p_word_lstm = _prevStackState == 0 ? NULL : &(_prevStackState->_nextscores.word_lstm);
    _atomFeat.p_tag_lstm = _prevStackState == 0 ? NULL : &(_prevStackState->_nextscores.tag_lstm);
    _atomFeat.p_action_lstm = _prevState == 0 ? NULL : &(_prevState->_nextscores.action_lstm);
    _atomFeat.p_char_left_lstm = global_nodes == NULL ? NULL : &(global_nodes->char_left_lstm);
    _atomFeat.p_char_right_lstm = global_nodes == NULL ? NULL : &(global_nodes->char_right_lstm);
  }

public:
  inline string getStateDictTag(HyperParams* hyper_params, CStateItem* pState) {
    string curTag = "";
    if (pState->_wstart >= 0) {
      stringstream ss;
      int curLength = pState->_wend - pState->_wstart + 1;
      if (curLength > 5) curLength = 5;
      if (hyper_params->usualWords.find(pState->_word) != hyper_params->usualWords.end()) {
        ss << pState->_tag << curLength;
      }
      else {
        ss << pState->_tag << 0;
      }
      curTag = ss.str();
    }
    else {
      stringstream ss;
      ss << pState->_tag << 5;
      curTag = ss.str();
    }

    return curTag;
  }
};


class CScoredState {
public:
  CStateItem *item;
  CAction ac;
  dtype score;
  bool bGold;
  int position;

public:
  CScoredState() : item(0), score(0), ac(), bGold(0), position(-1) {
  }

  CScoredState(const CScoredState& other) : item(other.item), score(other.score), ac(other.ac), bGold(other.bGold), position(other.position) {

  }

public:
  bool operator <(const CScoredState &a1) const {
    return score < a1.score;
  }
  bool operator >(const CScoredState &a1) const {
    return score > a1.score;
  }
  bool operator <=(const CScoredState &a1) const {
    return score <= a1.score;
  }
  bool operator >=(const CScoredState &a1) const {
    return score >= a1.score;
  }
};

class CScoredState_Compare {
public:
  int operator()(const CScoredState &o1, const CScoredState &o2) const {
    if (o1.score < o2.score)
      return -1;
    else if (o1.score > o2.score)
      return 1;
    else
      return 0;
  }
};


#endif /* SEG_STATE_H_ */
