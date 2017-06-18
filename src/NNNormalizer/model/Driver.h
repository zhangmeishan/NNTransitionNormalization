/*
 * Driver.h
 *
 *  Created on: June 15, 2017
 *      Author: mszhang
 */

#ifndef SRC_Driver_H_
#define SRC_Driver_H_

#include "N3LDG.h"
#include "State.h"
#include "ActionedNodes.h"
#include "Action.h"
#include "ComputionGraph.h"

class Driver {
public:
  Driver(size_t memsize) : aligned_mem(memsize) {
    _batch = 0;
  }

  ~Driver() {
    _batch = 0;
  }

public:
  Graph _cg;  // build neural graphs
  vector<GraphBuilder> _builders;
  ModelParams _modelparams;  // model parameters
  HyperParams _hyperparams;

  Metric _eval;
  ModelUpdate _ada;  // model update

  AlignedMemoryPool aligned_mem;
  int _batch;

public:

  inline void initial() {
    if (!_hyperparams.bValid()) {
      std::cout << "hyper parameter initialization Error, Please check!" << std::endl;
      return;
    }
    if (!_modelparams.initial(_hyperparams, &aligned_mem)) {
      std::cout << "model parameter initialization Error, Please check!" << std::endl;
      return;
    }
    _hyperparams.print();

    _builders.resize(_hyperparams.batch);

    for (int idx = 0; idx < _hyperparams.batch; idx++) {
      _builders[idx].initial(&_cg, _modelparams, _hyperparams, &aligned_mem);
    }

    std::cout << "allocated memory: " << aligned_mem.capacity << ", total required memory: " << aligned_mem.required
      << ", perc = " << aligned_mem.capacity * 1.0 / aligned_mem.required << std::endl;

    setUpdateParameters(_hyperparams.nnRegular, _hyperparams.adaAlpha, _hyperparams.adaEps);
    _batch = 0;
  }


public:
  dtype train(const std::vector<std::vector<string> >& sentences, const vector<vector<CAction> >& goldACs) {
    _eval.reset();
    dtype cost = 0.0;
    _cg.clearValue(true);
    int num = sentences.size();
    if (num > _builders.size()) {
      std::cout << "input example number is larger than predefined batch number" << std::endl;
      return -1;
    }

    for (int idx = 0; idx < num; idx++) {
      _builders[idx].encode(&sentences[idx]);
    }
    _cg.compute();

    for (int idx = 0; idx < num; idx++) {
      _builders[idx].decode(&sentences[idx], 1.0, &goldACs[idx]);
      _eval.overall_label_count += goldACs[idx].size();
      cost += loss_google(idx, num);
    }

    _cg.backward();
    return cost;
  }

  void decode(const std::vector<std::vector<string> >& sentences, dtype ratio, vector<vector<string> >& segouts, vector<vector<string> >& tagouts) {
    int num = sentences.size();
    if (num > _builders.size()) {
      std::cout << "input example number is larger than predefined batch number" << std::endl;
      return;
    }
    _cg.clearValue();
    for (int idx = 0; idx < num; idx++) {
      _builders[idx].encode(&sentences[idx]);
    }
    _cg.compute();


    segouts.resize(num);
    tagouts.resize(num);
    for (int idx = 0; idx < num; idx++) {
      _builders[idx].decode(&sentences[idx], ratio);
      int step = _builders[idx].outputs.size();
      _builders[idx].states[step - 1][0].getSegResults(segouts[idx], tagouts[idx]);
    }
  }

  void updateModel() {
    if (_ada._params.empty()) {
      _modelparams.exportModelParams(_ada);
    }
    //_ada.rescaleGrad(1.0 / _batch);
    _ada.update(10);
    //_ada.updateAdam(10);
    _batch = 0;
  }


  void writeModel();

  void loadModel();

private:
  // max-margin
  dtype loss(int batch_index, int num) {
    int step = _builders[batch_index].outputs.size();
    _eval.correct_label_count += step;
    static PNode pBestNode = NULL;
    static PNode pGoldNode = NULL;
    static PNode pCurNode;

    int offset = _builders[batch_index].outputs[step - 1].size();
    for (int idx = 0; idx < offset; idx++) {
      pCurNode = _builders[batch_index].outputs[step - 1][idx].in;
      if (pBestNode == NULL || pCurNode->val[0] > pBestNode->val[0]) {
        pBestNode = pCurNode;
      }
      if (_builders[batch_index].outputs[step - 1][idx].bGold) {
        pGoldNode = pCurNode;
      }
    }

    _batch++;

    if (pGoldNode != pBestNode) {
      pGoldNode->loss[0] = -1.0 / num;
      pBestNode->loss[0] = 1.0 / num;

      return 1.0;
    }

    return 0.0;
  }

  dtype loss_google(int batch_index, int num) {
    int maxstep = _builders[batch_index].outputs.size();
    if (maxstep == 0) return 1.0;
    _eval.correct_label_count += maxstep;
    static PNode pBestNode = NULL;
    static PNode pGoldNode = NULL;
    static PNode pCurNode;
    static dtype sum, max;
    static int curcount, goldIndex;
    static vector<dtype> scores;
    dtype cost = 0.0;

    for (int step = 0; step < maxstep; step++) {
      curcount = _builders[batch_index].outputs[step].size();
      max = 0.0;
      goldIndex = -1;
      pBestNode = pGoldNode = NULL;
      for (int idx = 0; idx < curcount; idx++) {
        pCurNode = _builders[batch_index].outputs[step][idx].in;
        if (pBestNode == NULL || pCurNode->val[0] > pBestNode->val[0]) {
          pBestNode = pCurNode;
        }
        if (_builders[batch_index].outputs[step][idx].bGold) {
          pGoldNode = pCurNode;
          goldIndex = idx;
        }
      }

      if (goldIndex == -1) {
        std::cout << "impossible" << std::endl;
      }
      pGoldNode->loss[0] = -1.0 / num;

      max = pBestNode->val[0];
      sum = 0.0;
      scores.resize(curcount);
      for (int idx = 0; idx < curcount; idx++) {
        pCurNode = _builders[batch_index].outputs[step][idx].in;
        scores[idx] = exp(pCurNode->val[0] - max);
        sum += scores[idx];
      }

      for (int idx = 0; idx < curcount; idx++) {
        pCurNode = _builders[batch_index].outputs[step][idx].in;
        pCurNode->loss[0] += scores[idx] / (sum * num);
      }

      cost += -log(scores[goldIndex] / sum);

      if (std::isnan(cost)) {
        std::cout << "debug: " << goldIndex << ", sum = " << sum << ", num = " << num << std::endl;
        for (int idx = 0; idx < curcount; idx++) {
          std::cout << scores[idx] << " ";
        }
        std::cout << std::endl;
      }

      _batch++;

    }

    return cost;
  }


public:
  inline void setUpdateParameters(dtype nnRegular, dtype adaAlpha, dtype adaEps) {
    _ada._alpha = adaAlpha;
    _ada._eps = adaEps;
    _ada._reg = nnRegular;
  }

};

#endif /* SRC_Driver_H_ */
