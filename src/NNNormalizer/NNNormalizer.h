/*
 * Normalizer.h
 *
 *  Created on: Jan 25, 2016
 *      Author: mszhang
 */

#ifndef SRC_PARSER_H_
#define SRC_PARSER_H_

#include "N3LDG.h"

#include "Driver.h"
#include "Options.h"
#include "Pipe.h"
#include "Utf.h"
#include <vld.h> 

using namespace nr;
using namespace std;


class Normalizer {
public:
	Normalizer(size_t memsize);
	virtual ~Normalizer();

public:
	Driver m_driver;
	Options m_options;
	Pipe m_pipe;

public:
	int createAlphabet(const vector<Instance>& vecInsts);
	int initialActionWordMap();


public:
	void train(const string& trainFile, const string& devFile, const string& testFile, const string& modelFile, const string& optionFile);
	void predict(const vector<Instance>& inputs, dtype ratio, vector<vector<string> >& segouts, vector<vector<string> >& tagouts);
	void test(const string& testFile, dtype ratio, const string& outputFile, const string& modelFile);

	// static training
	void getGoldActions(const vector<Instance>& vecInsts, vector<vector<CAction> >& vecActions);
	void getGoldActions(const Instance& inst, vector<CAction>& actions);

public:
	void writeModelFile(const string& outputModelFile);
	void loadModelFile(const string& inputModelFile);

protected:
	inline void randomInput(const Instance& inst, Instance& outInst) {
		outInst.clear();
		dtype rpRation = m_options.rpRatio;
        vector<string> charInfo;
        static unordered_set<string>::iterator iter;
        for (int idx = 0; idx < inst.words.size(); idx++) {
            dtype curRand = dtype(rand()) / RAND_MAX;
            string orgWord = inst.words[idx];
            string newWord = inst.words[idx];
            if (curRand < rpRation) {
                vector<string> & candidate_words = m_driver._hyperparams.word2inform[orgWord];
                int count = candidate_words.size();
                if (count > 0) {
                    int selId = rand() % count;
                    newWord = candidate_words[selId];
                }
            }

            outInst.words.push_back(newWord);
            if (newWord.compare(orgWord) == 0) {
                outInst.tags.push_back("1");
            }
            else {
                outInst.tags.push_back("0");
            }
            getCharactersFromUTF8String(newWord, charInfo);
            for (int idy = 0; idy < charInfo.size(); idy++) {
                outInst.chars.push_back(charInfo[idy]);
            }
        }
	}

};

#endif /* SRC_PARSER_H_ */
