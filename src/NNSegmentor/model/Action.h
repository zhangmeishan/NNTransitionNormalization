/*
 * CAction.h
 *
 *  Created on: Jan 25, 2016
 *      Author: mszhang
 */

#ifndef BASIC_CAction_H_
#define BASIC_CAction_H_



/*===============================================================
 *
 * scored actions
 *
 *==============================================================*/
// for segmentation, there are only threee valid operations
class CAction {
public:
	enum CODE { SEP = 0, APP = 1, FIN = 2, NO_ACTION = 3};
  unsigned long _code;

public:
   CAction() : _code(NO_ACTION){
   }

   CAction(int code) : _code(code){
   }

   CAction(const CAction &ac) : _code(ac._code){
   }

public:
	inline void clear() { _code = NO_ACTION; }

   inline void set(int code){
     _code = code;
   }

   inline void set(const CAction &ac) {
       _code = ac._code;
   }

   inline bool isNone() const { return _code==NO_ACTION; }
   inline bool isSeparate() const { return _code==SEP; }
   inline bool isAppend() const { return _code==APP; }
   inline bool isFinish() const { return _code==FIN; }

public:
   inline std::string str() const {
     if (isNone()) { return nullkey; }
     if (isSeparate()) { return "SEP"; }
     if (isAppend()) { return "APP"; }
     if (isFinish()) { return "FIN"; }
	 return nullkey;
   }

public:
   bool operator == (const CAction &a1) const { return _code == a1._code; }
   bool operator != (const CAction &a1) const { return _code != a1._code; }

};


#endif /* BASIC_CAction_H_ */
