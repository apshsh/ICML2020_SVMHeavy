
//
// Anionic regression GP
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

//
// Currently this is basically a wrap-around for a LS-SVR with C mapped to
// 1/sigma for noise regularisation.  This is equivalent to the standard
// GP regressor assuming Gaussian measurement noise.
//

#ifndef _gpr_anions_h
#define _gpr_anions_h

#include "gpr_generic.h"
#include "lsv_anions.h"




class GPR_Anions;

// Swap and zeroing (restarting) functions

inline void qswap(GPR_Anions &a, GPR_Anions &b);
inline GPR_Anions &setzero(GPR_Anions &a);

class GPR_Anions : public GPR_Generic
{
public:

    // Constructors, destructors, assignment etc..

    GPR_Anions();
    GPR_Anions(const GPR_Anions &src);
    GPR_Anions(const GPR_Anions &src, const ML_Base *srcx);
    GPR_Anions &operator=(const GPR_Anions &src) { assign(src); return *this; }
    virtual ~GPR_Anions() { return; }

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0);
    virtual void semicopy(const ML_Base &src);
    virtual void qswapinternal(ML_Base &b);

    virtual std::ostream &printstream(std::ostream &output, int dep) const;
    virtual std::istream &inputstream(std::istream &input );


    // ================================================================
    //     Common functions for all ML
    // ================================================================

    // Information functions

    virtual int type(void)    const { return 402; }
    virtual int subtype(void) const { return 0;   }

    virtual int isClassifier(void) const { return 0; }




    // ================================================================
    //     Common functions for all GPs
    // ================================================================

    // Information functions (training data):

    virtual       GPR_Generic &getGPR(void)            { return *this; }
    virtual const GPR_Generic &getGPRconst(void) const { return *this; }

    // General modification and autoset functions

    virtual       ML_Base &getML(void)            { return static_cast<      ML_Base &>(getGPR());      }
    virtual const ML_Base &getMLconst(void) const { return static_cast<const ML_Base &>(getGPRconst()); }



    // Base-level stuff

    virtual       LSV_Generic &getQ(void)            { return QQ; }
    virtual const LSV_Generic &getQconst(void) const { return QQ; }




    // don't use this

    virtual LSV_Generic &getQcheat(void) const { return (**thisthisthis).QQ; }

private:

    LSV_Anions QQ;

    GPR_Anions *thisthis;
    GPR_Anions **thisthisthis;
};

inline void qswap(GPR_Anions &a, GPR_Anions &b)
{
    a.qswapinternal(b);

    return;
}

inline GPR_Anions &setzero(GPR_Anions &a)
{
    a.restart();

    return a;
}

inline void GPR_Anions::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    GPR_Anions &b = dynamic_cast<GPR_Anions &>(bb.getML());

    GPR_Generic::qswapinternal(b);

    qswap(getQ(),b.getQ());

    return;
}

inline void GPR_Anions::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const GPR_Anions &b = dynamic_cast<const GPR_Anions &>(bb.getMLconst());

    GPR_Generic::semicopy(b);

    getQ().semicopy(b.getQconst());

    return;
}

inline void GPR_Anions::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const GPR_Anions &src = dynamic_cast<const GPR_Anions &>(bb.getMLconst());

    GPR_Generic::assign(src,onlySemiCopy);

    getQ().assign(src.getQconst(),onlySemiCopy);

    return;
}

#endif
