//FIXMERKHS: mlinter: isProjection == 5
//           reProjections sets number of restarts per round


// see FIXMEFIX - see also smboopt.h

//FIXME: totiters == -2 means smart-stop.  In smboopt you need: model_err() = ( min_x f(x) + sigma(x) ) - ( min_x f(x) - sigma(x) ).  This can use directopt to calculate.  Then use model_err() <= eps and totiters == -2 then we 
//       exit at that point.
//FIXME: ALMOST DONE! in bayesopt.cc, have == -2 condition, just need to put target on this and exit when target reached!

//FIXME: option to tune weight on current best?

//
// Global optimisation options base-class
//
// Date: 29/09/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "ml_base.h"
#include "gpr_scalar.h"
#include "gpr_vector.h"
#include "ml_mutable.h"
#include "errortest.h"
#include "FNVector.h"
#include "mlcommon.h"

#ifndef _globalopt_h
#define _globalopt_h



extern int godebug; // to debug vector.h


#define DEFAULT_NUMOPTS 10
#define DEFAULT_LOGZTOL 1e-8

// Near zero, near inf values for distMode 4 in global optimiser

#define NEARZEROVAL 0.1
#define NEARINFVAL  1e6
#define RATIOMAX    1e-15

// ML registration index

#define DEFAULT_MLREGIND 1024


void overfn(gentype &res, Vector<gentype> &x, void *arg);
void hhyperfn(gentype &res, Vector<gentype> &x, void *arg);

// returns maximum width of x

double calcabc(int dim, 
               Vector<gentype> &fakexmin, Vector<gentype> &fakexmax, 
               Vector<double> &a, Vector<double> &b, Vector<double> &c,
               const Vector<gentype> &xmin, const Vector<gentype> &xmax, 
               const Vector<int> &distMode);


class GridOptions;
class SMBOOptions;

class GlobalOptions
{
    friend class GridOptions;
    friend class SMBOOptions;

public:

    // maxtraintime: maximum training time (sec).  0 for unlimited.  Note
    //           that this is not precise - for example in bayesian optim
    //           this applies to both the inner (DIRect) and outer loops,
    //           so actual upper bound could be >2x this number.
    // softmin: min value of objective function 
    // softmax: max value of objective function (currently unused) 
    // hardmin: min value of objective function, terminate if found.
    // hardmin: max value of objective function (currently unused)

    double maxtraintime;
    double softmin;
    double softmax;
    double hardmin;
    double hardmax;

    // xwidth - set by optim to maximum width of x range

    double xwidth;

    // Penalty term: penterm(p(x)) is added to the evaluation

    gentype penterm;

    // if isProjection set then x = projOp.g(x), where projOp is a weighted
    // combination of random directions specified by subDef, which are a vector
    // of dim versions of randDirtemplate, where the user must set the distribution
    // in randDir.  By default defrandDirtemplateVec is used, which corresponds
    // to finite dimensional random directions, but you can change this to 
    // defrandDirtemplateFnGP to specify function drawn from a GPR.
    //
    // isProjection  - 0 for normal (no fancy projection stuff)
    //                 1 for vector
    //                 2 for projection to function via draws from a GP
    //                 3 for projection to function via Bernstein polynomials
    //                 4 for projection to function via Bernstein polynomials with complexity schedule
    //                 5 for projection to function via RKHSVector
    // includeConst  - 0 for normal (no constant terms in projection)
    //                 1 means that the final term in the projection is a constant.
    // whatConst     - constant included under includeConst
    // randReproject - 0 for normal
    //                 1 for new random projection after each actual evaluation
    // useScalarFn   - 0 treat functional results as distributions
    //                 1 treat functional results as scalar functions (default).  Target must be 1-d
    // xNsamp        - number of samples for approximate integration for functional approx
    // fnDim         - if we're trying to find an optimal function then this is the
    //                 dimension of the function.  Variables are x,y,z,... - that is, 
    //                 in order of increasing fnDim, f(), f(x), f(x,y), f(x,y,z) ...
    // bernstart     - for scheduled Bernstein projection (isProjection == 4), this is the
    //                 initial polynomial order.  The maximum is min(dim,bernStart+randReproject)

    int isProjection;
    int includeConst;
    double whatConst;
    int randReproject;
    int useScalarFn;
    int xNsamp;
    int fnDim;
    int bernstart;

    BLK_UsrFnB defrandDirtemplateVec;
    GPR_Scalar defrandDirtemplateFnGP;
    BLK_Bernst defrandDirtemplateFnBern;
    RKHSVector defrandDirtemplateFnRKHS;

    // Used by scheduled Bernstein (isProjection == 4 )
    //
    // stopearly:  usually 0, set 1 during scheduled Bernstein if maxdiff in best x >= 0.95
    //             (keeps track of best result etc)
    // firsttest:  1 for first test
    // spOverride: set for all but the first run to make sure startpoints are only added for the first for scheduled Bernstein
    // bestyet:    best result yet

    int stopearly;
    int firsttest;
    int spOverride;
    int berndim;
    gentype bestyet;

    // ML registration stuff (for functional optimisation)
    //
    // MLregind:  index where ML is registered (unless full, in which case this
    //            gets incremented until an empty one is found)
    // MLregfn:   function to register ML.  ind is suggested, actual assigned
    //            index is returned.
    // MLreglist: list of all registered MLs (so they can be deleted)
    // MLregltyp: list of all registered ML types
    //            0  = randDirtemplate (copy of defrandDirtemplateVec in constructor)
    //            1  = randDirtemplate (copy of defrandDirtemplateFnGP in constructor)
    //            2  = subDef (in makeSubspace)
    //            3  = projOp (in makeSubspace)
    //            4  = projOpNow (in convertx)
    //            5  = fnapprox (copy of altfnapprox in smboopt constructor)
    //            6  = fnapprox (copy of altfnapproxmoo in smboopt constructor)
    //            7  = sigmaapprox (in smboopt constructor)
    //            8  = subDef (in makeSubspace)
    //            9  = projOp (in makeSubspace)
    //            10 = source model in diff-GP
    //            11 = difference model in diff-GP

    int MLregind;
    int (*MLregfn)(int ind, ML_Mutable *MLcase, void *fnarg);

    Vector<int> MLreglist;
    Vector<int> MLregltyp;

    int regML(ML_Mutable *MLcase, void *fnarg, int ltyp)
    {
        int nres = -1;

        if ( MLregfn )
        {
            nres = ( MLregind = MLregfn(MLregind,MLcase,fnarg) );

            MLreglist.append(MLreglist.size(),nres);
            MLregltyp.append(MLregltyp.size(),ltyp);

            MLregind++;
        }

        return nres;
    }

    // Constructors and assignment operators

    GlobalOptions()
    {
        thisthis = this;
        thisthisthis = &thisthis;

        randDirtemplate    = NULL;
        randDirtemplateInd = -1;
        projOp             = NULL;
        projOpInd          = -1;

        MLregind = DEFAULT_MLREGIND;
        MLregfn  = NULL;

        maxtraintime = 0;

        softmin = valninf();
        softmax = valpinf();
        hardmin = valninf();
        hardmax = valpinf();

        xwidth = 0;

        penterm = 0.0;

        isProjection  = 0;
        includeConst  = 0;
        whatConst     = 2.0;
        randReproject = 0;
        useScalarFn   = 1;
        xNsamp        = DEFAULT_SAMPLES_SAMPLE;
        fnDim         = 1;
        bernstart     = 1;

        return;
    }

    GlobalOptions(const GlobalOptions &src)
    {
        thisthis = this;
        thisthisthis = &thisthis;

        *this = src;

        return;
    }

    // It is important that this not be made virtual!
    GlobalOptions &operator=(const GlobalOptions &src)
    {
        maxtraintime = src.maxtraintime;

        softmin = src.softmin;
        softmax = src.softmax;
        hardmin = src.hardmin;
        hardmax = src.hardmax;

        xwidth = src.xwidth;

        penterm = src.penterm;

        isProjection  = src.isProjection;
        includeConst  = src.includeConst;
        whatConst     = src.whatConst;
        randReproject = src.randReproject;
        useScalarFn   = src.useScalarFn;
        xNsamp        = src.xNsamp;
        fnDim         = src.fnDim;
        bernstart     = src.bernstart;

        defrandDirtemplateVec    = src.defrandDirtemplateVec;
        defrandDirtemplateFnGP   = src.defrandDirtemplateFnGP;
        defrandDirtemplateFnBern = src.defrandDirtemplateFnBern;
        defrandDirtemplateFnRKHS = src.defrandDirtemplateFnRKHS;

        stopearly  = src.stopearly;
        firsttest  = src.firsttest;
        spOverride = src.spOverride;
        berndim    = src.berndim;
        bestyet    = src.bestyet;

        MLregind = src.MLregind;
        MLregfn  = src.MLregfn;

        MLreglist = src.MLreglist;
        MLregltyp = src.MLregltyp;

        a = src.a;
        b = src.b;
        c = src.c;

        locdistMode = src.locdistMode;
        locvarsType = src.locvarsType;

        projOptemplate = src.projOptemplate;
        projOp         = src.projOp;
        projOpRaw      = src.projOpRaw;
        projOpInd      = src.projOpInd;

        randDirtemplate    = src.randDirtemplate;
        randDirtemplateInd = src.randDirtemplateInd;
        subDef             = src.subDef;
        subDefInd          = src.subDefInd;

        addSubDim = src.addSubDim;

        locfnarg = src.locfnarg;

        xpweight         = src.xpweight;
        xpweightIsWeight = src.xpweightIsWeight;
        xbasis           = src.xbasis;
        xbasisprod       = src.xbasisprod;

        return *this;
    }

    // Generate a copy of the relevant optimisation class.

    virtual GlobalOptions *makeDup(void) const
    {
        GlobalOptions *newver;

        MEMNEW(newver,GlobalOptions(*this));

        return newver;
    }

    // virtual Destructor to get rid of annoying warnings

    virtual ~GlobalOptions() { return; }

    // Optimisation function stubs
    //
    // dim: problem dimension.
    // xres: x result.
    // Xres: x result (raw, unprocessed format).
    // fres: f(x) result.
    // ires: index of result.
    // mInd: for functional optimisation, this returns the (registered) index of the ML model found
    // muInd: index of mu model approximation (if any)
    // sigInd: index of sigma model approximation (if any)
    // srcmodInd: index of source model (if used, see diff-GP and env-GP)
    // diffmodInd: index of diff model (if used, see diff-GP)
    // allxres: all x results.
    // allfres: all f(x) results.
    // allmres: all f(x) results, modified (eg scaled by probability of 
    //     feasibility etc, hypervolume etc - what you should be judging 
    //     performance on).
    // allsres: suplementary results (timing etc - see specific method)
    // s_score: stability score of x (default 1)
    // xmin: lower bound on x.
    // xmax: upper bound on x.
    // distMode: method of forming each axis.
    //     0 = linear distribution
    //     1 = logarithmic distribution
    //     2 = anti-logarithmic distribution
    //     3 = random distribution (grid only)
    //     4 = inverse logistic distribution of points
    //     5 = REMBO
    // varsType: (grid only) variable types.
    //     0 = integer
    //     1 = real.
    // fn: callback for function being evaluated.
    // fnarg: arguments for fn.
    // killSwitch: usually zero, set 1 to force early exit.
    // numReps: number of repeats
    //
    // If numReps > 1 then the results are from the final run, and the
    // following variables reflect statistics
    //
    // meanfres, varfres: mean and variance of fres
    // meanires, varires: mean and variance of time to best result
    // meantres, vartres: mean and variance of time to softmin
    // meanTres, varTres: mean and variance of time to hardmin
    // meanallfres, varallfres: mean and variance of allfres
    // meanallmres, varallmres: mean and variance of allfres

    virtual int optim(int dim,
                      Vector<gentype> &Xres,
                      gentype &fres,
                      int &ires,
                      Vector<Vector<gentype> > &allXres,
                      Vector<gentype> &allfres,
                      Vector<gentype> &allmres,
                      Vector<gentype> &allsres,
                      Vector<double>  &s_score,
                      const Vector<gentype> &xmin,
                      const Vector<gentype> &xmax,
                      void (*fn)(gentype &res, Vector<gentype> &x, void *arg),
                      void *fnarg,
                      svmvolatile int &)
    {
        (void) dim;
        (void) Xres;
        (void) fres;
        (void) ires;
        (void) allXres;
        (void) allfres;
        (void) allmres;
        (void) allsres;
        (void) s_score;
        (void) xmin;
        (void) xmax;
        (void) fn;
        (void) fnarg;

        throw("You should not be here - it's just a stub!");

        return -42;
    }

    gentype &findfirstLT(gentype &res, const Vector<gentype> &src, double target) const
    {
        int k;
        int nores = 1;

        res = valpinf();

        for ( k = 0 ; k < src.size() ; k++ )
        {
            if ( (double) src(k) <= target )
            {
                res = k+1;
                nores = 0;
                break;
            }
        }

        // Observation: BO can get stuck.  Even BO with built-in exploration (like GP-UCB) can 
        //              get stuck.  It's obvious when you see it, but means that "time to target"
        //              invariably skews towards infinity.  This is uninformative.  Hence the
        //              following "inf -> bignum" hack

        if ( nores )
        {
            res = src.size();
        }

        return res;
    }

    void calcmeanvar(gentype &meanres, gentype &varires, const Vector<gentype> &src)
    {
        mean(meanres,src);
        vari(varires,src);

        // Actually we want the variance of the sample mean here, so...

        varires *= 1/((double) src.size());

        return;
    }

    void calcmeanvar(Vector<gentype> &meanres, Vector<gentype> &varires, const Vector<Vector<gentype> > &src)
    {
        Vector<gentype> srcsum;
        Vector<gentype> srcsqsum;

        sum(  srcsum,  src);
        sqsum(srcsqsum,src);

        int numReps = src.size();

        meanres = srcsum;
        meanres.scale(1/((double) numReps));

        varires =  srcsum;
        varires *= srcsum;
        varires.scale(1/((double) numReps));
        varires -= srcsqsum;
        varires.scale(-1/((double) numReps));

        // Actually we want the variance of the sample mean here, so...

        varires.scale(1/((double) numReps));

        return;
    }

    virtual int optim(int dim,
                      Vector<gentype> &xres,
                      Vector<gentype> &Xres,
                      gentype &fres,
                      int &ires,
                      int &mInd,
                      int &muInd,
                      int &sigInd,
                      int &srcmodInd,
                      int &diffmodInd,
                      Vector<Vector<gentype> > &allxres,
                      Vector<Vector<gentype> > &allXres,
                      Vector<gentype> &allfres,
                      Vector<gentype> &allmres,
                      Vector<gentype> &allsres,
                      Vector<double>  &s_score,
                      const Vector<gentype> &xmin,
                      const Vector<gentype> &xmax,
                      const Vector<int> &distMode,
                      const Vector<int> &varsType,
                      void (*fn)(gentype &res, Vector<gentype> &x, void *arg),
                      void *fnarg,
                      svmvolatile int &killSwitch,
                      unsigned int numReps, 
                      gentype &meanfres, gentype &varfres,
                      gentype &meanires, gentype &varires,
                      gentype &meantres, gentype &vartres,
                      gentype &meanTres, gentype &varTres,
                      Vector<gentype> &meanallfres, Vector<gentype> &varallfres,
                      Vector<gentype> &meanallmres, Vector<gentype> &varallmres)
    {
        // numReps:  1 = normal behaviour
        //          >1 = do multiple runs.  Everything returned is for the final run, except
        //               allfres and allmres are now vectors, where the first element is
        //               an average and the second element is a variance.

        int res = 0;


        if ( numReps == 1 )
        {
            int k;

            gentype nullval;
            nullval.makeNull();

            res = realOptim(dim,xres,Xres,fres,ires,mInd,muInd,sigInd,srcmodInd,diffmodInd,allxres,allXres,allfres,allmres,allsres,s_score,xmin,xmax,distMode,varsType,fn,fnarg,killSwitch);

            // Sort allmres to be strictly decreasing!

            for ( k = 0 ; k < allmres.size() ; k++ )
            {
                if ( k )
                {
                    allmres("&",k) = ( allmres(k) < allmres(k-1) ) ? allmres(k) : allmres(k-1);
                }
            }

            meanfres = fres; varfres = nullval;
            meanires = ires; varires = nullval;

            meanallfres = allfres; varallfres.resize(allfres.size()) = nullval;
            meanallmres = allmres; varallmres.resize(allmres.size()) = nullval;

            findfirstLT(meantres,allmres,softmin); vartres = nullval;
            findfirstLT(meanTres,allmres,hardmin); vartres = nullval;
        }

        else if ( numReps > 1 )
        {
            unsigned int j;
            int k;

            Vector<gentype> vecfres(numReps);
            Vector<gentype> vecires(numReps);

            Vector<Vector<gentype> > vecallfres(numReps);
            Vector<Vector<gentype> > vecallmres(numReps);

            Vector<gentype> vectres(numReps);
            Vector<gentype> vecTres(numReps);

            int maxxlen = 0;

            for ( j = 0 ; j < numReps ; j++ )
            {
                GlobalOptions *locopt = makeDup();

                allxres.resize(0);
                allXres.resize(0);
                allsres.resize(0);
                s_score.resize(0);

                res += (*locopt).realOptim(dim,xres,Xres,vecfres("&",j),vecires("&",j).force_int(),mInd,muInd,sigInd,srcmodInd,diffmodInd,allxres,allXres,vecallfres("&",j),vecallmres("&",j),allsres,s_score,xmin,xmax,distMode,varsType,fn,fnarg,killSwitch);

                if ( j == numReps-1 )
                {
                    fres = vecfres(j);
                    ires = (int) vecires(j);

                    allfres = vecallfres(j);
                    allmres = vecallmres(j);
                }

                // Sort vecallmres(j) to be strictly decreasing!

                for ( k = 0 ; k < vecallfres(j).size() ; k++ )
                {
                    if ( k )
                    {
                        vecallmres("&",j)("&",k) = ( vecallmres(j)(k) < vecallmres(j)(k-1) ) ? vecallmres(j)(k) : vecallmres(j)(k-1);
                    }
                }

                maxxlen = ( vecallmres(j).size() > maxxlen ) ? vecallmres(j).size() : maxxlen; 

                findfirstLT(vectres("&",j),vecallmres(j),softmin);
                findfirstLT(vecTres("&",j),vecallmres(j),hardmin);

                MEMDEL(locopt);
            }

            retVector<gentype> tmpva;

            for ( j = 0 ; j < numReps ; j++ )
            {
                if ( ( k = vecallmres(j).size() ) < maxxlen )
                {
                    (vecallfres("&",j).resize(maxxlen))("&",k,1,maxxlen-1,tmpva) = vecallfres(j)(k-1);
                    (vecallmres("&",j).resize(maxxlen))("&",k,1,maxxlen-1,tmpva) = vecallmres(j)(k-1);
                }
            }

            calcmeanvar(meanfres,varfres,vecfres);
            calcmeanvar(meanires,varires,vecires);
            calcmeanvar(meantres,vartres,vectres);
            calcmeanvar(meanTres,varTres,vecTres);

            calcmeanvar(meanallfres,varallfres,vecallfres);
            calcmeanvar(meanallmres,varallmres,vecallmres);
        }

        return res;
    }

    virtual int realOptim(int dim,
                      Vector<gentype> &xres,
                      Vector<gentype> &Xres,
                      gentype &fres,
                      int &ires,
                      int &mInd,
                      int &muInd,
                      int &sigInd,
                      int &srcmodInd,
                      int &diffmodInd,
                      Vector<Vector<gentype> > &allxres,
                      Vector<Vector<gentype> > &allXres,
                      Vector<gentype> &allfres,
                      Vector<gentype> &allmres,
                      Vector<gentype> &allsres,
                      Vector<double>  &s_score,
                      const Vector<gentype> &xmin,
                      const Vector<gentype> &xmax,
                      const Vector<int> &distMode,
                      const Vector<int> &varsType,
                      void (*fn)(gentype &res, Vector<gentype> &x, void *arg),
                      void *fnarg,
                      svmvolatile int &killSwitch)
    {
        (void) muInd;
        (void) sigInd;
        (void) srcmodInd;
        (void) diffmodInd;

        locfnarg = fnarg;

        // dim: dimension of problem
        // xres: x at minimum
        // Xres: xres in format seen by specific method, without projection or distortion
        // fres: f(x) at minimum
        // ires: index of optimal solution in allres
        // allxres: vector of all vectors evaluated
        // allXres: vector of all vectors evaluated, in raw (unprojected/undistorted) format
        // allfres: vector of all f(x) evaluations
        // xmin: lower bound on variables
        // xmax: upper bound on variables
        // distMode: 0 = linear distribution of points
        //           1 = logarithmic distribution of points
        //           2 = anti-logarithmic distribution of points
        //           3 = uniform random distribution of points (grid only)
        //           4 = inverse logistic distribution of points
        //           5 = REMBO
        // varsType: 0 = integer
        //           1 = double
        // fn: function being optimised
        // fnarg: argument passed to fn
        // killSwitch: set nonzero to force stop

        NiceAssert( dim >= 0 );
        NiceAssert( optdefed() );
        NiceAssert( distMode.size() == dim );
        NiceAssert( varsType.size() == dim );

        addSubDim = 0;

        // For schedules Bernstein projection, effdim is the (reduced) dimension for this "round"

        int effdim = dim;

        stopearly  = 0;
        firsttest  = 1;
        spOverride = 0;
        bestyet    = 0.0;

        // The following variables are used to normalise the axis to [0,1].
        // Their exact meaning depends on distMode.

        Vector<gentype> allfakexmin(dim);
        Vector<gentype> allfakexmax(dim);

        xwidth = calcabc(dim,allfakexmin,allfakexmax,a,b,c,xmin,xmax,distMode);

        Vector<gentype> fakexmin(allfakexmin);
        Vector<gentype> fakexmax(allfakexmax);

        locdistMode = distMode;
        locvarsType = varsType;

        // These need to be passed back

        Vector<int> &MLnumbers = *((Vector<int> *) ((void **) fnarg)[15]);
        Vector<int> locMLnumbers(6); // See mlinter definition of MLnumbers

        locMLnumbers = -1;

        // overfn setup

        Vector<gentype> xmod(dim);

        void *overfnargs[16]; // It is very important that overfnargs[15] is defined here!

        overfnargs[0] = (void *)  fnarg;
        overfnargs[1] = (void *) &dim;
        overfnargs[2] = (void *) &xmod;
        overfnargs[3] = (void *)  this;
        overfnargs[4] = (void *) &allxres;
        overfnargs[5] = (void *) &penterm;
        overfnargs[15] = (void *) &locMLnumbers; // if the optimiser gets recursed (which only happens in gridOpt) then this is required to ensure thet MLnumbers is defined and doesn't overlap!

        overfnargs[6] = (void *) fn;

        // Projection templates

        MLnumbers("&",3) = -1;

        if ( isProjection == 0 )
        {
            // No vector/function projection, so do nothing

            ;
        }

        else if ( isProjection == 1 )
        {
            // Create random subspace - vector-valued subspace

            ML_Mutable *temprandDirtemplate;

            MEMNEW(temprandDirtemplate,ML_Mutable);
            (*temprandDirtemplate).setMLTypeClean(defrandDirtemplateVec.type());

            (*temprandDirtemplate).getML() = defrandDirtemplateVec;
            randDirtemplateInd = regML(temprandDirtemplate,fnarg,0);

            randDirtemplate = &((*temprandDirtemplate).getML());

            MLnumbers("&",3) = randDirtemplateInd;
        }

        else if ( isProjection == 2 )
        {
            // Create random subspace - function-valued subspace

            ML_Mutable *temprandDirtemplate;

            MEMNEW(temprandDirtemplate,ML_Mutable);
            (*temprandDirtemplate).setMLTypeClean(defrandDirtemplateFnGP.type());

            (*temprandDirtemplate).getML() = defrandDirtemplateFnGP;
            randDirtemplateInd = regML(temprandDirtemplate,fnarg,1);

            randDirtemplate = &((*temprandDirtemplate).getML());
errstream() << "phantomx 0: randDirtemplate = " << *randDirtemplate << "\n";

            MLnumbers("&",3) = randDirtemplateInd;

            // Need to fill in x distributions.

            Vector<int> sampleInd(fnDim);
            Vector<gentype> sampleDist(fnDim);

            int j;

            for ( j = 0 ; j < fnDim ; j++ )
            {
                sampleInd("&",j)  = j;
                sampleDist("&",j) = "urand(x,y)";

                SparseVector<SparseVector<gentype> > xy;

                xy("&",zeroint())("&",zeroint()) = 0.0;
                xy("&",zeroint())("&",1)         = 1.0;

                sampleDist("&",j).substitute(xy);
            }

            initModelDistr(sampleInd,sampleDist);
        }

        else if ( ( isProjection == 3 ) || ( isProjection == 4 ) )
        {
            // Create non-random subspace - function-valued subspace, Bernstein basis

            ML_Mutable *temprandDirtemplate;

            MEMNEW(temprandDirtemplate,ML_Mutable);
            (*temprandDirtemplate).setMLTypeClean(defrandDirtemplateFnBern.type());

            (*temprandDirtemplate).getML() = defrandDirtemplateFnBern;
            randDirtemplateInd = regML(temprandDirtemplate,fnarg,1);

            randDirtemplate = &((*temprandDirtemplate).getML());

            MLnumbers("&",3) = randDirtemplateInd;

            if ( isProjection == 4 )
            {
                effdim = bernstart;
            }
        }

        else if ( isProjection == 5 )
        {
            // Projection via RKHSVector, so do nothing

            ;
        }

        // Optimisation loop
        //
        // substatus: 0 = this is the first optimisation
        //            1 = not the first optimisation, but the "best" subspace wasn't changed by the most recent optimisation
        //            2 = not the first optimisation, "best" subspace given by subsequent x vector

        int res = 0;

        int ii,jj;
        int substatus = 0;

        // We can't afford to call prelim operations between optimisation and
        // making a subspace, we we pre-make it here, then after each optim

        // Trigger preliminary "setup" operations

        ii = 0;
        {
            gentype trigval;
            Vector<gentype> xdummy;
            SparseVector<gentype> altlocxres;

            trigval.force_int() = (-ii*10000)-1; // -1,-10001,-20001,-30001,...
            (*fn)(trigval,xdummy,fnarg);

            MLnumbers("&",2) = makeSubspace(dim,fnarg,substatus,altlocxres); // until we actually have a projection, at least

            trigval.force_int() = (-ii*10000)-2; // -2,-10002,-20002,-30002,...
            (*fn)(trigval,xdummy,fnarg);
        }

        int locrandReproject = randReproject;

        for ( ii = 0 ; ii <= locrandReproject ; ii++ )
        {
            outstream() << "---\n";

            Vector<gentype> locxres;
            Vector<gentype> locXres;
            SparseVector<gentype> altlocxres;
            SparseVector<gentype> altlocXres;
            gentype locfres;
            int locires;

            Vector<Vector<gentype> > locallxres;
            Vector<Vector<gentype> > locallXres;
            Vector<gentype> locallfres;
            Vector<gentype> locallmres;
            Vector<gentype> locallsres;
            Vector<double>  locs_score;

            stopearly = 0;

            if ( isProjection == 4 )
            {
                spOverride = ii; // only want one initial random batch

                // Update bernstein schedule (note that if isProjection == 3 this intentionally does nothing, and that effdim is bounded by dim)

                if ( ii && ( effdim < dim ) )
                {
                    effdim++;
                }

                // Fake-out unused dimensions for scheduled Bernstein optimisation

                if ( effdim )
                {
                    for ( jj = 0 ; jj < effdim ; jj++ )
                    {
                        fakexmin("&",jj) = allfakexmin(jj);
                        fakexmax("&",jj) = allfakexmax(jj);
                    }
                }

                if ( effdim < dim )
                {
                    for ( jj = effdim ; jj < dim ; jj++ )
                    {
                        fakexmin("&",jj) = allfakexmin(jj);
                        fakexmax("&",jj) = allfakexmin(jj);
                    }
                }
            }

            else
            {
                effdim = dim;
            }

            berndim = effdim;

            // Optimisation

            res = optim(dim,locXres,locfres,locires,locallXres,locallfres,locallmres,locallsres,locs_score,
                        fakexmin,fakexmax,overfn,overfnargs,killSwitch);

            // Update subspace if/as required

            if ( substatus == 0 )
            {
                // Grab both x result *and* put subspace in optimal state

                convertx(dim,locxres,locXres);
                model_convertx(altlocxres,( altlocXres = locXres ));

                // Just completed the first optimisation, so the solution must be the best so far by definition

                xres = locxres;
                Xres = locXres;
                fres = locfres;
                ires = locires+(allfres.size());
                mInd = projOpInd;

                allxres.append(allxres.size(),locallxres);
                allXres.append(allXres.size(),locallXres);
                allfres.append(allfres.size(),locallfres);
                allmres.append(allmres.size(),locallmres);
                allsres.append(allsres.size(),locallsres);
                s_score.append(s_score.size(),locs_score);

                substatus = 2;
            }

            else if ( locfres < fres )
            {
                // Grab both x result *and* put subspace in optimal state

                convertx(dim,locxres,locXres);
                model_convertx(altlocxres,( altlocXres = locXres ));

                // Not first optimisation, but there is a new best solution

                xres = locxres;
                Xres = locXres;
                fres = locfres;
                ires = locires+(allfres.size());
                mInd = projOpInd;

                allxres.append(allxres.size(),locallxres);
                allXres.append(allXres.size(),locallXres);
                allfres.append(allfres.size(),locallfres);
                allmres.append(allmres.size(),locallmres);
                allsres.append(allsres.size(),locallsres);
                s_score.append(s_score.size(),locs_score);

                substatus = 2;
            }

            else
            {
                // Set "x" to zero so that it just reverts to previous best

                locXres = zerogentype();

                // Grab both x result *and* put subspace in optimal state

                convertx(dim,locxres,locXres,1); // Note that useOrigin is set here, so locXres is not actually used!
                model_convertx(altlocxres,( altlocXres = locXres ),1); // Note that useOrigin is set here, so locXres is not actually used!

                // Not first optimisation, no improvement found (but zero set anyhow)

                //xres = locxres;
                //Xres = locXres;
                //fres = locfres;
                //ires = locires+(allfres.size());
                //mInd = projOpInd;

                allxres.append(allxres.size(),locallxres);
                allXres.append(allXres.size(),locallXres);
                allfres.append(allfres.size(),locallfres);
                allmres.append(allmres.size(),locallmres);
                allsres.append(allsres.size(),locallsres);
                s_score.append(s_score.size(),locs_score);

                substatus = 1;
            }

            // Increase iterations if stopearly set (bernstein scheduled stopearly)

            if ( stopearly )
            {
                NiceAssert( isProjection == 4 );

                locrandReproject++;
            }

            // Construct random subspace if required (but not for scheduled Bernstein)

            if ( ii < locrandReproject )
            {
                if ( isProjection != 4 )
                {
                    MLnumbers("&",2) = makeSubspace(dim,fnarg,substatus,altlocxres);
                }

                // Trigger intermediate (hyperparameter tuning) operations

                gentype trigval;
                Vector<gentype> xdummy;

                trigval.force_int() = (-(ii+1)*10000)-2; // -2,-10002,-20002,-30002,...
                (*fn)(trigval,xdummy,fnarg);
            }
        }

        // And we're done

        a.resize(0);
        b.resize(0);
        c.resize(0);

        return res;
    }

    // Overload to set sample distributions in models (if any)

    virtual int initModelDistr(const Vector<int> &sampleInd, const Vector<gentype> &sampleDist) { (void) sampleInd; (void) sampleDist; return 0; }

    // Make random subspace

    int makeSubspace(int dim, void *fnarg, int substatus, const SparseVector<gentype> &locxres)
    {
//errstream() << "phantomx -1\n";
        // substatus zero on first call, 1 or 2 otherwise

        // Note we don't delete any projections blocks as they may be
        // required in the functional optimisation case where return
        // is of form fnB(projOpInd,500,x).  We do register them though
        // so they will be deleted on exit.

        if ( substatus )
        {
            // This function is used by smboopt to clear models
            // if the model is built on x rather than model_convertx(x)

            model_clear();
        }

        if ( isProjection == 0 )
        {
            // No projection, nothing to see here

            addSubDim = 0;

            projOpInd = -1;
        }

        else if ( ( isProjection == 1 ) || ( isProjection == 2 ) )
        {
            // Projection onto either random vectors or random functions.  Both of
            // these are distributions, so we can clone and finalise them to get a
            // random projection.

            // If not first run then we add a dimension (zero point), permanently
            // weighted to 1, that represents the current best solution.

            addSubDim = substatus ? 1 : 0;

            errstream() << "Constructing random subspace... (" << dim+addSubDim << ") ";

            subDef.resize(dim+addSubDim);
            subDefInd.resize(dim+addSubDim);

            int i,j;

            ML_Mutable *randDir;

            errstream() << "creating random direction prototypes... ";

            int rdim = addSubDim ? dim+1 : dim;

            for ( i = 0 ; i < dim ; i++ )
            {
                if ( ( i < dim-1 ) || !( !addSubDim && includeConst ) )
                {
                    MEMNEW(randDir,ML_Mutable);
                    (*randDir).setMLTypeClean((*randDirtemplate).type());

                    (*randDir).getML() = (*randDirtemplate);
                    subDefInd("&",i) = regML(randDir,fnarg,2);

                    subDef("&",i) = &((*randDir).getML());

                    if ( substatus )
                    {
                        consultTheOracle(*randDir,dim,locxres,!i);
                    }
errstream() << "phantomx 1: randDir (" << i << ") = " << *randDir << "\n";
                }

                else
                {
                    errstream() << "include constant term (weighted)... ";

                    // First round, so final variable controls a *constant* offset

                    gentype outfnhere(whatConst);

                    MEMNEW(randDir,ML_Mutable);
                    (*randDir).setMLTypeClean(207);  // BLK_UsrFnb

                    (*randDir).setoutfn(outfnhere);
                    subDefInd("&",i) = regML(randDir,fnarg,2);

                    subDef("&",i) = &((*randDir).getML());
                }
            }

            if ( addSubDim )
            {
                errstream() << "include previous best term (fixed zero point)... ";

                // Put previous best as "fixed" point in new subspace.  projOp
                // has been made the best result from the most recent sub-optimisation.

                randDir = projOpRaw;
                subDefInd("&",dim) = projOpInd;
                subDef("&",dim) = &((*randDir).getML());
            }

            // Make projOp a projection onto this subspace

            MEMNEW(projOpRaw,ML_Mutable);
            (*projOpRaw).setMLTypeClean(projOptemplate.type());

            (*projOpRaw).getML() = projOptemplate;
            projOpInd = regML(projOpRaw,fnarg,3);

            projOp = &(dynamic_cast<BLK_Conect &>((*projOpRaw).getBLK()));
//errstream() << "phantomx 2: projOp = " << *projOp << "\n";

            errstream() << "sampling....";

            Vector<gentype> subWeight(dim+addSubDim);

            gentype biffer;

            subWeight = ( biffer = 0.0 );

            if ( addSubDim )
            {
                subWeight("&",dim) = 1.0;
            }

            (*projOp).setmlqlist(subDef);
            (*projOp).setmlqweight(subWeight);

//FIXMEFIXME fnDim
            errstream() << "combining... ";

            Vector<gentype> xmin(fnDim);
            Vector<gentype> xmax(fnDim);

            gentype buffer;

            xmin = ( buffer = 0.0 );
            xmax = ( buffer = 1.0 );

            (*projOp).setSampleMode(1,xmin,xmax,xNsamp); // Need this even when subDef has been sampled to set variables used to construct y in projOp, when required
//errstream() << "phantomx 3: subDef (0) = " << *subDef(0) << "\n";
errstream() << "phantomx 4: projOp = " << (*projOp).y() << "\n";

            if ( useScalarFn && !includeConst && ( isProjection == 2 ) )
            {
                xbasis.resize(rdim);

                for ( i = 0 ; i < rdim ; i++ )
                {
                    xbasis("&",i) = (*subDef(i)).y();
                    xbasis("&",i).scale(sqrt(1/((double) (*subDef(i)).y().size())));
                }
//errstream() << "phantomx 5: xbasis = " << xbasis << "\n";

                gentype tempxp;

                xbasisprod.resize(rdim,rdim);

                for ( i = 0 ; i < rdim ; i++ )
                {
                    for ( j = 0 ; j < rdim ; j++ )
                    {
                        twoProduct(xbasisprod("&",i,j),xbasis(i),xbasis(j));
                    }
                }
//errstream() << "phantomx 6: xbasisprod = " << xbasisprod << "\n";

                model_update();
            }

            errstream() << "done with weight " << ((*projOp).mlqweight()) << "\n";
        }

        else if ( ( isProjection == 3 ) || ( isProjection == 4 ) )
        {
            // We set up the same dimension, even though we don't use some of it

            NiceAssert( !includeConst );

            addSubDim = 0;

            errstream() << "Constructing Bernstein basis... ";

            subDef.resize(dim);
            subDefInd.resize(dim);

            int i,j,k;
            int axisdim = (int) pow((double) dim+1,1/((double) fnDim))-1; // floor

            Vector<double> berndim(fnDim);
            Vector<double> bernind(fnDim);

            berndim = (double) axisdim;
            bernind = 0.0;

            ML_Mutable *randDir;

            errstream() << "setting parameters... ";

            for ( i = 0 ; i < dim ; i++ )
            {
                MEMNEW(randDir,ML_Mutable);
                (*randDir).setMLTypeClean((*randDirtemplate).type());

                (*randDir).getML() = (*randDirtemplate);
                subDefInd("&",i) = regML(randDir,fnarg,8);

                subDef("&",i) = &((*randDir).getML());

                k = i;

                for ( j = 0 ; j < fnDim ; j++ )
                {
                    bernind("&",j) = k%(((int) berndim(j))+1);
                    k /= ((int) berndim(j))+1;
                }

                gentype gberndim(berndim);
                gentype gbernind(bernind);

//FIXMEFIXME fnDim
                dynamic_cast<BLK_Bernst &>((*(subDef("&",i)))).setBernDegree(gberndim);
                dynamic_cast<BLK_Bernst &>((*(subDef("&",i)))).setBernIndex(gbernind);
            }

            errstream() << "combining... ";

            // Make projOp a projection onto this subspace

            MEMNEW(projOpRaw,ML_Mutable);
            (*projOpRaw).setMLTypeClean(projOptemplate.type());

            (*projOpRaw).getML() = projOptemplate;
            projOpInd = regML(projOpRaw,fnarg,9);

            projOp = &(dynamic_cast<BLK_Conect &>((*projOpRaw).getBLK()));

            errstream() << "sampling....";

            Vector<gentype> subWeight(dim);

            gentype biffer;

            subWeight = ( biffer = 0.0 );

            (*projOp).setmlqlist(subDef);
            (*projOp).setmlqweight(subWeight);

//FIXMEFIXME fnDim
            Vector<gentype> xmin(fnDim);
            Vector<gentype> xmax(fnDim);

            gentype buffer;

            xmin = ( buffer = 0.0 );
            xmax = ( buffer = 1.0 );

            (*projOp).setSampleMode(1,xmin,xmax,xNsamp); // Need this even when subDef has been sampled to set variables used to construct y in projOp, when required

            errstream() << "done with weight " << ((*projOp).mlqweight()) << "\n";
        }

        else if ( isProjection == 5 )
        {
            // Projection via RKHSVector, nothing to see here

            addSubDim = 0;

            defrandDirtemplateFnRKHS.resizeN(dim);

            int i,j;

            for ( i = 0 ; i < dim ; i++ )
            {
                defrandDirtemplateFnRKHS.x("&",i).zero();

                for ( j = 0 ; j < fnDim ; j++ )
                {
                    randfill(defrandDirtemplateFnRKHS.x("&",i)("&",j).force_double());
                }
            }

            projOpInd = -1;
        }

        return projOpInd;
    }

    // Results analysis.  In general the optim function will return one
    // result only, or none if this is a multi-objective bayesian optimisation
    // and/or there are not optima.  The following function goes through the
    // list of all results and returns the set of minima (or the Pareto set
    // if this is a vector optimisation problem).  Returns the number of
    // points in optimal set and sets optres index vector that contains the
    // indices of the Pareto set.
    //
    // hypervol: set to the hypervolume (or just best result so far).  This
    //           is actually based on -allfres in the scalar case, hypervolume
    //           dominated in the negative quadrant otherwise.
    // fdecreasing: min f up to iteration.

    int analyse(const Vector<Vector<gentype> > &allxres,
                const Vector<gentype> &allmres,
                Vector<double> &hypervol,
                Vector<int> &parind,
                int calchypervol) const; // = 1) const;

    // Internal function to differentiate between the base class and the
    // actual optimisation problem classes.  Will return zero for base
    // class, unique index for various global optimisers.

    virtual int optdefed(void)
    {
        return 0;
    }

    // Return 1 if x conversion is non-trivial

    int isXconvertNonTrivial(void) const
    {
        return isProjection || sum(locdistMode);
    }

    // model_clear: clear data from model if relevant.
    // model_update: update any pre-calculated inner-products with model vectors (only called if relevant)

    virtual void model_clear(void) { return; }
    virtual void model_update(void) { return; }

    // Direction oracle: this one is a simple random selection oracle.
    // Specialise this to implement more specialised oracles (direction etc).
    // See eg smboopt
    //
    // isFirstAxis: the first axis may be treated differently than the rest in some cases

    virtual void consultTheOracle(ML_Mutable &randDir, int dim, const SparseVector<gentype> &locxres, int isFirstAxis)
    {
        (void) dim;
        (void) locxres;
        (void) isFirstAxis;

//FIXMEFIXME fnDim
        errstream() << "combining... ";

        Vector<gentype> xmin(fnDim);
        Vector<gentype> xmax(fnDim);

        gentype buffer;

        xmin = ( buffer = 0.0 );
        xmax = ( buffer = 1.0 );

        randDir.setSampleMode(1,xmin,xmax,xNsamp);

        return;
    }

    // Convert x to p(x) (we want to optimise f(p(x)))
    //
    // useShortcut: 0 usual - calculate res
    //              1 only calculate xpweight , not res (if possible: otherwise calculate res)
    //              2 calculate both xpweight and res (where possible)

    Vector<gentype> xpweight; // We assume that this is single-threaded
    int xpweightIsWeight; // if 1 then xpweight contains the actual weights
    Vector<SparseVector<gentype> > xbasis; // when xpweightIsWeight set this will be the basis vectors themselves
    Matrix<gentype> xbasisprod; // when xpweightIsWeight set this will be the inner products of the basis vectors (function) themselves

    template <class S>
    const SparseVector<gentype> &model_convertx(SparseVector<gentype> &res, const SparseVector<S> &x, int useOrigin = 0, int useShortcut = 0, int givefeedback = 0) const
    {
//errstream() << "phantomxyz 0: " << a << "\n";
//errstream() << "phantomxyz 0: " << b << "\n";
//errstream() << "phantomxyz 0: " << c << "\n";
//givefeedback = 1;
if ( givefeedback )
{
errstream() << "phantomxyzqqq 0\n";
}
        (**thisthisthis).xpweightIsWeight = 0;

        if ( (void *) &res == (void *) &x )
        {
            SparseVector<S> tempx(x);

            return model_convertx(res,tempx,useOrigin);
        }

if ( givefeedback )
{
errstream() << "phantomxyzqqq 1\n";
}
        int dim = x.indsize();

if ( givefeedback )
{
errstream() << "phantomxyzqqq 2: " << dim << "\n";
}
        res.indalign(x);
if ( givefeedback )
{
errstream() << "phantomxyzqqq 3\n";
}

        if ( dim )
        {
            if ( a.size() )
            {
                int i,j,ii;

                for ( ii = 0 ; ii < dim ; ii++ )
                {
                    (res.direref(ii)).scalarfn_setisscalarfn(0);

                    i = x.ind(ii);

                    if ( locdistMode(i) == 1 )
                    {
                        // 1: v = a + e^(b+c.t)

                        //res.direref(ii) = a(i)+exp(b(i)+(c(i)*x.direcref(ii)));

                        res.direref(ii)  = c(i);
                        res.direref(ii) *= x.direcref(ii);
                        res.direref(ii) += b(i);
                        OP_exp(res.direref(ii));
                        res.direref(ii) += a(i);
                    }

                    else if ( locdistMode(i) == 2 )
                    {
                        // 2: v = (1/c) log(t-a) - (b/c)

                        //res.direref(ii) = (log(x.direcref(ii)-a(i))-b(i))/c(i);

                        res.direref(ii)  = x.direcref(ii);
                        res.direref(ii) -= a(i);
                        OP_log(res.direref(ii));
                        res.direref(ii) -= b(i);
                        res.direref(ii) /= c(i);
                    }

                    else if ( locdistMode(i) == 4 )
                    {
                        // 4: v = a - (1/b) log( 1/(0.5+(c*(t-0.5))) - 1 )
                        //      = a - (1/b) log(0.5-(c*(t-0.5))) + (1/b) log(0.5+(c*(t-0.5)))
                        //      = a - (1/b) log(1/(0.5+(c*(t-0.5)))) + (1/b) log(1/(0.5-(c*(t-0.5))))
                        //      = a + (1/b) log(1/(0.5-(c*(t-0.5)))) - (1/b) log(1/(0.5+(c*(t-0.5))))
                        //      = a + (1/b) log(1/tm) - (1/b) log(1/tp)

                        //res.direref(ii) = a(i)-(( log(1.0/(0.5+(c(i)*(x.direcref(ii)-0.5)))) - log(1.0/(0.5-(c(i)*(x.direcref(ii)-0.5)))) )/b(i));

                        gentype tm,tp;

                        tp  = x.direcref(ii);
                        tp -= 0.5;
                        tp *= c(i);

                        tm  = 0.5;
                        tm -= tp;

                        tp += 0.5;

                        tm.inverse();
                        OP_log(tm);

                        tp.inverse();
                        OP_log(tp);
                        
                        res.direref(ii)  = tm;
                        res.direref(ii) -= tp;
                        res.direref(ii) /= b(i);
                        res.direref(ii) += a(i);
                    }

                    else
                    {
                        //res.direref(ii) = a(i)+(b(i)*x.direcref(ii));

                        res.direref(ii)  = b(i);
                        res.direref(ii) *= x.direcref(ii);
                        res.direref(ii) += a(i);
                    }

                    if ( locvarsType(i) == 0 )
                    {
                        j = roundnearest((double) x.direcref(ii));

                        res.direref(ii).force_int() = j;
                    }

                    if ( useOrigin )
                    {
                        res.direref(ii).force_double() = 0.0;
                    }
                }
            }

            if ( ( isProjection == 4 ) && ( berndim < dim ) )
            {
                // Project up to equivalent (rather than messing with dim of bernstein polynomials

                int i,j;

//errstream() << "phantomx 0b start: " << res << "\n";
                for ( i = berndim ; i < dim ; i++ )
                {
                    for ( j = i ; j >= 0 ; j-- )
                    {
                        if ( j == 0 )
                        {
                            res.direref(j) = ( 1 - (((double) j)/((double) i)) )*res.direcref(j);
                        }

                        else if ( j < i )
                        {
                            res.direref(j) = (( (((double) j)/((double) i)) )*res.direcref(j-1)) + (( 1 - (((double) j)/((double) i)) )*res.direcref(j));
                        }

                        else
                        {
                            res.direref(j) = (( (((double) j)/((double) i)) )*res.direcref(j-1));
                        }
                    }
                }
//errstream() << "phantomx 0b end: " << res << "\n";
            }

//errstream() << "phantomxyz 1: " << res << "\n";
            if ( isProjection && ( isProjection <= 4 ) )
            {
//errstream() << "phantomxyz 2\n";
                // This version is used to maintain models in smboopt.h, so we need to ensure that the result only includes the projection

                Vector<gentype> &pweight = (**thisthisthis).xpweight;

                retVector<gentype> tmpva;
                retVector<gentype> tmpvb;

                pweight.resize(addSubDim ? dim+1 : dim);
                pweight("&",0,1,dim-1,tmpvb) = res(tmpva);

                if ( addSubDim )
                {
                    pweight("&",dim) = 1.0;
                }

if ( givefeedback )
{
errstream() << "ML weight = " << pweight << "\n";
}
                (*((**thisthisthis).projOp)).setmlqweight(pweight);

//errstream() << "phantomxyz 2.0\n";
                if ( useShortcut && useScalarFn && !includeConst && ( isProjection == 2 ) )
                {
                    (**thisthisthis).xpweightIsWeight = 1;
                }

//errstream() << "phantomxyz 2.1\n";
                if ( ( useShortcut != 1 ) || ( useShortcut && useScalarFn && !includeConst && ( isProjection == 2 ) ) )
                {
//errstream() << "phantomxyz 2.2\n";
                    if ( isProjection == 1 )
                    {
                        SparseVector<gentype> xxmod;
                        static SparseVector<gentype> xdummy;

                        (*((**thisthisthis).projOp)).gg(xxmod("&",(res.ind())(zeroint())),xdummy);

                        res = xxmod;
                    }

                    else if ( isProjection == 2 )
                    {
                        if ( useScalarFn )
                        {
                            // Faster version: approximate function with vector evaluated on grid.

                            res = (*projOp).y(); // BLK_Conect does the work here.
                            res.scale(sqrt(1/((double) (*projOp).y().size()))); // To ensure the inner product approximates the true integral.  Note also, we use y().size() as N() won't work as you might expect for blk_bernst
                        }

                        else
                        {
                            // We need to duplicate projOp here so that each direction is kept

                            SparseVector<gentype> xxmod;

                            ML_Mutable *projOpNowRaw;
                            int projOpNowInd;

                            MEMNEW(projOpNowRaw,ML_Mutable);
                            (*projOpNowRaw).setMLTypeClean((*projOp).type());

                            (*projOpNowRaw).getML() = *projOp;
                            projOpNowInd = (**thisthisthis).regML(projOpNowRaw,locfnarg,4);

                            // Default (work anywhere) version

//FIXMEFIXME fnDim
                            //xxmod("&",(res.ind())(zeroint())) = "fnB(var(1,0),500,x)"; // now gg(x) of ML var(1,0) (see instructvar.txt)
                            xxmod("&",(res.ind())(zeroint())) = "fnB(var(1,0),500,var(2,0))"; // now gg([x y z ...]) of ML var(1,0) (see instructvar.txt)

                            retVector<int> tmpva;

                            xxmod("&",(res.ind())(zeroint())).scalarfn_setisscalarfn(useScalarFn);
                            xxmod("&",(res.ind())(zeroint())).scalarfn_seti(zerointvec(fnDim,tmpva));
                            xxmod("&",(res.ind())(zeroint())).scalarfn_setj(cntintvec(fnDim,tmpva));
                            xxmod("&",(res.ind())(zeroint())).scalarfn_setnumpts(xNsamp);

                            Vector<gentype> varlist(fnDim);

                            int ii;

                            for ( ii = 0 ; ii < fnDim ; ii++ )
                            {
                                std::stringstream resbuffer;

                                resbuffer << "var(0," << ii << ")";
                                resbuffer >> varlist("&",ii);
                            }

                            SparseVector<SparseVector<gentype> > xy;
                            xy("&",1)("&",0) = projOpNowInd; // fill in var(1,0) with registered projOp index
                            xy("&",2)("&",0) = varlist; // replace var(0,ivect(0,1,var(2,0)-1)) with [ x y ... ]
                            xxmod("&",(res.ind())(zeroint())).substitute(xy); // now gg(x,y,...) for ML projOp (that is, a function)

                            res = xxmod;
                        }
                    }

                    else if ( ( isProjection == 3 ) || ( isProjection == 4 ) )
                    {
//errstream() << "phantomxyz 3\n";
                        if ( useScalarFn )
                        {
//errstream() << "phantomxyz 4\n";
                            // Faster version: approximate function with vector evaluated on grid.

                            res = (*projOp).y(); // BLK_Conect does the work here.
//errstream() << "phantomxyz 5: " << res << "\n";
                            res.scale(sqrt(1/((double) (*projOp).y().size()))); // To ensure the inner product approximates the true integral.  Note also, we use y().size() as N() won't work as you might expect for blk_bernst
//errstream() << "phantomxyz 6\n";
                        }

                        else
                        {
//errstream() << "phantomxyz 7\n";
                            // We need to duplicate projOp here so that each direction is kept

                            SparseVector<gentype> xxmod;

                            ML_Mutable *projOpNowRaw;
                            int projOpNowInd;

                            MEMNEW(projOpNowRaw,ML_Mutable);
                            (*projOpNowRaw).setMLTypeClean((*projOp).type());

                            (*projOpNowRaw).getML() = *projOp;
                            projOpNowInd = (**thisthisthis).regML(projOpNowRaw,locfnarg,4);

                            // Default (work anywhere) version

//FIXMEFIXME: fnDim
                            //xxmod("&",(res.ind())(zeroint())) = "fnB(var(1,0),500,x)"; // now gg(x) of ML var(1,0) (see instructvar.txt)
                            xxmod("&",(res.ind())(zeroint())) = "fnB(var(1,0),500,var(2,0))"; // now gg([x y z ...]) of ML var(1,0) (see instructvar.txt)

                            retVector<int> tmpva;

                            xxmod("&",(res.ind())(zeroint())).scalarfn_setisscalarfn(useScalarFn);
                            xxmod("&",(res.ind())(zeroint())).scalarfn_seti(zerointvec(fnDim,tmpva));
                            xxmod("&",(res.ind())(zeroint())).scalarfn_setj(cntintvec(fnDim,tmpva));
                            xxmod("&",(res.ind())(zeroint())).scalarfn_setnumpts(xNsamp);

                            Vector<gentype> varlist(fnDim);

                            int ii;

                            for ( ii = 0 ; ii < fnDim ; ii++ )
                            {
                                std::stringstream resbuffer;

                                resbuffer << "var(0," << ii << ")";
                                resbuffer >> varlist("&",ii);
                            }

                            SparseVector<SparseVector<gentype> > xy;
                            xy("&",1)("&",0) = projOpNowInd; // fill in var(1,0) with registered projOp index
                            xy("&",2)("&",0) = varlist; // replace var(0,ivect(0,1,var(2,0)-1)) with [ x y ... ]
                            xxmod("&",(res.ind())(zeroint())).substitute(xy); // now gg(x) for ML projOp (that is, a function)

                            res = xxmod;
                        }
                    }
//errstream() << "phantomxyz 2.20\n";
                }
            }

            else if ( isProjection == 5 )
            {
//errstream() << "phantomxy 0\n";
                NiceAssert( res.indsize() == defrandDirtemplateFnRKHS.N() );

//errstream() << "phantomxy 1\n";
                SparseVector<gentype> xxmod;

//errstream() << "phantomxy 2\n";
                retVector<gentype> tmpva;
                retVector<gentype> tmpvb;

//errstream() << "phantomxy 3\n";
                RKHSVector realres(defrandDirtemplateFnRKHS);

//errstream() << "phantomxy 4\n";
                realres.a("&",tmpva) = res(tmpvb); // strip off sparseness (RHS), assign non-sparse version to alpha (weights) in RKHS
//errstream() << "phantomxy 5\n";
                xxmod("&",(res.ind())(zeroint())) = realres; // set zeroth index of xxmod as RKHSVector using magic

//errstream() << "phantomxy 6\n";
                res = xxmod; // Assign result
//errstream() << "phantomxy 7\n";
            }
//errstream() << "phantomxyz 2.40\n";
        }

        return res;
    }

    template <class S>
    const Vector<gentype> &convertx(int dim, Vector<gentype> &res, const Vector<S> &x, int useOrigin = 0, int givefeedback = 0) const
    {
        if ( (void *) &res == (void *) &x )
        {
            Vector<S> tempx(x);

            return convertx(dim,res,tempx);
        }

        res.resize(x.size());

        if ( x.size() )
        {
            if ( dim )
            {
                if ( a.size() )
                {
                    int i,j;

                    //for ( i = 0 ; ( i < dim ) && ( i < x.size() ) ; i++ )
                    for ( i = 0 ; i < x.size() ; i++ )
                    {
                        if ( i < dim )
                        {
                            res("&",i).scalarfn_setisscalarfn(0);

                            if ( locdistMode(i) == 1 )
                            {
                                // 1: v = a + e^(b+c.t)

                                //res("&",i) = a(i)+exp(b(i)+(c(i)*x(i)));

                                res("&",i)  = c(i);
                                res("&",i) *= x(i);
                                res("&",i) += b(i);
                                OP_exp(res("&",i));
                                res("&",i) += a(i);
                            }

                            else if ( locdistMode(i) == 2 )
                            {
                                // 2: v = (1/c) log(t-a) - (b/c)

                                //res("&",i) = (log(x(i)-a(i))-b(i))/c(i);

                                res("&",i)  = x(i);
                                res("&",i) -= a(i);
                                OP_log(res("&",i));
                                res("&",i) -= b(i);
                                res("&",i) /= c(i);
                            }

                            else if ( locdistMode(i) == 4 )
                            {
                                // 4: v = a - (1/b) log( 1/(0.5+(c*(t-0.5))) - 1 )
                                //      = a - (1/b) log(0.5-(c*(t-0.5))) + (1/b) log(0.5+(c*(t-0.5)))
                                //      = a - (1/b) log(1/(0.5+(c*(t-0.5)))) + (1/b) log(1/(0.5-(c*(t-0.5))))

                                //res("&",i) = a(i)-(( log(1.0/(0.5+(c(i)*(x(i)-0.5)))) - log(1.0/(0.5-(c(i)*(x(i)-0.5)))) )/b(i));

                                gentype tm,tp;

                                tp  = x(i);
                                tp -= 0.5;
                                tp *= c(i);

                                tm  = 0.5;
                                tm -= tp;

                                tp += 0.5;

                                tm.inverse();
                                OP_log(tm);

                                tp.inverse();
                                OP_log(tp);

                                res("&",i)  = tm;
                                res("&",i) -= tp;
                                res("&",i) /= b(i);
                                res("&",i) += a(i);
                            }

                            else
                            {
                                //res("&",i) = a(i)+(b(i)*x(i));

                                res("&",i)  = b(i);
                                res("&",i) *= x(i);
                                res("&",i) += a(i);
                            }

                            if ( locvarsType(i) == 0 )
                            {
                                j = roundnearest((double) x(i));

                                res("&",i).force_int() = j;
                            }

                            if ( useOrigin )
                            {
                                res("&",i).force_double() = 0.0;
                            }
                        }

                        else if ( i == dim )
                        {
                            // for grid optimisation, this hold the expected (pre-processing) y value

                            res("&",i).scalarfn_setisscalarfn(0);
                            res("&",i) = x(i);
                        }
                    }
                }
            }

            if ( ( isProjection == 4 ) && ( berndim < dim ) )
            {
                // Project up to equivalent (rather than messing with dim of bernstein polynomials

                int i,j;

//errstream() << "phantomx 0 start: " << res << "\n";
                for ( i = berndim ; i < dim ; i++ )
                {
                    for ( j = i ; j >= 0 ; j-- )
                    {
                        if ( j == 0 )
                        {
                            res("&",j) = ( 1 - (((double) j)/((double) i)) )*res(j);
                        }

                        else if ( j < i )
                        {
                            res("&",j) = (( (((double) j)/((double) i)) )*res(j-1)) + (( 1 - (((double) j)/((double) i)) )*res(j));
                        }

                        else
                        {
                            res("&",j) = (( (((double) j)/((double) i)) )*res(j-1));
                        }
                    }
                }
//errstream() << "phantomx 0 end: " << res << "\n";
            }

            if ( isProjection && ( isProjection <= 4 ) )
            {
                // This version is used to send data back to mlinter, so xxmod needs to 
                // be "filled out" to the same size as x *even though only the first bit is 
                // relevant*.  Note also that this is "after", so projOp is "fixed" from
                // here, meaning that we don't need to duplicate.

                Vector<gentype> &pweight = (**thisthisthis).xpweight;

                retVector<gentype> tmpva;

                pweight.resize(addSubDim ? dim+1 : dim);
                pweight("&",0,1,dim-1,tmpva) = res;

                if ( addSubDim )
                {
                    pweight("&",dim) = 1.0;
                }

if ( givefeedback )
{
errstream() << "ML weight (2) = " << pweight << "\n";
}
                (*((**thisthisthis).projOp)).setmlqweight(pweight);

                if ( isProjection == 1 )
                {
                    Vector<gentype> xxmod(res);
                    static SparseVector<gentype> xdummy;

                    (*((**thisthisthis).projOp)).gg(xxmod("&",zeroint()),xdummy);

                    res = xxmod;
                }

                else if ( isProjection == 2 )
                {
//errstream() << "phantomxyglob 0: y = " << (*projOp).y() << "\n";
                    Vector<gentype> xxmod(res);

//FIXMEFIXME fnDim
                    //xxmod("&",zeroint()) = "fnB(var(1,0),500,x)"; // now gg(x) of ML y (see instructvar.txt)
                    xxmod("&",zeroint()) = "fnB(var(1,0),500,var(2,0))"; // now gg([x y z ...]) of ML var(1,0) (see instructvar.txt)

                    retVector<int> tmpva;

                    xxmod("&",zeroint()).scalarfn_setisscalarfn(useScalarFn);
                    xxmod("&",zeroint()).scalarfn_seti(zerointvec(fnDim,tmpva));
                    xxmod("&",zeroint()).scalarfn_setj(cntintvec(fnDim,tmpva));
                    xxmod("&",zeroint()).scalarfn_setnumpts(xNsamp);

                    Vector<gentype> varlist(fnDim);

                    int ii;

                    for ( ii = 0 ; ii < fnDim ; ii++ )
                    {
                        std::stringstream resbuffer;

                        resbuffer << "var(0," << ii << ")";
//errstream() << "var(0," << ii << ")";
                        resbuffer >> varlist("&",ii);
                    }
//errstream() << "\n";

                    SparseVector<SparseVector<gentype> > xy;
                    xy("&",1)("&",0) = projOpInd; // fill in var(1,0) with registered projOp index
                    xy("&",2)("&",0) = varlist; // replace var(0,ivect(0,1,var(2,0)-1)) with [ x y ... ]
                    xxmod("&",zeroint()).substitute(xy);
//errstream() << "phantomxyglob 1 xxmod = " << xxmod << "\n";

                    res = xxmod;
                }

                else if ( ( isProjection == 3 ) || ( isProjection == 4 ) )
                {
                    Vector<gentype> xxmod(res);

//FIXMEFIXME fnDim
                    //xxmod("&",zeroint()) = "fnB(var(1,0),500,x)"; // now gg(x) of ML y (see instructvar.txt)
                    xxmod("&",zeroint()) = "fnB(var(1,0),500,var(2,0))"; // now gg([x y z ...]) of ML var(1,0) (see instructvar.txt)

                    retVector<int> tmpva;

                    xxmod("&",zeroint()).scalarfn_setisscalarfn(useScalarFn);
                    xxmod("&",zeroint()).scalarfn_seti(zerointvec(fnDim,tmpva));
                    xxmod("&",zeroint()).scalarfn_setj(cntintvec(fnDim,tmpva));
                    xxmod("&",zeroint()).scalarfn_setnumpts(xNsamp);

                    Vector<gentype> varlist(fnDim);

                    int ii;

                    for ( ii = 0 ; ii < fnDim ; ii++ )
                    {
                        std::stringstream resbuffer;

                        resbuffer << "var(0," << ii << ")";
                        resbuffer >> varlist("&",ii);
                    }

                    SparseVector<SparseVector<gentype> > xy;
                    xy("&",1)("&",0) = projOpInd; // fill in var(1,0) with registered projOp index
                    xy("&",2)("&",0) = varlist; // replace var(0,ivect(0,1,var(2,0)-1)) with [ x y ... ]
                    xxmod("&",zeroint()).substitute(xy);

                    res = xxmod;
                }
            }

            else if ( isProjection == 5 )
            {
//errstream() << "phantomxy 10\n";
                NiceAssert( res.size() == defrandDirtemplateFnRKHS.N() );

//errstream() << "phantomxy 11\n";
                Vector<gentype> xxmod(res);

//errstream() << "phantomxy 12\n";
                retVector<gentype> tmpva;

//errstream() << "phantomxy 13\n";
                RKHSVector realres(defrandDirtemplateFnRKHS);

//errstream() << "phantomxy 14\n";
                realres.a("&",tmpva) = res; // assign weights to alpha in RKHS
//errstream() << "phantomxy 15\n";
                xxmod("&",0) = realres; // set zeroth index of xxmod as RKHSVector using magic
//errstream() << "phantomxy 16: " << res << "\n";
//errstream() << "phantomxy 16: " << xxmod << "\n";
//errstream() << "phantomxy 16 go\n";
//godebug = 1;

                res = xxmod; // Assign result
//errstream() << "phantomxy 17\n";
//errstream() << "phantomxy 18: " << res << "\n";
//errstream() << "phantomxy 19\n";
            }
        }

        return res;
    }

    template <class S>
    const Vector<SparseVector<gentype> > &model_convertx(Vector<SparseVector<gentype> > &res, const Vector<SparseVector<S> > &x) const
    {
        res = x;

        if ( x.size() )
        {
            int i;

            for ( i = 0 ; i < x.size() ; i++ )
            {
                model_convertx(res("&",i),x(i));
            }
        }

        return res;
    }

    template <class S>
    const Vector<Vector<gentype> > &convertx(int dim, Vector<Vector<gentype> > &res, const Vector<Vector<S> > &x) const
    {
        res = x;

        if ( x.size() )
        {
            int i;

            for ( i = 0 ; i < x.size() ; i++ )
            {
                convertx(dim,res("&",i),x(i));
            }
        }

        return res;
    }

private:
    // Convert vectors from optimisation space to "real" space.  The optimisers 
    // work on (see) the "optimisation" space, which is then converted to "real"
    // space when evaluating f(x).

    Vector<double> a;
    Vector<double> b;
    Vector<double> c;

    Vector<int> locdistMode;
    Vector<int> locvarsType;

    // Private variables

    BLK_Conect projOptemplate;
    BLK_Conect *projOp;
    ML_Mutable *projOpRaw;
    int projOpInd;

    ML_Base *randDirtemplate;
    int randDirtemplateInd;
    Vector<ML_Base *> subDef;
    Vector<int> subDefInd;

    int addSubDim; // additional (fixed) dimensions on subspace

    void *locfnarg;

    GlobalOptions *thisthis;
    GlobalOptions **thisthisthis;
};





#endif



