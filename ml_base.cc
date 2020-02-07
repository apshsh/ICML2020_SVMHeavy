/*
methodkey 4: set if gradorder > 0.  For everything but K2 assert that if 4 set then 2 should be set
FIXME: when igradorder > 0 but no farfar then need to return a vector or matrix of appropriate size.  Basic method is to use
a helper to try to (a) turn T into vector/matrix (throw if not gentype) and (b) access element i (or i,j) of T (throw if not gentype).
Then jam the vectorised result (or de-vectorised if it's a full covariance) into this.  This result then naturally filters through
ghTrainingVector(-1) of various models (though need to make sure ghTrainginVector doesn't *assume* K double for -1 as per
svm_scalar) and the result naturally comes out at the end.  Also note in instructions that setting a_6 without gradient will
give a vector or matrix, which could potentially be used as a matrix-valued kernel.
*/
//
// ML (machine learning) base type
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "mercer.h"
#include "ml_base.h"



svmvolatile SparseVector<int>* ML_Base::xvernumber = NULL;
svmvolatile SparseVector<int>* ML_Base::gvernumber = NULL;
svmvolatile svm_mutex ML_Base::mleyelock;


// Training vector conversion

int convertSetToSparse(SparseVector<gentype> &res, const gentype &srrc, int idiv)
{
    if ( srrc.isValSet() )
    {
        const Set<gentype> &src = srrc.cast_set(2); // Note use of 2 here to finalise globals but not randoms!

        res.zero();

        NiceAssert( src.size() <= 5 );

        int i = 0;
        int j;

        if ( src.size() >= 1 )
        {
            if ( !((src.all())(i).isValNull()) )
            {
                NiceAssert( (src.all())(i).isCastableToVectorWithoutLoss() );

                for ( j = 0 ; j < (src.all())(i).size() ; j++ )
                {
                    res("&",j) = ((const Vector<gentype> &) (src.all())(i))(j);
                }
            }

            i++;
        }

        if ( src.size() >= 2 )
        {
            if ( !((src.all())(i).isValNull()) )
            {
                NiceAssert( (src.all())(i).isCastableToVectorWithoutLoss() );

                for ( j = 0 ; j < (src.all())(i).size() ; j++ )
                {
                    res.f("&",j) = ((const Vector<gentype> &) (src.all())(i))(j);
                }
            }

            i++;
        }

        if ( src.size() >= 3 )
        {
            if ( !((src.all())(i).isValNull()) )
            {
                NiceAssert( (src.all())(i).isCastableToVectorWithoutLoss() );

                for ( j = 0 ; j < (src.all())(i).size() ; j++ )
                {
                    res.ff("&",j) = ((const Vector<gentype> &) (src.all())(i))(j);
                }
            }

            i++;
        }

        if ( src.size() >= 4 )
        {
            if ( !((src.all())(i).isValNull()) )
            {
                NiceAssert( (src.all())(i).isCastableToVectorWithoutLoss() );

                for ( j = 0 ; j < (src.all())(i).size() ; j++ )
                {
                    res.fff("&",j) = ((const Vector<gentype> &) (src.all())(i))(j);
                }
            }

            i++;
        }
    }

    else
    {
//        const Vector<gentype> &src = (const Vector<gentype> &) srrc;
//
//        res = src;
        res = srrc.cast_vector(0); // Note use of 2 here to finalise globals but not randoms!
    }

    if ( idiv > 0 )
    {
        res.fff("&",6) += idiv;
    }

//    return ( srrc.isValEqn() & 0x10 ) && !(srrc.scalarfn_isscalarfn());
//    return srrc.isValEqn() && !(srrc.scalarfn_isscalarfn());
    return srrc.scalarfn_isscalarfn() ? 0 : ( srrc.isValEqn() & 8 );
}

int convertSparseToSet(gentype &rres, const SparseVector<gentype> &src)
{
    if ( src.indsize() == src.nearindsize() )
    {
        Set<gentype> &res = rres.force_set();

        res.zero();

        int doit = 0;

        Vector<gentype> temp;
        gentype nulldummy('N');

             if ( src.farfarfarindsize() ) { doit = 4; }
        else if ( src.farfarindsize()    ) { doit = 3; }
        else if ( src.farindsize()       ) { doit = 2; }
        else if ( src.nearindsize()      ) { doit = 1; }

        if ( doit >= 1 )
        {
            if ( src.nearindsize() )
            {
                retVector<gentype> tmpva;
                gentype toadd(src.nearref()(tmpva));

                res.add(toadd);
            }

            else
            {
                res.add(nulldummy);
            }
        }

        if ( doit >= 2 )
        {
            if ( src.farindsize() )
            {
                retVector<gentype> tmpva;
                gentype toadd(src.farref()(tmpva));

                res.add(toadd);
            }

            else
            {
                res.add(nulldummy);
            }
        }

        if ( doit >= 3 )
        {
            if ( src.farfarindsize() )
            {
                retVector<gentype> tmpva;
                gentype toadd(src.farfarref()(tmpva));

                res.add(toadd);
            }

            else
            {
                res.add(nulldummy);
            }
        }

        if ( doit >= 4 )
        {
            if ( src.farfarfarindsize() )
            {
                retVector<gentype> tmpva;
                gentype toadd(src.farfarfarref()(tmpva));

                res.add(toadd);
            }

            else
            {
                res.add(nulldummy);
            }
        }
    }

    else
    {
        Vector<gentype> &res = rres.force_vector();

        retVector<gentype> tmpva;

        res = src(tmpva);
    }

    return rres.isValEqn() && !(rres.scalarfn_isscalarfn());
}



// First we define those functions that never need to be polymorphed, and do
// any changes via functions that may be polymorphed.

ML_Base::ML_Base(int _isIndPrune) : kernPrecursor()
{
    {
        svm_mutex_lock(mleyelock); 

        int newMLid = xmlid; // this is a member of kernPrecursor, and it has been set in constructor call kernPrecursor()

        SparseVector<int>* locxvernumber = NULL;
        SparseVector<int>* locgvernumber = NULL;

        if ( xvernumber == NULL )
        {
            MEMNEW(locxvernumber,SparseVector<int>);
            MEMNEW(locgvernumber,SparseVector<int>);

            NiceAssert(locxvernumber);
            NiceAssert(locgvernumber);

            xvernumber = locxvernumber;
            gvernumber = locgvernumber;
        }

        (*const_cast<SparseVector<int>*>(xvernumber))("&",newMLid) = 0; 
        (*const_cast<SparseVector<int>*>(gvernumber))("&",newMLid) = 0; 

        svm_mutex_unlock(mleyelock); 
    }

    thisthis = this;
    thisthisthis = &thisthis;

    assumeReal       = 1;
    trainingDataReal = 1;
    wildxaReal       = 1;
    wildxbReal       = 1;
    wildxcReal       = 1;
    wildxdReal       = 1;

    wildxinfoa = NULL;
    wildxinfob = NULL;
    wildxinfoc = NULL;
    wildxinfod = NULL;

    wildxtanga = 0;
    wildxtangb = 0;
    wildxtangc = 0;
    wildxtangd = 0;

    xpreallocsize = 0;

    UUoutkernel.setType(1);

    isBasisUserUU = 0;
    defbasisUU = -1;
    isBasisUserVV = 0;
    defbasisVV = -1;
    UUcallback = UUcallbackdef;
    VVcallback = VVcallbackdef;

    wildxgenta = NULL;
    wildxgentb = NULL;
    wildxgentc = NULL;
    wildxgentd = NULL;
    wildxxgent = NULL;

    wildxdim = 0;

    wildxdima = 0;
    wildxdimb = 0;
    wildxdimc = 0;
    wildxdimd = 0;
    wildxxdim = 0;

    altxsrc = NULL;

    MEMNEW(that,ML_Base *);
    NiceAssert(that);
    *that = this;

    isIndPrune      = _isIndPrune;
    xassumedconsist = 0;
    xconsist        = 1;

    xdzero = 0;

    globalzerotol = DEFAULT_ZTOL;

    return;
}

ML_Base::~ML_Base() 
{ 
    {
        svm_mutex_lock(mleyelock); 

        int oldMLid = xmlid; // this is an accessible member of kernPrecursor, which hasn't been destroyed (yet)

        (*const_cast<SparseVector<int>*>(xvernumber)).remove(oldMLid); 
        (*const_cast<SparseVector<int>*>(gvernumber)).remove(oldMLid); 

        if ( !((*const_cast<SparseVector<int>*>(xvernumber)).indsize()) )
        {
            MEMDEL(const_cast<SparseVector<int>*>(xvernumber));
            MEMDEL(const_cast<SparseVector<int>*>(gvernumber));

            xvernumber = NULL;
            gvernumber = NULL;
        }

        svm_mutex_unlock(mleyelock); 
    }

    MEMDEL(that); 

    return; 
}

const SparseVector<gentype> &ML_Base::xsum(SparseVector<gentype> &res) const
{
    res.zero();

    if ( N() )
    {
        // Use x so function polymorphs correctly

        res.nearassign(x(zeroint()));

        if ( N() > 1 )
        {
            int i;

            for ( i = 1 ; i < N() ; i++ )
            {
                res.nearadd(x(i));
            }
        }
    }

    return res;
}

const SparseVector<gentype> &ML_Base::xsqsum(SparseVector<gentype> &res) const
{
    res.zero();

    if ( N() )
    {
        // Use x so function polymorphs correctly

        res.nearassign(x(zeroint()));

        int i,j,k;

        for ( i = 0 ; i < N() ; i++ )
        {
            const SparseVector<gentype> &xx = x(i);

            if ( xx.nearindsize() )
            {
                for ( j = 0 ; j < xx.nearindsize() ; j++ )
                {
                    k = xx.ind(j);

                    if ( xx(k).isValNull()    ||
                         xx(k).isValInteger() ||
                         xx(k).isValReal()    ||
                         xx(k).isValAnion()   ||
                         xx(k).isValVector()  ||
                         xx(k).isValMatrix()     )
                    {
                        if ( i )
                        {
                            res("&",k) += outerProd(xx(k),xx(k));
                        }

                        else
                        {
                            res("&",k) = outerProd(xx(k),xx(k));
                        }
                    }

                    else
                    {
                        res("&",k) = "\"wtf\"";
                    }
                }
            }
        }
    }

    return res;
}

const SparseVector<gentype> &ML_Base::xmean(SparseVector<gentype> &res) const
{
    // NB: these are designed with data normalisation in mind!!!  Hence
    // setting things to zero (or one) when the sum doesn't make sense,
    // and saying the inverse of 0 is 1 for the invstddev

    xsum(res);

    if ( xspaceDim() )
    {
        int i;
        int indkeyscale = isXAssumedConsistent() ? N() : 1;

        for ( i = 0 ; i < xspaceDim() ; i++ )
        {
            if ( res(indKey()(i)).isValNull()    ||
                 res(indKey()(i)).isValInteger() ||
                 res(indKey()(i)).isValReal()    ||
                 res(indKey()(i)).isValAnion()   ||
                 res(indKey()(i)).isValVector()  ||
                 res(indKey()(i)).isValMatrix()     )
            {
                res("&",indKey()(i)) *= (1.0/(indkeyscale*indKeyCount()(i)));
            }

            else
            {
                res("&",indKey()(i)) = zeroint();
            }
        }
    }

    return res;
}

const SparseVector<gentype> &ML_Base::xsqmean(SparseVector<gentype> &res) const
{
    xsqsum(res);

    if ( xspaceDim() )
    {
        int i;
        int indkeyscale = isXAssumedConsistent() ? N() : 1;

        for ( i = 0 ; i < xspaceDim() ; i++ )
        {
            if ( res(indKey()(i)).isValNull()    ||
                 res(indKey()(i)).isValInteger() ||
                 res(indKey()(i)).isValReal()    ||
                 res(indKey()(i)).isValAnion()   ||
                 res(indKey()(i)).isValVector()  ||
                 res(indKey()(i)).isValMatrix()     )
            {
                res("&",indKey()(i)) *= (1.0/(indkeyscale*indKeyCount()(i)));
            }

            else
            {
                res("&",indKey()(i)) = zeroint();
            }
        }
    }

    return res;
}

const SparseVector<gentype> &ML_Base::xmeansq(SparseVector<gentype> &res) const
{
    xmean(res);

    if ( res.nearindsize() )
    {
        int i;

        for ( i = 0 ; i < res.nearindsize() ; i++ )
        {
            if ( res.direcref(i).isValNull()    ||
                 res.direcref(i).isValInteger() ||
                 res.direcref(i).isValReal()    ||
                 res.direcref(i).isValAnion()   ||
                 res.direcref(i).isValVector()  ||
                 res.direcref(i).isValMatrix()     )
            {
                res.direref(i) = outerProd(res.direcref(i),res.direcref(i));
            }

            else
            {
                res.direref(i) = zeroint();
            }
        }
    }

    return res;
}

const SparseVector<gentype> &ML_Base::xmedian(SparseVector<gentype> &res) const
{
    res.zero();

    if ( xspaceDim() && N() )
    {
        int i,j,k;
        int indkeyscale = isXAssumedConsistent() ? N() : 1;

        for ( i = 0 ; i < xspaceDim() ; i++ )
        {
            Vector<gentype> featvec(indkeyscale*indKeyCount()(i));

            k = 0;

            for ( j = 0 ; j < N() ; j++ )
            {
                SparseVector<gentype> xx = x(j);

                if ( xx.isindpresent(indKey()(i)) )
                {
                    featvec("&",k) = xx(indKey()(i));
                    k++;
                }
            }

            gentype featveccomp(featvec);

            res("&",indKey()(i)) = median(featveccomp);
        }
    }

    if ( res.nearindsize() )
    {
        int i;

        for ( i = 0 ; i < res.nearindsize() ; i++ )
        {
            if ( !( res.direcref(i).isValNull()    ||
                    res.direcref(i).isValInteger() ||
                    res.direcref(i).isValReal()    ||
                    res.direcref(i).isValAnion()   ||
                    res.direcref(i).isValVector()  ||
                    res.direcref(i).isValMatrix()     ) )
            {
                res.direref(i) = zeroint();
            }
        }
    }

    return res;
}

const SparseVector<gentype> &ML_Base::xvar(SparseVector<gentype> &res) const
{
    // var(x) = 1/N sum_i (x_i-m)^2
    //          1/N sum_i x_i^2 + 1/N sum_i m^2 - 2/N sum_i x_i 1/N sum_j x_j
    //          1/N sum_i x_i^2 + m^2 - 2 1/N sum_i x_i 1/N sum_j x_j
    //          1/N sum_i x_i^2 + m^2 - 2 m^2
    //          1/N sum_i x_i^2 - m^2 
    //          xsqmean - xmeansq

    SparseVector<gentype> xmeansqval;

    xsqmean(res);
    xmeansq(xmeansqval);

    xmeansqval.negate();

    res += xmeansqval;

    return res;
}

const SparseVector<gentype> &ML_Base::xstddev(SparseVector<gentype> &res) const
{
    xvar(res);

    if ( res.nearindsize() )
    {
        int i;

        for ( i = 0 ; i < res.nearindsize() ; i++ )
        {
            if ( res.direcref(i).isValNull()    ||
                 res.direcref(i).isValInteger() ||
                 res.direcref(i).isValReal()    ||
                 res.direcref(i).isValAnion()       )
            {
                if ( ( (double) norm2(res.direcref(i)) ) >= zerotol() )
                {
                    res.direref(i) = sqrt(res.direcref(i));
                }

                else
                {
                    res.direref(i) = 1;
                }
            }

            else if ( res.direcref(i).isValMatrix() )
            {
                if ( ( (double) det(res.direcref(i)) ) >= zerotol() )
                {
                    Matrix<gentype> temp((const Matrix<gentype> &) res.direcref(i));

                    ((const Matrix<gentype> &) res.direref(i)).naiveChol(temp,1);
                    res.direref(i) = temp;
                }

                else
                {
                    res.direref(i) = 1;
                }
            }

            else
            {
                res.direref(i) = "\"wtf\"";
            }
        }
    }

    return res;
}

const SparseVector<gentype> &ML_Base::xmax(SparseVector<gentype> &res) const
{
    res.zero();

    if ( xspaceDim() && N() )
    {
        int i,j,k;
        int indkeyscale = isXAssumedConsistent() ? N() : 1;

        for ( i = 0 ; i < xspaceDim() ; i++ )
        {
            Vector<gentype> featvec(indkeyscale*indKeyCount()(i));

            k = 0;

            for ( j = 0 ; j < N() ; j++ )
            {
                SparseVector<gentype> xx = x(j);

                if ( xx.isindpresent(indKey()(i)) )
                {
                    featvec("&",k) = xx(indKey()(i));
                    k++;
                }
            }

            gentype featveccomp(featvec);

            res("&",indKey()(i)) = max(featveccomp);
        }
    }

    if ( res.nearindsize() )
    {
        int i;

        for ( i = 0 ; i < res.nearindsize() ; i++ )
        {
            if ( !( res.direcref(i).isValNull()    ||
                    res.direcref(i).isValInteger() ||
                    res.direcref(i).isValReal()    ||
                    res.direcref(i).isValAnion()   ||
                    res.direcref(i).isValVector()  ||
                    res.direcref(i).isValMatrix()     ) )
            {
                res.direref(i) = zeroint();
            }
        }
    }

    return res;
}


const SparseVector<gentype> &ML_Base::xmin(SparseVector<gentype> &res) const
{
    res.zero();

    if ( xspaceDim() && N() )
    {
        int i,j,k;
        int indkeyscale = isXAssumedConsistent() ? N() : 1;

        for ( i = 0 ; i < xspaceDim() ; i++ )
        {
            Vector<gentype> featvec(indkeyscale*indKeyCount()(i));

            k = 0;

            for ( j = 0 ; j < N() ; j++ )
            {
                SparseVector<gentype> xx = x(j);

                if ( xx.isindpresent(indKey()(i)) )
                {
                    featvec("&",k) = xx(indKey()(i));
                    k++;
                }
            }

            gentype featveccomp(featvec);

            res("&",indKey()(i)) = min(featveccomp);
        }
    }

    if ( res.nearindsize() )
    {
        int i;

        for ( i = 0 ; i < res.nearindsize() ; i++ )
        {
            if ( !( res.direcref(i).isValNull()    ||
                    res.direcref(i).isValInteger() ||
                    res.direcref(i).isValReal()    ||
                    res.direcref(i).isValAnion()   ||
                    res.direcref(i).isValVector()  ||
                    res.direcref(i).isValMatrix()     ) )
            {
                res.direref(i) = zeroint();
            }
        }
    }

    return res;
}

int ML_Base::normKernelNone(void)
{
    int res = 0;

    if ( N() )
    {
        res = 1;

        getKernel_unsafe().setUnShiftedScaled();
        resetKernel(1);
    }

    return res;
}

int ML_Base::normKernelZeroMeanUnitVariance(int flatnorm, int noshift)
{
    // Normalise to zero mean, unit variance

    int res = 0;

    if ( N() )  
    {
        res = 1;

        SparseVector<gentype> xmeanis;
        SparseVector<gentype> xstddevis;

        // Calculate mean and variance

        xmean(xmeanis);
        xstddev(xstddevis);

        errstream() << ".";

        // Calculate shift and scale from mean and variance
        // Also sanitise by removing non-numeric entries

        SparseVector<gentype> xshift;
        SparseVector<gentype> xscale;

        xshift = xmeanis;
        xshift.negate();
        xscale = xstddevis;

        if ( flatnorm )
        {
            gentype totalscale;
            int iii;

            totalscale = min(xscale,iii);

            xscale = totalscale;
        }

        if ( noshift )
        {
            xshift.zero();
        }

        // Remove errors from shift and scale (probably related to
        // non-numeric features).

        errstream() << "Shift: " << xshift << "\n";
        errstream() << "Scale: " << xscale << "\n";

        getKernel_unsafe().setShift(xshift);
        getKernel_unsafe().setScale(xscale);
        resetKernel(1);
    }

    return res;
}

int ML_Base::normKernelZeroMedianUnitVariance(int flatnorm, int noshift)
{
    // Normalise to zero median, unit variance

    int res = 0;

    if ( N() )  
    {
        res = 1;

        SparseVector<gentype> xmedianis;
        SparseVector<gentype> xstddevis;

        // Calculate median and variance

        xmedian(xmedianis);
        xstddev(xstddevis);

        errstream() << ".";

        // Calculate shift and scale from median and variance
        // Also sanitise by removing non-numeric entries

        SparseVector<gentype> xshift;
        SparseVector<gentype> xscale;

        xshift = xmedianis;
        xshift.negate();
        xscale = xstddevis;

        if ( flatnorm )
        {
            gentype totalscale;
            int iii;

            totalscale = min(xscale,iii);

            xscale = totalscale;
        }

        if ( noshift )
        {
            xshift.zero();
        }

        // Remove errors from shift and scale (probably related to
        // non-numeric features).

        errstream() << "Shift: " << xshift << "\n";
        errstream() << "Scale: " << xscale << "\n";

        getKernel_unsafe().setShift(xshift);
        getKernel_unsafe().setScale(xscale);
        resetKernel(1);
    }

    return res;
}


int ML_Base::normKernelUnitRange(int flatnorm, int noshift)
{
    // Normalise range to 0-1

    int res = 0;

    if ( N() )  
    {
        res = 1;

        SparseVector<gentype> xminis;
        SparseVector<gentype> xmaxis;

        xmin(xminis);
        xmax(xmaxis);

        // Calculate shift and scale

        SparseVector<gentype> xshift(xminis);
        SparseVector<gentype> xscale(xmaxis);

        xshift.negate();
        xscale -= xminis;

        if ( xscale.nearindsize() )
        {
            int j;

            for ( j = 0 ; j < xscale.nearindsize() ; j++ )
            {
                if ( abs2((double) xscale.direcref(j)) > 0 )
                {
                    ;
                }

                else
                {
                    xscale.direref(j) = 1;
                }
            }
        }

        if ( flatnorm )
        {
            gentype totalscale;
            int iii;

            totalscale = min(xscale,iii);

            xscale = totalscale;
        }

        if ( noshift )
        {
            xshift.zero();
        }

        // Shift and scale

        errstream() << "Shift: " << xshift << "\n";
        errstream() << "Scale: " << xscale << "\n";

        getKernel_unsafe().setShift(xshift);
        getKernel_unsafe().setScale(xscale);
        resetKernel(1);
    }

    return res;
}

SparseVector<gentype> &ML_Base::xlateToSparse(SparseVector<gentype> &dest, const Vector<gentype> &src) const
{
        NiceAssert( src.size() == xspaceDim() );

        dest.zero();

        if ( xspaceDim() )
        {
            int i;

            for ( i = 0 ; i < xspaceDim() ; i++ )
            {
                dest("&",indKey()(i)) = src(i);
            }
        }

        return dest;
}

SparseVector<gentype> &ML_Base::xlateToSparse(SparseVector<gentype> &dest, const Vector<double> &src) const
{
        NiceAssert( src.size() == xspaceDim() );

        dest.zero();

        if ( xspaceDim() )
        {
            int i;

            for ( i = 0 ; i < xspaceDim() ; i++ )
            {
                dest("&",indKey()(i)) = src(i);
            }
        }

        return dest;
}

SparseVector<gentype> &ML_Base::xlateToSparse(SparseVector<gentype> &dest, const SparseVector<double> &src) const
{
    dest.indalign(src);

    NiceAssert( dest.nearindsize() == src.nearindsize() );

    if ( dest.nearindsize() )
    {
        int i;

        for ( i = 0 ; i < dest.nearindsize() ; i++ )
        {
            dest.direref(i) = src.direcref(i);
        }
    }

    return dest;
}

Vector<gentype> &ML_Base::xlateFromSparse(Vector<gentype> &dest, const SparseVector<gentype> &src) const
{
        dest.resize(xspaceDim());
        dest.zero();

        if ( src.nearindsize() )
        {
            int ikInd = 0;
            int i;

            for ( i = 0 ; i < src.nearindsize() ; i++ )
            {
                int oob = 1;

                if ( ikInd < xspaceDim() )
                {
                    oob = 0;

                    while ( indKey()(ikInd) < src.ind(i) )
                    {
                        ++ikInd;

                        if ( ikInd >= xspaceDim() )
                        {
                            oob = 1;
                            break;
                        }
                    }
                }

                (void) oob;
                // Design change: don't add unknown indices at all.  That
                // way if you are using real vectors to represent sparse
                // (for example in ml_serial) then then the inner product
                // calls will have aligned sizes, and hence won't throw
                // an exception.  This is OK, as the indices that are not
                // in the ML already will simply be multiplied by zero and
                // hence leaving them out should technically make no diff.

                //if ( oob )
                //{
                //    // General rule: put unknown indices at the end
                //
                //    int j;
                //
                //    for ( j = i ; j < src.nearindsize() ; j++ )
                //    {
                //        dest.add(dest.size());
                //        dest("&",dest.size()-1) = src.direcref(j);
                //    }
                //
                //    return dest;
                //}
                //
                //if ( indKey()(ikInd) != src.ind(i) )
                //{
                //    dest.add(dest.size());
                //    dest("&",dest.size()-1) = src.direcref(i);
                //}
                //
                //else
                {
                    dest("&",ikInd) = src.direcref(i);
                }
            }
        }

        return dest;
}

Vector<double> &ML_Base::xlateFromSparse(Vector<double> &dest, const SparseVector<gentype> &src) const
{
        dest.resize(xspaceDim());
        dest.zero();

        if ( src.nearindsize() )
        {
            int ikInd = 0;
            int i;

            for ( i = 0 ; i < src.nearindsize() ; i++ )
            {
                int oob = 1;

                if ( ikInd < xspaceDim() )
                {
                    oob = 0;

                    while ( indKey()(ikInd) < src.ind(i) )
                    {
                        ++ikInd;

                        if ( ikInd >= xspaceDim() )
                        {
                            oob = 1;
                            break;
                        }
                    }
                }

                (void) oob;
                // Design change: don't add unknown indices at all.  That
                // way if you are using real vectors to represent sparse
                // (for example in ml_serial) then then the inner product
                // calls will have aligned sizes, and hence won't throw
                // an exception.  This is OK, as the indices that are not
                // in the ML already will simply be multiplied by zero and
                // hence leaving them out should technically make no diff.

                //if ( oob )
                //{
                //    // General rule: put unknown indices at the end
                //
                //    int j;
                //
                //    for ( j = i ; j < src.nearindsize() ; j++ )
                //    {
                //        dest.add(dest.size());
                //        dest("&",dest.size()-1) = (double) src.direcref(j);
                //    }
                //
                //    return dest;
                //}
                //
                //if ( indKey()(ikInd) != src.ind(i) )
                //{
                //    dest.add(dest.size());
                //    dest("&",dest.size()-1) = (double) src.direcref(i);
                //}
                //
                //else
                {
                    dest("&",ikInd) = (double) src.direcref(i);
                }
            }
        }

        return dest;
}

Vector<double> &ML_Base::xlateFromSparse(Vector<double> &dest, const SparseVector<double> &src) const
{
        dest.resize(xspaceDim());
        dest.zero();

        if ( src.nearindsize() )
        {
            int ikInd = 0;
            int i;

            for ( i = 0 ; i < src.nearindsize() ; i++ )
            {
                int oob = 1;

                if ( ikInd < xspaceDim() )
                {
                    oob = 0;

                    while ( indKey()(ikInd) < src.ind(i) )
                    {
                        ++ikInd;

                        if ( ikInd >= xspaceDim() )
                        {
                            oob = 1;
                            break;
                        }
                    }
                }

                (void) oob;
                // Design change: don't add unknown indices at all.  That
                // way if you are using real vectors to represent sparse
                // (for example in ml_serial) then then the inner product
                // calls will have aligned sizes, and hence won't throw
                // an exception.  This is OK, as the indices that are not
                // in the ML already will simply be multiplied by zero and
                // hence leaving them out should technically make no diff.

                //if ( oob )
                //{
                //    // General rule: put unknown indices at the end
                //
                //    int j;
                //
                //    for ( j = i ; j < src.nearindsize() ; j++ )
                //    {
                //        dest.add(dest.size());
                //        dest("&",dest.size()-1) = (double) src.direcref(j);
                //    }
                //
                //    return dest;
                //}
                //
                //if ( indKey()(ikInd) != src.ind(i) )
                //{
                //    dest.add(dest.size());
                //    dest("&",dest.size()-1) = (double) src.direcref(i);
                //}
                //
                //else
                {
                    dest("&",ikInd) = (double) src.direcref(i);
                }
            }
        }

        return dest;
}

Vector<double> &ML_Base::xlateFromSparse(Vector<double> &dest, const Vector<gentype> &src) const
{
        dest.resize(src.size());

        int i;

        for ( i = 0 ; i < src.size() ; i++ )
        {
            dest("&",i) = src(i);
        }

        return dest;
}

Vector<double> &ML_Base::xlateFromSparse(Vector<double> &dest, const Vector<double> &src) const
{
        dest = src;

        return dest;
}

SparseVector<gentype> &ML_Base::makeFullSparse(SparseVector<gentype> &dest) const
{
        dest.zero();

        if ( xspaceDim() )
        {
            int i;

            for ( i = 0 ; i < xspaceDim() ; i++ )
            {
                dest("&",indKey()(i));  // No need to actually give a value, we just want the index to exist
            }
        }

        return dest;
}


















int ML_Base::setMLid(int nv)
{
    svm_mutex_lock(mleyelock);

    int oldMLid = MLid();
    int res = kernPrecursor::setMLid(nv);

    if ( !res )
    {
        NiceAssert( !(*const_cast<SparseVector<int>*>(xvernumber)).isindpresent(nv) );
        NiceAssert( !(*const_cast<SparseVector<int>*>(gvernumber)).isindpresent(nv) );

        int xvn = (*const_cast<SparseVector<int>*>(xvernumber))(oldMLid);
        int gvn = (*const_cast<SparseVector<int>*>(gvernumber))(oldMLid);

        (*const_cast<SparseVector<int>*>(xvernumber)).zero(oldMLid);
        (*const_cast<SparseVector<int>*>(gvernumber)).zero(oldMLid);

        (*const_cast<SparseVector<int>*>(xvernumber))("&",nv) = xvn;
        (*const_cast<SparseVector<int>*>(gvernumber))("&",nv) = gvn;
    }

    svm_mutex_unlock(mleyelock);

    return res;
}

































// private functions that are only called locally

void ML_Base::recalcIndTypFromScratch(void)
{
    indexKey.resize(0);
    indexKeyCount.resize(0);
    typeKey.resize(0);
    typeKeyBreak.resize(0);

    int j;

    if ( ML_Base::N() )
    {
        for ( j = 0 ; j < ML_Base::N() ; j++ )
        {
            addToIndexKeyAndUpdate(x(j));
        }
    }

    calcSetAssumeReal();

    return;
}

void ML_Base::unfillIndex(int i)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < indexKey.size() );

    if ( !altxsrc )
    {
        if ( indexKeyCount(i) )
        {
            int j,k,m,n;

            for ( j = 0 ; ( j < ML_Base::N() ) && indexKeyCount(i) ; j++ )
            {
                SparseVector<gentype> &xx = allxdatagent("&",j);

                m = xx.nearupsize();

                for ( n = 0 ; n < m ; n++ )
                {
                    if ( xx.isnearindpresent(indexKey(i)+(n*DEFAULT_TUPLE_INDEX_STEP)) )
                    {
                        k = gettypeind(xx.n(indexKey(i)+(n*DEFAULT_TUPLE_INDEX_STEP)));

                        xx.zero(indexKey(i)+(n*DEFAULT_TUPLE_INDEX_STEP));
                        --(indexKeyCount("&",i));
                        --(typeKeyBreak("&",i)("&",k));
                    }
                }

                m = xx.farupsize();

                for ( n = 0 ; n < m ; n++ )
                {
                    if ( xx.isfarindpresent(indexKey(i)+(n*DEFAULT_TUPLE_INDEX_STEP)) )
                    {
                        k = gettypeind(xx.f(indexKey(i)+(n*DEFAULT_TUPLE_INDEX_STEP)));

                        xx.zero(INDFAROFFSTART+indexKey(i)+(n*DEFAULT_TUPLE_INDEX_STEP));
                        --(indexKeyCount("&",i));
                        --(typeKeyBreak("&",i)("&",k));
                    }
                }
            }
        }

        typeKey("&",i) = 1;
    }

    calcSetAssumeReal();

    return;
}

void ML_Base::fillIndex(int i)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < indexKey.size() );

    if ( !altxsrc )
    {
        //if ( indexKeyCount(i) < ML_Base::N() ) - can exceed N if multivectors and near/far are present!
        {
            int j,m,n,k = 1;

            //for ( j = 0 ; ( j < ML_Base::N() ) && ( indexKeyCount(i) < ML_Base::N() ) ; j++ )
            for ( j = 0 ; j < ML_Base::N() ; j++ )
            {
                SparseVector<gentype> &xx = allxdatagent("&",j);

                m = xx.nearupsize();

                for ( n = 0 ; n < m ; n++ )
                {
                    if ( !(xx.isnearindpresent(indexKey(i)+(n*DEFAULT_TUPLE_INDEX_STEP))) )
                    {
                        //k = 1;

                        (xx("&",indexKey(i)+(n*DEFAULT_TUPLE_INDEX_STEP))).makeNull();
                        ++(indexKeyCount("&",i));
                        ++(typeKeyBreak("&",i)("&",k));
                    }
                }

                m = xx.farupsize();

                for ( n = 0 ; n < m ; n++ )
                {
                    if ( !(xx.isfarindpresent(indexKey(i)+(n*DEFAULT_TUPLE_INDEX_STEP))) )
                    {
                        //k = 1;

                        (xx("&",INDFAROFFSTART+indexKey(i)+(n*DEFAULT_TUPLE_INDEX_STEP))).makeNull();
                        ++(indexKeyCount("&",i));
                        ++(typeKeyBreak("&",i)("&",k));
                    }
                }
            }
        }
    }

    calcSetAssumeReal();

    return;
}

void ML_Base::addToIndexKeyAndUpdate(const SparseVector<gentype> &newz)
{
    if ( ( newz.nearupsize() > 1 ) || ( newz.farupsize() > 1 ) || newz.isfaroffindpresent() || newz.isfarfaroffindpresent() || newz.isfarfarfaroffindpresent() )
    {
        int i,s;

        s = newz.nearupsize();

        for ( i = 0 ; i < s ; i++ )
        {
            addToIndexKeyAndUpdate(newz.nearrefup(i));
        }

        s = newz.farupsize();

        for ( i = 0 ; i < s ; i++ )
        {
            addToIndexKeyAndUpdate(newz.farrefup(i));
        }
    }

    else
    {
        if ( newz.nearindsize() )
        {
            int zInd;
            int ikInd = indexKey.size()-1;
            int needaddxspacedim;

            for ( zInd = newz.nearindsize()-1 ; zInd >= 0 ; --zInd )
            {
                needaddxspacedim = 0;

                // Find index zInd in new vector
                // Set ikInd such that indexKey(ikInd) == newz.ind(zInd).
                // If ikInd == -1 or indexKey(ikInd) != newz.ind(zInd) then
                // we need to add the new index.

                if ( ikInd >= 0 )
                {
                    while ( indexKey(ikInd) > newz.ind(zInd) )
                    {
                        --ikInd;

                        if ( ikInd < 0 )
                        {
                            break;
                        }
                    }
                }

                if ( ikInd == -1 )
                {
                    addInd:

                    // Index not found, so add it.

                    ++ikInd;

                    indexKey.add(ikInd);
                    indexKey("&",ikInd) = newz.ind(zInd);
                    indexKeyCount.add(ikInd);
                    indexKeyCount("&",ikInd) = 0;
                    typeKey.add(ikInd);
                    typeKey("&",ikInd) = 1;
                    typeKeyBreak.add(ikInd);
                    typeKeyBreak("&",ikInd).resize(NUMXTYPES);
                    typeKeyBreak("&",ikInd) = 0;

                    needaddxspacedim = 1;
                }

                else if ( indexKey(ikInd) != newz.ind(zInd) )
                {
                    goto addInd;
                }

                NiceAssert( indexKey(ikInd) == newz.ind(zInd) );

                // Update index information

                indexKeyCount("&",ikInd)++;

                int indType = gettypeind(newz.direcref(zInd));

                NiceAssert( indType );

                typeKeyBreak("&",ikInd)("&",indType)++;
                typeKeyBreak("&",ikInd)("&",0)++;

                int indchange = 1;

                if ( indType == typeKey(ikInd) )
                {
                    indchange = 0;
                }

                else if ( ( indType <= typeKey(ikInd) ) && ( typeKey(ikInd) <= 5 ) )
                {
                    indchange = 0;
                    indType = typeKey(ikInd);
                }

                if ( indchange )
                {
                    int runsum = typeKeyBreak(ikInd)(1);

                    if ( typeKeyBreak(ikInd)(zeroint()) == runsum )
                    {
                        // null
                        indType = 1;
                    }

                    else if ( typeKeyBreak(ikInd)(zeroint()) == ( runsum += typeKeyBreak(ikInd)(2) ) )
                    {
                        // null or binary
                        indType = 2;
                    }

                    else if ( typeKeyBreak(ikInd)(zeroint()) == ( runsum += typeKeyBreak(ikInd)(3) ) )
                    {
                        // null or binary or integer
                        indType = 3;
                    }

                    else if ( typeKeyBreak(ikInd)(zeroint()) == ( runsum += typeKeyBreak(ikInd)(4) ) )
                    {
                        // null or binary or integer or double
                        indType = 4;
                    }

                    else if ( typeKeyBreak(ikInd)(zeroint()) == ( runsum += typeKeyBreak(ikInd)(5) ) )
                    {
                        // null or binary or integer or double or anion
                        indType = 5;
                    }

                    else if ( typeKeyBreak(ikInd)(zeroint()) == ( typeKeyBreak(ikInd)(zeroint()) + typeKeyBreak(ikInd)(6) ) )
                    {
                        // null or vector
                        indType = 6;
                    }

                    else if ( typeKeyBreak(ikInd)(zeroint()) == ( typeKeyBreak(ikInd)(zeroint()) + typeKeyBreak(ikInd)(7) ) )
                    {
                        // null or matrix
                        indType = 7;
                    }

                    else if ( typeKeyBreak(ikInd)(zeroint()) == ( typeKeyBreak(ikInd)(zeroint()) + typeKeyBreak(ikInd)(8) ) )
                    {
                        // null or set
                        indType = 8;
                    }

                    else if ( typeKeyBreak(ikInd)(zeroint()) == ( typeKeyBreak(ikInd)(zeroint()) + typeKeyBreak(ikInd)(9) ) )
                    {
                        // null or dgraph
                        indType = 9;
                    }

                    else if ( typeKeyBreak(ikInd)(zeroint()) == ( typeKeyBreak(ikInd)(zeroint()) + typeKeyBreak(ikInd)(10) ) )
                    {
                        // null or string
                        indType = 10;
                    }

                    else if ( typeKeyBreak(ikInd)(zeroint()) == ( typeKeyBreak(ikInd)(zeroint()) + typeKeyBreak(ikInd)(11) ) )
                    {
                        // null or string
                        indType = 11;
                    }

                    else
                    {
                        // unknown
                        indType = 0;
                    }
                }

                typeKey("&",ikInd) = indType;

                if ( indPrune() && ( indexKeyCount(ikInd) < ML_Base::N() ) )
                {
                    // Make sure index is in all vectors, as required when
                    // pruning is set.

                    fillIndex(ikInd);
                }

                if ( needaddxspacedim )
                {
                    addxspaceFeat(ikInd);
                }
            }
        }

        // Need this to make sure the newly added training vector has
        // all relevant indices

        if ( indexKey.size() && indPrune() )
        {
            int i;

            for ( i = 0 ; i < indexKey.size() ; i++ )
            {
                if ( indPrune() && ( indexKeyCount(i) < ML_Base::N() ) )
                {
                    fillIndex(i);
                }
            }
        }
    }

    calcSetAssumeReal();

    return;
}

void ML_Base::removeFromIndexKeyAndUpdate(const SparseVector<gentype> &oldz)
{
    if ( ( oldz.nearupsize() > 1 ) || ( oldz.farupsize() > 1 ) || oldz.isfaroffindpresent() || oldz.isfarfaroffindpresent() || oldz.isfarfarfaroffindpresent() )
    {
        int i,s;

        s = oldz.nearupsize();

        for ( i = 0 ; i < s ; i++ )
        {
            removeFromIndexKeyAndUpdate(oldz.nearrefup(i));
        }

        s = oldz.farupsize();

        for ( i = 0 ; i < s ; i++ )
        {
            removeFromIndexKeyAndUpdate(oldz.farrefup(i));
        }
    }

    else
    {
        if ( oldz.nearindsize() )
        {
            int zInd;
            int ikInd = indexKey.size()-1;

            for ( zInd = oldz.nearindsize()-1 ; zInd >= 0 ; --zInd )
            {
                while ( indexKey(ikInd) > oldz.ind(zInd) )
                {
                    --ikInd;
                    NiceAssert( ikInd >= 0 );
                }

                int indType = gettypeind(oldz.direcref(zInd));

                NiceAssert( indType );

                typeKeyBreak("&",ikInd)("&",indType)--;
                typeKeyBreak("&",ikInd)("&",0)--;

                int indchange = 0;

                if ( typeKeyBreak(ikInd)(indType) != typeKeyBreak(ikInd)(0) )
                {
                    indchange = 1;
                }

                if ( indchange )
                {
                    int runsum = typeKeyBreak(ikInd)(1);

                    if ( typeKeyBreak(ikInd)(0) == runsum )
                    {
                        // null
                        indType = 1;
                    }

                    else if ( typeKeyBreak(ikInd)(0) == ( runsum += typeKeyBreak(ikInd)(2) ) )
                    {
                        // null or binary
                        indType = 2;
                    }

                    else if ( typeKeyBreak(ikInd)(0) == ( runsum += typeKeyBreak(ikInd)(3) ) )
                    {
                        // null or binary or integer
                        indType = 3;
                    }

                    else if ( typeKeyBreak(ikInd)(0) == ( runsum += typeKeyBreak(ikInd)(4) ) )
                    {
                        // null or binary or integer or double
                        indType = 4;
                    }

                    else if ( typeKeyBreak(ikInd)(0) == ( runsum += typeKeyBreak(ikInd)(5) ) )
                    {
                        // null or binary or integer or double or anion
                        indType = 5;
                    }

                    else if ( typeKeyBreak(ikInd)(0) == ( typeKeyBreak(ikInd)(0) + typeKeyBreak(ikInd)(6) ) )
                    {
                        // null or vector
                        indType = 6;
                    }

                    else if ( typeKeyBreak(ikInd)(0) == ( typeKeyBreak(ikInd)(0) + typeKeyBreak(ikInd)(7) ) )
                    {
                        // null or matrix
                        indType = 7;
                    }

                    else if ( typeKeyBreak(ikInd)(0) == ( typeKeyBreak(ikInd)(0) + typeKeyBreak(ikInd)(8) ) )
                    {
                        // null or set
                        indType = 8;
                    }

                    else if ( typeKeyBreak(ikInd)(0) == ( typeKeyBreak(ikInd)(0) + typeKeyBreak(ikInd)(9) ) )
                    {
                        // null or dgraph
                        indType = 9;
                    }

                    else if ( typeKeyBreak(ikInd)(0) == ( typeKeyBreak(ikInd)(0) + typeKeyBreak(ikInd)(10) ) )
                    {
                        // null or string
                        indType = 10;
                    }

                    else if ( typeKeyBreak(ikInd)(0) == ( typeKeyBreak(ikInd)(0) + typeKeyBreak(ikInd)(11) ) )
                    {
                        // null or string
                        indType = 11;
                    }

                    else
                    {
                        // unknown
                        indType = 0;
                    }
                }

                typeKey("&",ikInd) = indType;

                NiceAssert( indexKey(ikInd) == oldz.ind(zInd) );

                indexKeyCount("&",ikInd)--;

                if ( indPrune() && ( typeKey(ikInd) == 1 ) )
                {
                    // Remove null feature as required.  Next if statement
                    // will finish operation

                    unfillIndex(ikInd);
                }

                if ( !indexKeyCount(ikInd) )
                {
                    // Remove unused feature from indexing

                    removexspaceFeat(ikInd);
                    indexKey.remove(ikInd);
                    indexKeyCount.remove(ikInd);

                    if ( ikInd > indexKey.size()-1 )
                    {
                        ikInd = indexKey.size()-1;
                    }
                }
            }
        }
    }

    calcSetAssumeReal();

    return;
}

int ML_Base::gettypeind(const gentype &y) const
{
        int indType = 0; // unknown type

             if ( y.isValNull()    ) { indType = 1;  }
        else if ( y.isValInteger() ) { indType = 3;  }
        else if ( y.isValReal()    ) { indType = 4;  }
        else if ( y.isValAnion()   ) { indType = 5;  }
        else if ( y.isValVector()  ) { indType = 6;  }
        else if ( y.isValMatrix()  ) { indType = 7;  }
        else if ( y.isValSet()     ) { indType = 8;  }
        else if ( y.isValDgraph()  ) { indType = 9;  }
        else if ( y.isValString()  ) { indType = 10; }
        else if ( y.isValEqnDir()  ) { indType = 11; }

        if ( y.isValInteger() && ( ( ( (int) y ) == 0 ) ||
                                   ( ( (int) y ) == 1 )    ) )
        {
            indType = 2;
        }

        return indType;
}













































// Functions that *should* be polymorphed.

int ML_Base::addTrainingVector(int i, const gentype &y, const SparseVector<gentype> &xx, double nCweight, double nepsweight)
{
    if ( i != N() )
    {
        incxvernum();
    }

    NiceAssert( i >= 0 );
    NiceAssert( i <= ML_Base::N() );

    if ( isBasisUserUU )
    {
        isBasisUserUU = 0;
        addToBasisUU(i,y);
        isBasisUserUU = 1;
    }

    if ( isBasisUserVV )
    {
        isBasisUserVV = 0;
        addToBasisVV(i,y);
        isBasisUserVV = 1;
    }

    xd.add(i);
    xCweight.add(i);
    xCweightfuzz.add(i);
    xepsweight.add(i);

    xd("&",i) = 2;
    xCweight("&",i)     = nCweight;
    xCweightfuzz("&",i) = 1.0;
    xepsweight("&",i)   = nepsweight;

    allxdatagent.add(i);
    alltraintarg.add(i);

    allxdatagent("&",i) = xx;
    alltraintarg("&",i) = y;

    if ( !(allxdatagent(i).altcontent) )
    {
        allxdatagent("&",i).makealtcontent();
    }

    if ( !isXAssumedConsistent() || !i )
    {
        addToIndexKeyAndUpdate(x(i));

        if ( xconsist || !i )
        {
            xconsist = testxconsist();
        }
    }

    // Note that we do this after we have done everything else.  This is
    // because x(i) may be redirected through any number of caches and
    // misdirections, which we need to have set up before we can proceed.
    // Note also the use of getKernel(), which is presumed to have been
    // appropriately polymorphed by inheriting class(es).

    traininfo.add(i);
    traintang.add(i);
    xalphaState.add(i);
    // If x() is not accurate don't worry as it will be done via callback
    getKernel().getvecInfo(traininfo("&",i),x(i),NULL,isXConsistent(),assumeReal); //x()(i));
    xalphaState("&",i) = 1;
    traintang("&",i) = detangle_x(i);

//FIXME    allxdatagentp.add(i);
//FIXME    traininfop.add(i);

//FIXME    allxdatagentp("&",i) = &allxdatagent(i);
//FIXME    traininfop("&",i)    = &traininfo(i);

    return 0;
}

int ML_Base::qaddTrainingVector(int i, const gentype &y,       SparseVector<gentype> &xx, double nCweight, double nepsweight)
{
    if ( i != N() )
    {
        incxvernum();
    }

    NiceAssert( i >= 0 );
    NiceAssert( i <= ML_Base::N() );

    if ( isBasisUserUU )
    {
        isBasisUserUU = 0;
        addToBasisUU(i,y);
        isBasisUserUU = 1;
    }

    if ( isBasisUserVV )
    {
        isBasisUserVV = 0;
        addToBasisVV(i,y);
        isBasisUserVV = 1;
    }

    xd.add(i);
    xCweight.add(i);
    xCweightfuzz.add(i);
    xepsweight.add(i);

    xd("&",i) = 2;
    xCweight("&",i)     = nCweight;
    xCweightfuzz("&",i) = 1.0;
    xepsweight("&",i)   = nepsweight;

    allxdatagent.add(i);
    alltraintarg.add(i);

    qswap(allxdatagent("&",i),xx);
    alltraintarg("&",i) = y;

    if ( !(allxdatagent(i).altcontent) )
    {
        allxdatagent("&",i).makealtcontent();
    }

    if ( !isXAssumedConsistent() || !i )
    {
        addToIndexKeyAndUpdate(x(i));

        if ( xconsist || !i )
        {
            xconsist = testxconsist();
        }
    }

    // Note that we do this after we have done everything else.  This is
    // because x(i) may be redirected through any number of caches and
    // misdirections, which we need to have set up before we can proceed.
    // Note also the use of getKernel(), which is presumed to have been
    // appropriately polymorphed by inheriting class(es).

    traininfo.add(i);
    traintang.add(i);
    xalphaState.add(i);
    // If x() is not accurate don't worry as it will be done via callback
    getKernel().getvecInfo(traininfo("&",i),x(i),NULL,isXConsistent(),assumeReal); //x()(i));
    traintang("&",i) = detangle_x(i);

    xalphaState("&",i) = 1;

//FIXME    allxdatagentp.add(i);
//FIXME    traininfop.add(i);

//FIXME    allxdatagentp("&",i) = &allxdatagent(i);
//FIXME    traininfop("&",i)    = &traininfo(i);

    return 0;
}

int ML_Base::addTrainingVector (int i, const Vector<gentype> &y, const Vector<SparseVector<gentype> > &xx, const Vector<double> &nCweigh, const Vector<double> &nepsweigh)
{
    if ( i != N() )
    {
        incxvernum();
    }

    NiceAssert( i >= 0 );
    NiceAssert( i <= ML_Base::N() );
    NiceAssert( y.size() == xx.size() );
    NiceAssert( y.size() == nCweigh.size() );
    NiceAssert( y.size() == nepsweigh.size() );

    if ( y.size() )
    {
        int j;

        for ( j = 0 ; j < y.size() ; j++ )
        {
            if ( isBasisUserUU )
            {
                isBasisUserUU = 0;
                addToBasisUU(i,y(i+j));
                isBasisUserUU = 1;
            }

            if ( isBasisUserVV )
            {
                isBasisUserVV = 0;
                addToBasisVV(i,y(i+j));
                isBasisUserVV = 1;
            }

            xd.add(i+j);
            xCweight.add(i+j);
            xCweightfuzz.add(i+j);
            xepsweight.add(i+j);

            xd("&",i+j) = 2;
            xCweight("&",i+j)     = nCweigh(j);
            xCweightfuzz("&",i+j) = 1.0;
            xepsweight("&",i+j)   = nepsweigh(j);

            allxdatagent.add(i+j);
            alltraintarg.add(i+j);

            allxdatagent("&",i+j) = xx(j);
            alltraintarg("&",i+j) = y(j);

            if ( !(allxdatagent(i+j).altcontent) )
            {
                allxdatagent("&",i+j).makealtcontent();
            }

            if ( !isXAssumedConsistent() || !(i+j) )
            {
                addToIndexKeyAndUpdate(x(i+j));

                if ( xconsist || !(i+j) )
                {
                    xconsist = testxconsist();
                }
            }

            // Note that we do this after we have done everything else.  This is
            // because x(i) may be redirected through any number of caches and
            // misdirections, which we need to have set up before we can proceed.
            // Note also the use of getKernel(), which is presumed to have been
            // appropriately polymorphed by inheriting class(es).

            traininfo.add(i+j);
            traintang.add(i+j);
            xalphaState.add(i+j);
            // If x() is not accurate don't worry as it will be done via callback
            getKernel().getvecInfo(traininfo("&",i+j),x(i+j),NULL,isXConsistent(),assumeReal); //()(i+j));
            xalphaState("&",i+j) = 1;
            traintang("&",i+j) = detangle_x(i+j);

//FIXME            allxdatagentp.add(i+j);
//FIXME            traininfop.add(i+j);

//FIXME            allxdatagentp("&",i+j) = &allxdatagent(i+j);
//FIXME            traininfop("&",i+j)    = &traininfo(i+j);
        }
    }

    return 0;
}

int ML_Base::qaddTrainingVector(int i, const Vector<gentype> &y, Vector<SparseVector<gentype> > &xx, const Vector<double> &nCweigh, const Vector<double> &nepsweigh)
{
    if ( i != N() )
    {
        incxvernum();
    }

    NiceAssert( i >= 0 );
    NiceAssert( i <= ML_Base::N() );
    NiceAssert( y.size() == xx.size() );
    NiceAssert( y.size() == nCweigh.size() );
    NiceAssert( y.size() == nepsweigh.size() );

    if ( y.size() )
    {
        int j;

        for ( j = 0 ; j < y.size() ; j++ )
        {
            if ( isBasisUserUU )
            {
                isBasisUserUU = 0;
                addToBasisUU(i,y(i+j));
                isBasisUserUU = 1;
            }

            if ( isBasisUserVV )
            {
                isBasisUserVV = 0;
                addToBasisVV(i,y(i+j));
                isBasisUserVV = 1;
            }

            xd.add(i+j);
            xCweight.add(i+j);
            xCweightfuzz.add(i+j);
            xepsweight.add(i+j);

            xd("&",i+j) = 2;
            xCweight("&",i+j)     = nCweigh(j);
            xCweightfuzz("&",i+j) = 1.0;
            xepsweight("&",i+j)   = nepsweigh(j);

            allxdatagent.add(i+j);
            alltraintarg.add(i+j);

            qswap(allxdatagent("&",i+j),xx("&",j));
            alltraintarg("&",i+j) = y(j);

            if ( !(allxdatagent(i+j).altcontent) )
            {
                allxdatagent("&",i+j).makealtcontent();
            }

            if ( !isXAssumedConsistent() || !(i+j) )
            {
                addToIndexKeyAndUpdate(x(i+j));

                if ( xconsist || !(i+j) )
                {
                    xconsist = testxconsist();
                }
            }

            // Note that we do this after we have done everything else.  This is
            // because x(i) may be redirected through any number of caches and
            // misdirections, which we need to have set up before we can proceed.
            // Note also the use of getKernel(), which is presumed to have been
            // appropriately polymorphed by inheriting class(es).

            traininfo.add(i+j);
            traintang.add(i+j);
            xalphaState.add(i+j);
            // If x() is not accurate don't worry as it will be done via callback
            getKernel().getvecInfo(traininfo("&",i+j),x(i+j),NULL,isXConsistent(),assumeReal); //()(i+j));
            xalphaState("&",i+j) = 1;
            traintang("&",i+j) = detangle_x(i+j);

//FIXME            allxdatagentp.add(i+j);
//FIXME            traininfop.add(i+j);

//FIXME            allxdatagentp("&",i+j) = &allxdatagent(i+j);
//FIXME            traininfop("&",i+j)    = &traininfo(i+j);
        }
    }

    return 0;
}

int ML_Base::removeTrainingVector(int i, gentype &y, SparseVector<gentype> &xx)
{
    incxvernum();
    incgvernum();

    NiceAssert( i >= 0 );
    NiceAssert( i <  ML_Base::N() );

    if ( isBasisUserUU )
    {
        isBasisUserUU = 0;
        removeFromBasisUU(i);
        isBasisUserUU = 1;
    }

    if ( isBasisUserVV )
    {
        isBasisUserVV = 0;
        removeFromBasisVV(i);
        isBasisUserVV = 1;
    }

    xdzero -= xd(i) ? 0 : 1;

    if ( !isXAssumedConsistent() || !i )
    {
        removeFromIndexKeyAndUpdate(x(i));

        if ( xconsist || !i )
        {
            xconsist = testxconsist();
        }
    }

    qswap(xx,allxdatagent("&",i));
    qswap(y,alltraintarg("&",i));

    allxdatagent.remove(i);
    alltraintarg.remove(i);

    traininfo.remove(i);
    traintang.remove(i);
    xalphaState.remove(i);

    xd.remove(i);
    xCweight.remove(i);
    xCweightfuzz.remove(i);
    xepsweight.remove(i);

//FIXME    allxdatagentp.remove(i);
//FIXME    traininfop.remove(i);

    return 0;
}

int ML_Base::removeTrainingVector(int i, int num)
{
    incxvernum();
    incgvernum();

    NiceAssert( i < ML_Base::N() );
    NiceAssert( num >= 0 );
    NiceAssert( num <= ML_Base::N()-i );

    int res = 0;
    gentype y;
    SparseVector<gentype> x;

    while ( num )
    {
        num--; 
        res |= removeTrainingVector(i+num,y,x);
    }

    return res;
}

int ML_Base::resetKernel(int modind, int onlyChangeRowI, int updateInfo)
{
    NiceAssert( onlyChangeRowI >= -1 );
    NiceAssert( onlyChangeRowI < ML_Base::N() );

    calcSetAssumeReal();

    int res = 1;

    if ( ML_Base::N() && ( onlyChangeRowI == -1 ) && updateInfo )
    {
        res = 1;

        int i;

        for ( i = 0 ; i < ML_Base::N() ; i++ )
	{
            if ( modind )
	    {
                // If x() is not accurate don't worry as it will be done via callback
                getKernel().getvecInfo(traininfo("&",i),x(i),NULL,isXConsistent(),assumeReal); //()(i));
                traintang("&",i) = detangle_x(i);
	    }
	}
    }

    else if ( ( onlyChangeRowI >= 0 ) && updateInfo )
    {
        res = 1;

        // If x() is not accurate don't worry as it will be done via callback
        getKernel().getvecInfo(traininfo("&",onlyChangeRowI),x(onlyChangeRowI),NULL,isXConsistent(),assumeReal); //()(onlyChangeRowI));
        traintang("&",onlyChangeRowI) = detangle_x(onlyChangeRowI);
    }

    return res;
}

int ML_Base::setKernel(const MercerKernel &xkernel, int modind, int onlyChangeRowI)
{
    kernel = xkernel;

    NiceAssert( onlyChangeRowI >= -1 );
    NiceAssert( onlyChangeRowI < ML_Base::N() );

    calcSetAssumeReal();

    int res = 1;

    if ( ML_Base::N() && ( onlyChangeRowI == -1 ) )
    {
        res = 1;

        int i;

        for ( i = 0 ; i < ML_Base::N() ; i++ )
	{
            if ( modind )
	    {
                // If x() is not accurate don't worry as it will be done via callback
                getKernel().getvecInfo(traininfo("&",i),x(i),NULL,isXConsistent(),assumeReal); //()(i));
                traintang("&",i) = detangle_x(i);
	    }
	}
    }

    else if ( onlyChangeRowI >= 0 )
    {
        res = 1;

        // If x() is not accurate don't worry as it will be done via callback
        getKernel().getvecInfo(traininfo("&",onlyChangeRowI),x(onlyChangeRowI),NULL,isXConsistent(),assumeReal); //()(onlyChangeRowI));
        traintang("&",onlyChangeRowI) = detangle_x(onlyChangeRowI);
    }

    return res;
}

int ML_Base::setx(int i, const SparseVector<gentype> &xx)
{
    incxvernum();
    incgvernum();

    NiceAssert( i >= 0 );
    NiceAssert( i < ML_Base::N() );

    int res = 0;

    if ( !isXAssumedConsistent() || !i )
    {
        removeFromIndexKeyAndUpdate(x(i));
        allxdatagent("&",i) = xx;
        addToIndexKeyAndUpdate(x(i));

        if ( xconsist || !i )
        {
            xconsist = testxconsist();
        }

        res = 1;
    }

    else
    {
        allxdatagent("&",i) = xx;
    }

    if ( !(allxdatagent(i).altcontent) )
    {
        allxdatagent("&",i).makealtcontent();
    }

    getKernel_unsafe().setIPdiffered(1);

    return res |= resetKernel(1,i);
}

int ML_Base::setx(const Vector<int> &i, const Vector<SparseVector<gentype> > &xx)
{
    incxvernum();
    incgvernum();

    NiceAssert( i.size() == xx.size() );

    int j;
    int res = 0;

    for ( j = 0 ; j < i.size() ; j++ )
    {
        if ( !isXAssumedConsistent() || !i(j) )
        {
            removeFromIndexKeyAndUpdate(x(i(j)));
            allxdatagent("&",i(j)) = xx(j);
            addToIndexKeyAndUpdate(x(i(j)));

            if ( xconsist || !i(j) )
            {
                xconsist = testxconsist();
            }

            res = 1;
        }

        else
        {
            allxdatagent("&",i(j)) = xx(j);
        }

        if ( !(allxdatagent(i(j)).altcontent) )
        {
            allxdatagent("&",i(j)).makealtcontent();
        }

        getKernel_unsafe().setIPdiffered(1);

        res |= resetKernel(1,i(j));
    }

    return res;
}

int ML_Base::setx(const Vector<SparseVector<gentype> > &xx)
{
    incxvernum();
    incgvernum();

    NiceAssert( xx.size() == ML_Base::N() );

    int j;
    int res = 0;

    for ( j = 0 ; j < ML_Base::N() ; j++ )
    {
        if ( !isXAssumedConsistent() || !j )
        {
            removeFromIndexKeyAndUpdate(x(j));
            allxdatagent("&",j) = xx(j);
            addToIndexKeyAndUpdate(x(j));

            if ( xconsist || !j )
            {
                xconsist = testxconsist();
            }

            res = 1;
        }

        else
        {
            allxdatagent("&",j) = xx(j);
        }
    }

    getKernel_unsafe().setIPdiffered(1);

    return res |= resetKernel(1,-1);
}

int ML_Base::qswapx(int i, SparseVector<gentype> &xx, int dontupdate)
{
    incxvernum();
    incgvernum();

    NiceAssert( i >= 0 );
    NiceAssert( i < ML_Base::N() );

    int res = 0;

    if ( !dontupdate && ( !isXAssumedConsistent() || !i ) )
    {
        removeFromIndexKeyAndUpdate(x(i));
        qswap(allxdatagent("&",i),xx);
        addToIndexKeyAndUpdate(x(i));

        if ( xconsist || !i )
        {
            xconsist = testxconsist();
        }

        res = 1;
    }

    else
    {
        qswap(allxdatagent("&",i),xx);
    }

    if ( !(allxdatagent(i).altcontent) )
    {
        allxdatagent("&",i).makealtcontent();
    }

    getKernel_unsafe().setIPdiffered(1);

    return res |= resetKernel(1,i);
}

int ML_Base::qswapx(const Vector<int> &i, Vector<SparseVector<gentype> > &xx, int dontupdate)
{
    incxvernum();
    incgvernum();

    NiceAssert( i.size() == xx.size() );

    int j;
    int res = 0;

    for ( j = 0 ; j < i.size() ; j++ )
    {
        if ( !dontupdate && ( !isXAssumedConsistent() || !i(j) ) )
        {
            removeFromIndexKeyAndUpdate(x(i(j)));
            qswap(allxdatagent("&",i(j)),xx("&",j));
            addToIndexKeyAndUpdate(x(i(j)));

            if ( xconsist || !i(j) )
            {
                xconsist = testxconsist();
            }

            res = 1;
        }

        else
        {
            qswap(allxdatagent("&",i(j)),xx("&",j));
        }

        if ( !(allxdatagent(i(j)).altcontent) )
        {
            allxdatagent("&",i(j)).makealtcontent();
        }

        getKernel_unsafe().setIPdiffered(1);

        res |= resetKernel(1,i(j));
    }

    return res;
}

int ML_Base::qswapx(Vector<SparseVector<gentype> > &xx, int dontupdate)
{
    incxvernum();
    incgvernum();

    NiceAssert( xx.size() == ML_Base::N() );

    int j;
    int res = 0;

    for ( j = 0 ; j < ML_Base::N() ; j++ )
    {
        if ( !dontupdate && ( !isXAssumedConsistent() || !j ) )
        {
            removeFromIndexKeyAndUpdate(x(j));
            qswap(allxdatagent("&",j),xx("&",j));
            addToIndexKeyAndUpdate(x(j));

            if ( xconsist || !j )
            {
                xconsist = testxconsist();
            }

            res = 1;
        }

        else
        {
            qswap(allxdatagent("&",j),xx("&",j));
        }

        if ( !(allxdatagent(j).altcontent) )
        {
            allxdatagent("&",j).makealtcontent();
        }
    }

    getKernel_unsafe().setIPdiffered(1);

    return res |= resetKernel(1,-1);
}

int ML_Base::sety(int i, const gentype &y)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < ML_Base::N() );

    if ( isBasisUserUU )
    {
        isBasisUserUU = 0;
        setBasisUU(i,y);
        isBasisUserUU = 1;
    }

    if ( isBasisUserVV )
    {
        isBasisUserVV = 0;
        setBasisVV(i,y);
        isBasisUserVV = 1;
    }

    alltraintarg("&",i) = y;

    return 0;
}

int ML_Base::sety(const Vector<int> &j, const Vector<gentype> &yn)
{
    NiceAssert( j.size() == yn.size() );

    if ( isBasisUserUU && yn.size() )
    {
        int i;

        isBasisUserUU = 0;

        for ( i = 0 ; i < yn.size() ; i++ )
        {
            setBasisUU(i,yn(j(i)));
        }

        isBasisUserUU = 1;
    }

    if ( isBasisUserVV && yn.size() )
    {
        int i;

        isBasisUserVV = 0;

        for ( i = 0 ; i < yn.size() ; i++ )
        {
            setBasisVV(i,yn(j(i)));
        }

        isBasisUserVV = 1;
    }

    retVector<gentype> tmpva;

    alltraintarg("&",j,tmpva) = yn;

    return 0;
}

int ML_Base::sety(const Vector<gentype> &yn)
{
    NiceAssert( yn.size() == ML_Base::N() );

    if ( isBasisUserUU && N() )
    {
        int i;

        isBasisUserUU = 0;

        for ( i = 0 ; i < N() ; i++ )
        {
            setBasisUU(i,yn(i));
        }

        isBasisUserUU = 1;
    }

    if ( isBasisUserVV && N() )
    {
        int i;

        isBasisUserVV = 0;

        for ( i = 0 ; i < N() ; i++ )
        {
            setBasisVV(i,yn(i));
        }

        isBasisUserVV = 1;
    }

    alltraintarg = yn;

    return 0;
}

int ML_Base::sety(int i, double z)
{
    gentype y(z);

    return sety(i,y);
}

int ML_Base::sety(const Vector<int> &i, const Vector<double> &z)
{
    Vector<gentype> y(z.size());

    if ( y.size() )
    {
        int j;

        for ( j = 0 ; j < y.size() ; j++ )
        {
            y("&",j) = z(j);
        }
    }

    return sety(i,y);
}

int ML_Base::sety(const Vector<double> &z)
{
    Vector<gentype> y(z.size());

    if ( y.size() )
    {
        int j;

        for ( j = 0 ; j < y.size() ; j++ )
        {
            y("&",j) = z(j);
        }
    }

    return sety(y);
}

int ML_Base::sety(int i, const Vector<double> &z)
{
    gentype y(z);

    return sety(i,z);
}

int ML_Base::sety(const Vector<int> &i, const Vector<Vector<double> > &z)
{
    Vector<gentype> y(z.size());

    if ( y.size() )
    {
        int j;

        for ( j = 0 ; j < y.size() ; j++ )
        {
            y("&",j) = z(j);
        }
    }

    return sety(i,y);
}

int ML_Base::sety(const Vector<Vector<double> > &z)
{
    Vector<gentype> y(z.size());

    if ( y.size() )
    {
        int j;

        for ( j = 0 ; j < y.size() ; j++ )
        {
            y("&",j) = z(j);
        }
    }

    return sety(y);
}

int ML_Base::sety(int i, const d_anion &z)
{
    gentype y(z);

    return sety(i,z);
}

int ML_Base::sety(const Vector<int> &i, const Vector<d_anion> &z)
{
    Vector<gentype> y(z.size());

    if ( y.size() )
    {
        int j;

        for ( j = 0 ; j < y.size() ; j++ )
        {
            y("&",j) = z(j);
        }
    }

    return sety(i,y);
}

int ML_Base::sety(const Vector<d_anion> &z)
{
    Vector<gentype> y(z.size());

    if ( y.size() )
    {
        int j;

        for ( j = 0 ; j < y.size() ; j++ )
        {
            y("&",j) = z(j);
        }
    }

    return sety(y);
}


int ML_Base::setzerotol(double zt)
{
    NiceAssert( zt >= 0 );

    globalzerotol = zt;

    return 0;
}

int ML_Base::setCweight(int i, double nv)
{
    xCweight("&",i) = nv;

    return 0;
}

int ML_Base::setCweight(const Vector<int> &i, const Vector<double> &nv)
{
    retVector<double> tmpva;

    xCweight("&",i,tmpva) = nv;

    return 0;
}

int ML_Base::setCweight(const Vector<double> &nv)
{
    xCweight = nv;

    return 0;
}

int ML_Base::setCweightfuzz(int i, double nv)
{
    xCweightfuzz("&",i) = nv;

    return 0;
}

int ML_Base::setCweightfuzz(const Vector<int> &i, const Vector<double> &nv)
{
    retVector<double> tmpva;

    xCweightfuzz("&",i,tmpva) = nv;

    return 0;
}

int ML_Base::setCweightfuzz(const Vector<double> &nv)
{
    xCweightfuzz = nv;

    return 0;
}

int ML_Base::setsigmaweight(int i, double nv)
{
    return setCweight(i,1.0/nv);
}

int ML_Base::setsigmaweight(const Vector<int> &i, const Vector<double> &nv)
{
    Vector<double> xnv(nv);

    if ( xnv.size() )
    {
        int j;

        for ( j = 0 ; j < xnv.size() ; j++ )
        {
            xnv("&",j) = 1.0/nv(j);
        }
    }

    return setCweight(i,xnv);
}

int ML_Base::setsigmaweight(const Vector<double> &nv)
{
    Vector<double> xnv(nv);

    if ( xnv.size() )
    {
        int j;

        for ( j = 0 ; j < xnv.size() ; j++ )
        {
            xnv("&",j) = 1.0/nv(j);
        }
    }

    return setCweight(xnv);
}

int ML_Base::setepsweight(int i, double nv)
{
    xepsweight("&",i) = nv;

    return 0;
}

int ML_Base::setepsweight(const Vector<int> &i, const Vector<double> &nv)
{
    retVector<double> tmpva;

    xepsweight("&",i,tmpva) = nv;

    return 0;
}

int ML_Base::setepsweight(const Vector<double> &nv)
{
    xepsweight = nv;

    return 0;
}

int ML_Base::scaleCweight(double s)
{
    NiceAssert( s >= 0 );

    xCweight *= s;

    return 0;
}

int ML_Base::scaleCweightfuzz(double s)
{
    NiceAssert( s >= 0 );

    xCweightfuzz *= s;

    return 0;
}

int ML_Base::scaleepsweight(double s)
{
    NiceAssert( s >= 0 );

    xepsweight *= s;

    return 0;
}

int ML_Base::setd(int i, int nd)
{
    xdzero -= xd(i) ? 0 : 1;
    xd("&",i) = nd;
    xdzero += xd(i) ? 0 : 1;

    return 0;
}

int ML_Base::setd(const Vector<int> &i, const Vector<int> &nd)
{
    retVector<int> tmpva;

    xd("&",i,tmpva) = nd;

    if ( ML_Base::N() )
    {
        int j;

        for ( j = 0 ; j < ML_Base::N() ; j++ )
        {
            xdzero += xd(j) ? 0 : 1;
        }
    }

    return 0;
}

int ML_Base::setd(const Vector<int> &nd)
{
    xd = nd;

    if ( ML_Base::N() )
    {
        int j;

        for ( j = 0 ; j < ML_Base::N() ; j++ )
        {
            xdzero += xd(j) ? 0 : 1;
        }
    }

    return 0;
}

const Vector<double> &ML_Base::sigmaweight(void) const
{
    ML_Base &reallythis = **thisthisthis;

    reallythis.xsigmaweightonfly = xCweight;

    int i;

    for ( i = 0 ; i < reallythis.xsigmaweightonfly.size() ; i++ )
    {
        reallythis.xsigmaweightonfly("&",i) = 1.0/reallythis.xsigmaweightonfly(i);
    }

    return xsigmaweightonfly;
}

void ML_Base::stabProbTrainingVector(double &res, int i, int p, double pnrm, int rot, double mu, double B) const
{
    NiceAssert( p >= 0 );
    NiceAssert( pnrm >= 0 );
    NiceAssert( mu >= 0 );
    NiceAssert( B >= 0 );

    SparseVector<gentype> xaug(xgetloc(i));

    int k,l;

    res = 1;

    if ( ( p > 0 ) || rot )
    {
        int n = xspaceDim();

        if ( rot )
        {
            // Special case: can calculate in closed form.

            Vector<gentype> eigvals;
            Matrix<gentype> eigvects;
            Vector<gentype> fv1;

            gentype xmeanvec;
            gentype xvarmat;

            for ( k = 1 ; k <= p ; k++ )
            {
                // Setting fff(6) talls the model to evaluate the kth derivative

                xaug.fff("&",6) = k;

                // Calculate mean (vector) and variance (matrix) for kth derivative

                //gg(xmeanvec,xaug);
                var(xvarmat,xmeanvec,xaug);

                // Scale mean (vector) and variance (matrix) as required

                xmeanvec *= pow(B,k)/xnfact(k);
                xvarmat  *= pow(pow(B,k)/xnfact(k),2);

                // Do eigendecomposition of variance matrix

                ((const Matrix<gentype> &) xvarmat).eig(eigvals,eigvects,fv1);

                // Calculate rotated means

                Vector<gentype> &meanvals = (xmeanvec.dir_vector());
                meanvals *= eigvects; // eigvects has eigenvectors in columns

                // Calculate stability score

                int meandim = (int) pow(n,k);

                NiceAssert( meanvals.size() == meandim );

                for ( l = 0 ; l < meandim ; l++ )
                {
                    if ( testisvnan((double) eigvals(l)) || testisinf((double) eigvals(l)) )
                    {
                        eigvals("&",l) = 1.0;
                    }

                    if ( (double) eigvals(l) <= 1e-8 )
                    {
                        eigvals("&",l) = 1e-8;
                    }

                    OP_sqrt(eigvals("&",l));

                    double Phil = 0;
                    double Phir = 0;

                    numbase_Phi(Phil,( ((double) meanvals(l)) + mu ) / ((double) eigvals(l)) );
                    numbase_Phi(Phir,( ((double) meanvals(l)) - mu ) / ((double) eigvals(l)) );

                    res *= (Phil-Phir);
                }
            }
        }

        else
        {
            // General case: generate a bunch of samples and test

            // First calculate mean and variance

            gentype xmeanvec;
            gentype xvarmat;
            Matrix<gentype> xvarchol;

            for ( k = 1 ; k <= p ; k++ )
            {
                int meandim = (int) pow(n,k);

                // Setting fff(6) tells the model to evaluate the kth derivative

                xaug.fff("&",6) = k;

                // Calculate mean (vector) and variance (matrix) for kth derivative

                //gg(xmeanvec,xaug);
                var(xvarmat,xmeanvec,xaug);

                // Scale mean (vector) and variance (matrix) as required

                xmeanvec *= pow(B,k)/xnfact(k);
                xvarmat  *= pow(pow(B,k)/xnfact(k),2);

                Vector<double> xxmeanvec(meandim);
                Matrix<double> xxvarmat(meandim,meandim);

                int r,s;

                const Vector<gentype> &ghgh = (const Vector<gentype> &) xmeanvec;

                for ( r = 0 ; r < meandim ; r++ )
                {
                    xxmeanvec("&",r) = (double) ghgh(r);

                    const Matrix<gentype> &ghgh = (const Matrix<gentype> &) xvarmat;

                    for ( s = 0 ; s < meandim ; s++ )
                    {
                        xxvarmat("&",r,s) = (double) ghgh(r,s);
                    }
                }

                // Do Cholesky decomposition of variance matrix

                Matrix<double> xxvarchol(meandim,meandim);

                xxvarmat.naiveCholNoConj(xxvarchol);

                int totsamp = DEFAULT_BAYES_STABSTABREP;
                int goodsamp = 0;

                Vector<double> v(meandim);

                for ( l = 0 ; l < totsamp ; l++ )
                {
                    randnfill(v);
                    rightmult(xxvarchol,v);
                    v += xxmeanvec;

                    if ( absp(v,pnrm) <= mu )
                    {
                        goodsamp++;
                    }
                }

                res *= ((double) goodsamp)/((double) totsamp);
            }
        }
    }

    return;
}





































std::ostream &ML_Base::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << " ML Base Class\n";
    repPrint(output,'>',dep) << " =============\n";
    repPrint(output,'>',dep) << " \n";
    repPrint(output,'>',dep) << " Base kernel:                      " << kernel           << "\n";
    repPrint(output,'>',dep) << " Base kernel bypass:               " << K2mat            << "\n";
    repPrint(output,'>',dep) << " Base training data:               " << allxdatagent     << "\n";
    repPrint(output,'>',dep) << " Base wild target:                 " << ytargdata        << "\n";
    repPrint(output,'>',dep) << " Base training targets:            " << alltraintarg     << "\n";
    repPrint(output,'>',dep) << " Base training info:               " << traininfo        << "\n";
    repPrint(output,'>',dep) << " Base training tangles:            " << traintang        << "\n";
    repPrint(output,'>',dep) << " Base d:                           " << xd               << "\n";
    repPrint(output,'>',dep) << " Base dnz:                         " << xdzero           << "\n";
    repPrint(output,'>',dep) << " Base zero tol:                    " << globalzerotol    << "\n";
    repPrint(output,'>',dep) << " Base C weight:                    " << xCweight         << "\n";
    repPrint(output,'>',dep) << " Base C weight (fuzz):             " << xCweightfuzz     << "\n";
    repPrint(output,'>',dep) << " Base eps weight:                  " << xepsweight       << "\n";
    repPrint(output,'>',dep) << " Base \"alpha\" state:             " << xalphaState      << "\n";
    repPrint(output,'>',dep) << " Base training index key:          " << indexKey         << "\n";
    repPrint(output,'>',dep) << " Base training index count:        " << indexKeyCount    << "\n";
    repPrint(output,'>',dep) << " Base training type key:           " << typeKey          << "\n";
    repPrint(output,'>',dep) << " Base training type key breakdown: " << typeKeyBreak     << "\n";
    repPrint(output,'>',dep) << " Base index pruning:               " << isIndPrune       << "\n";
    repPrint(output,'>',dep) << " Base x assumed consistency:       " << xassumedconsist  << "\n";
    repPrint(output,'>',dep) << " Base x actual consistency:        " << xconsist         << "\n";
    repPrint(output,'>',dep) << " Base x data real:                 " << trainingDataReal << "\n";
    repPrint(output,'>',dep) << " Base x data assume real:          " << assumeReal       << "\n";
    repPrint(output,'>',dep) << " Base output kernel U:             " << UUoutkernel      << "\n";
    repPrint(output,'>',dep) << " Base local basis U:               " << isBasisUserUU    << "\n";
    repPrint(output,'>',dep) << " Base y basis U:                   " << locbasisUU       << "\n";
    repPrint(output,'>',dep) << " Base default projection U:        " << defbasisUU       << "\n";
    repPrint(output,'>',dep) << " Base local basis V:               " << isBasisUserVV    << "\n";
    repPrint(output,'>',dep) << " Base y basis V:                   " << locbasisVV       << "\n";
    repPrint(output,'>',dep) << " Base default projection V:        " << defbasisVV       << "\n";

    return output;
}

std::istream &ML_Base::inputstream(std::istream &input)
{
    wait_dummy dummy;

    input >> dummy; input >> kernel;
    input >> dummy; input >> K2mat;
    input >> dummy; input >> allxdatagent;
    input >> dummy; input >> ytargdata;
    input >> dummy; input >> alltraintarg;
    input >> dummy; input >> traininfo;
    input >> dummy; input >> traintang;
    input >> dummy; input >> xd;
    input >> dummy; input >> xdzero;
    input >> dummy; input >> globalzerotol;
    input >> dummy; input >> xCweight;
    input >> dummy; input >> xCweightfuzz;
    input >> dummy; input >> xepsweight;
    input >> dummy; input >> xalphaState;
    input >> dummy; input >> indexKey;
    input >> dummy; input >> indexKeyCount;
    input >> dummy; input >> typeKey;
    input >> dummy; input >> typeKeyBreak;
    input >> dummy; input >> isIndPrune;
    input >> dummy; input >> xassumedconsist;
    input >> dummy; input >> xconsist;
    input >> dummy; input >> trainingDataReal;
    input >> dummy; input >> assumeReal;
    input >> dummy; input >> UUoutkernel;
    input >> dummy; input >> isBasisUserUU;
    input >> dummy; input >> locbasisUU;
    input >> dummy; input >> defbasisUU;
    input >> dummy; input >> isBasisUserVV;
    input >> dummy; input >> locbasisVV;
    input >> dummy; input >> defbasisVV;

    fixpvects();

    incxvernum();
    incgvernum();

    calcSetAssumeReal();

    return input;
}

void ML_Base::setmemsize(int memsize)
{
   (void) memsize;

   return;
}

int ML_Base::prealloc(int expectedN)
{
    NiceAssert( ( expectedN == -1 ) || ( expectedN > 0 ) );

    xpreallocsize = expectedN;

    allxdatagent.prealloc(expectedN);
//FIXME    allxdatagentp.prealloc(expectedN);
    alltraintarg.prealloc(expectedN);
    traininfo.prealloc(expectedN);
    traintang.prealloc(expectedN);
//FIXME    traininfop.prealloc(expectedN);
    xd.prealloc(expectedN);
    xCweight.prealloc(expectedN);
    xCweightfuzz.prealloc(expectedN);
    xepsweight.prealloc(expectedN);
    xalphaState.prealloc(expectedN);
    indexKey.prealloc(expectedN);
    indexKeyCount.prealloc(expectedN);
    typeKey.prealloc(expectedN);
    typeKeyBreak.prealloc(expectedN);

    return 0;
}







































































// Global functions

int ML_Base::disable(int i)
{
    return setd(i,0);
}

int ML_Base::disable(const Vector<int> &i)
{
    retVector<int> tmpva;

    return setd(i,zerointvec(i.size(),tmpva));
}

int ML_Base::renormalise(void)
{
    int res = 0;

    if ( ML_Base::N() )
    {
        int i;
        double maxout = 0.0;
        gentype gres;
        double gtemp = 0;

        for ( i = 0 ; i < ML_Base::N() ; i++ )
        {
            ggTrainingVector(gres,i);

            if ( gres.isValNull() )
            {
                ;
            }

            else if ( gres.isCastableToRealWithoutLoss() )
            {
                gtemp = abs2((double) gres);
            }

            else if ( gres.isCastableToVectorWithoutLoss() )
            {
                gtemp = absinf((const Vector<gentype> &) gres);
            }

            else if ( gres.isCastableToAnionWithoutLoss() )
            {
                gtemp = absinf((const d_anion &) gres);
            }

            else
            {
                throw("Unrecognised output type in renormalisation");
            }

            maxout = ( gtemp > maxout ) ? gtemp : maxout;
        }

        if ( maxout > 1 )
        {
            scale(1/maxout);
            res = 1;
        }
    }

    return res;
}


int ML_Base::realign(void)
{
    int res = 0;

    if ( ML_Base::N() )
    {
        Vector<gentype> locy(y());

        int i;
        gentype hres;

        for ( i = 0 ; i < ML_Base::N() ; i++ )
        {
            hhTrainingVector(hres,i);
            locy("&",i) = hres;
        }

        res = sety(locy);
    }

    return res;
}

int ML_Base::autoen(void)
{
    NiceAssert( hOutType() != 'A' );

    int res = 0;

    if ( ML_Base::N() )
    {
        Vector<gentype> locy(y());
        int i,j;

        for ( i = 0 ; i < ML_Base::N() ; i++ )
        {
            gentype hres = y()(i);

            switch ( hOutType() )
            {
                case 'R':
                {
                    {
                        Vector<gentype> xtemp;
                        xlateFromSparseTrainingVector(xtemp,i);

                        if ( xtemp.size() )
                        {
                            hres = (double) xtemp(zeroint());
                        }

                        else
                        {
                            setrand(hres);
                            hres *= 2.0;
                            hres -= 1.0;
                        }
                    }

                    break;
                }

                case 'V':
                {
                    {
                        Vector<gentype> xtemp;
                        xlateFromSparseTrainingVector(xtemp,i);

                        if ( xtemp.size() )
                        {
                            for ( j = 0 ; j < ( xtemp.size() <= hres.dir_vector().size() ? xtemp.size() : hres.dir_vector().size() ) ; j++ )
                            {
                                hres.dir_vector()("&",j) = xtemp(j);
                            }
                        }

                        if ( hres.dir_vector().size() > xtemp.size() )
                        {
                            for ( j = xtemp.size() ; j < hres.dir_vector().size() ; j++ )
                            {
                                setrand(hres.dir_vector()("&",j));
                                hres.dir_vector()("&",j) *= 2.0;
                                hres.dir_vector()("&",j) -= 1.0;
                            }
                        }
                    }

                    break;
                }

                default:
                {
                    // Don't throw - this includes NULL target
                    break;
                }
            }

            hhTrainingVector(hres,i);
            locy("&",i) = hres;
        }

        res = sety(locy);
    }

    return res;
}










































void ML_Base::fillCache(void)
{
    if ( N() > 0 )
    {
        gentype dummy;
        int i,j;

        for ( i = 0 ; i < N() ; i++ )
        {
            for ( j = 0 ; j < N() ; j++ )
            {
                K2(dummy,i,j);
            }
        }
    }

    return;
}


int ML_Base::isKVarianceNZ(void) const
{
    return getKernel().isKVarianceNZ();
}

void ML_Base::K0xfer(gentype &res, int &minmaxind, int typeis,
                     int xdim, int densetype, int resmode, int mlid) const
{
    (void) mlid;

    NiceAssert( !densetype );

    (void) densetype;
    (void) xdim;
    (void) minmaxind;

    res = 0.0;

    switch ( typeis )
    {
        case 800:
        case 810:
        {
            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    K0(res,NULL,resmode);

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    throw("K0xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        case 806:
        case 816:
        {
            res = 0.0;

            break;
        }

        default:
        {
            throw("K0xfer precursor type requested undefined at this level (only 800,806 at ML_Base).");

            break;
        }
    }

    return;
}

void ML_Base::K1xfer(gentype &res, int &minmaxind, int typeis,
                    const SparseVector<gentype> &xa, 
                    const vecInfo &xainfo, 
                    int ia, 
                    int xdim, int densetype, int resmode, int mlid) const
{
    (void) mlid;

    NiceAssert( !densetype );

    (void) densetype;
    (void) xdim;
    (void) minmaxind;

    res = 0.0;

    ia = (typeis-(100*(typeis/100)))/10 ? ia : -42;

    switch ( typeis )
    {
        case 800:
        case 810:
        {
            if ( ia < 0 ) { setInnerWildpa(&xa,&xainfo); ia = -1; }

            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    K1(res,ia,NULL,&xa,&xainfo,resmode);

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    throw("K1xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }
 
            if ( ia < 0 ) { resetInnerWildp(); }

            break;
        }

        case 806:
        {
            gentype ra;

            gg(ra,xa,&xainfo);

            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    oneProduct(res,ra.cast_vector());

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    throw("K1xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        case 816:
        {
            gentype ra;

            ggTrainingVector(ra,ia);

            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    oneProduct(res,ra.cast_vector());

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    throw("K1xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        default:
        {
            throw("K1xfer precursor type requested undefined at this level (only 800 at ML_Base).");

            break;
        }
    }

    return;
}

void ML_Base::K2xfer(gentype &dxyprod, gentype &ddiffis, gentype &res, int &minmaxind, int typeis,
                     const SparseVector<gentype> &xa, const SparseVector<gentype> &xb,
                     const vecInfo &xainfo, const vecInfo &xbinfo,
                     int ia, int ib,
                     int xdim, int densetype, int resmode, int mlid) const
{
    (void) mlid;

    NiceAssert( !densetype );

    (void) densetype;
    (void) xdim;
    (void) minmaxind;

    dxyprod = 0.0;
    ddiffis = 0.0;
    res = 0.0;

    ia = (typeis-(100*(typeis/100)))/10 ? ia : -42;
    ib = (typeis-(100*(typeis/100)))/10 ? ib : -43;

    switch ( typeis )
    {
        case 800:
        case 810:
        {
//errstream() << "phantomxy 1: " << ia << "," << ib << "\t" << xa << "," << xb << "\n";
            if ( ia < 0 ) { setInnerWildpa(&xa,&xainfo); ia = -1; }
            if ( ib < 0 ) { setInnerWildpb(&xb,&xbinfo); ib = -2; }

            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    K2(res,ia,ib,NULL,&xa,&xb,&xainfo,&xbinfo,resmode);

                    break;
                }

                case 16:
                case 32:
                case 48:
                {
                    // K, dK/dxyprod, dK/dxnorm

                    K2(res,ia,ib,NULL,&xa,&xb,&xainfo,&xbinfo,resmode);
                    dK(dxyprod,ddiffis,ia,ib,NULL,&xa,&xb,&xainfo,&xbinfo,1); // deep derivative required

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    throw("K2xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            if ( ( ia < 0 ) || ( ib < 0 ) ) { resetInnerWildp(); }

            break;
        }

        case 806:
        {
            gentype ra;
            gentype rb;

            gg(ra,xa,&xainfo);
            gg(rb,xb,&xbinfo);

            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    twoProduct(res,ra.cast_vector(),rb.cast_vector());

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    throw("K1xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        case 816:
        {
            gentype ra;
            gentype rb;

            ggTrainingVector(ra,ia);
            ggTrainingVector(rb,ib);

            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    twoProduct(res,ra.cast_vector(),rb.cast_vector());

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    throw("K1xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        default:
        {
            throw("K2xfer precursor type requested undefined at this level (only 800 at ML_Base).");

            break;
        }
    }

    return;
}

void ML_Base::K3xfer(gentype &res, int &minmaxind, int typeis,
                    const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, 
                    const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, 
                    int ia, int ib, int ic, 
                    int xdim, int densetype, int resmode, int mlid) const
{
    (void) mlid;

    NiceAssert( !densetype );

    (void) densetype;
    (void) xdim;
    (void) minmaxind;

    res = 0.0;

    ia = (typeis-(100*(typeis/100)))/10 ? ia : -42;
    ib = (typeis-(100*(typeis/100)))/10 ? ib : -43;
    ic = (typeis-(100*(typeis/100)))/10 ? ic : -44;

    switch ( typeis )
    {
        case 800:
        case 810:
        {
            if ( ia < 0 ) { setInnerWildpa(&xa,&xainfo); ia = -1; }
            if ( ib < 0 ) { setInnerWildpb(&xb,&xbinfo); ib = -2; }
            if ( ic < 0 ) { setInnerWildpc(&xc,&xcinfo); ic = -3; }

            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    K3(res,ia,ib,ic,NULL,&xa,&xb,&xc,&xainfo,&xbinfo,&xcinfo,resmode);

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    throw("K3xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            if ( ( ia < 0 ) || ( ib < 0 ) || ( ic < 0 ) ) { resetInnerWildp(); }

            break;
        }

        case 806:
        {
            gentype ra;
            gentype rb;
            gentype rc;

            gg(ra,xa,&xainfo);
            gg(rb,xb,&xbinfo);
            gg(rc,xc,&xcinfo);

            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    threeProduct(res,ra.cast_vector(),rb.cast_vector(),rc.cast_vector());

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    throw("K1xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        case 816:
        {
            gentype ra;
            gentype rb;
            gentype rc;

            ggTrainingVector(ra,ia);
            ggTrainingVector(rb,ib);
            ggTrainingVector(rc,ic);

            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    threeProduct(res,ra.cast_vector(),rb.cast_vector(),rc.cast_vector());

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    throw("K1xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        default:
        {
            throw("K3xfer precursor type requested undefined at this level (only 800 at ML_Base).");

            break;
        }
    }

    return;
}

void ML_Base::K4xfer(gentype &res, int &minmaxind, int typeis,
                    const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd,
                    const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo,
                    int ia, int ib, int ic, int id,
                    int xdim, int densetype, int resmode, int mlid) const
{
    (void) mlid;

    NiceAssert( !densetype );

    (void) densetype;
    (void) xdim;
    (void) minmaxind;

    res = 0.0;

    ia = (typeis-(100*(typeis/100)))/10 ? ia : -42;
    ib = (typeis-(100*(typeis/100)))/10 ? ib : -43;
    ic = (typeis-(100*(typeis/100)))/10 ? ic : -44;
    id = (typeis-(100*(typeis/100)))/10 ? id : -45;

    switch ( typeis )
    {
        case 800:
        case 810:
        {
            if ( ia < 0 ) { setInnerWildpa(&xa,&xainfo); ia = -1; }
            if ( ib < 0 ) { setInnerWildpb(&xb,&xbinfo); ib = -2; }
            if ( ic < 0 ) { setInnerWildpc(&xc,&xcinfo); ic = -3; }
            if ( id < 0 ) { setInnerWildpd(&xd,&xdinfo); id = -4; }

            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    K4(res,ia,ib,ic,id,NULL,&xa,&xb,&xc,&xd,&xainfo,&xbinfo,&xcinfo,&xdinfo,resmode);

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    throw("K4xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            if ( ( ia < 0 ) || ( ib < 0 ) || ( ic < 0 ) || ( id < 0 ) ) { resetInnerWildp(); }

            break;
        }

        case 806:
        {
            gentype ra;
            gentype rb;
            gentype rc;
            gentype rd;

            gg(ra,xa,&xainfo);
            gg(rb,xb,&xbinfo);
            gg(rc,xc,&xcinfo);
            gg(rd,xd,&xdinfo);

            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    fourProduct(res,ra.cast_vector(),rb.cast_vector(),rc.cast_vector(),rd.cast_vector());

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    throw("K1xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        case 816:
        {
            gentype ra;
            gentype rb;
            gentype rc;
            gentype rd;

            ggTrainingVector(ra,ia);
            ggTrainingVector(rb,ib);
            ggTrainingVector(rc,ic);
            ggTrainingVector(rd,id);

            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    fourProduct(res,ra.cast_vector(),rb.cast_vector(),rc.cast_vector(),rd.cast_vector());

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    throw("K1xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        default:
        {
            throw("K4xfer precursor type requested undefined at this level (only 800 at ML_Base).");

            break;
        }
    }

    return;
}

void ML_Base::Kmxfer(gentype &res, int &minmaxind, int typeis,
                    Vector<const SparseVector<gentype> *> &x,
                    Vector<const vecInfo *> &xinfo,
                    Vector<int> &ii,
                    int xdim, int m, int densetype, int resmode, int mlid) const
{
    (void) mlid;

    if ( ( m == 0 ) || ( m == 2 ) || ( m == 4 ) )
    {
        kernPrecursor::Kmxfer(res,minmaxind,typeis,x,xinfo,ii,xdim,m,densetype,resmode,mlid);
        return;
    }

    NiceAssert( !densetype );

    res = 0.0;

    int iq;

    Vector<int> i(ii);

    for ( iq = 0 ; iq < m ; iq++ )
    {
        i("&",iq) = (typeis-(100*(typeis/100)))/10 ? i(iq) : -42;
    }

    switch ( typeis )
    {
        case 800:
        case 810:
        {
            Vector<SparseVector<gentype> > *xx = NULL;

            if ( !( i >= zeroint() ) )
            {
                MEMNEW(xx,Vector<SparseVector<gentype> >(x.size()));

                int ir;

                for ( ir = 0 ; ir < x.size() ; ir++ )
                {
                    (*xx)("&",ir) = *(x(ir));
                }

                retVector<int> tmpva; 

                setInnerWildpx(xx); 
                i = cntintvec(m,tmpva);
                i += 1; 
                i *= -100;
            }

            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    Km(m,res,i,NULL,&x,&xinfo,resmode);

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    throw("Kmxfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            if ( !( i >= zeroint() ) ) { resetInnerWildp(); MEMDEL(xx); }

            break;
        }

        case 806:
        {
            Vector<gentype> r(m);
            Vector<const Vector<gentype> *> rr(m);

            for ( iq = 0 ; iq < m ; iq++ )
            {
                gg(r("&",iq),*(x(iq)),xinfo(iq));
                rr("&",iq) = &(r(iq).cast_vector());
            }

            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    mProduct(res,rr);

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    throw("K1xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        case 816:
        {
            Vector<gentype> r(m);
            Vector<const Vector<gentype> *> rr(m);

            for ( iq = 0 ; iq < m ; iq++ )
            {
                ggTrainingVector(r("&",iq),i(iq));
                rr("&",iq) = &(r(iq).cast_vector());
            }

            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    mProduct(res,rr);

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    throw("K1xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        default:
        {
            throw("Kmxfer precursor type requested undefined at this level (only 800 at ML_Base).");

            break;
        }
    }

    return;
}

void ML_Base::K0xfer(double &res, int &minmaxind, int typeis,
                     int xdim, int densetype, int resmode, int mlid) const
{
    (void) mlid;

    NiceAssert( !densetype );

    res = 0.0;

    switch ( typeis )
    {
        case 800:
        case 810:
        {
            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    K0(res,NULL,resmode);

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    throw("K0xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            break;
        }

        case 806:
        case 816:
        {
            gentype tempres;

            K0xfer(tempres,minmaxind,typeis,xdim,densetype,resmode,mlid);

            res = (double) tempres;

            break;
        }

        default:
        {
            kernPrecursor::K0xfer(res,minmaxind,typeis,xdim,densetype,resmode,mlid);

            break;
        }
    }

    return;
}

void ML_Base::K1xfer(double &res, int &minmaxind, int typeis,
                    const SparseVector<gentype> &xa, 
                    const vecInfo &xainfo, 
                    int ia, 
                    int xdim, int densetype, int resmode, int mlid) const
{
    (void) mlid;

    NiceAssert( !densetype );

    res = 0.0;

    ia = (typeis-(100*(typeis/100)))/10 ? ia : -42;

    switch ( typeis )
    {
        case 800:
        case 810:
        {
            if ( ia < 0 ) { setInnerWildpa(&xa,&xainfo); ia = -1; }

            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    K1(res,ia,NULL,&xa,&xainfo,resmode);

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    throw("K1xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            if ( ( ia < 0 ) ) { resetInnerWildp(); }

            break;
        }

        case 806:
        case 816:
        {
            gentype tempres;

            K1xfer(tempres,minmaxind,typeis,xa,xainfo,ia,xdim,densetype,resmode,mlid);

            res = (double) tempres;

            break;
        }

        default:
        {
            kernPrecursor::K1xfer(res,minmaxind,typeis,xa,xainfo,ia,xdim,densetype,resmode,mlid);

            break;
        }
    }

    return;
}

void ML_Base::K2xfer(double &dxyprod, double &ddiffis, double &res, int &minmaxind, int typeis,
                     const SparseVector<gentype> &xa, const SparseVector<gentype> &xb,
                     const vecInfo &xainfo, const vecInfo &xbinfo,
                     int ia, int ib,
                     int xdim, int densetype, int resmode, int mlid) const
{
    (void) mlid;

    NiceAssert( !densetype );

    dxyprod = 0.0;
    ddiffis = 0.0;
    res = 0.0;

    ia = (typeis-(100*(typeis/100)))/10 ? ia : -42;
    ib = (typeis-(100*(typeis/100)))/10 ? ib : -43;

    switch ( typeis )
    {
        case 800:
        case 810:
        {
//errstream() << "phantomxy 0: " << ia << "," << ib << "\n";
            if ( ia < 0 ) { setInnerWildpa(&xa,&xainfo); ia = -1; }
            if ( ib < 0 ) { setInnerWildpb(&xb,&xbinfo); ib = -2; }

            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    K2(res,ia,ib,NULL,&xa,&xb,&xainfo,&xbinfo,resmode);

                    break;
                }

                case 16:
                case 32:
                case 48:
                {
                    // K, dK/dxyprod, dK/dxnorm

                    K2(res,ia,ib,NULL,&xa,&xb,&xainfo,&xbinfo,resmode);
                    dK(dxyprod,ddiffis,ia,ib,NULL,&xa,&xb,&xainfo,&xbinfo,1); // deep derivative required

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    throw("K2xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            if ( ( ia < 0 ) || ( ib < 0 ) ) { resetInnerWildp(); }

            break;
        }

        case 806:
        case 816:
        {
            gentype tempdxyprod(dxyprod);
            gentype tempddiffis(ddiffis);

            gentype tempres;

            K2xfer(tempdxyprod,tempddiffis,tempres,minmaxind,typeis,xa,xb,xainfo,xbinfo,ia,ib,xdim,densetype,resmode,mlid);

            res = (double) tempres;

            break;
        }

        default:
        {
            kernPrecursor::K2xfer(dxyprod,ddiffis,res,minmaxind,typeis,xa,xb,xainfo,xbinfo,ia,ib,xdim,densetype,resmode,mlid);

            break;
        }
    }

    return;
}

void ML_Base::K3xfer(double &res, int &minmaxind, int typeis,
                    const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, 
                    const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, 
                    int ia, int ib, int ic, 
                    int xdim, int densetype, int resmode, int mlid) const
{
    (void) mlid;

    NiceAssert( !densetype );

    res = 0.0;

    ia = (typeis-(100*(typeis/100)))/10 ? ia : -42;
    ib = (typeis-(100*(typeis/100)))/10 ? ib : -43;
    ic = (typeis-(100*(typeis/100)))/10 ? ic : -44;

    switch ( typeis )
    {
        case 800:
        case 810:
        {
            if ( ia < 0 ) { setInnerWildpa(&xa,&xainfo); ia = -1; }
            if ( ib < 0 ) { setInnerWildpb(&xb,&xbinfo); ib = -2; }
            if ( ic < 0 ) { setInnerWildpc(&xc,&xcinfo); ic = -3; }

            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    K3(res,ia,ib,ic,NULL,&xa,&xb,&xc,&xainfo,&xbinfo,&xcinfo,resmode);

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    throw("K3xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            if ( ( ia < 0 ) || ( ib < 0 ) || ( ic < 0 ) ) { resetInnerWildp(); }

            break;
        }

        case 806:
        case 816:
        {
            gentype tempres;

            K3xfer(tempres,minmaxind,typeis,xa,xb,xc,xainfo,xbinfo,xcinfo,ia,ib,ic,xdim,densetype,resmode,mlid);

            res = (double) tempres;

            break;
        }

        default:
        {
            kernPrecursor::K3xfer(res,minmaxind,typeis,xa,xb,xc,xainfo,xbinfo,xcinfo,ia,ib,ic,xdim,densetype,resmode,mlid);

            break;
        }
    }

    return;
}

void ML_Base::K4xfer(double &res, int &minmaxind, int typeis,
                    const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd,
                    const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo,
                    int ia, int ib, int ic, int id,
                    int xdim, int densetype, int resmode, int mlid) const
{
    (void) mlid;

    NiceAssert( !densetype );

    res = 0.0;

    ia = (typeis-(100*(typeis/100)))/10 ? ia : -42;
    ib = (typeis-(100*(typeis/100)))/10 ? ib : -43;
    ic = (typeis-(100*(typeis/100)))/10 ? ic : -44;
    id = (typeis-(100*(typeis/100)))/10 ? id : -45;

    switch ( typeis )
    {
        case 800:
        case 810:
        {
            if ( ia < 0 ) { setInnerWildpa(&xa,&xainfo); ia = -1; }
            if ( ib < 0 ) { setInnerWildpb(&xb,&xbinfo); ib = -2; }
            if ( ic < 0 ) { setInnerWildpc(&xc,&xcinfo); ic = -3; }
            if ( id < 0 ) { setInnerWildpd(&xd,&xdinfo); id = -4; }

            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    K4(res,ia,ib,ic,id,NULL,&xa,&xb,&xc,&xd,&xainfo,&xbinfo,&xcinfo,&xdinfo,resmode);

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    throw("K4xfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            if ( ( ia < 0 ) || ( ib < 0 ) || ( ic < 0 ) || ( id < 0 ) ) { resetInnerWildp(); }

            break;
        }

        case 806:
        case 816:
        {
            gentype tempres;

            K4xfer(tempres,minmaxind,typeis,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,ia,ib,ic,id,xdim,densetype,resmode,mlid);

            res = (double) tempres;

            break;
        }

        default:
        {
            kernPrecursor::K4xfer(res,minmaxind,typeis,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,ia,ib,ic,id,xdim,densetype,resmode,mlid);

            break;
        }
    }

    return;
}

void ML_Base::Kmxfer(double &res, int &minmaxind, int typeis,
                    Vector<const SparseVector<gentype> *> &x,
                    Vector<const vecInfo *> &xinfo,
                    Vector<int> &ii,
                    int xdim, int m, int densetype, int resmode, int mlid) const
{
    (void) mlid;

    if ( ( m == 0 ) || ( m == 2 ) || ( m == 4 ) )
    {
        kernPrecursor::Kmxfer(res,minmaxind,typeis,x,xinfo,ii,xdim,m,densetype,resmode,mlid);
        return;
    }

    NiceAssert( !densetype );

    res = 0.0;

    int iq;

    Vector<int> i(ii);

    for ( iq = 0 ; iq < m ; iq++ )
    {
        i("&",iq) = (typeis-(100*(typeis/100)))/10 ? i(iq) : -42;
    }

    switch ( typeis )
    {
        case 800:
        case 810:
        {
            Vector<SparseVector<gentype> > *xx = NULL;

            if ( !( i >= zeroint() ) )
            {
                MEMNEW(xx,Vector<SparseVector<gentype> >(x.size()));

                int ir;

                for ( ir = 0 ; ir < x.size() ; ir++ )
                {
                    (*xx)("&",ir) = *(x(ir));
                }

                retVector<int> tmpva; 

                setInnerWildpx(xx); 
                i = cntintvec(m,tmpva);
                i += 1; 
                i *= -100;
            }

            switch ( resmode )
            {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                {
                    Km(m,res,i,NULL,&x,&xinfo,resmode);

                    break;
                }

                case 128:
                {
                    res = 0.0;

                    break;
                }

                default:
                {
                    throw("Kmxfer precursor specified resmode undefined at this level.");

                    break;
                }
            }

            if ( !( i >= zeroint() ) ) { resetInnerWildp(); MEMDEL(xx); }

            break;
        }

        case 806:
        case 816:
        {
            gentype tempres;

            Kmxfer(tempres,minmaxind,typeis,x,xinfo,i,xdim,m,densetype,resmode,mlid);

            res = (double) tempres;

            break;
        }

        default:
        {
            kernPrecursor::Kmxfer(res,minmaxind,typeis,x,xinfo,i,xdim,m,densetype,resmode,mlid);

            break;
        }
    }

    return;
}

double ML_Base::getvalIfPresent(int numi, int numj, int &isgood) const
{
    (void) numi;
    (void) numj;

    isgood = 0;

    static double dummy = 0;

    return dummy;
}





















































gentype &ML_Base::K0(gentype &res, 
                     const gentype **pxyprod, 
                     int resmode) const
{
    return K0(res,zerogentype(),getKernel(),pxyprod,resmode);
}

double &ML_Base::K0(double &res,
                   const gentype **pxyprod, 
                   int resmode) const
{
    static double zeroval = 0.0;

    return K0(res,zeroval,getKernel(),pxyprod,resmode);
}

gentype &ML_Base::K0(gentype &res, 
                     const gentype &bias, const gentype **pxyprod, 
                     int resmode) const
{
    return K0(res,bias,getKernel(),pxyprod,resmode);
}

gentype &ML_Base::K0(gentype &res, 
                     const MercerKernel &altK, const gentype **pxyprod, 
                     int resmode) const
{
    return K0(res,zerogentype(),altK,pxyprod,resmode);
}

Matrix<double> &ML_Base::K0(int spaceDim, Matrix<double> &res, 
                            const gentype **pxyprod, 
                            int resmode) const
{
    gentype tempres;

    K0(tempres,pxyprod,resmode);

    if ( spaceDim >= 1 )
    {
        gentypeToMatrixRep(res,tempres,spaceDim);
    }

    return res;
}

d_anion &ML_Base::K0(int order, d_anion &res, 
                     const gentype **pxyprod, 
                     int resmode) const
{
    gentype tempres;

    K0(tempres,pxyprod,resmode);

    res = (const d_anion &) tempres;

    if ( order != -1 )
    {
        res.setorder(order);
    }

    return res;
}

gentype &ML_Base::K1(gentype &res, 
                     int ia, 
                     const gentype **pxyprod, 
                     const SparseVector<gentype> *xa, 
                     const vecInfo *xainfo, 
                     int resmode) const
{
    int iaset = 0;

    return K1(res,ia,zerogentype(),getKernel(),pxyprod,xa,xainfo,resmode,iaset);
}

double &ML_Base::K1(double &res,
                   int ia, 
                   const gentype **pxyprod, 
                   const SparseVector<gentype> *xa, 
                   const vecInfo *xainfo, 
                   int resmode) const
{
    static double zeroval = 0.0;

    int iaset = 0;

    return K1(res,ia,zeroval,getKernel(),pxyprod,xa,xainfo,resmode,iaset);
}

gentype &ML_Base::K1(gentype &res, 
                     int ia, 
                     const gentype &bias, const gentype **pxyprod, 
                     const SparseVector<gentype> *xa, 
                     const vecInfo *xainfo, 
                     int resmode) const
{
    int iaset = 0;

    return K1(res,ia,bias,getKernel(),pxyprod,xa,xainfo,resmode,iaset);
}

gentype &ML_Base::K1(gentype &res, 
                     int ia, 
                     const MercerKernel &altK, const gentype **pxyprod, 
                     const SparseVector<gentype> *xa, 
                     const vecInfo *xainfo, 
                     int resmode) const
{
    int iaset = 0;

    return K1(res,ia,zerogentype(),altK,pxyprod,xa,xainfo,resmode,iaset);
}

Matrix<double> &ML_Base::K1(int spaceDim, Matrix<double> &res, 
                            int ia, 
                            const gentype **pxyprod, 
                            const SparseVector<gentype> *xa, 
                            const vecInfo *xainfo, 
                            int resmode) const
{
    gentype tempres;

    K1(tempres,ia,pxyprod,xa,xainfo,resmode);

    if ( spaceDim >= 1 )
    {
        gentypeToMatrixRep(res,tempres,spaceDim);
    }

    return res;
}

d_anion &ML_Base::K1(int order, d_anion &res, 
                     int ia, 
                     const gentype **pxyprod, 
                     const SparseVector<gentype> *xa, 
                     const vecInfo *xainfo, 
                     int resmode) const
{
    gentype tempres;

    K1(tempres,ia,pxyprod,xa,xainfo,resmode);

    res = (const d_anion &) tempres;

    if ( order != -1 )
    {
        res.setorder(order);
    }

    return res;
}

gentype &ML_Base::K2(gentype &res, 
                    int ib, int jb, 
                    const gentype **pxyprod, 
                    const SparseVector<gentype> *xx, const SparseVector<gentype> *yy, 
                    const vecInfo *xxinfo, const vecInfo *yyinfo, 
                    int resmode) const
{
    int iaset = 0;
    int ibset = 0;

    return K2(res,ib,jb,zerogentype(),getKernel(),pxyprod,xx,yy,xxinfo,yyinfo,resmode,iaset,ibset);
}

gentype &ML_Base::K2(gentype &res, 
                    int ib, int jb, 
                    const gentype &bias, const gentype **pxyprod, 
                    const SparseVector<gentype> *xx, const SparseVector<gentype> *yy, 
                    const vecInfo *xxinfo, const vecInfo *yyinfo, 
                    int resmode) const
{
    int iaset = 0;
    int ibset = 0;

    return K2(res,ib,jb,bias,getKernel(),pxyprod,xx,yy,xxinfo,yyinfo,resmode,iaset,ibset);
}

double &ML_Base::K2(double &res, 
                   int ib, int jb, 
                   const gentype **pxyprod, 
                   const SparseVector<gentype> *xx, const SparseVector<gentype> *yy, 
                   const vecInfo *xxinfo, const vecInfo *yyinfo, 
                   int resmode) const
{
    static double zeroval = 0.0;

    int iaset = 0;
    int ibset = 0;

    return K2(res,ib,jb,zeroval,getKernel(),pxyprod,xx,yy,xxinfo,yyinfo,resmode,iaset,ibset);
}

gentype &ML_Base::K2(gentype &res, 
                    int ib, int jb, 
                    const MercerKernel &altK, const gentype **pxyprod, 
                    const SparseVector<gentype> *xx, const SparseVector<gentype> *yy, 
                    const vecInfo *xxinfo, const vecInfo *yyinfo, 
                    int resmode) const
{
    int iaset = 0;
    int ibset = 0;

    return K2(res,ib,jb,zerogentype(),altK,pxyprod,xx,yy,xxinfo,yyinfo,resmode,iaset,ibset);
}

Matrix<double> &ML_Base::K2(int spaceDim, Matrix<double> &res, 
                           int i, int j, 
                           const gentype **pxyprod, 
                           const SparseVector<gentype> *xx, const SparseVector<gentype> *yy, 
                           const vecInfo *xxinfo, const vecInfo *yyinfo, 
                           int resmode) const
{
    gentype tempres;

    K2(tempres,i,j,pxyprod,xx,yy,xxinfo,yyinfo,resmode);

    if ( spaceDim >= 1 )
    {
        gentypeToMatrixRep(res,tempres,spaceDim);
    }

    return res;
}

d_anion &ML_Base::K2(int order, d_anion &res, 
                    int i, int j, 
                    const gentype **pxyprod, 
                    const SparseVector<gentype> *xx, const SparseVector<gentype> *yy, 
                    const vecInfo *xxinfo, const vecInfo *yyinfo, 
                    int resmode) const
{
    gentype tempres;

    K2(tempres,i,j,pxyprod,xx,yy,xxinfo,yyinfo,resmode);

    res = (const d_anion &) tempres;

    if ( order != -1 )
    {
        res.setorder(order);
    }

    return res;
}

gentype &ML_Base::K3(gentype &res, 
                     int ia, int ib, int ic, 
                     const gentype **pxyprod, 
                     const SparseVector<gentype> *xa, const SparseVector<gentype> *xb, const SparseVector<gentype> *xc, 
                     const vecInfo *xainfo, const vecInfo *xbinfo, const vecInfo *xcinfo, 
                     int resmode) const
{
    int iaset = 0;
    int ibset = 0;
    int icset = 0;

    return K3(res,ia,ib,ic,zerogentype(),getKernel(),pxyprod,xa,xb,xc,xainfo,xbinfo,xcinfo,resmode,iaset,ibset,icset);
}

double &ML_Base::K3(double &res,
                   int ia, int ib, int ic, 
                   const gentype **pxyprod, 
                   const SparseVector<gentype> *xa, const SparseVector<gentype> *xb, const SparseVector<gentype> *xc, 
                   const vecInfo *xainfo, const vecInfo *xbinfo, const vecInfo *xcinfo, 
                   int resmode) const
{
    static double zeroval = 0.0;

    int iaset = 0;
    int ibset = 0;
    int icset = 0;

    return K3(res,ia,ib,ic,zeroval,getKernel(),pxyprod,xa,xb,xc,xainfo,xbinfo,xcinfo,resmode,iaset,ibset,icset);
}

gentype &ML_Base::K3(gentype &res, 
                     int ia, int ib, int ic, 
                     const gentype &bias, const gentype **pxyprod, 
                     const SparseVector<gentype> *xa, const SparseVector<gentype> *xb, const SparseVector<gentype> *xc, 
                     const vecInfo *xainfo, const vecInfo *xbinfo, const vecInfo *xcinfo, 
                     int resmode) const
{
    int iaset = 0;
    int ibset = 0;
    int icset = 0;

    return K3(res,ia,ib,ic,bias,getKernel(),pxyprod,xa,xb,xc,xainfo,xbinfo,xcinfo,resmode,iaset,ibset,icset);
}

gentype &ML_Base::K3(gentype &res, 
                     int ia, int ib, int ic, 
                     const MercerKernel &altK, const gentype **pxyprod, 
                     const SparseVector<gentype> *xa, const SparseVector<gentype> *xb, const SparseVector<gentype> *xc, 
                     const vecInfo *xainfo, const vecInfo *xbinfo, const vecInfo *xcinfo, 
                     int resmode) const
{
    int iaset = 0;
    int ibset = 0;
    int icset = 0;

    return K3(res,ia,ib,ic,zerogentype(),altK,pxyprod,xa,xb,xc,xainfo,xbinfo,xcinfo,resmode,iaset,ibset,icset);
}

Matrix<double> &ML_Base::K3(int spaceDim, Matrix<double> &res, 
                            int ia, int ib, int ic, 
                            const gentype **pxyprod, 
                            const SparseVector<gentype> *xa, const SparseVector<gentype> *xb, const SparseVector<gentype> *xc, 
                            const vecInfo *xainfo, const vecInfo *xbinfo, const vecInfo *xcinfo, 
                            int resmode) const
{
    gentype tempres;

    K3(tempres,ia,ib,ic,pxyprod,xa,xb,xc,xainfo,xbinfo,xcinfo,resmode);

    if ( spaceDim >= 1 )
    {
        gentypeToMatrixRep(res,tempres,spaceDim);
    }

    return res;
}

d_anion &ML_Base::K3(int order, d_anion &res, 
                     int ia, int ib, int ic, 
                     const gentype **pxyprod, 
                     const SparseVector<gentype> *xa, const SparseVector<gentype> *xb, const SparseVector<gentype> *xc, 
                     const vecInfo *xainfo, const vecInfo *xbinfo, const vecInfo *xcinfo, 
                     int resmode) const
{
    gentype tempres;

    K3(tempres,ia,ib,ic,pxyprod,xa,xb,xc,xainfo,xbinfo,xcinfo,resmode);

    res = (const d_anion &) tempres;

    if ( order != -1 )
    {
        res.setorder(order);
    }

    return res;
}

gentype &ML_Base::K4(gentype &res, 
                     int ia, int ib, int ic, int id, 
                     const gentype **pxyprod, 
                     const SparseVector<gentype> *xa, const SparseVector<gentype> *xb, const SparseVector<gentype> *xc, const SparseVector<gentype> *xd, 
                     const vecInfo *xainfo, const vecInfo *xbinfo, const vecInfo *xcinfo, const vecInfo *xdinfo, 
                     int resmode) const
{
    int iaset = 0;
    int ibset = 0;
    int icset = 0;
    int idset = 0;

    return K4(res,ia,ib,ic,id,zerogentype(),getKernel(),pxyprod,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,resmode,iaset,ibset,icset,idset);
}

double &ML_Base::K4(double &res,
                   int ia, int ib, int ic, int id, 
                   const gentype **pxyprod, 
                   const SparseVector<gentype> *xa, const SparseVector<gentype> *xb, const SparseVector<gentype> *xc, const SparseVector<gentype> *xd, 
                   const vecInfo *xainfo, const vecInfo *xbinfo, const vecInfo *xcinfo, const vecInfo *xdinfo, 
                   int resmode) const
{
    static double zeroval = 0.0;

    int iaset = 0;
    int ibset = 0;
    int icset = 0;
    int idset = 0;

    return K4(res,ia,ib,ic,id,zeroval,getKernel(),pxyprod,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,resmode,iaset,ibset,icset,idset);
}

gentype &ML_Base::K4(gentype &res, 
                     int ia, int ib, int ic, int id,
                     const gentype &bias, const gentype **pxyprod, 
                     const SparseVector<gentype> *xa, const SparseVector<gentype> *xb, const SparseVector<gentype> *xc, const SparseVector<gentype> *xd, 
                     const vecInfo *xainfo, const vecInfo *xbinfo, const vecInfo *xcinfo, const vecInfo *xdinfo, 
                     int resmode) const
{
    int iaset = 0;
    int ibset = 0;
    int icset = 0;
    int idset = 0;

    return K4(res,ia,ib,ic,id,bias,getKernel(),pxyprod,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,resmode,iaset,ibset,icset,idset);
}

gentype &ML_Base::K4(gentype &res, 
                     int ia, int ib, int ic, int id, 
                     const MercerKernel &altK, const gentype **pxyprod, 
                     const SparseVector<gentype> *xa, const SparseVector<gentype> *xb, const SparseVector<gentype> *xc, const SparseVector<gentype> *xd, 
                     const vecInfo *xainfo, const vecInfo *xbinfo, const vecInfo *xcinfo, const vecInfo *xdinfo, 
                     int resmode) const
{
    int iaset = 0;
    int ibset = 0;
    int icset = 0;
    int idset = 0;

    return K4(res,ia,ib,ic,id,zerogentype(),altK,pxyprod,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,resmode,iaset,ibset,icset,idset);
}

Matrix<double> &ML_Base::K4(int spaceDim, Matrix<double> &res, 
                            int ia, int ib, int ic, int id, 
                            const gentype **pxyprod, 
                            const SparseVector<gentype> *xa, const SparseVector<gentype> *xb, const SparseVector<gentype> *xc, const SparseVector<gentype> *xd, 
                            const vecInfo *xainfo, const vecInfo *xbinfo, const vecInfo *xcinfo, const vecInfo *xdinfo, 
                            int resmode) const
{
    gentype tempres;

    K4(tempres,ia,ib,ic,id,pxyprod,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,resmode);

    if ( spaceDim >= 1 )
    {
        gentypeToMatrixRep(res,tempres,spaceDim);
    }

    return res;
}

d_anion &ML_Base::K4(int order, d_anion &res, 
                     int ia, int ib, int ic, int id, 
                     const gentype **pxyprod, 
                     const SparseVector<gentype> *xa, const SparseVector<gentype> *xb, const SparseVector<gentype> *xc, const SparseVector<gentype> *xd, 
                     const vecInfo *xainfo, const vecInfo *xbinfo, const vecInfo *xcinfo, const vecInfo *xdinfo, 
                     int resmode) const
{
    gentype tempres;

    K4(tempres,ia,ib,ic,id,pxyprod,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,resmode);

    res = (const d_anion &) tempres;

    if ( order != -1 )
    {
        res.setorder(order);
    }

    return res;
}

gentype &ML_Base::Km(int m, gentype &res, 
                     Vector<int> &i, 
                     const gentype **pxyprod, 
                     Vector<const SparseVector<gentype> *> *xxx, 
                     Vector<const vecInfo *> *xxxinfo, 
                     int resmode) const
{
    Vector<int> iset(m);

    iset = zeroint();

    return Km(m,res,i,zerogentype(),getKernel(),pxyprod,xxx,xxxinfo,resmode,NULL,NULL,-1,-1,0,0,iset);
}

gentype &ML_Base::Km(int m, gentype &res, 
                     Vector<int> &i, 
                     const gentype &bias, const gentype **pxyprod, 
                     Vector<const SparseVector<gentype> *> *xxx, 
                     Vector<const vecInfo *> *xxxinfo, 
                     int resmode) const
{
    Vector<int> iset(m);

    iset = zeroint();

    return Km(m,res,i,bias,getKernel(),pxyprod,xxx,xxxinfo,resmode,NULL,NULL,-1,-1,0,0,iset);
}

gentype &ML_Base::Km(int m, gentype &res, 
                     Vector<int> &i, 
                     const MercerKernel &altK, const gentype **pxyprod, 
                     Vector<const SparseVector<gentype> *> *xxx, 
                     Vector<const vecInfo *> *xxxinfo, 
                     int resmode) const
{
    Vector<int> iset(m);

    iset = zeroint();

    return Km(m,res,i,zerogentype(),altK,pxyprod,xxx,xxxinfo,resmode,NULL,NULL,-1,-1,0,0,iset);
}

double &ML_Base::Km(int m, double &res, 
                    Vector<int> &i, 
                    const gentype **pxyprod, 
                    Vector<const SparseVector<gentype> *> *xxx, 
                    Vector<const vecInfo *> *xxxinfo, 
                    int resmode) const
{
    static double zeroval = 0.0;

    Vector<int> iset(m);

    iset = zeroint();

    return Km(m,res,i,zeroval,getKernel(),pxyprod,xxx,xxxinfo,resmode,NULL,NULL,-1,-1,0,0,iset);
}

Matrix<double> &ML_Base::Km(int m, int spaceDim, Matrix<double> &res, 
                            Vector<int> &i, 
                            const gentype **pxyprod, 
                            Vector<const SparseVector<gentype> *> *xx, 
                            Vector<const vecInfo *> *xxinfo, 
                            int resmode) const
{
    gentype tempres;

    Km(m,tempres,i,pxyprod,xx,xxinfo,resmode);

    if ( spaceDim >= 1 )
    {
        gentypeToMatrixRep(res,tempres,spaceDim);
    }

    return res;
}

d_anion &ML_Base::Km(int m, int order, d_anion &res, 
                     Vector<int> &i, 
                     const gentype **pxyprod, 
                     Vector<const SparseVector<gentype> *> *xx, 
                     Vector<const vecInfo *> *xxinfo, 
                     int resmode) const
{
    gentype tempres;

    Km(m,tempres,i,pxyprod,xx,xxinfo,resmode);

    res = (const d_anion &) tempres;

    if ( order != -1 )
    {
        res.setorder(order);
    }

    return res;
}

void ML_Base::dK(gentype &xygrad, gentype &xnormgrad, 
                 int ib, int jb, 
                 const gentype **pxyprod, 
                 const SparseVector<gentype> *xx, const SparseVector<gentype> *yy, 
                 const vecInfo *xxinfo, const vecInfo *yyinfo, 
                 int deepDeriv) const
{
    int iaset = 0;
    int ibset = 0;

    dK(xygrad,xnormgrad,ib,jb,zerogentype(),getKernel(),pxyprod,xx,yy,xxinfo,yyinfo,deepDeriv,iaset,ibset);

    return;
}

void ML_Base::dK(double &xygrad, double &xnormgrad, 
                 int ib, int jb, 
                 const gentype **pxyprod, 
                 const SparseVector<gentype> *xx, const SparseVector<gentype> *yy, 
                 const vecInfo *xxinfo, const vecInfo *yyinfo, 
                 int deepDeriv) const
{
    static double zeroval = 0.0;

    int iaset = 0;
    int ibset = 0;

    dK(xygrad,xnormgrad,ib,jb,zeroval,getKernel(),pxyprod,xx,yy,xxinfo,yyinfo,deepDeriv,iaset,ibset);

    return;
}

void ML_Base::dK2delx(gentype &xscaleres, gentype &yscaleres, int &minmaxind, 
                     int ib, int jb, 
                     const gentype **pxyprod, 
                     const SparseVector<gentype> *xx, const SparseVector<gentype> *yy, 
                     const vecInfo *xxinfo, const vecInfo *yyinfo) const
{
    int iset = 0;
    int jset = 0;

    dK2delx(xscaleres,yscaleres,minmaxind,ib,jb,zerogentype(),getKernel(),pxyprod,xx,yy,xxinfo,yyinfo,iset,jset);

    return;
}

void ML_Base::dK2delx(double &xscaleres, double &yscaleres, int &minmaxind, 
                     int ib, int jb, 
                     const gentype **pxyprod, 
                     const SparseVector<gentype> *xx, const SparseVector<gentype> *yy, 
                     const vecInfo *xxinfo, const vecInfo *yyinfo) const
{
    static double zeroval = 0.0;

    int iset = 0;
    int jset = 0;

    dK2delx(xscaleres,yscaleres,minmaxind,ib,jb,zeroval,getKernel(),pxyprod,xx,yy,xxinfo,yyinfo,iset,jset);

    return;
}

void ML_Base::d2K(gentype &xygrad, gentype &xnormgrad, gentype &xyxygrad, gentype &xyxnormgrad, gentype &xyynormgrad, gentype &xnormxnormgrad, gentype &xnormynormgrad, gentype &ynormynormgrad, int &minmaxind, 
                  int i, int j, 
                  const gentype **pxyprod, 
                  const SparseVector<gentype> *xx, const SparseVector<gentype> *yy, 
                  const vecInfo *xxinfo, const vecInfo *yyinfo) const
{
    int iaset = 0;
    int ibset = 0;

    d2K(xygrad,xnormgrad,xyxygrad,xyxnormgrad,xyynormgrad,xnormxnormgrad,xnormynormgrad,ynormynormgrad,minmaxind,i,j,zerogentype(),getKernel(),pxyprod,xx,yy,xxinfo,yyinfo,iaset,ibset);

    return;
}

void ML_Base::d2K(double  &xygrad, double  &xnormgrad, double  &xyxygrad, double  &xyxnormgrad, double  &xyynormgrad, double  &xnormxnormgrad, double  &xnormynormgrad, double  &ynormynormgrad, int &minmaxind, 
                  int i, int j, 
                  const gentype **pxyprod, 
                  const SparseVector<gentype> *xx, const SparseVector<gentype> *yy, 
                  const vecInfo *xxinfo, const vecInfo *yyinfo) const
{
    static double zeroval = 0.0;

    int iaset = 0;
    int ibset = 0;

    d2K(xygrad,xnormgrad,xyxygrad,xyxnormgrad,xyynormgrad,xnormxnormgrad,xnormynormgrad,ynormynormgrad,minmaxind,i,j,zeroval,getKernel(),pxyprod,xx,yy,xxinfo,yyinfo,iaset,ibset);

    return;
}

void ML_Base::d2K2delxdelx(gentype &xxscaleres, gentype &yyscaleres, gentype &xyscaleres, gentype &yxscaleres, gentype &constres, int &minmaxind, 
                          int i, int j, 
                          const gentype **pxyprod, 
                          const SparseVector<gentype> *xx, const SparseVector<gentype> *yy, 
                          const vecInfo *xxinfo, const vecInfo *yyinfo) const
{
    int iaset = 0;
    int ibset = 0;

    d2K2delxdelx(xxscaleres,yyscaleres,xyscaleres,yxscaleres,constres,minmaxind,i,j,zerogentype(),getKernel(),pxyprod,xx,yy,xxinfo,yyinfo,iaset,ibset);

    return;
}

void ML_Base::d2K2delxdely(gentype &xxscaleres, gentype &yyscaleres, gentype &xyscaleres, gentype &yxscaleres, gentype &constres, int &minmaxind, 
                          int i, int j, 
                          const gentype **pxyprod, 
                          const SparseVector<gentype> *xx, const SparseVector<gentype> *yy, 
                          const vecInfo *xxinfo, const vecInfo *yyinfo) const
{
    int iaset = 0;
    int ibset = 0;

    d2K2delxdely(xxscaleres,yyscaleres,xyscaleres,yxscaleres,constres,minmaxind,i,j,zerogentype(),getKernel(),pxyprod,xx,yy,xxinfo,yyinfo,iaset,ibset);

    return;
}

void ML_Base::d2K2delxdelx(double  &xxscaleres, double  &yyscaleres, double  &xyscaleres, double  &yxscaleres, double  &constres, int &minmaxind, 
                          int i, int j, 
                          const gentype **pxyprod, 
                          const SparseVector<gentype> *xx, const SparseVector<gentype> *yy, 
                          const vecInfo *xxinfo, const vecInfo *yyinfo) const
{
    static double zeroval = 0.0;

    int iaset = 0;
    int ibset = 0;

    d2K2delxdelx(xxscaleres,yyscaleres,xyscaleres,yxscaleres,constres,minmaxind,i,j,zeroval,getKernel(),pxyprod,xx,yy,xxinfo,yyinfo,iaset,ibset);

    return;
}

void ML_Base::d2K2delxdely(double  &xxscaleres, double  &yyscaleres, double  &xyscaleres, double  &yxscaleres, double  &constres, int &minmaxind, 
                          int i, int j, 
                          const gentype **pxyprod, 
                          const SparseVector<gentype> *xx, const SparseVector<gentype> *yy, 
                          const vecInfo *xxinfo, const vecInfo *yyinfo) const
{
    static double zeroval = 0.0;

    int iaset = 0;
    int ibset = 0;

    d2K2delxdely(xxscaleres,yyscaleres,xyscaleres,yxscaleres,constres,minmaxind,i,j,zeroval,getKernel(),pxyprod,xx,yy,xxinfo,yyinfo,iaset,ibset);

    return;
}

void ML_Base::dnK2del(Vector<gentype> &sc, Vector<Vector<int> > &n, int &minmaxind, 
                     const Vector<int> &q, 
                     int i, int j, 
                     const gentype **pxyprod, 
                     const SparseVector<gentype> *xx, const SparseVector<gentype> *yy, 
                     const vecInfo *xxinfo, const vecInfo *yyinfo) const
{
    int iaset = 0;
    int ibset = 0;

    dnK2del(sc,n,minmaxind,q,i,j,zerogentype(),getKernel(),pxyprod,xx,yy,xxinfo,yyinfo,iaset,ibset);

    return;
}

void ML_Base::dnK2del(Vector<double> &sc, Vector<Vector<int> > &n, int &minmaxind, 
                     const Vector<int> &q, 
                     int i, int j, 
                     const gentype **pxyprod, 
                     const SparseVector<gentype> *xx, const SparseVector<gentype> *yy, 
                     const vecInfo *xxinfo, const vecInfo *yyinfo) const
{
    static double zeroval = 0.0;

    int iaset = 0;
    int ibset = 0;

    dnK2del(sc,n,minmaxind,q,i,j,zeroval,getKernel(),pxyprod,xx,yy,xxinfo,yyinfo,iaset,ibset);

    return;
}





// Kronecker product:
//
// res = b(0) \otimes b(1) ...
// where b(i) has dim dim.
// use NULLs to indicate vectorised identity matrix.  These are labelled (paired) as negative indices in the nn vector.
//
// Second form: dissimilar dimensions dima and dimb

template <class T>
void kronprod(SparseVector<T> &res, Vector<const SparseVector<T> *> &b, const Vector<int> &nn, int dim)
{
    int i,j,k,l,p;
    int n = b.size();
    int ressize = (int) pow(dim,n);

    SparseVector<int> idset;

    res.zero();

    for ( i = 0 ; i < ressize ; i++ )
    {
        res("&",i) = 1.0;

        k = i;
        l = ressize;

        idset.zero();

        for ( j = 0 ; j < n ; j++ )
        {
            l = l/dim;
            p = k/l;
            k = k%l;

            if ( b(j) )
            {
                res("&",i) *= (*b(j))(p);
            }

            else
            {
                NiceAssert( nn(j) < 0 );

                if ( !(idset.isindpresent(-nn(j))) )
                {
                    idset("&",-nn(j)) = p; // store first index of kronecker-delta, but don't do anything yet
                }

                else if ( idset(-nn(j)) != p )
                {
                    res("&",i) = 0.0; // second index doesn't match first, so result is zero
                    break; // break out of inner for loop for speed
                }
            }
        }
    }

    return;
}

//template <class T>
//void kronprod(SparseVector<T> &res, const SparseVector<T> &a, const SparseVector<T> &b, int dima, int dimb)
//{
//    int i,j;
//
//    res.zero();
//
//    for ( i = 0 ; i < dima ; i++ )
//    {
//        for ( j = 0 ; j < dimb ; j++ )
//        { 
//            res("&",(i*dimb)+j) = a(i)*b(j);
//        }
//    }
//
//    return;
//}

// get unique (kinda, cycling) pseudo non-training-vector index between -10 and -90

int UPNTVI(int i, int off);
int UPNTVI(int i, int off)
{
//    static unsigned int ind = 0;
//
//    return -(((ind++)%81)+10);
    return -((10*off)+i);
}

// return 1 if index indicates vector in training set, 0 otherwise

int istrv(int i);
int istrv(int i)
{
    return ( i >= 0 ) || ( i <= -100 );
    //return ( i >= 0 ) || ( i <= -10 );
}

int istrv(const Vector<int> &i);
int istrv(const Vector<int> &i)
{
    //NB: this won't work exactly in the case where there
    //    is a mix of points >=0 and <=-100, but in
    //    general its good enough (the result of the above
    //    case is just slow operation, not incorrect operation).

    return ( i >= zeroint() ) || ( i <= -100 );
    //return ( i >= zeroint() ) || ( i <= -10 );
}

// Functions to allow different types to be treated "as if" they were matrices or vectors

double  &resizeZeroMat(double  &x, int i, int j);
gentype &resizeZeroMat(gentype &x, int i, int j);

double  &resizeZeroVec(double  &x, int i);
gentype &resizeZeroVec(gentype &x, int i);

double  &getMatElm(double  &x, int i, int j);
gentype &getMatElm(gentype &x, int i, int j);

double  &getVecElm(double  &x, int i);
gentype &getVecElm(gentype &x, int i);

double &resizeZeroMat(double &x, int i, int j)
{
    if ( ( i != 1 ) || ( j != 1 ) )
    {
        throw("Attempt to resize double as matrix");
    }

    return x = 0.0;
}

gentype &resizeZeroMat(gentype &x, int i, int j)
{
    x.force_matrix().resize(i,j);
    x.dir_matrix().zero();

    return x;
}

double &resizeZeroVec(double &x, int i)
{
    if ( i != 1 )
    {
        throw("Attempt to resize double as vector");
    }

    return x = 0.0;
}

gentype &resizeZeroVec(gentype &x, int i)
{
    x.force_vector().resize(i);
    x.dir_vector().zero();

    return x;
}

double &getMatElm(double &x, int i, int j)
{
    if ( ( i != 0 ) || ( j != 0 ) )
    {
        throw("Attempt to dereference double as matrix");
    }

    return x;
}

gentype &getMatElm(gentype &x, int i, int j)
{
    return x("&",i,j);
}

double &getVecElm(double &x, int i)
{
    if ( i != 0 )
    {
        throw("Attempt to dereference double as vector");
    }

    return x;
}

gentype &getVecElm(gentype &x, int i)
{
    return x("&",i);
}





gentype &ML_Base::Keqn(gentype &res, int resmode) const
{
    return Keqn(res,getKernel(),resmode);
}

gentype &ML_Base::Keqn(gentype &res, const MercerKernel &altK, int resmode) const
{
    return altK.Keqn(res,resmode);
}






































template <class T>
T &ML_Base::K0(T &res, 
               const T &bias, const MercerKernel &altK, const gentype **pxyprod, 
               int resmode) const
{
//phantomx
    altK.K0(res,bias,pxyprod,xspaceDim(),isXConsistent(),resmode,MLid(),assumeReal);

    return res;
}

template <class T>
T &ML_Base::K1(T &res, 
               int ib, 
               const T &bias, const MercerKernel &altK, const gentype **pxyprod, 
               const SparseVector<gentype> *xx, 
               const vecInfo *xxinfo, 
               int resmode, int &ibset) const
{
//phantomx
    if ( xx && !xxinfo )
    {
        vecInfo xinfox;

        (**thisthisthis).getKernel().getvecInfo(xinfox,*xx);

        return K1(res,ib,bias,altK,pxyprod,xx,&xinfox,resmode,ibset);
    }

    if ( xx && xx->isnofaroffindpresent() )
    {
        int iupm = (*xx).nearupsize();

        if ( iupm == 1 )
        {
            const Matrix<double> *xyp = NULL;

            if ( altK.suggestXYcache() && getxymat() && ( ib >= 0 ) )
            {
                const Matrix<double> &xy = *getxymat();
                Vector<int> xyderef(1);

                int z = 0;

                xyderef("&",z) = ib;

                retMatrix<double> tmpma;

                xyp = &(xy(xyderef,xyderef,tmpma));
            }

            altK.K1(res,*xx,*xxinfo,bias,pxyprod,ib,xspaceDim(),isXConsistent() && istrv(ib),resmode,MLid(),xyp,ibset,assumeReal);
        }

        else if ( iupm == 2 )
        {
            const SparseVector<gentype> &rxa = xx->nearrefup(0);
            const SparseVector<gentype> &rxb = xx->nearrefup(1);

            const vecInfo &rxainfo = (*xxinfo)(-1,0);
            const vecInfo &rxbinfo = (*xxinfo)(-1,1);

            int ixa = ib;
            int ixb = UPNTVI(ib,1);

            int ixaset = ibset;
            int ixbset = ibset;

            K2(res,ixa,ixb,bias,altK,NULL,&rxa,&rxb,&rxainfo,&rxbinfo,resmode,ixaset,ixbset);
        }

        else if ( iupm == 3 )
        {
            const SparseVector<gentype> &rxa = xx->nearrefup(0);
            const SparseVector<gentype> &rxb = xx->nearrefup(1);
            const SparseVector<gentype> &rxc = xx->nearrefup(2);

            const vecInfo &rxainfo = (*xxinfo)(-1,0);
            const vecInfo &rxbinfo = (*xxinfo)(-1,1);
            const vecInfo &rxcinfo = (*xxinfo)(-1,2);

            int ixa = ib;
            int ixb = UPNTVI(ib,1);
            int ixc = UPNTVI(ib,2);

            int ixaset = ibset;
            int ixbset = ibset;
            int ixcset = ibset;

            K3(res,ixa,ixb,ixc,bias,altK,NULL,&rxa,&rxb,&rxc,&rxainfo,&rxbinfo,&rxcinfo,resmode,ixaset,ixbset,ixcset);
        }

        else if ( iupm == 4 )
        {
            const SparseVector<gentype> &rxa = xx->nearrefup(0);
            const SparseVector<gentype> &rxb = xx->nearrefup(1);
            const SparseVector<gentype> &rxc = xx->nearrefup(2);
            const SparseVector<gentype> &rxd = xx->nearrefup(3);

            const vecInfo &rxainfo = (*xxinfo)(-1,0);
            const vecInfo &rxbinfo = (*xxinfo)(-1,1);
            const vecInfo &rxcinfo = (*xxinfo)(-1,2);
            const vecInfo &rxdinfo = (*xxinfo)(-1,3);

            int ixa = ib;
            int ixb = UPNTVI(ib,1);
            int ixc = UPNTVI(ib,2);
            int ixd = UPNTVI(ib,3);

            int ixaset = ibset;
            int ixbset = ibset;
            int ixcset = ibset;
            int ixdset = ibset;

            K4(res,ixa,ixb,ixc,ixd,bias,altK,NULL,&rxa,&rxb,&rxc,&rxd,&rxainfo,&rxbinfo,&rxcinfo,&rxdinfo,resmode,ixaset,ixbset,ixcset,ixdset);
        }

        else
        {
            int ii;

            Vector<int> iv(iupm);
            Vector<const SparseVector<gentype> *> xxv(iupm);
            Vector<const vecInfo *> xxvinfo(iupm);
            Vector<int> ivset(iupm);

            if ( iupm )
            {
                for ( ii = 0 ; ii < iupm ; ii++ )
                {
                    iv("&",ii) = ii ? UPNTVI(ib,ii) : ib;
                    xxv("&",ii) = &((*xx).nearrefup(ii));
                    xxvinfo("&",ii) = &((*xxinfo)(-1,ii));
                    ivset("&",ii) = ibset;
                }
            }

            Km(iupm,res,iv,bias,altK,NULL,&xxv,&xxvinfo,resmode,NULL,NULL,-1,-1,0,0,ivset);
        }

        return res;
    }

    else if ( !xx )
    {
        const SparseVector<gentype> &xib = xgetloc(ib);

        const vecInfo &xinfoi = xinfo(ib);

        return K1(res,ib,bias,altK,pxyprod,&xib,&xinfoi,resmode,ibset);
    }

    const SparseVector<gentype> *xinear = NULL;
    const SparseVector<gentype> *xifar = NULL;
    const SparseVector<gentype> *xifarfar = NULL;
    const vecInfo *xinearinfo = NULL;
    const vecInfo *xifarinfo = NULL;

    int i,ii,iokr,iok,idiagr,igradOrder,iplanr,iplan; // i is near, ii is far

    const gentype *itup = NULL;
    const gentype *iitup = NULL;

    int imethodkey = detangle_x(xinear,xifar,xifarfar,xinearinfo,xifarinfo,i,ii,itup,iitup,iokr,iok,ib,idiagr,xx,xxinfo,igradOrder,iplanr,iplan,ibset) & 7;

    if ( idiagr && ( i >= 0 ) )
    {
        res =  (T) (*xx).fff(4);

        return res;
    }

    else if ( idiagr )
    {
        res = 0.0;

        return res;
    }

    int z = 0;
    T tmp;

    if ( itup || iitup )
    {
        goto thunk;
    }

    switch ( imethodkey )
    {
        case 0:
        {
            K1(res,i, bias,altK,NULL,xinear,xinearinfo,resmode,ibset);
            
            break;
        }

        case 1:
        {
                   K1(res,i, bias,altK,NULL,xinear,xinearinfo,resmode,ibset);
            res -= K1(tmp,ii,bias,altK,NULL,xifar, xifarinfo, resmode,ibset);

            break;
        }

        default:
        {
            goto thunk;
            
            break;
        }
    }

    if ( iokr )
    {
        Vector<int> iiokr(1);
        Vector<int> iiok(1);
        Vector<int> iiset(1);
        Vector<const gentype *> xxalt(1);

        iiokr("&",0) = iokr;
        iiok("&",0)  = iok;
        iiset("&",0) = ibset;
        xxalt("&",0) = (*xx).isfarfarfarindpresent(3) ? &((*xx).fff(3)) : &nullgentype();

        gentype UUres;

        (*UUcallback)(UUres,1,*this,iiokr,iiok,iiset,xxalt,defbasisUU);

        res *= (T) UUres;
    }

    if ( iplanr )
    {
        Vector<int> iiplanr(1);
        Vector<int> iiplan(1);
        Vector<int> iiset(1);
        Vector<const gentype *> xxalt(1);

        iiplanr("&",0) = iplanr;
        iiplan("&",0)  = iplan;
        iiset("&",0)   = ibset;
        xxalt("&",0)   = (*xx).isfarfarfarindpresent(7) ? &((*xx).fff(7)) : &nullgentype();

        gentype VVres;
        gentype kval(res);

        (*VVcallback)(VVres,1,kval,*this,iiplanr,iiplan,iiset,xxalt,defbasisVV);

        res = (T) VVres;
    }

    return res;

thunk:
    Vector<int> lpi(1);
    Vector<const SparseVector<gentype> *> lpxx(1);
    Vector<const vecInfo *> lpxxinfo(1);
    Vector<int> lpiset(1);

    lpi("&",z) = ib;

    lpxx("&",z) = xx;

    lpxxinfo("&",z) = xxinfo;

    lpiset("&",z) = ibset;

    return Km(1,res,lpi,bias,altK,pxyprod,&lpxx,&lpxxinfo,resmode,NULL,NULL,-1,-1,0,0,lpiset);
}

Vector<gentype> &ML_Base::phi2(Vector<gentype> &res, int ia, const SparseVector<gentype> *xa, const vecInfo *xainf, int iaset) const
{
    if ( xa && !xainf )
    {
        vecInfo xinfoia;

        (**thisthisthis).getKernel().getvecInfo(xinfoia,*xa);

        return phi2(res,ia,xa,&xinfoia,iaset);
    }

    if ( !xa )
    {
        const SparseVector<gentype> &xia = xgetloc(ia);

        const vecInfo &xinfoia = xinfo(ia);

        return phi2(res,ia,&xia,&xinfoia,iaset);
    }

    return getKernel().phi2(res,*xa,ia,1,xspaceDim(),isXConsistent() && istrv(ia),iaset,assumeReal);
}

Vector<double> &ML_Base::phi2(Vector<double> &res, int ia, const SparseVector<gentype> *xa, const vecInfo *xainf, int iaset) const
{
    if ( xa && !xainf )
    {
        vecInfo xinfoia;

        (**thisthisthis).getKernel().getvecInfo(xinfoia,*xa);

        return phi2(res,ia,xa,&xinfoia,iaset);
    }

    if ( !xa )
    {
        const SparseVector<gentype> &xia = xgetloc(ia);

        const vecInfo &xinfoia = xinfo(ia);

        return phi2(res,ia,&xia,&xinfoia,iaset);
    }

    return getKernel().phi2(res,*xa,ia,1,xspaceDim(),isXConsistent() && istrv(ia),iaset,assumeReal);
}

template <class T>
T &ML_Base::K2(T &res, 
              int ib, int jb, 
              const T &bias, const MercerKernel &altK, const gentype **pxyprod, 
              const SparseVector<gentype> *xx, const SparseVector<gentype> *yy, 
              const vecInfo *xxinfo, const vecInfo *yyinfo, 
              int resmode, int &ibset, int &jbset) const
{
//phantomx
//errstream() << "phantomxyggghhh 0\n";
    if ( xx && !xxinfo )
    {
        vecInfo xinfox;

        (**thisthisthis).getKernel().getvecInfo(xinfox,*xx);

        return K2(res,ib,jb,bias,altK,pxyprod,xx,yy,&xinfox,yyinfo,resmode,ibset,jbset);
    }

    if ( yy && !yyinfo )
    {
        vecInfo xinfoy;

        (**thisthisthis).getKernel().getvecInfo(xinfoy,*yy);

        return K2(res,ib,jb,bias,altK,pxyprod,xx,yy,xxinfo,&xinfoy,resmode,ibset,jbset);
    }

    if ( ( ib >= 0 ) && ( jb >= 0 ) && ( K2mat.numRows() ) && ( K2mat.numCols() ) )
    {
        res = (T) K2mat(ib,jb);

        return res;
    }

    // Shortcut for speed

    int dummyind = -1;

    if ( xx && yy &&  xx->isnofaroffindpresent() && yy->isnofaroffindpresent() )
    {
//errstream() << "phantomxyggghhh 1\n";
        int iupm = (*xx).nearupsize();
        int jupm = (*yy).nearupsize();

        if ( ( iupm == 1 ) && ( jupm == 1 ) )
        {
            const Matrix<double> *xyp = NULL;

            if ( altK.suggestXYcache() && getxymat() && ( ib >= 0 ) && ( jb >= 0 ) )
            {
                const Matrix<double> &xy = *getxymat();
                Vector<int> xyderef(2);

                int z = 0;

                xyderef("&",z) = ib;
                xyderef("&",1) = jb;

                retMatrix<double> tmpma;

                xyp = &(xy(xyderef,xyderef,tmpma));
            }

//errstream() << "phantomxyggghhh 2\n";
            altK.K2(res,*xx,*yy,*xxinfo,*yyinfo,bias,pxyprod,ib,jb,xspaceDim(),isXConsistent() && istrv(ib) && istrv(jb),resmode,MLid(),xyp,ibset,jbset,assumeReal);
//errstream() << "phantomxyggghhh 3\n";
        }

        else if ( ( iupm == 1 ) && ( jupm == 2 ) )
        {
//NB: order of vectors is very important here (see splits, ns in mercer.h)
            const SparseVector<gentype> &rxa = xx->nearrefup(0);
            const SparseVector<gentype> &rxb = yy->nearrefup(0);
            const SparseVector<gentype> &rxc = yy->nearrefup(1);

            const vecInfo &rxainfo = (*xxinfo)(-1,0);
            const vecInfo &rxbinfo = (*yyinfo)(-1,0);
            const vecInfo &rxcinfo = (*yyinfo)(-1,1);

            int ixa = ib;
            int ixb = jb;
            int ixc = UPNTVI(jb,1);

            int ixaset = ibset;
            int ixbset = jbset;
            int ixcset = jbset;

            K3(res,ixa,ixb,ixc,bias,altK,NULL,&rxa,&rxb,&rxc,&rxainfo,&rxbinfo,&rxcinfo,resmode,ixaset,ixbset,ixcset);
        }

        else if ( ( iupm == 2 ) && ( jupm == 1 ) )
        {
            const SparseVector<gentype> &rxa = xx->nearrefup(0);
            const SparseVector<gentype> &rxb = xx->nearrefup(1);
            const SparseVector<gentype> &rxc = yy->nearrefup(0);

            const vecInfo &rxainfo = (*xxinfo)(-1,0);
            const vecInfo &rxbinfo = (*xxinfo)(-1,1);
            const vecInfo &rxcinfo = (*yyinfo)(-1,0);

            int ixa = ib;
            int ixb = UPNTVI(ib,1);
            int ixc = jb;

            int ixaset = ibset;
            int ixbset = ibset;
            int ixcset = jbset;

            K3(res,ixa,ixb,ixc,bias,altK,NULL,&rxa,&rxb,&rxc,&rxainfo,&rxbinfo,&rxcinfo,resmode,ixaset,ixbset,ixcset);
        }

        else if ( ( iupm == 1 ) && ( jupm == 3 ) )
        {
            const SparseVector<gentype> &rxa = xx->nearrefup(0);
            const SparseVector<gentype> &rxb = yy->nearrefup(0);
            const SparseVector<gentype> &rxc = yy->nearrefup(1);
            const SparseVector<gentype> &rxd = yy->nearrefup(2);

            const vecInfo &rxainfo = (*xxinfo)(-1,0);
            const vecInfo &rxbinfo = (*yyinfo)(-1,0);
            const vecInfo &rxcinfo = (*yyinfo)(-1,1);
            const vecInfo &rxdinfo = (*yyinfo)(-1,2);

            int ixa = ib;
            int ixb = jb;
            int ixc = UPNTVI(jb,1);
            int ixd = UPNTVI(jb,2);

            int ixaset = ibset;
            int ixbset = jbset;
            int ixcset = jbset;
            int ixdset = jbset;

            K4(res,ixa,ixb,ixc,ixd,bias,altK,NULL,&rxa,&rxb,&rxc,&rxd,&rxainfo,&rxbinfo,&rxcinfo,&rxdinfo,resmode,ixaset,ixbset,ixcset,ixdset);
        }

        else if ( ( iupm == 2 ) && ( jupm == 2 ) )
        {
            const SparseVector<gentype> &rxa = xx->nearrefup(0);
            const SparseVector<gentype> &rxb = xx->nearrefup(1);
            const SparseVector<gentype> &rxc = yy->nearrefup(0);
            const SparseVector<gentype> &rxd = yy->nearrefup(1);

            const vecInfo &rxainfo = (*xxinfo)(-1,0);
            const vecInfo &rxbinfo = (*xxinfo)(-1,1);
            const vecInfo &rxcinfo = (*yyinfo)(-1,0);
            const vecInfo &rxdinfo = (*yyinfo)(-1,1);

            int ixa = ib;
            int ixb = UPNTVI(ib,1);
            int ixc = jb;
            int ixd = UPNTVI(jb,1);

            int ixaset = ibset;
            int ixbset = ibset;
            int ixcset = jbset;
            int ixdset = jbset;

            K4(res,ixa,ixb,ixc,ixd,bias,altK,NULL,&rxa,&rxb,&rxc,&rxd,&rxainfo,&rxbinfo,&rxcinfo,&rxdinfo,resmode,ixaset,ixbset,ixcset,ixdset);
        }

        else if ( ( iupm == 3 ) && ( jupm == 1 ) )
        {
            const SparseVector<gentype> &rxa = xx->nearrefup(0);
            const SparseVector<gentype> &rxb = xx->nearrefup(1);
            const SparseVector<gentype> &rxc = xx->nearrefup(2);
            const SparseVector<gentype> &rxd = yy->nearrefup(0);

            const vecInfo &rxainfo = (*xxinfo)(-1,0);
            const vecInfo &rxbinfo = (*xxinfo)(-1,1);
            const vecInfo &rxcinfo = (*xxinfo)(-1,2);
            const vecInfo &rxdinfo = (*yyinfo)(-1,0);

            int ixa = ib;
            int ixb = UPNTVI(ib,1);
            int ixc = UPNTVI(ib,2);
            int ixd = jb;

            int ixaset = ibset;
            int ixbset = ibset;
            int ixcset = ibset;
            int ixdset = jbset;

            K4(res,ixa,ixb,ixc,ixd,bias,altK,NULL,&rxa,&rxb,&rxc,&rxd,&rxainfo,&rxbinfo,&rxcinfo,&rxdinfo,resmode,ixaset,ixbset,ixcset,ixdset);
        }

        else
        {
            int ii;

            Vector<int> iv(iupm+jupm);
            Vector<const SparseVector<gentype> *> xxv(iupm+jupm);
            Vector<const vecInfo *> xxvinfo(iupm+jupm);
            Vector<int> ivset(iupm+jupm);

            if ( iupm )
            {
                for ( ii = 0 ; ii < iupm ; ii++ )
                {
                    iv("&",ii) = ii ? UPNTVI(ib,ii) : ib;
                    xxv("&",ii) = &((*xx).nearrefup(ii));
                    xxvinfo("&",ii) = &((*xxinfo)(-1,ii));
                    ivset("&",ii) = ibset;
                }
            }

            if ( jupm )
            {
                for ( ii = 0 ; ii < jupm ; ii++ )
                {
                    iv("&",ii+iupm) = ii ? UPNTVI(jb,ii) : jb;
                    xxv("&",ii+iupm) = &((*yy).nearrefup(ii));
                    xxvinfo("&",ii+iupm) = &((*yyinfo)(-1,ii));
                    ivset("&",ii+iupm) = jbset;
                }
            }

            Km(iupm+jupm,res,iv,bias,altK,NULL,&xxv,&xxvinfo,resmode,NULL,NULL,-1,-1,0,0,ivset);
        }

        NiceAssert( !testisvnan(res) );
        NiceAssert( !testisinf(res) );

        return res;
    }

    else if ( xx && !yy )
    {
        const SparseVector<gentype> &xjb = xgetloc(jb);

        const vecInfo &xinfoj = xinfo(jb);

        return K2(res,ib,jb,bias,altK,pxyprod,xx,&xjb,xxinfo,&xinfoj,resmode,ibset,jbset);
    }

    else if ( !xx && yy )
    {
        const SparseVector<gentype> &xib = xgetloc(ib);

        const vecInfo &xinfoi = xinfo(ib);

        return K2(res,ib,jb,bias,altK,pxyprod,&xib,yy,&xinfoi,yyinfo,resmode,ibset,jbset);
    }

    else if ( !xx && !yy )
    {
        const SparseVector<gentype> &xib = xgetloc(ib);
        const SparseVector<gentype> &xjb = xgetloc(jb);

        const vecInfo &xinfoi = xinfo(ib);
        const vecInfo &xinfoj = xinfo(jb);

        return K2(res,ib,jb,bias,altK,pxyprod,&xib,&xjb,&xinfoi,&xinfoj,resmode,ibset,jbset);
    }

    const SparseVector<gentype> *xinear = NULL;
    const SparseVector<gentype> *xjnear = NULL;

    const SparseVector<gentype> *xifar = NULL;
    const SparseVector<gentype> *xjfar = NULL;

    const SparseVector<gentype> *xifarfar = NULL;
    const SparseVector<gentype> *xjfarfar = NULL;

    const vecInfo *xinearinfo = NULL;
    const vecInfo *xjnearinfo = NULL;

    const vecInfo *xifarinfo = NULL;
    const vecInfo *xjfarinfo = NULL;

    int i,ii,iokr,iok,idiagr,igradOrder,iplanr,iplan; // i is near, ii is far
    int j,jj,jokr,jok,jdiagr,jgradOrder,jplanr,jplan; // j is near, jj is far

    const gentype *itup = NULL;
    const gentype *iitup = NULL;

    const gentype *jtup = NULL;
    const gentype *jjtup = NULL;

    int imethodkey = detangle_x(xinear,xifar,xifarfar,xinearinfo,xifarinfo,i,ii,itup,iitup,iokr,iok,ib,idiagr,xx,xxinfo,igradOrder,iplanr,iplan,ibset) & 7;
    int jmethodkey = detangle_x(xjnear,xjfar,xjfarfar,xjnearinfo,xjfarinfo,j,jj,jtup,jjtup,jokr,jok,jb,jdiagr,yy,yyinfo,jgradOrder,jplanr,jplan,jbset) & 7;

    if ( idiagr && jdiagr && ( ib == jb ) && ( ib >= 0 ) )
    {
        res =  (T) (*xx).fff(4);
        setconj(res);
        res *= (T) (*yy).fff(4);

        return res;
    }

    else if ( idiagr || jdiagr )
    {
        res = 0.0;

        return res;
    }

    int z = 0;

    T tmp;

    if ( itup || jtup || iitup || jjtup )
    {
        goto thunk;
    }

    switch ( imethodkey+(10*jmethodkey)+(itup?100:0)+(jtup?1000:0) )
    {
        case  0:
        {
                   K2(res,i, j, bias,altK,NULL,xinear,xjnear,xinearinfo,xjnearinfo,resmode,ibset,jbset);
            
            break;
        }

        case  1:
        {
                   K2(res,i, j, bias,altK,NULL,xinear,xjnear,xinearinfo,xjnearinfo,resmode,ibset,jbset);
            res -= K2(tmp,ii,j, bias,altK,NULL,xifar, xjnear,xifarinfo, xjnearinfo,resmode,ibset,jbset);

            break;
        }

        case 10:
        {
                   K2(res,i, jj,bias,altK,NULL,xinear,xjnear,xinearinfo,xjnearinfo,resmode,ibset,jbset);
            res -= K2(tmp,i, jj,bias,altK,NULL,xinear,xjfar ,xinearinfo,xjfarinfo ,resmode,ibset,jbset);

            break;
        }

        case 11:
        {
                   K2(res,i, j, bias,altK,NULL,xinear,xjnear,xinearinfo,xjnearinfo,resmode,ibset,jbset);
            res -= K2(tmp,ii,j, bias,altK,NULL,xifar, xjnear,xifarinfo, xjnearinfo,resmode,ibset,jbset);
            res -= K2(tmp,i, jj,bias,altK,NULL,xinear,xjfar, xinearinfo,xjfarinfo, resmode,ibset,jbset);
            res += K2(tmp,ii,jj,bias,altK,NULL,xifar, xjfar, xifarinfo, xjfarinfo, resmode,ibset,jbset);

            break;
        }

        case  2:
        {
            NiceAssert( igradOrder > 0 );

            if ( igradOrder == 1 )
            {
                // <ex,dK/dx> = <ex,x.ax + y.ay>
                //            = <ex,x>.ax + <ex,y>.ay

                T xscaleres; 
                T yscaleres;

                dK2delx(xscaleres,yscaleres,dummyind,i, j, bias,altK,NULL,xinear,xjnear,xinearinfo,xjnearinfo,ibset,jbset);

                NiceAssert( dummyind < 0 );

                gentype exx;
                gentype exy;

                twoProduct(exx,*xifarfar,*xinear);
                twoProduct(exy,*xifarfar,*xjnear);

                res = (((T) exx)*xscaleres)+(((T) exy)*yscaleres);
            }

            else
            {
                // <(ex d/dx)^n,K>

                int xdm = xspaceDim();

                res = 0.0;

                Vector<T> sc;
                Vector<Vector<int> > nn;

                Vector<int> q(igradOrder);

                q = zeroint();

                int iqa,jqa;
                gentype tmp;

                SparseVector<gentype> kres;
                Vector<const SparseVector<gentype> *> bord;

                dnK2del(sc,nn,dummyind,q,i, j, bias,altK,NULL,xinear,xjnear,xinearinfo,xjnearinfo,ibset,jbset);

                NiceAssert( dummyind < 0 );

                for ( iqa = 0 ; iqa < sc.size() ; iqa++ )
                {
                    bord.resize(nn(iqa).size());

                    for ( jqa = 0 ; jqa < nn(iqa).size() ; jqa++ )
                    {
                        bord("&",jqa) = ( nn(iqa)(jqa) == 0 ) ? xinear : ( ( nn(iqa)(jqa) == 1 ) ? xjnear : NULL );
                    }

                    kronprod(kres,bord,nn(iqa),xdm);

                    twoProduct(tmp,*xifarfar,kres);

                    res += ((T) tmp)*((T) sc(iqa));
                }
            }

            break;
        }

        case 4:
        {
            NiceAssert( igradOrder > 0 );

            if ( igradOrder == 1 )
            {
                // <ex,dK/dx> = <ex,x.ax + y.ay>
                //            = <ex,x>.ax + <ex,y>.ay

                int xdm = xspaceDim();

                resizeZeroVec(res,xdm);

                T xscaleres; 
                T yscaleres;

                int jqa;

                dK2delx(xscaleres,yscaleres,dummyind,i, j, bias,altK,NULL,xinear,xjnear,xinearinfo,xjnearinfo,ibset,jbset);

                NiceAssert( dummyind < 0 );

                for ( jqa = 0 ; jqa < xdm ; jqa++ )
                {
                    getVecElm(res,jqa) += ((T) xscaleres)*((T) ((*xinear)(jqa)));
                    getVecElm(res,jqa) += ((T) yscaleres)*((T) ((*xjnear)(jqa)));
                }
            }

            else
            {
                // <(ex d/dx)^n,K>

                int xdm = xspaceDim();
                int resdim = (int) pow(xdm,igradOrder);

                resizeZeroVec(res,resdim);

                Vector<T> sc;
                Vector<Vector<int> > nn;

                Vector<int> q(igradOrder);

                q = zeroint();

                int iqa,jqa;

                SparseVector<gentype> kres;
                Vector<const SparseVector<gentype> *> bord;

                dnK2del(sc,nn,dummyind,q,i, j, bias,altK,NULL,xinear,xjnear,xinearinfo,xjnearinfo,ibset,jbset);

                NiceAssert( dummyind < 0 );

                for ( iqa = 0 ; iqa < sc.size() ; iqa++ )
                {
                    bord.resize(nn(iqa).size());

                    for ( jqa = 0 ; jqa < nn(iqa).size() ; jqa++ )
                    {
                        bord("&",jqa) = ( nn(iqa)(jqa) == 0 ) ? xinear : ( ( nn(iqa)(jqa) == 1 ) ? xjnear : NULL );
                    }

                    kronprod(kres,bord,nn(iqa),xdm);

                    for ( jqa = 0 ; jqa < resdim ; jqa++ )
                    {
                        getVecElm(res,jqa) += ((T) sc(iqa))*((T) kres(jqa));
                    }
                }
            }

            break;
        }

        case 20:
        {
            NiceAssert( jgradOrder > 0 );

            if ( jgradOrder == 1 )
            {
                // <dK/dy,ey> = <xa.x + ya.y,ey>
                //            = xa.<x,ey> + ya.<y,ey>

                T xscaleres;
                T yscaleres;

                // MOD: do by reversing x and y, assuming symmetry.
                //dK2dely(xscaleres,yscaleres,dummyind,i, j, bias,altK,NULL,xinear,xjnear,xinearinfo,xjnearinfo,ibset,jbset);
                dK2delx(yscaleres,xscaleres,dummyind,j, i, bias,altK,NULL,xjnear,xinear,xjnearinfo,xinearinfo,jbset,ibset);

                NiceAssert( dummyind < 0 );

                gentype xey;
                gentype yey;

                twoProduct(xey,*xinear,*xjfarfar);
                twoProduct(yey,*xjnear,*xjfarfar);

                res = (xscaleres*((T) xey))+(yscaleres*((T) yey));
            }

            else
            {
                // <(ex d/dy)^n,K>

                int xdm = xspaceDim();

                res = 0.0;

                Vector<T> sc;
                Vector<Vector<int> > nn;

                Vector<int> q(jgradOrder);

                q = 1;

                dnK2del(sc,nn,dummyind,q,i, j, bias,altK,NULL,xinear,xjnear,xinearinfo,xjnearinfo,ibset,jbset);

                NiceAssert( dummyind < 0 );

                int iqa,jqa;
                gentype tmp;

                SparseVector<gentype> kres;
                Vector<const SparseVector<gentype> *> bord;

                for ( iqa = 0 ; iqa < sc.size() ; iqa++ )
                {
                    bord.resize(nn(iqa).size());

                    for ( jqa = 0 ; jqa < nn(iqa).size() ; jqa++ )
                    {
                        bord("&",jqa) = ( nn(iqa)(jqa) == 0 ) ? xinear : ( ( nn(iqa)(jqa) == 1 ) ? xjnear : NULL );
                    }

                    kronprod(kres,bord,nn(iqa),xdm);

                    twoProduct(tmp,*xjfarfar,kres);

                    res += ((T) tmp)*((T) sc(iqa));
                }
            }
            
            break;
        }

        case 40:
        {
            NiceAssert( jgradOrder > 0 );

            if ( jgradOrder == 1 )
            {
                // <dK/dy,ey> = <xa.x + ya.y,ey>
                //            = xa.<x,ey> + ya.<y,ey>

                int xdm = xspaceDim();

                resizeZeroVec(res,xdm);

                T xscaleres;
                T yscaleres;

                int jqa;

                // MOD: do by reversing x and y, assuming symmetry.
                //dK2dely(xscaleres,yscaleres,dummyind,i, j, bias,altK,NULL,xinear,xjnear,xinearinfo,xjnearinfo,ibset,jbset);
                dK2delx(yscaleres,xscaleres,dummyind,j, i, bias,altK,NULL,xjnear,xinear,xjnearinfo,xinearinfo,jbset,ibset);

                NiceAssert( dummyind < 0 );

                for ( jqa = 0 ; jqa < xdm ; jqa++ )
                {
                    getVecElm(res,jqa) += ((T) xscaleres)*((T) ((*xinear)(jqa)));
                    getVecElm(res,jqa) += ((T) yscaleres)*((T) ((*xjnear)(jqa)));
                }
            }

            else
            {
                // <(ex d/dy)^n,K>

                int xdm = xspaceDim();
                int resdim = (int) pow(xdm,jgradOrder);

                resizeZeroVec(res,resdim);

                Vector<T> sc;
                Vector<Vector<int> > nn;

                Vector<int> q(jgradOrder);

                q = 1;

                dnK2del(sc,nn,dummyind,q,i, j, bias,altK,NULL,xinear,xjnear,xinearinfo,xjnearinfo,ibset,jbset);

                NiceAssert( dummyind < 0 );

                int iqa,jqa;

                SparseVector<gentype> kres;
                Vector<const SparseVector<gentype> *> bord;

                for ( iqa = 0 ; iqa < sc.size() ; iqa++ )
                {
                    bord.resize(nn(iqa).size());

                    for ( jqa = 0 ; jqa < nn(iqa).size() ; jqa++ )
                    {
                        bord("&",jqa) = ( nn(iqa)(jqa) == 0 ) ? xinear : ( ( nn(iqa)(jqa) == 1 ) ? xjnear : NULL );
                    }

                    kronprod(kres,bord,nn(iqa),xdm);

                    for ( jqa = 0 ; jqa < resdim ; jqa++ )
                    {
                        getVecElm(res,jqa) += ((T) sc(iqa))*((T) kres(jqa));
                    }
                }
            }
            
            break;
        }

        case 22:
        {
            NiceAssert( igradOrder > 0 );
            NiceAssert( jgradOrder > 0 );

            if ( ( igradOrder == 1 ) && ( jgradOrder == 1 ) )
            {
                // <ex,d2K/dxdy,ey>

                T xxscaleres;
                T xyscaleres;
                T yxscaleres;
                T yyscaleres;

                T constres;

                d2K2delxdely(xxscaleres,yyscaleres,xyscaleres,yxscaleres,constres,dummyind,i, j, bias,altK,NULL,xinear,xjnear,xinearinfo,xjnearinfo,ibset,jbset);

                NiceAssert( dummyind < 0 );

                gentype exx;
                gentype exy;
                gentype xey;
                gentype yey;

                gentype exey;

                twoProduct(exx,*xifarfar,*xinear);
                twoProduct(exy,*xifarfar,*xjnear);
                twoProduct(xey,*xinear,*xjfarfar);
                twoProduct(yey,*xjnear,*xjfarfar);

                twoProduct(exey,*xifarfar,*xjfarfar);

                res = (((T) exx)*xxscaleres*((T) xey)) 
                    + (((T) exx)*xyscaleres*((T) yey)) 
                    + (((T) exy)*yxscaleres*((T) xey)) 
                    + (((T) exy)*yyscaleres*((T) yey)) 
                    + (constres*((T) exey));
            }

            else
            {
                // <(ex d/dx)^n (ey d/dy)^n,K>

                int xdm = xspaceDim();

                int residim = (int) pow(xdm,igradOrder);
                int resjdim = (int) pow(xdm,jgradOrder);

                res = 0.0;

                Vector<T> sc;
                Vector<Vector<int> > nn;

                Vector<int> q(igradOrder+jgradOrder);

                retVector<int> tmpva;

                q("&",0         ,1,igradOrder-1           ,tmpva) = 0;
                q("&",igradOrder,1,igradOrder+jgradOrder-1,tmpva) = 1;

                // Here we use the result: vec(ABC) = vec(C' \otimes A).vec(B), so if
                // we let a = xifarfar, c = xjfarfar, then:
                // a'.B.c = vec(c' \otimes a').vec(B)
                //        = vec(c \otimes a)'.vec(B)
                // where vec(B) is the vectorised *transpose* gradient we usually calculate, so, getting rid of the transpose,
                // we need to take the inner product with vec(ex \otimes ey)
                // ORIGINAL INCORRECT VERSION: where vec(B) is precisely the vectorised gradient we usually calculate
                //
                // CLARIFICATION: see pdf in stable bayesian optimisation paper

                int iqa,jqa;
                gentype tmp;

                SparseVector<gentype> kres;
                SparseVector<gentype> farfarprod;
                Vector<const SparseVector<gentype> *> bord;

                // first calculate c \otimes a

                int dimmy = 0;

                //kronprod(farfarprod,*xjfarfar,*xifarfar,pow(xdm,jgradOrder),pow(xdm,igradOrder)); //- NB order wrong here, have tested and confirmed, following line is correct.
                kronprod(farfarprod,dimmy,*xifarfar,*xjfarfar,residim,resjdim);

               // Gradient

                dnK2del(sc,nn,dummyind,q,i, j, bias,altK,NULL,xinear,xjnear,xinearinfo,xjnearinfo,ibset,jbset);

                NiceAssert( dummyind < 0 );

                // Then proceed in fully vectorised form

                for ( iqa = 0 ; iqa < sc.size() ; iqa++ )
                {
                    bord.resize(nn(iqa).size());

                    for ( jqa = 0 ; jqa < nn(iqa).size() ; jqa++ )
                    {
                        bord("&",jqa) = ( nn(iqa)(jqa) == 0 ) ? xinear : ( ( nn(iqa)(jqa) == 1 ) ? xjnear : NULL );
                    }

                    kronprod(kres,bord,nn(iqa),xdm);

                    twoProduct(tmp,farfarprod,kres);

                    res += ((T) tmp)*((T) sc(iqa));
                }
            }
            
            break;
        }

        case 24:
        {
            NiceAssert( igradOrder > 0 );
            NiceAssert( jgradOrder > 0 );

            if ( ( igradOrder == 1 ) && ( jgradOrder == 1 ) )
            {
                // <ex,d2K/dxdy,ey>

                int xdm = xspaceDim();

                resizeZeroVec(res,xdm);

                T xxscaleres;
                T xyscaleres;
                T yxscaleres;
                T yyscaleres;

                T constres;

                int jqa;

                d2K2delxdely(xxscaleres,yyscaleres,xyscaleres,yxscaleres,constres,dummyind,i, j, bias,altK,NULL,xinear,xjnear,xinearinfo,xjnearinfo,ibset,jbset);

                NiceAssert( dummyind < 0 );

                gentype xey;
                gentype yey;

                twoProduct(xey,*xinear,*xjfarfar);
                twoProduct(yey,*xjnear,*xjfarfar);

                for ( jqa = 0 ; jqa < xdm ; jqa++ )
                {
                    getVecElm(res,jqa) += ((T) ((*xinear)(jqa)))*((T) xxscaleres)*((T) xey);
                    getVecElm(res,jqa) += ((T) ((*xinear)(jqa)))*((T) xyscaleres)*((T) yey);
                    getVecElm(res,jqa) += ((T) ((*xjnear)(jqa)))*((T) yxscaleres)*((T) xey);
                    getVecElm(res,jqa) += ((T) ((*xjnear)(jqa)))*((T) yyscaleres)*((T) yey);

                    getVecElm(res,jqa) += ((T) constres)*((T) ((*xjfarfar)(jqa)));
                }
            }

            else
            {
                // <(ex d/dx)^n (ey d/dy)^n,K>

                int xdm = xspaceDim();

                int residim = (int) pow(xdm,igradOrder);
                int resjdim = (int) pow(xdm,jgradOrder);

                resizeZeroVec(res,residim);

                Vector<T> sc;
                Vector<Vector<int> > nn;

                Vector<int> q(igradOrder+jgradOrder);

                retVector<int> tmpva;

                q("&",0         ,1,igradOrder-1           ,tmpva) = 0;
                q("&",igradOrder,1,igradOrder+jgradOrder-1,tmpva) = 1;

                // Here we use the result: vec(ABC) = vec(C' \otimes A).vec(B), so if
                // we let a = xifarfar, c = xjfarfar, then:
                // a'.B.c = vec(c' \otimes a').vec(B)
                //        = vec(c \otimes a)'.vec(B)
                // where vec(B) is the vectorised *transpose* gradient we usually calculate, so, getting rid of the transpose,
                // we need to take the inner product with vec(ex \otimes ey)
                // ORIGINAL INCORRECT VERSION: where vec(B) is precisely the vectorised gradient we usually calculate
                //
                // CLARIFICATION: see pdf in stable bayesian optimisation paper

                int iqa,jqa,kqa;
                gentype tmp;

                SparseVector<gentype> kres;
                Vector<const SparseVector<gentype> *> bord;

                // Gradient

                dnK2del(sc,nn,dummyind,q,i, j, bias,altK,NULL,xinear,xjnear,xinearinfo,xjnearinfo,ibset,jbset);

                NiceAssert( dummyind < 0 );

                // Then proceed in fully vectorised form

                for ( iqa = 0 ; iqa < sc.size() ; iqa++ )
                {
                    bord.resize(nn(iqa).size());

                    for ( jqa = 0 ; jqa < nn(iqa).size() ; jqa++ )
                    {
                        bord("&",jqa) = ( nn(iqa)(jqa) == 0 ) ? xinear : ( ( nn(iqa)(jqa) == 1 ) ? xjnear : NULL );
                    }

                    kronprod(kres,bord,nn(iqa),xdm);

                    for ( jqa = 0 ; jqa < residim ; jqa++ )
                    {
                        for ( kqa = 0 ; kqa < resjdim ; kqa++ )
                        {
                            getVecElm(res,jqa) += ((T) sc(iqa))*((T) (kres((jqa*resjdim)+kqa)))*((T) (*xjfarfar)(kqa));
                        }
                    }
                }
            }
            
            break;
        }

        case 42:
        {
            NiceAssert( igradOrder > 0 );
            NiceAssert( jgradOrder > 0 );

            if ( ( igradOrder == 1 ) && ( jgradOrder == 1 ) )
            {
                // <ex,d2K/dxdy,ey>

                int xdm = xspaceDim();

                resizeZeroVec(res,xdm);

                T xxscaleres;
                T xyscaleres;
                T yxscaleres;
                T yyscaleres;

                T constres;

                int jqa;

                d2K2delxdely(xxscaleres,yyscaleres,xyscaleres,yxscaleres,constres,dummyind,i, j, bias,altK,NULL,xinear,xjnear,xinearinfo,xjnearinfo,ibset,jbset);

                NiceAssert( dummyind < 0 );

                gentype exx;
                gentype exy;

                twoProduct(exx,*xifarfar,*xinear);
                twoProduct(exy,*xifarfar,*xjnear);

                for ( jqa = 0 ; jqa < xdm ; jqa++ )
                {
                    getVecElm(res,jqa) += ((T) exx)*((T) xxscaleres)*((T) ((*xinear)(jqa)));
                    getVecElm(res,jqa) += ((T) exx)*((T) xyscaleres)*((T) ((*xjnear)(jqa)));
                    getVecElm(res,jqa) += ((T) exy)*((T) yxscaleres)*((T) ((*xinear)(jqa)));
                    getVecElm(res,jqa) += ((T) exy)*((T) yyscaleres)*((T) ((*xjnear)(jqa)));

                    getVecElm(res,jqa) += ((T) constres)*((T) ((*xifarfar)(jqa)));
                }
            }

            else
            {
                // <(ex d/dx)^n (ey d/dy)^n,K>

                int xdm = xspaceDim();

                int residim = (int) pow(xdm,igradOrder);
                int resjdim = (int) pow(xdm,jgradOrder);

                resizeZeroVec(res,resjdim);

                Vector<T> sc;
                Vector<Vector<int> > nn;

                Vector<int> q(igradOrder+jgradOrder);

                retVector<int> tmpva;

                q("&",0         ,1,igradOrder-1           ,tmpva) = 0;
                q("&",igradOrder,1,igradOrder+jgradOrder-1,tmpva) = 1;

                // Here we use the result: vec(ABC) = vec(C' \otimes A).vec(B), so if
                // we let a = xifarfar, c = xjfarfar, then:
                // a'.B.c = vec(c' \otimes a').vec(B)
                //        = vec(c \otimes a)'.vec(B)
                // where vec(B) is the vectorised *transpose* gradient we usually calculate, so, getting rid of the transpose,
                // we need to take the inner product with vec(ex \otimes ey)
                // ORIGINAL INCORRECT VERSION: where vec(B) is precisely the vectorised gradient we usually calculate
                //
                // CLARIFICATION: see pdf in stable bayesian optimisation paper

                int iqa,jqa,kqa;
                gentype tmp;

                SparseVector<gentype> kres;
                Vector<const SparseVector<gentype> *> bord;

                // Gradient

                dnK2del(sc,nn,dummyind,q,i, j, bias,altK,NULL,xinear,xjnear,xinearinfo,xjnearinfo,ibset,jbset);

                NiceAssert( dummyind < 0 );

                // Then proceed in fully vectorised form

                for ( iqa = 0 ; iqa < sc.size() ; iqa++ )
                {
                    bord.resize(nn(iqa).size());

                    for ( jqa = 0 ; jqa < nn(iqa).size() ; jqa++ )
                    {
                        bord("&",jqa) = ( nn(iqa)(jqa) == 0 ) ? xinear : ( ( nn(iqa)(jqa) == 1 ) ? xjnear : NULL );
                    }

                    kronprod(kres,bord,nn(iqa),xdm);

                    for ( jqa = 0 ; jqa < residim ; jqa++ )
                    {
                        for ( kqa = 0 ; kqa < resjdim ; kqa++ )
                        {
                            getVecElm(res,kqa) += ((T) sc(iqa))*((T) (kres((jqa*resjdim)+kqa)))*((T) (*xifarfar)(jqa));
                        }
                    }
                }
            }
            
            break;
        }

        case 44:
        {
            NiceAssert( igradOrder > 0 );
            NiceAssert( jgradOrder > 0 );

            if ( ( igradOrder == 1 ) && ( jgradOrder == 1 ) )
            {
                // <ex,d2K/dxdy,ey>

                int xdm = xspaceDim();

                resizeZeroMat(res,xdm,xdm);

                T xxscaleres;
                T xyscaleres;
                T yxscaleres;
                T yyscaleres;

                T constres;

                int jqa,kqa;

                d2K2delxdely(xxscaleres,yyscaleres,xyscaleres,yxscaleres,constres,dummyind,i,j,bias,altK,NULL,xinear,xjnear,xinearinfo,xjnearinfo,ibset,jbset);

                NiceAssert( dummyind < 0 );

                for ( jqa = 0 ; jqa < xdm ; jqa++ )
                {
                    for ( kqa = 0 ; kqa < xdm ; kqa++ )
                    {
                        getMatElm(res,jqa,kqa) += ((T) xxscaleres)*((T) ((*xinear)(jqa)))*((T) ((*xinear)(kqa)));
                        getMatElm(res,jqa,kqa) += ((T) yyscaleres)*((T) ((*xjnear)(jqa)))*((T) ((*xjnear)(kqa)));
                        getMatElm(res,jqa,kqa) += ((T) xyscaleres)*((T) ((*xinear)(jqa)))*((T) ((*xjnear)(kqa)));
                        getMatElm(res,jqa,kqa) += ((T) yxscaleres)*((T) ((*xjnear)(jqa)))*((T) ((*xinear)(kqa)));
                    }

                    getMatElm(res,jqa,jqa) += ((T) constres);
                }
            }

            else
            {
                // <(ex d/dx)^n (ey d/dy)^n,K>

                int xdm = xspaceDim();

                int residim = (int) pow(xdm,igradOrder);
                int resjdim = (int) pow(xdm,jgradOrder);

                resizeZeroMat(res,residim,resjdim);

                Vector<T> sc;
                Vector<Vector<int> > nn;

                Vector<int> q(igradOrder+jgradOrder);

                retVector<int> tmpva;

                q("&",0         ,1,igradOrder-1           ,tmpva) = 0;
                q("&",igradOrder,1,igradOrder+jgradOrder-1,tmpva) = 1;

                // Here we use the result: vec(ABC) = vec(C' \otimes A).vec(B), so if
                // we let a = xifarfar, c = xjfarfar, then:
                // a'.B.c = vec(c' \otimes a').vec(B)
                //        = vec(c \otimes a)'.vec(B)
                // where vec(B) is the vectorised *transpose* gradient we usually calculate, so, getting rid of the transpose,
                // we need to take the inner product with vec(ex \otimes ey)
                // ORIGINAL INCORRECT VERSION: where vec(B) is precisely the vectorised gradient we usually calculate
                //
                // CLARIFICATION: see pdf in stable bayesian optimisation paper

                int iqa,jqa,kqa;

                SparseVector<gentype> kres;
                SparseVector<gentype> farfarprod;
                Vector<const SparseVector<gentype> *> bord;

                // Gradient

                dnK2del(sc,nn,dummyind,q,i,j,bias,altK,NULL,xinear,xjnear,xinearinfo,xjnearinfo,ibset,jbset);

                NiceAssert( dummyind < 0 );

                // Then proceed in fully vectorised form

                for ( iqa = 0 ; iqa < sc.size() ; iqa++ )
                {
                    bord.resize(nn(iqa).size());

                    for ( jqa = 0 ; jqa < nn(iqa).size() ; jqa++ )
                    {
                        bord("&",jqa) = ( nn(iqa)(jqa) == 0 ) ? xinear : ( ( nn(iqa)(jqa) == 1 ) ? xjnear : NULL );
                    }

                    kronprod(kres,bord,nn(iqa),xdm);

                    for ( jqa = 0 ; jqa < residim ; jqa++ )
                    {
                        for ( kqa = 0 ; kqa < resjdim ; kqa++ )
                        {
                            getMatElm(res,jqa,kqa) += ((T) sc(iqa))*((T) (kres((jqa*resjdim)+kqa)));
                        }
                    }
                }
            }
            
            break;
        }

        case 12:
        {
            NiceAssert( igradOrder > 0 );

            if ( igradOrder == 1 )
            {
                // <ex,dK/dx> = <ex,x.ax + y.ay>
                //            = <ex,x>.ax + <ex,y>.ay

                T xscaleres; 
                T yscaleres;

                gentype exx;
                gentype exy;

                dK2delx(xscaleres,yscaleres,dummyind,i, j, bias,altK,NULL,xinear,xjnear,xinearinfo,xjnearinfo,ibset,jbset);

                NiceAssert( dummyind < 0 );

                twoProduct(exx,*xifarfar,*xinear);
                twoProduct(exy,*xifarfar,*xjnear);

                res = (((T) exx)*xscaleres)+(((T) exy)*yscaleres);

                dK2delx(xscaleres,yscaleres,dummyind,i, jj,bias,altK,NULL,xinear,xjfar ,xinearinfo,xjfarinfo ,ibset,jbset);

                NiceAssert( dummyind < 0 );

                //twoProduct(exx,*xifarfar,*xinear);
                twoProduct(exy,*xifarfar,*xjfar );

                res -= (((T) exx)*xscaleres)+(((T) exy)*yscaleres);
            }

            else
            {
                // <(ex d/dx)^n,K>

                res = 0.0;

                int xdm = xspaceDim();

                Vector<T> sc;
                Vector<Vector<int> > nn;

                Vector<int> q(igradOrder);

                q = zeroint();

                int iqa,jqa;
                gentype tmp;

                SparseVector<gentype> kres;
                Vector<const SparseVector<gentype> *> bord;

                dnK2del(sc,nn,dummyind,q,i, j, bias,altK,NULL,xinear,xjnear,xinearinfo,xjnearinfo,ibset,jbset);

                NiceAssert( dummyind < 0 );

                for ( iqa = 0 ; iqa < sc.size() ; iqa++ )
                {
                    bord.resize(nn(iqa).size());

                    for ( jqa = 0 ; jqa < nn(iqa).size() ; jqa++ )
                    {
                        bord("&",jqa) = ( nn(iqa)(jqa) == 0 ) ? xinear : ( ( nn(iqa)(jqa) == 1 ) ? xjnear : NULL );
                    }

                    kronprod(kres,bord,nn(iqa),xdm);

                    twoProduct(tmp,*xifarfar,kres);

                    res += ((T) tmp)*((T) sc(iqa));
                }

                dnK2del(sc,nn,dummyind,q,i, jj,bias,altK,NULL,xinear,xjfar ,xinearinfo,xjfarinfo ,ibset,jbset);

                NiceAssert( dummyind < 0 );

                for ( iqa = 0 ; iqa < sc.size() ; iqa++ )
                {
                    bord.resize(nn(iqa).size());

                    for ( jqa = 0 ; jqa < nn(iqa).size() ; jqa++ )
                    {
                        bord("&",jqa) = ( nn(iqa)(jqa) == 0 ) ? xinear : ( ( nn(iqa)(jqa) == 1 ) ? xjfar  : NULL );
                    }

                    kronprod(kres,bord,nn(iqa),xdm);

                    twoProduct(tmp,*xifarfar,kres);

                    res -= ((T) tmp)*((T) sc(iqa));
                }
            }
            
            break;
        }

        case 14:
        {
            NiceAssert( igradOrder > 0 );

            if ( igradOrder == 1 )
            {
                // <ex,dK/dx> = <ex,x.ax + y.ay>
                //            = <ex,x>.ax + <ex,y>.ay

                int xdm = xspaceDim();

                resizeZeroVec(res,xdm);

                T xscaleres; 
                T yscaleres;

                int jqa;

                dK2delx(xscaleres,yscaleres,dummyind,i, j, bias,altK,NULL,xinear,xjnear,xinearinfo,xjnearinfo,ibset,jbset);

                NiceAssert( dummyind < 0 );

                for ( jqa = 0 ; jqa < xdm ; jqa++ )
                {
                    getVecElm(res,jqa) += ((T) xscaleres)*((T) ((*xinear)(jqa)));
                    getVecElm(res,jqa) += ((T) yscaleres)*((T) ((*xjnear)(jqa)));
                }

                dK2delx(xscaleres,yscaleres,dummyind,i, jj,bias,altK,NULL,xinear,xjfar ,xinearinfo,xjfarinfo ,ibset,jbset);

                NiceAssert( dummyind < 0 );

                for ( jqa = 0 ; jqa < xdm ; jqa++ )
                {
                    getVecElm(res,jqa) -= ((T) xscaleres)*((T) ((*xinear)(jqa)));
                    getVecElm(res,jqa) -= ((T) yscaleres)*((T) ((*xjfar )(jqa)));
                }
            }

            else
            {
                // <(ex d/dx)^n,K>

                int xdm = xspaceDim();

                int resdim = (int) pow(xdm,igradOrder);

                resizeZeroVec(res,resdim);

                Vector<T> sc;
                Vector<Vector<int> > nn;

                Vector<int> q(igradOrder);

                q = zeroint();

                int iqa,jqa;

                SparseVector<gentype> kres;
                Vector<const SparseVector<gentype> *> bord;

                dnK2del(sc,nn,dummyind,q,i, j, bias,altK,NULL,xinear,xjnear,xinearinfo,xjnearinfo,ibset,jbset);

                NiceAssert( dummyind < 0 );

                for ( iqa = 0 ; iqa < sc.size() ; iqa++ )
                {
                    bord.resize(nn(iqa).size());

                    for ( jqa = 0 ; jqa < nn(iqa).size() ; jqa++ )
                    {
                        bord("&",jqa) = ( nn(iqa)(jqa) == 0 ) ? xinear : ( ( nn(iqa)(jqa) == 1 ) ? xjnear : NULL );
                    }

                    kronprod(kres,bord,nn(iqa),xdm);

                    for ( jqa = 0 ; jqa < resdim ; jqa++ )
                    {
                        getVecElm(res,jqa) += ((T) sc(iqa))*((T) kres(jqa));
                    }
                }

                dnK2del(sc,nn,dummyind,q,i ,jj,bias,altK,NULL,xinear,xjfar ,xinearinfo,xjfarinfo ,ibset,jbset);

                NiceAssert( dummyind < 0 );

                for ( iqa = 0 ; iqa < sc.size() ; iqa++ )
                {
                    bord.resize(nn(iqa).size());

                    for ( jqa = 0 ; jqa < nn(iqa).size() ; jqa++ )
                    {
                        bord("&",jqa) = ( nn(iqa)(jqa) == 0 ) ? xinear : ( ( nn(iqa)(jqa) == 1 ) ? xjfar  : NULL );
                    }

                    kronprod(kres,bord,nn(iqa),xdm);

                    for ( jqa = 0 ; jqa < resdim ; jqa++ )
                    {
                        getVecElm(res,jqa) += ((T) sc(iqa))*((T) kres(jqa));
                    }
                }
            }

            break;
        }

        case 21:
        {
            NiceAssert( jgradOrder > 0 );

            if ( jgradOrder == 1 )
            {
                // <dK/dy,ey> = <xa.x + ya.y,ey>
                //            = xa.<x,ey> + ya.<y,ey>

                T xscaleres;
                T yscaleres;

                gentype xey;
                gentype yey;

                // MOD: do by reversing x and y, assuming symmetry.
                //dK2dely(xscaleres,yscaleres,dummyind,i, j, bias,altK,NULL,xinear,xjnear,xinearinfo,xjnearinfo,ibset,jbset);
                dK2delx(yscaleres,xscaleres,dummyind,j, i, bias,altK,NULL,xjnear,xinear,xjnearinfo,xinearinfo,jbset,ibset);

                NiceAssert( dummyind < 0 );

                twoProduct(xey,*xinear,*xjfarfar);
                twoProduct(yey,*xjnear,*xjfarfar);

                res = (xscaleres*((T) xey))+(yscaleres*((T) yey));

                // MOD: do by reversing x and y, assuming symmetry.
                //dK2dely(xscaleres,yscaleres,dummyind,ii,j, bias,altK,NULL,xifar ,xjnear,xifarinfo ,xjnearinfo,ibset,jbset);
                dK2delx(yscaleres,xscaleres,dummyind,j, ii,bias,altK,NULL,xjnear,xifar ,xjnearinfo,xifarinfo ,jbset,ibset);

                NiceAssert( dummyind < 0 );

                twoProduct(xey,*xifar ,*xjfarfar);
                //twoProduct(yey,*xjnear,*xjfarfar);

                res -= (xscaleres*((T) xey))+(yscaleres*((T) yey));
            }

            else
            {
                // <(ex d/dy)^n,K>

                int xdm = xspaceDim();

                res = 0.0;

                Vector<T> sc;
                Vector<Vector<int> > nn;

                Vector<int> q(jgradOrder);

                q = 1;

                int iqa,jqa;
                gentype tmp;

                SparseVector<gentype> kres;
                Vector<const SparseVector<gentype> *> bord;

                dnK2del(sc,nn,dummyind,q,i, j, bias,altK,NULL,xinear,xjnear,xinearinfo,xjnearinfo,ibset,jbset);

                NiceAssert( dummyind < 0 );

                for ( iqa = 0 ; iqa < sc.size() ; iqa++ )
                {
                    bord.resize(nn(iqa).size());

                    for ( jqa = 0 ; jqa < nn(iqa).size() ; jqa++ )
                    {
                        bord("&",jqa) = ( nn(iqa)(jqa) == 0 ) ? xinear : ( ( nn(iqa)(jqa) == 1 ) ? xjnear : NULL );
                    }

                    kronprod(kres,bord,nn(iqa),xdm);

                    twoProduct(tmp,*xjfarfar,kres);

                    res += ((T) tmp)*((T) sc(iqa));
                }

                dnK2del(sc,nn,dummyind,q,ii,j, bias,altK,NULL,xifar ,xjnear,xifarinfo ,xjnearinfo,ibset,jbset);

                NiceAssert( dummyind < 0 );

                for ( iqa = 0 ; iqa < sc.size() ; iqa++ )
                {
                    bord.resize(nn(iqa).size());

                    for ( jqa = 0 ; jqa < nn(iqa).size() ; jqa++ )
                    {
                        bord("&",jqa) = ( nn(iqa)(jqa) == 0 ) ? xifar  : ( ( nn(iqa)(jqa) == 1 ) ? xjnear : NULL );
                    }

                    kronprod(kres,bord,nn(iqa),xdm);

                    twoProduct(tmp,*xjfarfar,kres);

                    res -= ((T) tmp)*((T) sc(iqa));
                }
            }
            
            break;
        }

        case 41:
        {
            NiceAssert( jgradOrder > 0 );

            if ( jgradOrder == 1 )
            {
                // <dK/dy,ey> = <xa.x + ya.y,ey>
                //            = xa.<x,ey> + ya.<y,ey>

                int xdm = xspaceDim();

                resizeZeroVec(res,xdm);

                T xscaleres;
                T yscaleres;

                int jqa;

                // MOD: do by reversing x and y, assuming symmetry.
                //dK2dely(xscaleres,yscaleres,dummyind,i, j, bias,altK,NULL,xinear,xjnear,xinearinfo,xjnearinfo,ibset,jbset);
                dK2delx(yscaleres,xscaleres,dummyind,j, i, bias,altK,NULL,xjnear,xinear,xjnearinfo,xinearinfo,jbset,ibset);

                NiceAssert( dummyind < 0 );

                for ( jqa = 0 ; jqa < xdm ; jqa++ )
                {
                    getVecElm(res,jqa) += ((T) xscaleres)*((T) ((*xinear)(jqa)));
                    getVecElm(res,jqa) += ((T) yscaleres)*((T) ((*xjnear)(jqa)));
                }

                dK2delx(yscaleres,xscaleres,dummyind,j, ii,bias,altK,NULL,xjnear,xifar ,xjnearinfo,xifarinfo ,jbset,ibset);

                NiceAssert( dummyind < 0 );

                for ( jqa = 0 ; jqa < xdm ; jqa++ )
                {
                    getVecElm(res,jqa) += ((T) xscaleres)*((T) ((*xifar )(jqa)));
                    getVecElm(res,jqa) += ((T) yscaleres)*((T) ((*xjnear)(jqa)));
                }
            }

            else
            {
                // <(ex d/dy)^n,K>

                int xdm = xspaceDim();

                int resdim = (int) pow(xdm,jgradOrder);

                resizeZeroVec(res,resdim);

                Vector<T> sc;
                Vector<Vector<int> > nn;

                Vector<int> q(jgradOrder);

                q = 1;

                int iqa,jqa;

                SparseVector<gentype> kres;
                Vector<const SparseVector<gentype> *> bord;

                dnK2del(sc,nn,dummyind,q,i, j, bias,altK,NULL,xinear,xjnear,xinearinfo,xjnearinfo,ibset,jbset);
 
                NiceAssert( dummyind < 0 );

                for ( iqa = 0 ; iqa < sc.size() ; iqa++ )
                {
                    bord.resize(nn(iqa).size());

                    for ( jqa = 0 ; jqa < nn(iqa).size() ; jqa++ )
                    {
                        bord("&",jqa) = ( nn(iqa)(jqa) == 0 ) ? xinear : ( ( nn(iqa)(jqa) == 1 ) ? xjnear : NULL );
                    }

                    kronprod(kres,bord,nn(iqa),xdm);

                    for ( jqa = 0 ; jqa < resdim ; jqa++ )
                    {
                        getVecElm(res,jqa) += ((T) sc(iqa))*((T) kres(jqa));
                    }
                }

                dnK2del(sc,nn,dummyind,q,ii,j, bias,altK,NULL,xifar ,xjnear,xifarinfo ,xjnearinfo,ibset,jbset);
 
                NiceAssert( dummyind < 0 );

                for ( iqa = 0 ; iqa < sc.size() ; iqa++ )
                {
                    bord.resize(nn(iqa).size());

                    for ( jqa = 0 ; jqa < nn(iqa).size() ; jqa++ )
                    {
                        bord("&",jqa) = ( nn(iqa)(jqa) == 0 ) ? xifar  : ( ( nn(iqa)(jqa) == 1 ) ? xjnear : NULL );
                    }

                    kronprod(kres,bord,nn(iqa),xdm);

                    for ( jqa = 0 ; jqa < resdim ; jqa++ )
                    {
                        getVecElm(res,jqa) += ((T) sc(iqa))*((T) kres(jqa));
                    }
                }
            }
            
            break;
        }

        default:
        {
            goto thunk;

            break;
        }
    }

    if ( iokr || jokr )
    {
        Vector<int> iiokr(2);
        Vector<int> iiok(2);
        Vector<int> iiset(2);
        Vector<const gentype *> xxalt(2);

        iiokr("&",0) = iokr;
        iiok("&",0)  = iok;
        iiset("&",i) = ibset;
        xxalt("&",0) = (*xx).isfarfarfarindpresent(3) ? &((*xx).fff(3)) : &nullgentype();

        iiokr("&",1) = jokr;
        iiok("&",1)  = jok;
        iiset("&",1) = jbset;
        xxalt("&",1) = (*yy).isfarfarfarindpresent(3) ? &((*yy).fff(3)) : &nullgentype();

        gentype UUres;

        (*UUcallback)(UUres,2,*this,iiokr,iiok,iiset,xxalt,defbasisUU);

        res *= (T) UUres;
    }

    if ( iplanr || jplanr )
    {
        Vector<int> iiplanr(2);
        Vector<int> iiplan(2);
        Vector<int> iiset(2);
        Vector<const gentype *> xxalt(2);

        iiplanr("&",0) = iplanr;
        iiplan("&",0)  = iplan;
        iiset("&",0)   = ibset;
        xxalt("&",0)   = (*xx).isfarfarfarindpresent(7) ? &((*xx).fff(7)) : &nullgentype();

        iiplanr("&",1) = jplanr;
        iiplan("&",1)  = jplan;
        iiset("&",1)   = jbset;
        xxalt("&",1)   = (*yy).isfarfarfarindpresent(7) ? &((*yy).fff(7)) : &nullgentype();

        gentype VVres;
        gentype kval(res);

        (*VVcallback)(VVres,2,kval,*this,iiplanr,iiplan,iiset,xxalt,defbasisVV);

        res = (T) VVres;
    }

    return res;

thunk:
    Vector<int> lpi(2);
    Vector<const SparseVector<gentype> *> lpxx(2);
    Vector<const vecInfo *> lpxxinfo(2);
    Vector<int> lpiset(2);

    lpi("&",z) = ib;
    lpi("&",1) = jb;

    lpxx("&",z) = xx;
    lpxx("&",1) = yy;

    lpxxinfo("&",z) = xxinfo;
    lpxxinfo("&",1) = yyinfo;

    lpiset("&",z) = ibset;
    lpiset("&",1) = jbset;

    return Km(2,res,lpi,bias,altK,pxyprod,&lpxx,&lpxxinfo,resmode,NULL,NULL,-1,-1,0,0,lpiset);
}

template <class T>
T &ML_Base::K3(T &res, 
               int ia, int ib, int ic, 
               const T &bias, const MercerKernel &altK, const gentype **pxyprod, 
               const SparseVector<gentype> *xa, const SparseVector<gentype> *xb, const SparseVector<gentype> *xc, 
               const vecInfo *xainfo, const vecInfo *xbinfo, const vecInfo *xcinfo, 
               int resmode, int &iaset, int &ibset, int &icset) const
{
//phantomx
    if ( xa && !xainfo )
    {
        vecInfo xinfoa;

        (**thisthisthis).getKernel().getvecInfo(xinfoa,*xa);

        return K3(res,ia,ib,ic,bias,altK,pxyprod,xa,xb,xc,&xinfoa,xbinfo,xcinfo,resmode,iaset,ibset,icset);
    }

    if ( xb && !xbinfo )
    {
        vecInfo xinfob;

        (**thisthisthis).getKernel().getvecInfo(xinfob,*xb);

        return K3(res,ia,ib,ic,bias,altK,pxyprod,xa,xb,xc,xainfo,&xinfob,xcinfo,resmode,iaset,ibset,icset);
    }

    if ( xc && !xcinfo )
    {
        vecInfo xinfoc;

        (**thisthisthis).getKernel().getvecInfo(xinfoc,*xc);

        return K3(res,ia,ib,ic,bias,altK,pxyprod,xa,xb,xc,xainfo,xbinfo,&xinfoc,resmode,iaset,ibset,icset);
    }

    // Always simplify

         if ( xb && !xa ) { return K3(res,ib,ia,ic,bias,altK,pxyprod,xb,xa,xc,xbinfo,xainfo,xcinfo,resmode,ibset,iaset,icset); }
    else if ( xc && !xa ) { return K3(res,ic,ib,ia,bias,altK,pxyprod,xc,xb,xa,xcinfo,xbinfo,xainfo,resmode,icset,ibset,iaset); }
    else if ( xc && !xb ) { return K3(res,ia,ic,ib,bias,altK,pxyprod,xa,xc,xb,xainfo,xcinfo,xbinfo,resmode,iaset,icset,ibset); }

    // Shortcut for speed

    if ( xa && xb && xc && xa->isnofaroffindpresent() && xb->isnofaroffindpresent() && xc->isnofaroffindpresent() )
    {
        int aupm = (*xa).nearupsize();
        int bupm = (*xb).nearupsize();
        int cupm = (*xc).nearupsize();

        if ( ( aupm == 1 ) && ( bupm == 1 ) && ( cupm == 1 ) )
        {
            const Matrix<double> *xyp = NULL;

            if ( altK.suggestXYcache() && getxymat() && ( ia >= 0 ) && ( ib >= 0 ) && ( ic >= 0 ) )
            {
                const Matrix<double> &xy = *getxymat();
                Vector<int> xyderef(3);

                int z = 0;

                xyderef("&",z) = ia;
                xyderef("&",1) = ib;
                xyderef("&",2) = ic;

                retMatrix<double> tmpma;

                xyp = &(xy(xyderef,xyderef,tmpma));
            }

            altK.K3(res,*xa,*xb,*xc,*xainfo,*xbinfo,*xcinfo,bias,pxyprod,ia,ib,ic,xspaceDim(),isXConsistent() && istrv(ia) && istrv(ib) && istrv(ic),resmode,MLid(),xyp,iaset,ibset,icset,assumeReal);
        }

        else if ( ( aupm == 1 ) && ( bupm == 1 ) && ( cupm == 2 ) )
        {
            const SparseVector<gentype> &rxa = xa->nearrefup(0);
            const SparseVector<gentype> &rxb = xb->nearrefup(0);
            const SparseVector<gentype> &rxc = xc->nearrefup(0);
            const SparseVector<gentype> &rxd = xc->nearrefup(1);

            const vecInfo &rxainfo = (*xainfo)(-1,0);
            const vecInfo &rxbinfo = (*xbinfo)(-1,0);
            const vecInfo &rxcinfo = (*xcinfo)(-1,0);
            const vecInfo &rxdinfo = (*xcinfo)(-1,1);

            int ixa = ia;
            int ixb = ib;
            int ixc = ic;
            int ixd = UPNTVI(ic,1);

            int ixaset = iaset;
            int ixbset = ibset;
            int ixcset = icset;
            int ixdset = icset;

            K4(res,ixa,ixb,ixc,ixd,bias,altK,NULL,&rxa,&rxb,&rxc,&rxd,&rxainfo,&rxbinfo,&rxcinfo,&rxdinfo,resmode,ixaset,ixbset,ixcset,ixdset);
        }

        else if ( ( aupm == 1 ) && ( bupm == 2 ) && ( cupm == 1 ) )
        {
            const SparseVector<gentype> &rxa = xa->nearrefup(0);
            const SparseVector<gentype> &rxb = xb->nearrefup(0);
            const SparseVector<gentype> &rxc = xb->nearrefup(1);
            const SparseVector<gentype> &rxd = xc->nearrefup(0);

            const vecInfo &rxainfo = (*xainfo)(-1,0);
            const vecInfo &rxbinfo = (*xbinfo)(-1,0);
            const vecInfo &rxcinfo = (*xbinfo)(-1,1);
            const vecInfo &rxdinfo = (*xcinfo)(-1,0);

            int ixa = ia;
            int ixb = ib;
            int ixc = UPNTVI(ib,1);
            int ixd = ic;

            int ixaset = iaset;
            int ixbset = ibset;
            int ixcset = ibset;
            int ixdset = icset;

            K4(res,ixa,ixb,ixc,ixd,bias,altK,NULL,&rxa,&rxb,&rxc,&rxd,&rxainfo,&rxbinfo,&rxcinfo,&rxdinfo,resmode,ixaset,ixbset,ixcset,ixdset);
        }

        else if ( ( aupm == 2 ) && ( bupm == 1 ) && ( cupm == 1 ) )
        {
            const SparseVector<gentype> &rxa = xa->nearrefup(0);
            const SparseVector<gentype> &rxb = xa->nearrefup(1);
            const SparseVector<gentype> &rxc = xb->nearrefup(0);
            const SparseVector<gentype> &rxd = xc->nearrefup(0);

            const vecInfo &rxainfo = (*xainfo)(-1,0);
            const vecInfo &rxbinfo = (*xainfo)(-1,1);
            const vecInfo &rxcinfo = (*xbinfo)(-1,0);
            const vecInfo &rxdinfo = (*xcinfo)(-1,0);

            int ixa = ia;
            int ixb = UPNTVI(ia,1);
            int ixc = ib;
            int ixd = ic;

            int ixaset = iaset;
            int ixbset = iaset;
            int ixcset = ibset;
            int ixdset = icset;

            K4(res,ixa,ixb,ixc,ixd,bias,altK,NULL,&rxa,&rxb,&rxc,&rxd,&rxainfo,&rxbinfo,&rxcinfo,&rxdinfo,resmode,ixaset,ixbset,ixcset,ixdset);
        }

        else
        {
            int ii;

            Vector<int> iv(aupm+bupm+cupm);
            Vector<const SparseVector<gentype> *> xxv(aupm+bupm+cupm);
            Vector<const vecInfo *> xxvinfo(aupm+bupm+cupm);
            Vector<int> ivset(aupm+bupm+cupm);

            if ( aupm )
            {
                for ( ii = 0 ; ii < aupm ; ii++ )
                {
                    iv("&",ii) = ii ? UPNTVI(ia,ii) : ia;
                    xxv("&",ii) = &((*xa).nearrefup(ii));
                    xxvinfo("&",ii) = &((*xainfo)(-1,ii));
                    ivset("&",ii) = iaset;
                }
            }

            if ( bupm )
            {
                for ( ii = 0 ; ii < bupm ; ii++ )
                {
                    iv("&",ii+aupm) = ii ? UPNTVI(ib,ii) : ib;
                    xxv("&",ii+aupm) = &((*xb).nearrefup(ii));
                    xxvinfo("&",ii+aupm) = &((*xbinfo)(-1,ii));
                    ivset("&",ii+aupm) = ibset;
                }
            }

            if ( cupm )
            {
                for ( ii = 0 ; ii < cupm ; ii++ )
                {
                    iv("&",ii+aupm+bupm) = ii ? UPNTVI(ic,ii) : ic;
                    xxv("&",ii+aupm+bupm) = &((*xc).nearrefup(ii));
                    xxvinfo("&",ii+aupm+bupm) = &((*xcinfo)(-1,ii));
                    ivset("&",ii+aupm+bupm) = icset;
                }
            }

            Km(aupm+bupm+cupm,res,iv,bias,altK,NULL,&xxv,&xxvinfo,resmode,NULL,NULL,-1,-1,0,0,ivset);
        }

        return res;
    }

    else if ( xa && xb && !xc )
    {
        const SparseVector<gentype> &xcb = xgetloc(ic);

        const vecInfo &xinfoc = xinfo(ic);

        return K3(res,ia,ib,ic,bias,altK,pxyprod,xa,xb,&xcb,xainfo,xbinfo,&xinfoc,resmode,iaset,ibset,icset);
    }

    else if ( xa && !xb && !xc )
    {
        const SparseVector<gentype> &xbb = xgetloc(ib);
        const SparseVector<gentype> &xcb = xgetloc(ic);

        const vecInfo &xinfob = xinfo(ib);
        const vecInfo &xinfoc = xinfo(ic);

        return K3(res,ia,ib,ic,bias,altK,pxyprod,xa,&xbb,&xcb,xainfo,&xinfob,&xinfoc,resmode,iaset,ibset,icset);
    }

    else if ( !xa && !xb && !xc )
    {
        const SparseVector<gentype> &xab = xgetloc(ia);
        const SparseVector<gentype> &xbb = xgetloc(ib);
        const SparseVector<gentype> &xcb = xgetloc(ic);

        const vecInfo &xinfoa = xinfo(ia);
        const vecInfo &xinfob = xinfo(ib);
        const vecInfo &xinfoc = xinfo(ic);

        return K3(res,ia,ib,ic,bias,altK,pxyprod,&xab,&xbb,&xcb,&xinfoa,&xinfob,&xinfoc,resmode,iaset,ibset,icset);
    }

    const SparseVector<gentype> *xanear = NULL;
    const SparseVector<gentype> *xbnear = NULL;
    const SparseVector<gentype> *xcnear = NULL;

    const SparseVector<gentype> *xafar = NULL;
    const SparseVector<gentype> *xbfar = NULL;
    const SparseVector<gentype> *xcfar = NULL;

    const SparseVector<gentype> *xafarfar = NULL;
    const SparseVector<gentype> *xbfarfar = NULL;
    const SparseVector<gentype> *xcfarfar = NULL;

    const vecInfo *xanearinfo = NULL;
    const vecInfo *xbnearinfo = NULL;
    const vecInfo *xcnearinfo = NULL;

    const vecInfo *xafarinfo = NULL;
    const vecInfo *xbfarinfo = NULL;
    const vecInfo *xcfarinfo = NULL;

    int ixa,iia,iaokr,iaok,adiagr,agradOrder,iaplanr,iaplan;
    int ixb,iib,ibokr,ibok,bdiagr,bgradOrder,ibplanr,ibplan;
    int ixc,iic,icokr,icok,cdiagr,cgradOrder,icplanr,icplan;

    const gentype *ixatup = NULL;
    const gentype *iiatup = NULL;

    const gentype *ixbtup = NULL;
    const gentype *iibtup = NULL;

    const gentype *ixctup = NULL;
    const gentype *iictup = NULL;

    int iamethodkey = detangle_x(xanear,xafar,xafarfar,xanearinfo,xafarinfo,ixa,iia,ixatup,iiatup,iaokr,iaok,ia,adiagr,xa,xainfo,agradOrder,iaplanr,iaplan,iaset) & 7;
    int ibmethodkey = detangle_x(xbnear,xbfar,xbfarfar,xbnearinfo,xbfarinfo,ixb,iib,ixbtup,iibtup,ibokr,ibok,ib,bdiagr,xb,xbinfo,bgradOrder,ibplanr,ibplan,ibset) & 7;
    int icmethodkey = detangle_x(xcnear,xcfar,xcfarfar,xcnearinfo,xcfarinfo,ixc,iic,ixctup,iictup,icokr,icok,ic,cdiagr,xc,xcinfo,cgradOrder,icplanr,icplan,icset) & 7;

    if ( adiagr && bdiagr && cdiagr && ( ia == ib ) && ( ia == ic ) && ( ia >= 0 ) )
    {
        res =  (T) (*xa).fff(4);
        res *= (T) (*xb).fff(4);
        res *= (T) (*xc).fff(4);

        return res;
    }

    else if ( adiagr || bdiagr || cdiagr )
    {
        res = 0.0;

        return res;
    }

    int z = 0;

    T tmp;

    if ( ixatup || ixbtup || ixctup || iiatup || iibtup || iictup )
    {
        goto thunk;
    }

    switch ( iamethodkey+(10*ibmethodkey)+(100*icmethodkey) )
    {
        case    0:
        {
                   K3(res,ixa,ixb,ixc,bias,altK,NULL,xanear,xbnear,xcnear,xanearinfo,xbnearinfo,xcnearinfo,resmode,iaset,ibset,icset);
            
            break;
        }

        case    1:
        {
                   K3(res,ixa,ixb,ixc,bias,altK,NULL,xanear,xbnear,xcnear,xanearinfo,xbnearinfo,xcnearinfo,resmode,iaset,ibset,icset);
            res -= K3(tmp,iia,iib,ixc,bias,altK,NULL,xafar, xbnear,xcnear,xafarinfo, xbnearinfo,xcnearinfo,resmode,iaset,ibset,icset);

            break;
        }

        case   10:
        {
                   K3(res,ixa,iib,ixc,bias,altK,NULL,xanear,xbnear,xcnear,xanearinfo,xbnearinfo,xcnearinfo,resmode,iaset,ibset,icset);
            res -= K3(tmp,ixa,ixb,ixc,bias,altK,NULL,xanear,xbfar, xcnear,xanearinfo,xbfarinfo, xcnearinfo,resmode,iaset,ibset,icset);

            break;
        }

        case   11:
        {
                   K3(res,ixa,ixb,ixc,bias,altK,NULL,xanear,xbnear,xcnear,xanearinfo,xbnearinfo,xcnearinfo,resmode,iaset,ibset,icset);
            res -= K3(tmp,iia,ixb,ixc,bias,altK,NULL,xafar, xbnear,xcnear,xafarinfo, xbnearinfo,xcnearinfo,resmode,iaset,ibset,icset);
            res -= K3(tmp,ixa,iib,ixc,bias,altK,NULL,xanear,xbfar, xcnear,xanearinfo,xbfarinfo, xcnearinfo,resmode,iaset,ibset,icset);
            res += K3(tmp,iia,iib,ixc,bias,altK,NULL,xafar, xbfar, xcnear,xafarinfo, xbfarinfo, xcnearinfo,resmode,iaset,ibset,icset);

            break;
        }

        case  100:
        {
                   K3(res,ixa,ixb,ixc,bias,altK,NULL,xanear,xbnear,xcnear,xanearinfo,xbnearinfo,xcnearinfo,resmode,iaset,ibset,icset);
            res -= K3(tmp,ixa,ixb,iic,bias,altK,NULL,xanear,xbnear,xcfar, xanearinfo,xbnearinfo,xcfarinfo, resmode,iaset,ibset,icset);

            break;
        }

        case  101:
        {
                   K3(res,ixa,ixb,ixc,bias,altK,NULL,xanear,xbnear,xcnear,xanearinfo,xbnearinfo,xcnearinfo,resmode,iaset,ibset,icset);
            res -= K3(tmp,iia,iib,ixc,bias,altK,NULL,xafar, xbnear,xcnear,xafarinfo, xbnearinfo,xcnearinfo,resmode,iaset,ibset,icset);
            res -= K3(tmp,ixa,ixb,iic,bias,altK,NULL,xanear,xbnear,xcfar, xanearinfo,xbnearinfo,xcfarinfo, resmode,iaset,ibset,icset);
            res += K3(tmp,iia,iib,iic,bias,altK,NULL,xafar, xbnear,xcfar, xafarinfo, xbnearinfo,xcfarinfo, resmode,iaset,ibset,icset);

            break;
        }

        case  110:
        {
                   K3(res,ixa,iib,ixc,bias,altK,NULL,xanear,xbnear,xcnear,xanearinfo,xbnearinfo,xcnearinfo,resmode,iaset,ibset,icset);
            res -= K3(tmp,ixa,ixb,ixc,bias,altK,NULL,xanear,xbfar, xcnear,xanearinfo,xbfarinfo, xcnearinfo,resmode,iaset,ibset,icset);
            res -= K3(tmp,ixa,iib,iic,bias,altK,NULL,xanear,xbnear,xcfar, xanearinfo,xbnearinfo,xcfarinfo, resmode,iaset,ibset,icset);
            res += K3(tmp,ixa,ixb,iic,bias,altK,NULL,xanear,xbfar, xcfar, xanearinfo,xbfarinfo, xcfarinfo, resmode,iaset,ibset,icset);

            break;
        }

        case  111:
        {
                   K3(res,ixa,ixb,ixc,bias,altK,NULL,xanear,xbnear,xcnear,xanearinfo,xbnearinfo,xcnearinfo,resmode,iaset,ibset,icset);
            res -= K3(tmp,iia,ixb,ixc,bias,altK,NULL,xafar, xbnear,xcnear,xafarinfo, xbnearinfo,xcnearinfo,resmode,iaset,ibset,icset);
            res -= K3(tmp,ixa,iib,ixc,bias,altK,NULL,xanear,xbfar, xcnear,xanearinfo,xbfarinfo, xcnearinfo,resmode,iaset,ibset,icset);
            res += K3(tmp,iia,iib,ixc,bias,altK,NULL,xafar, xbfar, xcnear,xafarinfo, xbfarinfo, xcnearinfo,resmode,iaset,ibset,icset);
            res -= K3(tmp,ixa,ixb,iic,bias,altK,NULL,xanear,xbnear,xcfar, xanearinfo,xbnearinfo,xcfarinfo, resmode,iaset,ibset,icset);
            res += K3(tmp,iia,ixb,iic,bias,altK,NULL,xafar, xbnear,xcfar, xafarinfo, xbnearinfo,xcfarinfo, resmode,iaset,ibset,icset);
            res += K3(tmp,ixa,iib,iic,bias,altK,NULL,xanear,xbfar, xcfar, xanearinfo,xbfarinfo, xcfarinfo, resmode,iaset,ibset,icset);
            res -= K3(tmp,iia,iib,iic,bias,altK,NULL,xafar, xbfar, xcfar, xafarinfo, xbfarinfo, xcfarinfo, resmode,iaset,ibset,icset);

            break;
        }

        default:
        {
            goto thunk;

            break;
        }
    }

    if ( iaokr || ibokr || icokr )
    {
        Vector<int> iiokr(3);
        Vector<int> iiok(3);
        Vector<int> iiset(3);
        Vector<const gentype *> xxalt(3);

        iiokr("&",0) = iaokr;
        iiok("&",0)  = iaok;
        iiset("&",0) = iaset;
        xxalt("&",0) = (*xa).isfarfarfarindpresent(3) ? &((*xa).fff(3)) : &nullgentype();

        iiokr("&",1) = ibokr;
        iiok("&",1)  = ibok;
        iiset("&",1) = ibset;
        xxalt("&",1) = (*xb).isfarfarfarindpresent(3) ? &((*xb).fff(3)) : &nullgentype();

        iiokr("&",2) = icokr;
        iiok("&",2)  = icok;
        iiset("&",2) = icset;
        xxalt("&",2) = (*xc).isfarfarfarindpresent(3) ? &((*xc).fff(3)) : &nullgentype();

        gentype UUres;

        (*UUcallback)(UUres,3,*this,iiokr,iiok,iiset,xxalt,defbasisUU);

        res *= (T) UUres;
    }

    if ( iaplanr || ibplanr || icplanr )
    {
        Vector<int> iiplanr(3);
        Vector<int> iiplan(3);
        Vector<int> iiset(3);
        Vector<const gentype *> xxalt(3);

        iiplanr("&",0) = iaplanr;
        iiplan("&",0)  = iaplan;
        iiset("&",0)   = iaset;
        xxalt("&",0)   = (*xa).isfarfarfarindpresent(7) ? &((*xa).fff(7)) : &nullgentype();

        iiplanr("&",1) = ibplanr;
        iiplan("&",1)  = ibplan;
        iiset("&",1)   = ibset;
        xxalt("&",1)   = (*xb).isfarfarfarindpresent(7) ? &((*xb).fff(7)) : &nullgentype();

        iiplanr("&",2) = icplanr;
        iiplan("&",2)  = icplan;
        iiset("&",2)   = icset;
        xxalt("&",2)   = (*xc).isfarfarfarindpresent(7) ? &((*xc).fff(7)) : &nullgentype();

        gentype VVres;
        gentype kval(res);

        (*VVcallback)(VVres,3,kval,*this,iiplanr,iiplan,iiset,xxalt,defbasisVV);

        res = (T) VVres;
    }

    NiceAssert( !iaplanr );
    NiceAssert( !ibplanr );
    NiceAssert( !icplanr );

    return res;

thunk:
    Vector<int> lpi(3);
    Vector<const SparseVector<gentype> *> lpxx(3);
    Vector<const vecInfo *> lpxxinfo(3);
    Vector<int> lpiset(3);

    lpi("&",z) = ia;
    lpi("&",1) = ib;
    lpi("&",2) = ic;

    lpxx("&",z) = xa;
    lpxx("&",1) = xb;
    lpxx("&",2) = xc;

    lpxxinfo("&",z) = xainfo;
    lpxxinfo("&",1) = xbinfo;
    lpxxinfo("&",2) = xcinfo;

    lpiset("&",z) = iaset;
    lpiset("&",1) = ibset;
    lpiset("&",2) = icset;

    return Km(3,res,lpi,bias,altK,pxyprod,&lpxx,&lpxxinfo,resmode,NULL,NULL,-1,-1,0,0,lpiset);
}

template <class T>
T &ML_Base::K4(T &res, 
               int ia, int ib, int ic, int id, 
               const T &bias, const MercerKernel &altK, const gentype **pxyprod, 
               const SparseVector<gentype> *xa, const SparseVector<gentype> *xb, const SparseVector<gentype> *xc, const SparseVector<gentype> *xd, 
               const vecInfo *xainfo, const vecInfo *xbinfo, const vecInfo *xcinfo, const vecInfo *xdinfo, 
               int resmode, int &iaset, int &ibset, int &icset, int &idset) const
{
//phantomx
    if ( xa && !xainfo )
    {
        vecInfo xinfoa;

        (**thisthisthis).getKernel().getvecInfo(xinfoa,*xa);

        return K4(res,ia,ib,ic,id,bias,altK,pxyprod,xa,xb,xc,xd,&xinfoa,xbinfo,xcinfo,xdinfo,resmode,iaset,ibset,icset,idset);
    }

    if ( xb && !xbinfo )
    {
        vecInfo xinfob;

        (**thisthisthis).getKernel().getvecInfo(xinfob,*xb);

        return K4(res,ia,ib,ic,id,bias,altK,pxyprod,xa,xb,xc,xd,xainfo,&xinfob,xcinfo,xdinfo,resmode,iaset,ibset,icset,idset);
    }

    if ( xc && !xcinfo )
    {
        vecInfo xinfoc;

        (**thisthisthis).getKernel().getvecInfo(xinfoc,*xc);

        return K4(res,ia,ib,ic,id,bias,altK,pxyprod,xa,xb,xc,xd,xainfo,xbinfo,&xinfoc,xdinfo,resmode,iaset,ibset,icset,idset);
    }

    if ( xd && !xdinfo )
    {
        vecInfo xinfod;

        (**thisthisthis).getKernel().getvecInfo(xinfod,*xd);

        return K4(res,ia,ib,ic,id,bias,altK,pxyprod,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,&xinfod,resmode,iaset,ibset,icset,idset);
    }

    // Always simplify

         if ( xb && !xa ) { return K4(res,ib,ia,ic,id,bias,altK,pxyprod,xb,xa,xc,xd,xbinfo,xainfo,xcinfo,xdinfo,resmode,ibset,iaset,icset,idset); }
    else if ( xc && !xa ) { return K4(res,ic,ib,ia,id,bias,altK,pxyprod,xc,xb,xa,xd,xcinfo,xbinfo,xainfo,xdinfo,resmode,icset,ibset,iaset,idset); }
    else if ( xd && !xa ) { return K4(res,id,ib,ic,ia,bias,altK,pxyprod,xd,xb,xc,xa,xdinfo,xbinfo,xcinfo,xainfo,resmode,idset,ibset,icset,iaset); }
    else if ( xc && !xb ) { return K4(res,ia,ic,ib,id,bias,altK,pxyprod,xa,xc,xb,xd,xainfo,xcinfo,xbinfo,xdinfo,resmode,iaset,icset,ibset,idset); }
    else if ( xd && !xb ) { return K4(res,ia,id,ic,ib,bias,altK,pxyprod,xa,xd,xc,xb,xainfo,xdinfo,xcinfo,xbinfo,resmode,iaset,idset,icset,ibset); }
    else if ( xd && !xc ) { return K4(res,ia,ib,id,ic,bias,altK,pxyprod,xa,xb,xd,xc,xainfo,xbinfo,xdinfo,xcinfo,resmode,iaset,ibset,idset,icset); }

    // Shortcut for speed

    if ( xa && xb && xc && xd && xa->isnofaroffindpresent() && xb->isnofaroffindpresent() && xc->isnofaroffindpresent() && xd->isnofaroffindpresent() )
    {
        int aupm = (*xa).nearupsize();
        int bupm = (*xb).nearupsize();
        int cupm = (*xc).nearupsize();
        int dupm = (*xd).nearupsize();

        if ( ( aupm == 1 ) && ( bupm == 1 ) && ( cupm == 1 ) && ( dupm == 1 ) )
        {
            if ( altK.suggestXYcache() && getxymat() && ( ia >= 0 ) && ( ib >= 0 ) && ( ic >= 0 ) && ( id >= 0 ) )
            {
//                    const Matrix<double> &xy = *getxymat();
//                    Vector<int> xyderef(4);
//
//                    int z = 0;
//
//                    xyderef("&",z) = ia;
//                    xyderef("&",1) = ib;
//                    xyderef("&",2) = ic;
//                    xyderef("&",3) = id;
//
//                    altK.K4(res,*xa,*xb,*xc,*xd,*xainfo,*xbinfo,*xcinfo,*xdinfo,bias,pxyprod,ia,ib,ic,id,xspaceDim(),isXConsistent(),resmode,&(xy(xyderef,xyderef)));

                const Matrix<double> &xy = *getxymat();
                int z = 0;

                altK.K4(res,*xa,*xb,*xc,*xd,*xainfo,*xbinfo,*xcinfo,*xdinfo,bias,pxyprod,ia,ib,ic,id,xspaceDim(),isXConsistent() && istrv(ia) && istrv(ib) && istrv(ic) && istrv(id),resmode,MLid(),NULL,&xy(z,z),&xy(1,z),&xy(1,1),&xy(2,z),&xy(2,1),&xy(2,2),&xy(3,z),&xy(3,1),&xy(3,2),&xy(3,3),iaset,ibset,icset,idset,assumeReal);
            }

            else
            {
                altK.K4(res,*xa,*xb,*xc,*xd,*xainfo,*xbinfo,*xcinfo,*xdinfo,bias,pxyprod,ia,ib,ic,id,xspaceDim(),isXConsistent() && istrv(ia) && istrv(ib) && istrv(ic) && istrv(id),resmode,MLid(),NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iaset,ibset,icset,idset,assumeReal);
            }
        }

        else
        {
            int ii;

            Vector<int> iv(aupm+bupm+cupm+dupm);
            Vector<const SparseVector<gentype> *> xxv(aupm+bupm+cupm+dupm);
            Vector<const vecInfo *> xxvinfo(aupm+bupm+cupm+dupm);
            Vector<int> ivset(aupm+bupm+cupm+dupm);

            if ( aupm )
            {
                for ( ii = 0 ; ii < aupm ; ii++ )
                {
                    iv("&",ii) = ii ? UPNTVI(ia,ii) : ia;
                    xxv("&",ii) = &((*xa).nearrefup(ii));
                    xxvinfo("&",ii) = &((*xainfo)(-1,ii));
                    ivset("&",ii) = iaset;
                }
            }

            if ( bupm )
            {
                for ( ii = 0 ; ii < bupm ; ii++ )
                {
                    iv("&",ii+aupm) = ii ? UPNTVI(ib,ii) : ib;
                    xxv("&",ii+aupm) = &((*xb).nearrefup(ii));
                    xxvinfo("&",ii+aupm) = &((*xbinfo)(-1,ii));
                    ivset("&",ii+aupm) = ibset;
                }
            }

            if ( cupm )
            {
                for ( ii = 0 ; ii < cupm ; ii++ )
                {
                    iv("&",ii+aupm+bupm) = ii ? UPNTVI(ic,ii) : ic;
                    xxv("&",ii+aupm+bupm) = &((*xc).nearrefup(ii));
                    xxvinfo("&",ii+aupm+bupm) = &((*xcinfo)(-1,ii));
                    ivset("&",ii+aupm+bupm) = icset;
                }
            }

            if ( bupm )
            {
                for ( ii = 0 ; ii < dupm ; ii++ )
                {
                    iv("&",ii+aupm+bupm+cupm) = ii ? UPNTVI(id,ii) : id;
                    xxv("&",ii+aupm+bupm+cupm) = &((*xd).nearrefup(ii));
                    xxvinfo("&",ii+aupm+bupm+cupm) = &((*xdinfo)(-1,ii));
                    ivset("&",ii+aupm+bupm+cupm) = idset;
                }
            }

            Km(aupm+bupm+cupm+dupm,res,iv,bias,altK,NULL,&xxv,&xxvinfo,resmode,NULL,NULL,-1,-1,0,0,ivset);
        }

        return res;
    }

    else if ( xa && xb && xc && !xd )
    {
        const SparseVector<gentype> &xdb = xgetloc(id);

        const vecInfo &xinfod = xinfo(id);

        return K4(res,ia,ib,ic,id,bias,altK,pxyprod,xa,xb,xc,&xdb,xainfo,xbinfo,xcinfo,&xinfod,resmode,iaset,ibset,icset,idset);
    }

    else if ( xa && xb && !xc && !xd )
    {
        const SparseVector<gentype> &xcb = xgetloc(ic);
        const SparseVector<gentype> &xdb = xgetloc(id);

        const vecInfo &xinfoc = xinfo(ic);
        const vecInfo &xinfod = xinfo(id);

        return K4(res,ia,ib,ic,id,bias,altK,pxyprod,xa,xb,&xcb,&xdb,xainfo,xbinfo,&xinfoc,&xinfod,resmode,iaset,ibset,icset,idset);
    }

    else if ( xa && !xb && !xc && !xd )
    {
        const SparseVector<gentype> &xbb = xgetloc(ib);
        const SparseVector<gentype> &xcb = xgetloc(ic);
        const SparseVector<gentype> &xdb = xgetloc(id);

        const vecInfo &xinfob = xinfo(ib);
        const vecInfo &xinfoc = xinfo(ic);
        const vecInfo &xinfod = xinfo(id);

        return K4(res,ia,ib,ic,id,bias,altK,pxyprod,xa,&xbb,&xcb,&xdb,xainfo,&xinfob,&xinfoc,&xinfod,resmode,iaset,ibset,icset,idset);
    }

    else if ( !xa && !xb && !xc && !xd )
    {
        const SparseVector<gentype> &xab = xgetloc(ia);
        const SparseVector<gentype> &xbb = xgetloc(ib);
        const SparseVector<gentype> &xcb = xgetloc(ic);
        const SparseVector<gentype> &xdb = xgetloc(id);

        const vecInfo &xinfoa = xinfo(ia);
        const vecInfo &xinfob = xinfo(ib);
        const vecInfo &xinfoc = xinfo(ic);
        const vecInfo &xinfod = xinfo(id);

        return K4(res,ia,ib,ic,id,bias,altK,pxyprod,&xab,&xbb,&xcb,&xdb,&xinfoa,&xinfob,&xinfoc,&xinfod,resmode,iaset,ibset,icset,idset);
    }

    const SparseVector<gentype> *xanear = NULL;
    const SparseVector<gentype> *xbnear = NULL;
    const SparseVector<gentype> *xcnear = NULL;
    const SparseVector<gentype> *xdnear = NULL;

    const SparseVector<gentype> *xafar = NULL;
    const SparseVector<gentype> *xbfar = NULL;
    const SparseVector<gentype> *xcfar = NULL;
    const SparseVector<gentype> *xdfar = NULL;

    const SparseVector<gentype> *xafarfar = NULL;
    const SparseVector<gentype> *xbfarfar = NULL;
    const SparseVector<gentype> *xcfarfar = NULL;
    const SparseVector<gentype> *xdfarfar = NULL;

    const vecInfo *xanearinfo = NULL;
    const vecInfo *xbnearinfo = NULL;
    const vecInfo *xcnearinfo = NULL;
    const vecInfo *xdnearinfo = NULL;

    const vecInfo *xafarinfo = NULL;
    const vecInfo *xbfarinfo = NULL;
    const vecInfo *xcfarinfo = NULL;
    const vecInfo *xdfarinfo = NULL;

    int ixa,iia,iaokr,iaok,adiagr,agradOrder,iaplanr,iaplan;
    int ixb,iib,ibokr,ibok,bdiagr,bgradOrder,ibplanr,ibplan;
    int ixc,iic,icokr,icok,cdiagr,cgradOrder,icplanr,icplan;
    int ixd,iid,idokr,idok,ddiagr,dgradOrder,idplanr,idplan;

    const gentype *ixatup = NULL;
    const gentype *iiatup = NULL;

    const gentype *ixbtup = NULL;
    const gentype *iibtup = NULL;

    const gentype *ixctup = NULL;
    const gentype *iictup = NULL;

    const gentype *ixdtup = NULL;
    const gentype *iidtup = NULL;

    int iamethodkey = detangle_x(xanear,xafar,xafarfar,xanearinfo,xafarinfo,ixa,iia,ixatup,iiatup,iaokr,iaok,ia,adiagr,xa,xainfo,agradOrder,iaplanr,iaplan,iaset) & 7;
    int ibmethodkey = detangle_x(xbnear,xbfar,xbfarfar,xbnearinfo,xbfarinfo,ixb,iib,ixbtup,iibtup,ibokr,ibok,ib,bdiagr,xb,xbinfo,bgradOrder,ibplanr,ibplan,ibset) & 7;
    int icmethodkey = detangle_x(xcnear,xcfar,xcfarfar,xcnearinfo,xcfarinfo,ixc,iic,ixctup,iictup,icokr,icok,ic,cdiagr,xc,xcinfo,cgradOrder,icplanr,icplan,icset) & 7;
    int idmethodkey = detangle_x(xdnear,xdfar,xdfarfar,xdnearinfo,xdfarinfo,ixd,iid,ixdtup,iidtup,idokr,idok,id,ddiagr,xd,xdinfo,dgradOrder,idplanr,idplan,idset) & 7;

    if ( adiagr && bdiagr && cdiagr && ddiagr && ( ia == ib ) && ( ia == ic ) && ( ia == id ) && ( ia >= 0 ) )
    {
        res =  (T) (*xa).fff(4);
        res *= (T) (*xb).fff(4);
        res *= (T) (*xc).fff(4);
        res *= (T) (*xd).fff(4);

        return res;
    }

    else if ( adiagr || bdiagr || cdiagr || ddiagr )
    {
        res = 0.0;

        return res;
    }

    int z = 0;

    T tmp;

    if ( ixatup || ixbtup || ixctup || ixdtup || iiatup || iibtup || iictup || iidtup )
    {
        goto thunk;
    }

    switch ( iamethodkey+(10*ibmethodkey)+(100*icmethodkey)+(1000*idmethodkey) )
    {
        case    0:
        {
                   K4(res,ixa,ixb,ixc,ixd,bias,altK,NULL,xanear,xbnear,xcnear,xdnear,xanearinfo,xbnearinfo,xcnearinfo,xdnearinfo,resmode,iaset,ibset,icset,idset);
            
            break;
        }

        case    1:
        {
                   K4(res,ixa,ixb,ixc,ixd,bias,altK,NULL,xanear,xbnear,xcnear,xdnear,xanearinfo,xbnearinfo,xcnearinfo,xdnearinfo,resmode,iaset,ibset,icset,idset);
            res -= K4(tmp,iia,iib,ixc,ixd,bias,altK,NULL,xafar, xbnear,xcnear,xdnear,xafarinfo, xbnearinfo,xcnearinfo,xdnearinfo,resmode,iaset,ibset,icset,idset);

            break;
        }

        case   10:
        {
                   K4(res,ixa,iib,ixc,ixd,bias,altK,NULL,xanear,xbnear,xcnear,xdnear,xanearinfo,xbnearinfo,xcnearinfo,xdnearinfo,resmode,iaset,ibset,icset,idset);
            res -= K4(tmp,ixa,ixb,ixc,ixd,bias,altK,NULL,xanear,xbfar, xcnear,xdnear,xanearinfo,xbfarinfo, xcnearinfo,xdnearinfo,resmode,iaset,ibset,icset,idset);

            break;
        }

        case   11:
        {
                   K4(res,ixa,ixb,ixc,ixd,bias,altK,NULL,xanear,xbnear,xcnear,xdnear,xanearinfo,xbnearinfo,xcnearinfo,xdnearinfo,resmode,iaset,ibset,icset,idset);
            res -= K4(tmp,iia,ixb,ixc,ixd,bias,altK,NULL,xafar, xbnear,xcnear,xdnear,xafarinfo, xbnearinfo,xcnearinfo,xdnearinfo,resmode,iaset,ibset,icset,idset);
            res -= K4(tmp,ixa,iib,ixc,ixd,bias,altK,NULL,xanear,xbfar, xcnear,xdnear,xanearinfo,xbfarinfo, xcnearinfo,xdnearinfo,resmode,iaset,ibset,icset,idset);
            res += K4(tmp,iia,iib,ixc,ixd,bias,altK,NULL,xafar, xbfar, xcnear,xdnear,xafarinfo, xbfarinfo, xcnearinfo,xdnearinfo,resmode,iaset,ibset,icset,idset);

            break;
        }

        case  100:
        {
                   K4(res,ixa,ixb,ixc,ixd,bias,altK,NULL,xanear,xbnear,xcnear,xdnear,xanearinfo,xbnearinfo,xcnearinfo,xdnearinfo,resmode,iaset,ibset,icset,idset);
            res -= K4(tmp,ixa,ixb,iic,ixd,bias,altK,NULL,xanear,xbnear,xcfar, xdnear,xanearinfo,xbnearinfo,xcfarinfo, xdnearinfo,resmode,iaset,ibset,icset,idset);

            break;
        }

        case  101:
        {
                   K4(res,ixa,ixb,ixc,ixd,bias,altK,NULL,xanear,xbnear,xcnear,xdnear,xanearinfo,xbnearinfo,xcnearinfo,xdnearinfo,resmode,iaset,ibset,icset,idset);
            res -= K4(tmp,iia,iib,ixc,ixd,bias,altK,NULL,xafar, xbnear,xcnear,xdnear,xafarinfo, xbnearinfo,xcnearinfo,xdnearinfo,resmode,iaset,ibset,icset,idset);
            res -= K4(tmp,ixa,ixb,iic,ixd,bias,altK,NULL,xanear,xbnear,xcfar, xdnear,xanearinfo,xbnearinfo,xcfarinfo, xdnearinfo,resmode,iaset,ibset,icset,idset);
            res += K4(tmp,iia,iib,iic,ixd,bias,altK,NULL,xafar, xbnear,xcfar, xdnear,xafarinfo, xbnearinfo,xcfarinfo, xdnearinfo,resmode,iaset,ibset,icset,idset);

            break;
        }

        case  110:
        {
                   K4(res,ixa,iib,ixc,ixd,bias,altK,NULL,xanear,xbnear,xcnear,xdnear,xanearinfo,xbnearinfo,xcnearinfo,xdnearinfo,resmode,iaset,ibset,icset,idset);
            res -= K4(tmp,ixa,ixb,ixc,ixd,bias,altK,NULL,xanear,xbfar, xcnear,xdnear,xanearinfo,xbfarinfo, xcnearinfo,xdnearinfo,resmode,iaset,ibset,icset,idset);
            res -= K4(tmp,ixa,iib,iic,ixd,bias,altK,NULL,xanear,xbnear,xcfar, xdnear,xanearinfo,xbnearinfo,xcfarinfo, xdnearinfo,resmode,iaset,ibset,icset,idset);
            res += K4(tmp,ixa,ixb,iic,ixd,bias,altK,NULL,xanear,xbfar, xcfar, xdnear,xanearinfo,xbfarinfo, xcfarinfo, xdnearinfo,resmode,iaset,ibset,icset,idset);

            break;
        }

        case  111:
        {
                   K4(res,ixa,ixb,ixc,ixd,bias,altK,NULL,xanear,xbnear,xcnear,xdnear,xanearinfo,xbnearinfo,xcnearinfo,xdnearinfo,resmode,iaset,ibset,icset,idset);
            res -= K4(tmp,iia,ixb,ixc,ixd,bias,altK,NULL,xafar, xbnear,xcnear,xdnear,xafarinfo, xbnearinfo,xcnearinfo,xdnearinfo,resmode,iaset,ibset,icset,idset);
            res -= K4(tmp,ixa,iib,ixc,ixd,bias,altK,NULL,xanear,xbfar, xcnear,xdnear,xanearinfo,xbfarinfo, xcnearinfo,xdnearinfo,resmode,iaset,ibset,icset,idset);
            res += K4(tmp,iia,iib,ixc,ixd,bias,altK,NULL,xafar, xbfar, xcnear,xdnear,xafarinfo, xbfarinfo, xcnearinfo,xdnearinfo,resmode,iaset,ibset,icset,idset);
            res -= K4(tmp,ixa,ixb,iic,ixd,bias,altK,NULL,xanear,xbnear,xcfar, xdnear,xanearinfo,xbnearinfo,xcfarinfo, xdnearinfo,resmode,iaset,ibset,icset,idset);
            res += K4(tmp,iia,ixb,iic,ixd,bias,altK,NULL,xafar, xbnear,xcfar, xdnear,xafarinfo, xbnearinfo,xcfarinfo, xdnearinfo,resmode,iaset,ibset,icset,idset);
            res += K4(tmp,ixa,iib,iic,ixd,bias,altK,NULL,xanear,xbfar, xcfar, xdnear,xanearinfo,xbfarinfo, xcfarinfo, xdnearinfo,resmode,iaset,ibset,icset,idset);
            res -= K4(tmp,iia,iib,iic,ixd,bias,altK,NULL,xafar, xbfar, xcfar, xdnear,xafarinfo, xbfarinfo, xcfarinfo, xdnearinfo,resmode,iaset,ibset,icset,idset);

            break;
        }

        case 1000:
        {
                   K4(res,ixa,ixb,ixc,ixd,bias,altK,NULL,xanear,xbnear,xcnear,xdnear,xanearinfo,xbnearinfo,xcnearinfo,xdnearinfo,resmode,iaset,ibset,icset,idset);
            res -= K4(tmp,ixa,ixb,ixc,iid,bias,altK,NULL,xanear,xbnear,xcnear,xdfar, xanearinfo,xbnearinfo,xcnearinfo,xdfarinfo, resmode,iaset,ibset,icset,idset);

            break;
        }

        case 1001:
        {
                   K4(res,ixa,ixb,ixc,ixd,bias,altK,NULL,xanear,xbnear,xcnear,xdnear,xanearinfo,xbnearinfo,xcnearinfo,xdnearinfo,resmode,iaset,ibset,icset,idset);
            res -= K4(tmp,iia,iib,ixc,ixd,bias,altK,NULL,xafar, xbnear,xcnear,xdnear,xafarinfo, xbnearinfo,xcnearinfo,xdnearinfo,resmode,iaset,ibset,icset,idset);
            res -= K4(tmp,ixa,ixb,ixc,iid,bias,altK,NULL,xanear,xbnear,xcnear,xdfar, xanearinfo,xbnearinfo,xcnearinfo,xdfarinfo, resmode,iaset,ibset,icset,idset);
            res += K4(tmp,iia,iib,ixc,iid,bias,altK,NULL,xafar, xbnear,xcnear,xdfar, xafarinfo, xbnearinfo,xcnearinfo,xdfarinfo, resmode,iaset,ibset,icset,idset);

            break;
        }

        case 1010:
        {
                   K4(res,ixa,iib,ixc,ixd,bias,altK,NULL,xanear,xbnear,xcnear,xdnear,xanearinfo,xbnearinfo,xcnearinfo,xdnearinfo,resmode,iaset,ibset,icset,idset);
            res -= K4(tmp,ixa,ixb,ixc,ixd,bias,altK,NULL,xanear,xbfar, xcnear,xdnear,xanearinfo,xbfarinfo, xcnearinfo,xdnearinfo,resmode,iaset,ibset,icset,idset);
            res -= K4(tmp,ixa,iib,ixc,iid,bias,altK,NULL,xanear,xbnear,xcnear,xdfar, xanearinfo,xbnearinfo,xcnearinfo,xdfarinfo, resmode,iaset,ibset,icset,idset);
            res += K4(tmp,ixa,ixb,ixc,iid,bias,altK,NULL,xanear,xbfar, xcnear,xdfar, xanearinfo,xbfarinfo, xcnearinfo,xdfarinfo, resmode,iaset,ibset,icset,idset);

            break;
        }

        case 1011:
        {
                   K4(res,ixa,ixb,ixc,ixd,bias,altK,NULL,xanear,xbnear,xcnear,xdnear,xanearinfo,xbnearinfo,xcnearinfo,xdnearinfo,resmode,iaset,ibset,icset,idset);
            res -= K4(tmp,iia,ixb,ixc,ixd,bias,altK,NULL,xafar, xbnear,xcnear,xdnear,xafarinfo, xbnearinfo,xcnearinfo,xdnearinfo,resmode,iaset,ibset,icset,idset);
            res -= K4(tmp,ixa,iib,ixc,ixd,bias,altK,NULL,xanear,xbfar, xcnear,xdnear,xanearinfo,xbfarinfo, xcnearinfo,xdnearinfo,resmode,iaset,ibset,icset,idset);
            res += K4(tmp,iia,iib,ixc,ixd,bias,altK,NULL,xafar, xbfar, xcnear,xdnear,xafarinfo, xbfarinfo, xcnearinfo,xdnearinfo,resmode,iaset,ibset,icset,idset);
            res -= K4(tmp,ixa,ixb,ixc,iid,bias,altK,NULL,xanear,xbnear,xcnear,xdfar, xanearinfo,xbnearinfo,xcnearinfo,xdfarinfo, resmode,iaset,ibset,icset,idset);
            res += K4(tmp,iia,ixb,ixc,iid,bias,altK,NULL,xafar, xbnear,xcnear,xdfar, xafarinfo, xbnearinfo,xcnearinfo,xdfarinfo, resmode,iaset,ibset,icset,idset);
            res += K4(tmp,ixa,iib,ixc,iid,bias,altK,NULL,xanear,xbfar, xcnear,xdfar, xanearinfo,xbfarinfo, xcnearinfo,xdfarinfo, resmode,iaset,ibset,icset,idset);
            res -= K4(tmp,iia,iib,ixc,iid,bias,altK,NULL,xafar, xbfar, xcnear,xdfar, xafarinfo, xbfarinfo, xcnearinfo,xdfarinfo, resmode,iaset,ibset,icset,idset);

            break;
        }

        case 1100:
        {
                   K4(res,ixa,ixb,ixc,ixd,bias,altK,NULL,xanear,xbnear,xcnear,xdnear,xanearinfo,xbnearinfo,xcnearinfo,xdnearinfo,resmode,iaset,ibset,icset,idset);
            res -= K4(tmp,ixa,ixb,iic,ixd,bias,altK,NULL,xanear,xbnear,xcfar, xdnear,xanearinfo,xbnearinfo,xcfarinfo, xdnearinfo,resmode,iaset,ibset,icset,idset);
            res -= K4(tmp,ixa,ixb,ixc,iid,bias,altK,NULL,xanear,xbnear,xcnear,xdfar, xanearinfo,xbnearinfo,xcnearinfo,xdfarinfo, resmode,iaset,ibset,icset,idset);
            res += K4(tmp,ixa,ixb,iic,iid,bias,altK,NULL,xanear,xbnear,xcfar, xdfar, xanearinfo,xbnearinfo,xcfarinfo, xdfarinfo, resmode,iaset,ibset,icset,idset);
            
            break;
        }

        case 1101:
        {
                   K4(res,ixa,ixb,ixc,ixd,bias,altK,NULL,xanear,xbnear,xcnear,xdnear,xanearinfo,xbnearinfo,xcnearinfo,xdnearinfo,resmode,iaset,ibset,icset,idset);
            res -= K4(tmp,iia,iib,ixc,ixd,bias,altK,NULL,xafar, xbnear,xcnear,xdnear,xafarinfo, xbnearinfo,xcnearinfo,xdnearinfo,resmode,iaset,ibset,icset,idset);
            res -= K4(tmp,ixa,ixb,iic,ixd,bias,altK,NULL,xanear,xbnear,xcfar, xdnear,xanearinfo,xbnearinfo,xcfarinfo, xdnearinfo,resmode,iaset,ibset,icset,idset);
            res += K4(tmp,iia,iib,iic,ixd,bias,altK,NULL,xafar, xbnear,xcfar, xdnear,xafarinfo, xbnearinfo,xcfarinfo, xdnearinfo,resmode,iaset,ibset,icset,idset);
            res -= K4(tmp,ixa,ixb,ixc,iid,bias,altK,NULL,xanear,xbnear,xcnear,xdfar, xanearinfo,xbnearinfo,xcnearinfo,xdfarinfo, resmode,iaset,ibset,icset,idset);
            res += K4(tmp,iia,iib,ixc,iid,bias,altK,NULL,xafar, xbnear,xcnear,xdfar, xafarinfo, xbnearinfo,xcnearinfo,xdfarinfo, resmode,iaset,ibset,icset,idset);
            res += K4(tmp,ixa,ixb,iic,iid,bias,altK,NULL,xanear,xbnear,xcfar, xdfar, xanearinfo,xbnearinfo,xcfarinfo, xdfarinfo, resmode,iaset,ibset,icset,idset);
            res -= K4(tmp,iia,iib,iic,iid,bias,altK,NULL,xafar, xbnear,xcfar, xdfar, xafarinfo, xbnearinfo,xcfarinfo, xdfarinfo, resmode,iaset,ibset,icset,idset);

            break;
        }

        case 1110:
        {
                   K4(res,ixa,iib,ixc,ixd,bias,altK,NULL,xanear,xbnear,xcnear,xdnear,xanearinfo,xbnearinfo,xcnearinfo,xdnearinfo,resmode,iaset,ibset,icset,idset);
            res -= K4(tmp,ixa,ixb,ixc,ixd,bias,altK,NULL,xanear,xbfar, xcnear,xdnear,xanearinfo,xbfarinfo, xcnearinfo,xdnearinfo,resmode,iaset,ibset,icset,idset);
            res -= K4(tmp,ixa,iib,iic,ixd,bias,altK,NULL,xanear,xbnear,xcfar, xdnear,xanearinfo,xbnearinfo,xcfarinfo, xdnearinfo,resmode,iaset,ibset,icset,idset);
            res += K4(tmp,ixa,ixb,iic,ixd,bias,altK,NULL,xanear,xbfar, xcfar, xdnear,xanearinfo,xbfarinfo, xcfarinfo, xdnearinfo,resmode,iaset,ibset,icset,idset);
            res -= K4(tmp,ixa,iib,ixc,iid,bias,altK,NULL,xanear,xbnear,xcnear,xdfar, xanearinfo,xbnearinfo,xcnearinfo,xdfarinfo, resmode,iaset,ibset,icset,idset);
            res += K4(tmp,ixa,ixb,ixc,iid,bias,altK,NULL,xanear,xbfar, xcnear,xdfar, xanearinfo,xbfarinfo, xcnearinfo,xdfarinfo, resmode,iaset,ibset,icset,idset);
            res += K4(tmp,ixa,iib,iic,iid,bias,altK,NULL,xanear,xbnear,xcfar, xdfar, xanearinfo,xbnearinfo,xcfarinfo, xdfarinfo, resmode,iaset,ibset,icset,idset);
            res -= K4(tmp,ixa,ixb,iic,iid,bias,altK,NULL,xanear,xbfar, xcfar, xdfar, xanearinfo,xbfarinfo, xcfarinfo, xdfarinfo, resmode,iaset,ibset,icset,idset);

            break;
        }

        case 1111:
        {
                   K4(res,ixa,ixb,ixc,ixd,bias,altK,NULL,xanear,xbnear,xcnear,xdnear,xanearinfo,xbnearinfo,xcnearinfo,xdnearinfo,resmode,iaset,ibset,icset,idset);
            res -= K4(tmp,iia,ixb,ixc,ixd,bias,altK,NULL,xafar, xbnear,xcnear,xdnear,xafarinfo, xbnearinfo,xcnearinfo,xdnearinfo,resmode,iaset,ibset,icset,idset);
            res -= K4(tmp,ixa,iib,ixc,ixd,bias,altK,NULL,xanear,xbfar, xcnear,xdnear,xanearinfo,xbfarinfo, xcnearinfo,xdnearinfo,resmode,iaset,ibset,icset,idset);
            res += K4(tmp,iia,iib,ixc,ixd,bias,altK,NULL,xafar, xbfar, xcnear,xdnear,xafarinfo, xbfarinfo, xcnearinfo,xdnearinfo,resmode,iaset,ibset,icset,idset);
            res -= K4(tmp,ixa,ixb,iic,ixd,bias,altK,NULL,xanear,xbnear,xcfar, xdnear,xanearinfo,xbnearinfo,xcfarinfo, xdnearinfo,resmode,iaset,ibset,icset,idset);
            res += K4(tmp,iia,ixb,iic,ixd,bias,altK,NULL,xafar, xbnear,xcfar, xdnear,xafarinfo, xbnearinfo,xcfarinfo, xdnearinfo,resmode,iaset,ibset,icset,idset);
            res += K4(tmp,ixa,iib,iic,ixd,bias,altK,NULL,xanear,xbfar, xcfar, xdnear,xanearinfo,xbfarinfo, xcfarinfo, xdnearinfo,resmode,iaset,ibset,icset,idset);
            res -= K4(tmp,iia,iib,iic,ixd,bias,altK,NULL,xafar, xbfar, xcfar, xdnear,xafarinfo, xbfarinfo, xcfarinfo, xdnearinfo,resmode,iaset,ibset,icset,idset);
            res -= K4(tmp,ixa,ixb,ixc,iid,bias,altK,NULL,xanear,xbnear,xcnear,xdfar, xanearinfo,xbnearinfo,xcnearinfo,xdfarinfo, resmode,iaset,ibset,icset,idset);
            res += K4(tmp,iia,ixb,ixc,iid,bias,altK,NULL,xafar, xbnear,xcnear,xdfar, xafarinfo, xbnearinfo,xcnearinfo,xdfarinfo, resmode,iaset,ibset,icset,idset);
            res += K4(tmp,ixa,iib,ixc,iid,bias,altK,NULL,xanear,xbfar, xcnear,xdfar, xanearinfo,xbfarinfo, xcnearinfo,xdfarinfo, resmode,iaset,ibset,icset,idset);
            res -= K4(tmp,iia,iib,ixc,iid,bias,altK,NULL,xafar, xbfar, xcnear,xdfar, xafarinfo, xbfarinfo, xcnearinfo,xdfarinfo, resmode,iaset,ibset,icset,idset);
            res += K4(tmp,ixa,ixb,iic,iid,bias,altK,NULL,xanear,xbnear,xcfar, xdfar, xanearinfo,xbnearinfo,xcfarinfo, xdfarinfo, resmode,iaset,ibset,icset,idset);
            res -= K4(tmp,iia,ixb,iic,iid,bias,altK,NULL,xafar, xbnear,xcfar, xdfar, xafarinfo, xbnearinfo,xcfarinfo, xdfarinfo, resmode,iaset,ibset,icset,idset);
            res -= K4(tmp,ixa,iib,iic,iid,bias,altK,NULL,xanear,xbfar, xcfar, xdfar, xanearinfo,xbfarinfo, xcfarinfo, xdfarinfo, resmode,iaset,ibset,icset,idset);
            res += K4(tmp,iia,iib,iic,iid,bias,altK,NULL,xafar, xbfar, xcfar, xdfar, xafarinfo, xbfarinfo, xcfarinfo, xdfarinfo, resmode,iaset,ibset,icset,idset);

            break;
        }

        default:
        {
            goto thunk;

            break;
        }
    }

    if ( iaokr || ibokr || icokr || idokr )
    {
        Vector<int> iiokr(4);
        Vector<int> iiok(4);
        Vector<int> iiset(4);
        Vector<const gentype *> xxalt(4);

        iiokr("&",0) = iaokr;
        iiok("&",0)  = iaok;
        iiset("&",0) = iaset;
        xxalt("&",0) = (*xa).isfarfarfarindpresent(3) ? &((*xa).fff(3)) : &nullgentype();

        iiokr("&",1) = ibokr;
        iiok("&",1)  = ibok;
        iiset("&",1) = ibset;
        xxalt("&",1) = (*xb).isfarfarfarindpresent(3) ? &((*xb).fff(3)) : &nullgentype();

        iiokr("&",2) = icokr;
        iiok("&",2)  = icok;
        iiset("&",2) = icset;
        xxalt("&",2) = (*xc).isfarfarfarindpresent(3) ? &((*xc).fff(3)) : &nullgentype();

        iiokr("&",3) = idokr;
        iiok("&",3)  = idok;
        iiset("&",4) = idset;
        xxalt("&",3) = (*xd).isfarfarfarindpresent(3) ? &((*xd).fff(3)) : &nullgentype();

        gentype UUres;

        (*UUcallback)(UUres,4,*this,iiokr,iiok,iiset,xxalt,defbasisUU);

        res *= (T) UUres;
    }

    if ( iaplanr || ibplanr || icplanr || idplanr )
    {
        Vector<int> iiplanr(4);
        Vector<int> iiplan(4);
        Vector<int> iiset(4);
        Vector<const gentype *> xxalt(4);

        iiplanr("&",0) = iaplanr;
        iiplan("&",0)  = iaplan;
        iiset("&",0)   = iaset;
        xxalt("&",0)   = (*xa).isfarfarfarindpresent(7) ? &((*xa).fff(7)) : &nullgentype();

        iiplanr("&",1) = ibplanr;
        iiplan("&",1)  = ibplan;
        iiset("&",1)   = ibset;
        xxalt("&",1)   = (*xb).isfarfarfarindpresent(7) ? &((*xb).fff(7)) : &nullgentype();

        iiplanr("&",2) = icplanr;
        iiplan("&",2)  = icplan;
        iiset("&",2)   = icset;
        xxalt("&",2)   = (*xc).isfarfarfarindpresent(7) ? &((*xc).fff(7)) : &nullgentype();

        iiplanr("&",3) = idplanr;
        iiplan("&",3)  = idplan;
        iiset("&",3)   = idset;
        xxalt("&",3)   = (*xd).isfarfarfarindpresent(7) ? &((*xd).fff(7)) : &nullgentype();

        gentype VVres;
        gentype kval(res);

        (*VVcallback)(VVres,4,kval,*this,iiplanr,iiplan,iiset,xxalt,defbasisVV);

        res = (T) VVres;
    }

    NiceAssert( !iaplanr );
    NiceAssert( !ibplanr );
    NiceAssert( !icplanr );
    NiceAssert( !idplanr );

    return res;

thunk:
    Vector<int> lpi(4);
    Vector<const SparseVector<gentype> *> lpxx(4);
    Vector<const vecInfo *> lpxxinfo(4);
    Vector<int> lpiset(4);

    lpi("&",z) = ia;
    lpi("&",1) = ib;
    lpi("&",2) = ic;
    lpi("&",3) = id;

    lpxx("&",z) = xa;
    lpxx("&",1) = xb;
    lpxx("&",2) = xc;
    lpxx("&",3) = xd;

    lpxxinfo("&",z) = xainfo;
    lpxxinfo("&",1) = xbinfo;
    lpxxinfo("&",2) = xcinfo;
    lpxxinfo("&",3) = xdinfo;

    lpiset("&",z) = iaset;
    lpiset("&",1) = ibset;
    lpiset("&",2) = icset;
    lpiset("&",3) = idset;

    return Km(4,res,lpi,bias,altK,pxyprod,&lpxx,&lpxxinfo,resmode,NULL,NULL,-1,-1,0,0,lpiset);
}


template <class T>
T &ML_Base::Km(int m, T &res, 
               Vector<int> &i, 
               const T &bias, const MercerKernel &altK, const gentype **pxyprod, 
               Vector<const SparseVector<gentype> *> *xxx, 
               Vector<const vecInfo *> *xxxinfo, 
               int resmode, const SparseVector<gentype> *agrad, const SparseVector<gentype> *bgrad, int agradind, int bgradind, int agradorder, int bgradorder, Vector<int> &iset) const
{
//phantomx
    NiceAssert( m >= 0 );
    NiceAssert( ( xxx && xxxinfo ) || ( !xxx && !xxxinfo ) );

    if ( !xxx )
    {
        MEMNEW(xxx,Vector<const SparseVector<gentype> *>(m));
        MEMNEW(xxxinfo,Vector<const vecInfo *>(m));

        int ii;

        for ( ii = 0 ; ii < m ; ii++ )
        {
            (*xxx)("&",ii) = NULL;
            (*xxxinfo)("&",ii) = NULL;
        }

        Km(m,res,i,bias,altK,pxyprod,xxx,xxxinfo,resmode,agrad,bgrad,agradind,bgradind,agradorder,bgradorder,iset);

        MEMDEL(xxx);
        MEMDEL(xxxinfo);

        return res;
    }

    // Make sure all info are present

    int needpreproc = 0;
    int ii,jj;
    int z = 0;
    int u;

    if ( m )
    {
        for ( ii = 0 ; ii < m ; ii++ )
        {
            const SparseVector<gentype> *xxi = (*xxx)(ii);

            NiceAssert( ( xxi && (*xxxinfo)(ii) ) || ( !xxi && !(*xxxinfo)(ii) ) );

            if ( !xxi || !((*xxi).isnofaroffindpresent()) || ( (*xxi).nearupsize() > 1 ) )
            {
                needpreproc = 1;

                break;
            }
        }
    }

    if ( !needpreproc )
    {
        // *Only* do any possible back-passing after *all* pre-processing is complete!

        if ( m == 0 )
        {
            NiceAssert( !agrad );
            NiceAssert( !bgrad );

            K0(res,bias,altK,pxyprod,resmode);
        }

        else if ( m == 1 )
        {
            NiceAssert( !agrad );
            NiceAssert( !bgrad );

            K1(res,i(z),bias,altK,pxyprod,(*xxx)(z),(*xxxinfo)(z),resmode,iset("&",z));
        }

        else if ( m == 2 )
        {
            if ( !agrad && !bgrad )
            {
                nonogradbase:

                K2(res,i(z),i(1),bias,altK,pxyprod,(*xxx)(z),(*xxx)(1),(*xxxinfo)(z),(*xxxinfo)(1),resmode,iset("&",z),iset("&",1));
            }

            else if ( !bgrad )
            {
                gradabase:

                int dummyind = -1;

                NiceAssert( !resmode );
                NiceAssert( agradind >= 0 );
                NiceAssert( agradind <= 1 );

                int lind = agradind;
                int rind = agradind ? 0 : 1;

                if ( agradorder == 0 )
                {
                    goto nonogradbase;
                }

                else if ( agradorder == 1 )
                {
                    // <ex,dK/dx> = <ex,x.ax + y.ay>
                    //            = <ex,x>.ax + <ex,y>.ay

                    T xscaleres; 
                    T yscaleres;

                    int dummyind = -1;

                    dK2delx(xscaleres,yscaleres,dummyind,i(lind),i(rind),bias,altK,pxyprod,(*xxx)(lind),(*xxx)(rind),(*xxxinfo)(lind),(*xxxinfo)(rind),iset("&",lind),iset("&",rind));

                    NiceAssert( dummyind < 0 );

                    gentype exx;
                    gentype exy;

                    twoProduct(exx,*agrad,*((*xxx)(lind)));
                    twoProduct(exy,*agrad,*((*xxx)(rind)));

                    res = (((T) exx)*xscaleres)+(((T) exy)*yscaleres);
                }

                else
                {
                    // <(ex d/dx)^n,K>

                    int xdm = xspaceDim();

                    res = 0.0;

                    Vector<T> sc;
                    Vector<Vector<int> > nn;

                    Vector<int> q(agradorder);

                    q = zeroint();

                    dnK2del(sc,nn,dummyind,q,i(lind),i(rind),bias,altK,NULL,(*xxx)(lind),(*xxx)(rind),(*xxxinfo)(lind),(*xxxinfo)(rind),iset("&",lind),iset("&",rind));

                    NiceAssert( dummyind < 0 );

                    int iqa,jqa;
                    gentype tmp;

                    SparseVector<gentype> kres;
                    Vector<const SparseVector<gentype> *> bord;

                    for ( iqa = 0 ; iqa < sc.size() ; iqa++ )
                    {
                        bord.resize(nn(iqa).size());

                        for ( jqa = 0 ; jqa < nn(iqa).size() ; jqa++ )
                        {
                            bord("&",jqa) = ( nn(iqa)(jqa) == 0 ) ? (*xxx)(lind) : ( ( nn(iqa)(jqa) == 1 ) ? (*xxx)(rind) : NULL );
                        }

                        kronprod(kres,bord,nn(iqa),xdm);

                        twoProduct(tmp,*agrad,kres);

                        res += ((T) tmp)*((T) sc(iqa));
                    }
                }
            }

            else
            {
                int dummyind = -1;

                NiceAssert( !resmode );
                NiceAssert( agradind >= 0 );
                NiceAssert( agradind <= 1 );
                NiceAssert( bgradind >= 0 );
                NiceAssert( bgradind <= 1 );

                int lind = agradind;
                int rind = bgradind;

                if ( ( agradorder == 0 ) && ( bgradorder == 0 ) )
                {
                    goto nonogradbase;
                }

                else if ( bgradorder == 0 )
                {
                    goto gradabase;
                }

                else if ( agradorder == 0 )
                {
                    qswap(agrad,bgrad);
                    qswap(agradind,bgradind);
                    qswap(agradorder,bgradorder);

                    goto gradabase;
                }

                else if ( ( agradorder == 1 ) && ( bgradorder == 1 ) )
                {
                    // <ex,d2K/dxdy,ey>

                    T xxscaleres;
                    T xyscaleres;
                    T yxscaleres;
                    T yyscaleres;

                    T constres;

                    d2K2delxdely(xxscaleres,yyscaleres,xyscaleres,yxscaleres,constres,dummyind,i(lind),i(rind),bias,altK,NULL,(*xxx)(lind),(*xxx)(rind),(*xxxinfo)(lind),(*xxxinfo)(rind),iset("&",lind),iset("&",rind));

                    NiceAssert( dummyind < 0 );

                    gentype exx;
                    gentype exy;
                    gentype xey;
                    gentype yey;

                    gentype exey;

                    twoProduct(exx,*agrad,*((*xxx)(lind)));
                    twoProduct(exy,*agrad,*((*xxx)(rind)));
                    twoProduct(xey,*((*xxx)(lind)),*bgrad);
                    twoProduct(yey,*((*xxx)(rind)),*bgrad);

                    twoProduct(exey,*agrad,*bgrad);

                    res = (((T) exx)*xxscaleres*((T) xey)) 
                        + (((T) exx)*xyscaleres*((T) yey)) 
                        + (((T) exy)*yxscaleres*((T) xey)) 
                        + (((T) exy)*yyscaleres*((T) yey)) 
                        + (constres*((T) exey));
                }

                else
                {
                    // <(ex d/dx)^n (ey d/dy)^n,K>

                    int xdm = xspaceDim();

                    res = 0.0;

                    Vector<T> sc;
                    Vector<Vector<int> > nn;

                    Vector<int> q(agradorder+bgradorder);

                    retVector<int> tmpva;

                    q("&",0         ,1,agradorder-1           ,tmpva) = 0;
                    q("&",agradorder,1,agradorder+bgradorder-1,tmpva) = 1;

                    dnK2del(sc,nn,dummyind,q,i(lind),i(rind),bias,altK,NULL,(*xxx)(lind),(*xxx)(rind),(*xxxinfo)(lind),(*xxxinfo)(rind),iset("&",lind),iset("&",rind));

                    NiceAssert( dummyind < 0 );

                    // Here we use the result: vec(ABC) = vec(C' \otimes A).vec(B), so if
                    // we let a = xifarfar, c = xjfarfar, then:
                    // a'.B.c = vec(c' \otimes a').vec(B)
                    //        = vec(c \otimes a)'.vec(B)
                    // where vec(B) is the vectorised *transpose* gradient we usually calculate, so, getting rid of the transpose,
                    // we need to take the inner product with vec(ex \otimes ey)
                    // ORIGINAL INCORRECT VERSION: where vec(B) is precisely the vectorised gradient we usually calculate
                    //
                    // CLARIFICATION: see pdf in stable bayesian optimisation paper

                    int iqa,jqa;
                    gentype tmp;

                    SparseVector<gentype> kres;
                    SparseVector<gentype> farfarprod;
                    Vector<const SparseVector<gentype> *> bord;

                    // first calculate c \otimes a

                    int dimmy = 0;

                    //kronprod(farfarprod,*agrad,*bgrad,pow(xdm,bgradorder),pow(xdm,agradorder)); //- NB order wrong here, have tested and confirmed, following line is correct.
                    kronprod(farfarprod,dimmy,*agrad,*bgrad,(int) pow(xdm,agradorder),(int) pow(xdm,bgradorder));

                    // Then proceed in fully vectorised form

                    for ( iqa = 0 ; iqa < sc.size() ; iqa++ )
                    {
                        bord.resize(nn(iqa).size());

                        for ( jqa = 0 ; jqa < nn(iqa).size() ; jqa++ )
                        {
                            bord("&",jqa) = ( nn(iqa)(jqa) == 0 ) ? (*xxx)(lind) : ( ( nn(iqa)(jqa) == 1 ) ? (*xxx)(rind) : NULL );
                        }

                        kronprod(kres,bord,nn(iqa),xdm);

                        twoProduct(tmp,farfarprod,kres);

                        res += ((T) tmp)*((T) sc(iqa));
                    }
                }
            }
        }

        else if ( m == 3 )
        {
            NiceAssert( !agrad );
            NiceAssert( !bgrad );

            K3(res,i(z),i(1),i(2),bias,altK,pxyprod,(*xxx)(z),(*xxx)(1),(*xxx)(2),(*xxxinfo)(z),(*xxxinfo)(1),(*xxxinfo)(2),resmode,iset("&",z),iset("&",1),iset("&",2));
        }

        else if ( m == 4 )
        {
            NiceAssert( !agrad );
            NiceAssert( !bgrad );

            K4(res,i(z),i(1),i(2),i(3),bias,altK,pxyprod,(*xxx)(z),(*xxx)(1),(*xxx)(2),(*xxx)(3),(*xxxinfo)(z),(*xxxinfo)(1),(*xxxinfo)(2),(*xxxinfo)(3),resmode,iset("&",z),iset("&",1),iset("&",2),iset("&",3));
        }

        else
        {
            NiceAssert( !agrad );
            NiceAssert( !bgrad );

//            for ( ii = 0 ; ii < m ; ii++ )
//            {
//                if ( ( i(ii) >= 0 ) && ( !(d()(ii)) ) )
//                {
//                    break;
//                }
//            }
//
//            if ( ii < m )
//            {
//                res = 0.0;
//            }
//
//            else
            {
                if ( altK.suggestXYcache() && getxymat() && ( i >= z ) )
                {
                    const Matrix<double> &xy = *getxymat();

                    retMatrix<double> tmpma;

                    altK.Km(m,res,*xxx,*xxxinfo,bias,i,pxyprod,xspaceDim(),isXConsistent() && istrv(i),resmode,MLid(),&(xy(i,i,tmpma)),&iset,assumeReal);
                }

                else
                {
                    altK.Km(m,res,*xxx,*xxxinfo,bias,i,pxyprod,xspaceDim(),isXConsistent() && istrv(i),resmode,MLid(),NULL,&iset,assumeReal);
                }
            }
        }
    }

    else
    {
        Vector<int> j(i);
        Vector<const SparseVector<gentype> *> xx(*xxx);
        Vector<const vecInfo *> xxinfo(*xxxinfo);
        Vector<int> jset(iset);

        // Fill in any missing data vectors

        for ( ii = m-1 ; ii >= 0 ; ii-- )
        {
            if ( !xx(ii) )
            {
                xx("&",ii)     = &xgetloc(j(ii));
                xxinfo("&",ii) = &xinfo(j(ii));
            }
        }

        Vector<int> iokr(m);
        Vector<int> iok(m);

        Vector<int> iplanr(m);
        Vector<int> iplan(m);

        Vector<int> iset(m);

        iokr = z;
        iok  = z;

        iplanr = z;
        iplan  = z;

        iset = z;

        int indiagr = 0;
        int xdiagr  = 0;

        T tmp;

        for ( ii = 0 ; ii < m ; ii++ )
        {
            if ( !((*(xx(ii))).isnofaroffindpresent()) )
            {
                // Process rank constraints

                // Note that we process from 0 to m, in order.  Thus once
                // agradind and bgradind are set they need not be adjusted
                // as nothing before them will change.

                const SparseVector<gentype> *xanear   = NULL;
                const SparseVector<gentype> *xafar    = NULL;
                const SparseVector<gentype> *xafarfar = NULL;

                const vecInfo *xanearinfo = NULL;
                const vecInfo *xafarinfo  = NULL;

                int ixa,iia,adiagr,aagradOrder,iiaset;

                const gentype *ixatup = NULL;
                const gentype *iiatup = NULL;

                int iamethodkey = detangle_x(xanear,xafar,xafarfar,xanearinfo,xafarinfo,ixa,iia,ixatup,iiatup,iokr("&",ii),iok("&",ii),j(ii),adiagr,xx(ii),xxinfo(ii),aagradOrder,iplanr("&",ii),iplan("&",ii),iiaset) & 7;

                NiceAssert( !( iamethodkey & 4 ) );

                if ( !ii && adiagr && ( ixa >= 0 ) )
                {
                    indiagr = 1;
                    xdiagr  = ixa;
                    res     = (T) (*xanear).fff(4);
                }

                else if ( ii && indiagr && adiagr && ( ixa == xdiagr ) )
                {
                    res *= (T) (*xanear).fff(4);
                }

                else if ( ( !ii && adiagr ) || ( ii && ( indiagr || adiagr ) ) )
                {
                    res = 0.0;

                    return res;
                }

                if ( !indiagr && ( iamethodkey == 0 ) )
                {
                    // no rank, no gradient

                    if ( ixatup )
                    {
                        Vector<int> &lpj = j;
                        Vector<const SparseVector<gentype> *> &lpxx = xx;
                        Vector<const vecInfo *> &lpxxinfo = xxinfo;
                        Vector<int> &lpjset = jset;

                        const Vector<gentype> &iii = (const Vector<gentype> &) (*ixatup);

                        int vi,vp;

                        m--;

                        lpj.remove(ii);
                        lpxx.remove(ii);
                        lpxxinfo.remove(ii);
                        lpjset.remove(ii);

                        if ( iii.size() )
                        {
                            for ( vi = 0 ; vi < iii.size() ; vi++ )
                            {
                                vp = lpj.size();

                                lpj.add(vp);
                                lpxx.add(vp);
                                lpxxinfo.add(vp);
                                lpjset.add(vp);

                                lpj("&",vp) = (int) iii(vi);
                                lpxx("&",vp) = &(xgetloc(lpj(vp)));
                                lpxxinfo("&",vp) = &(((xinfo()(lpj(vp))))(-1,-1,iii.size()));
                                lpjset("&",vp) = iiaset;

                                m++;
                            }
                        }

                        // No need to return yet, overwrite suffices
                    }

                    else
                    {
                        j("&",ii) = ixa;
                        xx("&",ii) = xanear;
                        xxinfo("&",ii) = xanearinfo;
                        jset("&",ii) = iiaset;

                        // No need to return yet, overwrite suffices
                    }
                }

                else if ( !indiagr && ( iamethodkey == 1 ) )
                {
                    // rank, no gradient.

                    if ( ixatup && iiatup )
                    {
                        Vector<int> &lpj = j;
                        Vector<const SparseVector<gentype> *> &lpxx = xx;
                        Vector<const vecInfo *> &lpxxinfo = xxinfo;
                        Vector<int> &lpjset = jset;

                        Vector<int> rpj(j);
                        Vector<const SparseVector<gentype> *> rpxx(xx);
                        Vector<const vecInfo *> rpxxinfo(xxinfo);
                        Vector<int> rpjset(jset);

                        int lm = m;
                        int rm = m;

                        const Vector<gentype> &iii = (const Vector<gentype> &) (*ixatup);
                        const Vector<gentype> &jjj = (const Vector<gentype> &) (*iiatup);

                        int vi,vj,vp;

                        lm--;

                        lpj.remove(ii);
                        lpxx.remove(ii);
                        lpxxinfo.remove(ii);
                        lpjset.remove(ii);

                        rm--;
 
                        rpj.remove(ii);
                        rpxx.remove(ii);
                        rpxxinfo.remove(ii);
                        rpjset.remove(ii);

                        if ( iii.size() )
                        {
                            for ( vi = 0 ; vi < iii.size() ; vi++ )
                            {
                                vp = lpj.size();

                                lpj.add(vp);
                                lpxx.add(vp);
                                lpxxinfo.add(vp);
                                lpjset.add(vp);

                                lpj("&",vp) = (int) iii(vi);
                                lpxx("&",vp) = &(xgetloc(lpj(vp)));
                                lpxxinfo("&",vp) = &(((xinfo()(lpj(vp))))(-1,-1,iii.size()));
                                lpjset("&",vp) = iiaset;

                                lm++;
                            }
                        }

                        if ( jjj.size() )
                        {
                            for ( vj = 0 ; vj < jjj.size() ; vj++ )
                            {
                                vp = rpj.size();

                                rpj.add(vp);
                                rpxx.add(vp);
                                rpxxinfo.add(vp);
                                rpjset.add(vp);

                                rpj("&",vp) = (int) jjj(vj);
                                rpxx("&",vp) = &(xgetloc(rpj(vp)));
                                rpxxinfo("&",vp) = &(((xinfo()(rpj(vp))))(-1,-1,jjj.size()));
                                rpjset("&",vp) = iiaset;

                                rm++;
                            }
                        }

                               Km(lm,res,lpj,bias,altK,NULL,&lpxx,&lpxxinfo,resmode,agrad,bgrad,agradind,bgradind,agradorder,bgradorder,lpjset); 
                        res -= Km(rm,tmp,rpj,bias,altK,NULL,&rpxx,&rpxxinfo,resmode,agrad,bgrad,agradind,bgradind,agradorder,bgradorder,lpjset); 
                    }

                    else if ( ixatup )
                    {
                        Vector<int> &lpj = j;
                        Vector<const SparseVector<gentype> *> &lpxx = xx;
                        Vector<const vecInfo *> &lpxxinfo = xxinfo;
                        Vector<int> &lpjset = jset;

                        Vector<int> rpj(j);
                        Vector<const SparseVector<gentype> *> rpxx(xx);
                        Vector<const vecInfo *> rpxxinfo(xxinfo);
                        Vector<int> rpjset(jset);

                        int lm = m;
                        int rm = m;

                        const Vector<gentype> &iii = (const Vector<gentype> &) (*ixatup);

                        int vi,vp;

                        lm--;

                        lpj.remove(ii);
                        lpxx.remove(ii);
                        lpxxinfo.remove(ii);
                        lpjset.remove(ii);

                        if ( iii.size() )
                        {
                            for ( vi = 0 ; vi < iii.size() ; vi++ )
                            {
                                vp = lpj.size();

                                lpj.add(vp);
                                lpxx.add(vp);
                                lpxxinfo.add(vp);
                                lpjset.add(vp);

                                lpj("&",vp) = (int) iii(vi);
                                lpxx("&",vp) = &(xgetloc(lpj(vp)));
                                lpxxinfo("&",vp) = &(((xinfo()(lpj(vp))))(-1,-1,iii.size()));
                                lpjset("&",vp) = iiaset;

                                lm++;
                            }
                        }

                        rpj("&",ii) = iia;
                        rpxx("&",ii) = xafar;
                        rpxxinfo("&",ii) = xafarinfo;
                        rpjset("&",ii) = iiaset;

                               Km(lm,res,lpj,bias,altK,NULL,&lpxx,&lpxxinfo,resmode,agrad,bgrad,agradind,bgradind,agradorder,bgradorder,lpjset); 
                        res -= Km(rm,tmp,rpj,bias,altK,NULL,&rpxx,&rpxxinfo,resmode,agrad,bgrad,agradind,bgradind,agradorder,bgradorder,lpjset); 
                    }

                    else if ( iiatup )
                    {
                        Vector<int> &lpj = j;
                        Vector<const SparseVector<gentype> *> &lpxx = xx;
                        Vector<const vecInfo *> &lpxxinfo = xxinfo;
                        Vector<int> &lpjset = jset;

                        Vector<int> rpj(j);
                        Vector<const SparseVector<gentype> *> rpxx(xx);
                        Vector<const vecInfo *> rpxxinfo(xxinfo);
                        Vector<int> rpjset(jset);

                        int lm = m;
                        int rm = m;

                        const Vector<gentype> &jjj = (const Vector<gentype> &) (*iiatup);

                        int vj,vp;

                        rm--;
 
                        rpj.remove(ii);
                        rpxx.remove(ii);
                        rpxxinfo.remove(ii);
                        rpjset.remove(ii);

                        lpj("&",ii) = ixa;
                        lpxx("&",ii) = xanear;
                        lpxxinfo("&",ii) = xanearinfo;
                        lpjset("&",ii) = iiaset;

                        if ( jjj.size() )
                        {
                            for ( vj = 0 ; vj < jjj.size() ; vj++ )
                            {
                                vp = rpj.size();

                                rpj.add(vp);
                                rpxx.add(vp);
                                rpxxinfo.add(vp);
                                rpjset.add(vp);

                                rpj("&",vp) = (int) jjj(vj);
                                rpxx("&",vp) = &(xgetloc(rpj(vp)));
                                rpxxinfo("&",vp) = &(((xinfo()(rpj(vp))))(-1,-1,jjj.size()));
                                rpjset("&",vp) = iiaset;

                                rm++;
                            }
                        }

                               Km(lm,res,lpj,bias,altK,NULL,&lpxx,&lpxxinfo,resmode,agrad,bgrad,agradind,bgradind,agradorder,bgradorder,lpjset); 
                        res -= Km(rm,tmp,rpj,bias,altK,NULL,&rpxx,&rpxxinfo,resmode,agrad,bgrad,agradind,bgradind,agradorder,bgradorder,lpjset); 
                    }

                    else
                    {
                        Vector<int> &lpj = j;
                        Vector<const SparseVector<gentype> *> &lpxx = xx;
                        Vector<const vecInfo *> &lpxxinfo = xxinfo;
                        Vector<int> &lpjset = jset;

                        Vector<int> rpj(j);
                        Vector<const SparseVector<gentype> *> rpxx(xx);
                        Vector<const vecInfo *> rpxxinfo(xxinfo);
                        Vector<int> rpjset(jset);

                        lpj("&",ii) = ixa;
                        lpxx("&",ii) = xanear;
                        lpxxinfo("&",ii) = xanearinfo;
                        lpjset("&",ii) = iiaset;

                        rpj("&",ii) = iia;
                        rpxx("&",ii) = xafar;
                        rpxxinfo("&",ii) = xafarinfo;
                        rpjset("&",ii) = iiaset;

                               Km(m,res,lpj,bias,altK,NULL,&lpxx,&lpxxinfo,resmode,agrad,bgrad,agradind,bgradind,agradorder,bgradorder,lpjset); 
                        res -= Km(m,tmp,rpj,bias,altK,NULL,&rpxx,&rpxxinfo,resmode,agrad,bgrad,agradind,bgradind,agradorder,bgradorder,lpjset); 
                    }

                    return res;
                }

                else if ( !indiagr && ( iamethodkey == 2 ) )
                {
                    // Gradient, no rank

                    NiceAssert( !ixatup );
                    NiceAssert( (*xanear).nearupsize() == 1 );

                    j("&",ii) = ixa;
                    xx("&",ii) = xanear;
                    xxinfo("&",ii) = xanearinfo;
                    jset("&",ii) = iiaset;

                    if ( !agrad )
                    {
                        agrad      = xafarfar;
                        agradind   = ii;
                        agradorder = aagradOrder;
                    }

                    else if ( !bgrad )
                    {
                        bgrad      = xafarfar;
                        bgradind   = ii;
                        bgradorder = aagradOrder;
                    }

                    else
                    {
                        throw("Third or higher derivatives not implemented on Km");
                    }

                    return Km(m,res,j,bias,altK,NULL,&xx,&xxinfo,resmode,agrad,bgrad,agradind,bgradind,agradorder,bgradorder,jset);
                }

                else if ( !indiagr && ( iamethodkey == 3 ) )
                {
                    throw("Km cannot combine rank and gradient constraints on a single variable");
                }

                else if ( !indiagr )
                {
                    throw("Km asks: what is this detangle mode?\n");
                }
            }

            if ( !indiagr && ( u = (*(xx(ii))).nearupsize() ) > 1 )
            {
                const SparseVector<gentype> &xxv = (*(xx(ii)));
                const vecInfo &xxvinfo = (*(xxinfo(ii)));

                for ( jj = 0 ; jj < u ; jj++ )
                {
                    if ( jj )
                    {
                        m++;

                        j.add(ii+jj);
                        xx.add(ii+jj);
                        xxinfo.add(ii+jj);
                        jset.add(ii+jj);
                    }

                    j("&",ii+jj) = jj ? UPNTVI(j(ii+jj),jj) : j(ii+jj);
                    xx("&",ii+jj) =  &(xxv.nearrefup(jj));
                    xxinfo("&",ii+jj) = &(xxvinfo(ii+jj)(-1,jj));
                    jset("&",ii+jj) = jset(ii+jj);
                }
            }
        }

        if ( !indiagr )
        {
            Km(m,res,j,bias,altK,NULL,&xx,&xxinfo,resmode,agrad,bgrad,agradind,bgradind,agradorder,bgradorder,jset);

            if ( sum(iokr) )
            {
                Vector<const gentype *> xxalt(m);

                for ( jj = 0 ; jj < m ; jj++ )
                {
                    xxalt("&",jj) = (*(xx(jj))).isfarfarfarindpresent(3) ? &((*(xx(jj))).fff(3)) : &nullgentype();
                }

                gentype UUres;

                (*UUcallback)(UUres,m,*this,iokr,iok,iset,xxalt,defbasisUU);

                res *= (T) UUres;
            }

            if ( sum(iplanr) )
            {
                Vector<const gentype *> xxalt(m);

                for ( jj = 0 ; jj < m ; jj++ )
                {
                    xxalt("&",jj) = (*(xx(jj))).isfarfarfarindpresent(7) ? &((*(xx(jj))).fff(7)) : &nullgentype();
                }

                gentype VVres;
                gentype kval(res);

                (*VVcallback)(VVres,m,kval,*this,iplanr,iplan,iset,xxalt,defbasisVV);

                res = (T) VVres;
            }
        }
    }

    return res;
}

double &ML_Base::KK0ip(double &res, const double &bias, const gentype **pxyprod) const
{
//phantomx
    // Shortcut for speed

    getKernel().K0ip(res,bias,pxyprod,0,0,MLid(),assumeReal);

    return res;
}

double &ML_Base::KK2ip(double &res, int ib, int jb, const double &bias, const gentype **pxyprod, const SparseVector<gentype> *xx, const SparseVector<gentype> *yy, const vecInfo *xxinfo, const vecInfo *yyinfo, int &ibset, int &jbset) const
{
//phantomx
    NiceAssert( ( xx && xxinfo ) || ( !xx && !xxinfo ) );
    NiceAssert( ( yy && yyinfo ) || ( !yy && !yyinfo ) );

    // Shortcut for speed

    if ( xx && yy )
    {
        const SparseVector<gentype> &xxx  = (*xx).nearref();
        const SparseVector<gentype> &xxxx = xxx.nearrefup(0);

        const SparseVector<gentype> &yyy  = (*yy).nearref();
        const SparseVector<gentype> &yyyy = yyy.nearrefup(0);

        getKernel().K2ip(res,xxxx,yyyy,*xxinfo,*yyinfo,bias,pxyprod,ib,jb,0,0,MLid(),ibset,jbset,assumeReal);
    }

    else if ( xx && !yy )
    {
        const SparseVector<gentype> &xjb = xgetloc(jb);

        const vecInfo &xinfoj = xinfo(jb);

        KK2ip(res,ib,jb,bias,pxyprod,xx,&xjb,xxinfo,&xinfoj,ibset,jbset);
    }

    else if ( !xx && yy )
    {
        const SparseVector<gentype> &xib = xgetloc(ib);

        const vecInfo &xinfoi = xinfo(ib);

        KK2ip(res,ib,jb,bias,pxyprod,&xib,yy,&xinfoi,yyinfo,ibset,jbset);
    }

    else
    {
        const SparseVector<gentype> &xib = xgetloc(ib);
        const SparseVector<gentype> &xjb = xgetloc(jb);

        const vecInfo &xinfoi = xinfo(ib);
        const vecInfo &xinfoj = xinfo(jb);

        KK2ip(res,ib,jb,bias,pxyprod,&xib,&xjb,&xinfoi,&xinfoj,ibset,jbset);
    }

    return res;
}

double &ML_Base::KK4ip(double &res, int ia, int ib, int ic, int id, const double &bias, const gentype **pxyprod, const SparseVector<gentype> *xa, const SparseVector<gentype> *xb, const SparseVector<gentype> *xc, const SparseVector<gentype> *xd, const vecInfo *xainfo, const vecInfo *xbinfo, const vecInfo *xcinfo, const vecInfo *xdinfo, int &iaset, int &ibset, int &icset, int &idset) const
{
//phantomx
    NiceAssert( ( xa && xainfo ) || ( !xa && ( !xainfo || ( !xb && xc && xd ) ) ) );
    NiceAssert( ( xb && xbinfo ) || ( !xb && ( !xbinfo || ( !xa && xc && xd ) ) ) );
    NiceAssert( ( xc && xcinfo ) || ( !xc && !xcinfo ) );
    NiceAssert( ( xd && xdinfo ) || ( !xd && !xdinfo ) );

    // Always simplify

         if ( xb && !xa ) { return KK4ip(res,ib,ia,ic,id,bias,pxyprod,xb,xa,xc,xd,xbinfo,xainfo,xcinfo,xdinfo,ibset,iaset,icset,idset); }
    else if ( xc && !xa ) { return KK4ip(res,ic,ib,ia,id,bias,pxyprod,xc,xb,xa,xd,xcinfo,xbinfo,xainfo,xdinfo,icset,ibset,iaset,idset); }
    else if ( xc && !xb ) { return KK4ip(res,ia,ic,ib,id,bias,pxyprod,xa,xc,xb,xd,xainfo,xcinfo,xbinfo,xdinfo,iaset,icset,ibset,idset); }
    else if ( xd && !xa ) { return KK4ip(res,id,ib,ic,ia,bias,pxyprod,xd,xb,xc,xa,xdinfo,xbinfo,xcinfo,xainfo,idset,ibset,icset,iaset); }
    else if ( xd && !xb ) { return KK4ip(res,ia,id,ic,ib,bias,pxyprod,xa,xd,xc,xb,xainfo,xdinfo,xcinfo,xbinfo,iaset,idset,icset,ibset); }
    else if ( xd && !xc ) { return KK4ip(res,ia,ib,id,ic,bias,pxyprod,xa,xb,xd,xc,xainfo,xbinfo,xdinfo,xcinfo,iaset,ibset,idset,icset); }

    // Shortcut for speed

    if ( xa && xb && xc && xd )
    {
        const SparseVector<gentype> &xxa  = (*xa).nearref();
        const SparseVector<gentype> &xxxa = xxa.nearrefup(0);

        const SparseVector<gentype> &xxb  = (*xb).nearref();
        const SparseVector<gentype> &xxxb = xxb.nearrefup(0);

        const SparseVector<gentype> &xxc  = (*xc).nearref();
        const SparseVector<gentype> &xxxc = xxc.nearrefup(0);

        const SparseVector<gentype> &xxd  = (*xd).nearref();
        const SparseVector<gentype> &xxxd = xxd.nearrefup(0);

        getKernel().K4ip(res,xxxa,xxxb,xxxc,xxxd,*xainfo,*xbinfo,*xcinfo,*xdinfo,bias,pxyprod,ia,ib,ic,id,0,0,MLid(),iaset,ibset,icset,idset,assumeReal);
    }

    else if ( xa && xb && xc )
    {
        const SparseVector<gentype> &xdb = xgetloc(id);

        const vecInfo &xinfod = xinfo(id);

        KK4ip(res,ia,ib,ic,id,bias,pxyprod,xa,xb,xc,&xdb,xainfo,xbinfo,xcinfo,&xinfod,iaset,ibset,icset,idset);
    }

    else if ( xa && xb )
    {
        const SparseVector<gentype> &xcb = xgetloc(ic);
        const SparseVector<gentype> &xdb = xgetloc(id);

        const vecInfo &xinfoc = xinfo(ic);
        const vecInfo &xinfod = xinfo(id);

        KK4ip(res,ia,ib,ic,id,bias,pxyprod,xa,xb,&xcb,&xdb,xainfo,xbinfo,&xinfoc,&xinfod,iaset,ibset,icset,idset);
    }

    else if ( xa && xa->isnofaroffindpresent() )
    {
        const SparseVector<gentype> &xbb = xgetloc(ib);
        const SparseVector<gentype> &xcb = xgetloc(ic);
        const SparseVector<gentype> &xdb = xgetloc(id);

        const vecInfo &xinfob = xinfo(ib);
        const vecInfo &xinfoc = xinfo(ic);
        const vecInfo &xinfod = xinfo(id);

        KK4ip(res,ia,ib,ic,id,bias,pxyprod,xa,&xbb,&xcb,&xdb,xainfo,&xinfob,&xinfoc,&xinfod,iaset,ibset,icset,idset);
    }

    else
    {
        const SparseVector<gentype> &xab = xgetloc(ia);
        const SparseVector<gentype> &xbb = xgetloc(ib);
        const SparseVector<gentype> &xcb = xgetloc(ic);
        const SparseVector<gentype> &xdb = xgetloc(id);

        const vecInfo &xinfoa = xinfo(ia);
        const vecInfo &xinfob = xinfo(ib);
        const vecInfo &xinfoc = xinfo(ic);
        const vecInfo &xinfod = xinfo(id);

        KK4ip(res,ia,ib,ic,id,bias,pxyprod,&xab,&xbb,&xcb,&xdb,&xinfoa,&xinfob,&xinfoc,&xinfod,iaset,ibset,icset,idset);
    }

    return res;
}

double &ML_Base::KKmip(int m, double &res, Vector<int> &i, const double &bias, const gentype **pxyprod, Vector<const SparseVector<gentype> *> *xxx, Vector<const vecInfo *> *xxxinfo, Vector<int> &iset) const
{
//phantomx
    NiceAssert( m >= 0 );

    // Make sure all info are present

    int ii;

    Vector<const SparseVector<gentype> *> xx(*xxx);
    Vector<const vecInfo *> xxinfo(*xxxinfo);

    for ( ii = m-1 ; ii >= 0 ; ii-- )
    {
        if ( !xx(ii) )
        {
            // Fill in data

            xx("&",ii)     = &xgetloc(i(ii));
            xxinfo("&",ii) = &xinfo(i(ii));

            // Simplify data

            xx("&",ii) = &((*(xx(ii))).nearref());
            xx("&",ii) = &((*(xx(ii))).nearrefup(0));
        }
    }

    getKernel().Kmip(m,res,xx,xxinfo,i,bias,pxyprod,xspaceDim(),isXConsistent() && istrv(i),MLid(),&iset,assumeReal);

    return res;
}

template <class T>
void ML_Base::dK(T &xygrad, T &xnormgrad, 
                 int ib, int jb, 
                 const T &bias, const MercerKernel &altK, const gentype **pxyprod, 
                 const SparseVector<gentype> *xx, const SparseVector<gentype> *yy, 
                 const vecInfo *xxinfo, const vecInfo *yyinfo, 
                 int deepDeriv, int &iaset, int &ibset) const
{
//phantomx
    NiceAssert( ( xx && xxinfo ) || ( !xx && !xxinfo ) );
    NiceAssert( ( yy && yyinfo ) || ( !yy && !yyinfo ) );

    // Shortcut for speed

    int dummyind = -1;

    if ( xx && yy &&  xx->isnofaroffindpresent() && yy->isnofaroffindpresent() )
    {
        NiceAssert( (*xx).nearupsize() == 1 );
        NiceAssert( (*yy).nearupsize() == 1 );

        NiceAssert( (*xx).farupsize() == 1 );
        NiceAssert( (*yy).farupsize() == 1 );

        if ( altK.suggestXYcache() && getxymat() && ( ib >= 0 ) && ( jb >= 0 ) )
        {
            const Matrix<double> &xy = *getxymat();
            Vector<int> xyderef(2);

            xyderef("&",zeroint()) = ib;
            xyderef("&",1) = jb;

            retMatrix<double> tmpma;

            altK.dK(xygrad,xnormgrad,dummyind,*xx,*yy,*xxinfo,*yyinfo,bias,pxyprod,ib,jb,xspaceDim(),isXConsistent() && istrv(ib) && istrv(jb),MLid(),&(xy(xyderef,xyderef,tmpma)),deepDeriv,iaset,ibset,assumeReal);
        }

        else
        {
            altK.dK(xygrad,xnormgrad,dummyind,*xx,*yy,*xxinfo,*yyinfo,bias,pxyprod,ib,jb,xspaceDim(),isXConsistent() && istrv(ib) && istrv(jb),MLid(),NULL,deepDeriv,iaset,ibset,assumeReal);
        }

        NiceAssert( dummyind < 0 );

        return;
    }

    else if ( xx && !yy )
    {
        const SparseVector<gentype> &xjb = xgetloc(jb);

        const vecInfo &xinfoj = xinfo(jb);

        dK(xygrad,xnormgrad,ib,jb,bias,altK,pxyprod,xx,&xjb,xxinfo,&xinfoj,deepDeriv,iaset,ibset);

        return;
    }

    else if ( !xx && yy )
    {
        const SparseVector<gentype> &xib = xgetloc(ib);

        const vecInfo &xinfoi = xinfo(ib);

        dK(xygrad,xnormgrad,ib,jb,bias,altK,pxyprod,&xib,yy,&xinfoi,yyinfo,deepDeriv,iaset,ibset);

        return;
    }

    else if ( !xx && !yy )
    {
        const SparseVector<gentype> &xib = xgetloc(ib);
        const SparseVector<gentype> &xjb = xgetloc(jb);

        const vecInfo &xinfoi = xinfo(ib);
        const vecInfo &xinfoj = xinfo(jb);

        dK(xygrad,xnormgrad,ib,jb,bias,altK,pxyprod,&xib,&xjb,&xinfoi,&xinfoj,deepDeriv,iaset,ibset);

        return;
    }

    const SparseVector<gentype> *xinear = NULL;
    const SparseVector<gentype> *xjnear = NULL;

    const SparseVector<gentype> *xifar = NULL;
    const SparseVector<gentype> *xjfar = NULL;

    const SparseVector<gentype> *xifarfar = NULL;
    const SparseVector<gentype> *xjfarfar = NULL;

    const vecInfo *xinearinfo = NULL;
    const vecInfo *xjnearinfo = NULL;

    const vecInfo *xifarinfo = NULL;
    const vecInfo *xjfarinfo = NULL;

    int i,ii,iokr,iok,idiagr,igradOrder,iplanr,iplan;
    int j,jj,jokr,jok,jdiagr,jgradOrder,jplanr,jplan;

    const gentype *itup = NULL;
    const gentype *iitup = NULL;

    const gentype *jtup = NULL;
    const gentype *jjtup = NULL;

    int imethodkey = detangle_x(xinear,xifar,xifarfar,xinearinfo,xifarinfo,i,ii,itup,iitup,iokr,iok,ib,idiagr,xx,xxinfo,igradOrder,iplanr,iplan,iaset) & 7;
    int jmethodkey = detangle_x(xjnear,xjfar,xjfarfar,xjnearinfo,xjfarinfo,j,jj,jtup,jjtup,jokr,jok,jb,jdiagr,yy,yyinfo,jgradOrder,jplanr,jplan,ibset) & 7;

    if ( idiagr || jdiagr )
    {
        xygrad    = 0.0;
        xnormgrad = 0.0;

        return;
    }

    NiceAssert( !itup && !iitup );
    NiceAssert( !jtup && !jjtup );

    NiceAssert( !imethodkey ); (void) imethodkey;
    NiceAssert( !jmethodkey ); (void) jmethodkey;

    dK(xygrad,xnormgrad,i, j, bias,altK,NULL,xinear,xjnear,xinearinfo,xjnearinfo,deepDeriv,iaset,ibset);

    if ( iokr || jokr )
    {
        Vector<int> iiokr(2);
        Vector<int> iiok(2);
        Vector<int> iiset(2);
        Vector<const gentype *> xxalt(2);

        iiokr("&",0) = iokr;
        iiok("&",0)  = iok;
        iiset("&",0) = iaset;
        xxalt("&",0) = (*xx).isfarfarfarindpresent(3) ? &((*xx).fff(3)) : &nullgentype();

        iiokr("&",1) = jokr;
        iiok("&",1)  = jok;
        iiset("&",1) = ibset;
        xxalt("&",1) = (*yy).isfarfarfarindpresent(3) ? &((*yy).fff(3)) : &nullgentype();

        gentype UUres;

        (*UUcallback)(UUres,2,*this,iiokr,iiok,iiset,xxalt,defbasisUU);

        xygrad    *= (T) UUres;
        xnormgrad *= (T) UUres;
    }

    if ( iplanr || jplanr )
    {
        Vector<int> iiplanr(2);
        Vector<int> iiplan(2);
        Vector<int> iiset(2);
        Vector<const gentype *> xxalt(2);

        iiplanr("&",0) = iplanr;
        iiplan("&",0)  = iplan;
        iiset("&",0)   = iaset;
        xxalt("&",0)   = (*xx).isfarfarfarindpresent(7) ? &((*xx).fff(7)) : &nullgentype();

        iiplanr("&",1) = jplanr;
        iiplan("&",1)  = jplan;
        iiset("&",1)   = ibset;
        xxalt("&",1)   = (*yy).isfarfarfarindpresent(7) ? &((*yy).fff(7)) : &nullgentype();

        gentype VVres;
        gentype kval(1.0);

        (*VVcallback)(VVres,2,kval,*this,iiplanr,iiplan,iiset,xxalt,defbasisVV);

        xygrad    *= (T) VVres;
        xnormgrad *= (T) VVres;
    }

    return;
}

template <class T>
void ML_Base::d2K(T &xygrad, T &xnormgrad, T &xyxygrad, T &xyxnormgrad, T &xyynormgrad, T &xnormxnormgrad, T &xnormynormgrad, T &ynormynormgrad, int &minmaxind, 
                  int ib, int jb, 
                  const T &bias, const MercerKernel &altK, const gentype **pxyprod,
                  const SparseVector<gentype> *xx, const SparseVector<gentype> *yy, 
                  const vecInfo *xxinfo, const vecInfo *yyinfo, int &iset, int &jset) const
{
//phantomx
    NiceAssert( ( xx && xxinfo ) || ( !xx && !xxinfo ) );
    NiceAssert( ( yy && yyinfo ) || ( !yy && !yyinfo ) );

    // Shortcut for speed

    if ( xx && yy &&  xx->isnofaroffindpresent() && yy->isnofaroffindpresent() )
    {
        NiceAssert( (*xx).nearupsize() == 1 );
        NiceAssert( (*yy).nearupsize() == 1 );

        NiceAssert( (*xx).farupsize() == 1 );
        NiceAssert( (*yy).farupsize() == 1 );

        if ( altK.suggestXYcache() && getxymat() && ( ib >= 0 ) && ( jb >= 0 ) )
        {
            const Matrix<double> &xy = *getxymat();
            Vector<int> xyderef(2);

            xyderef("&",zeroint()) = ib;
            xyderef("&",1) = jb;

            retMatrix<double> tmpma;

            altK.d2K(xygrad,xnormgrad,xyxygrad,xyxnormgrad,xyynormgrad,xnormxnormgrad,xnormynormgrad,ynormynormgrad,minmaxind,*xx,*yy,*xxinfo,*yyinfo,bias,pxyprod,ib,jb,xspaceDim(),isXConsistent() && istrv(ib) && istrv(jb),MLid(),&(xy(xyderef,xyderef,tmpma)),0,iset,jset,assumeReal);
        }

        else
        {
            altK.d2K(xygrad,xnormgrad,xyxygrad,xyxnormgrad,xyynormgrad,xnormxnormgrad,xnormynormgrad,ynormynormgrad,minmaxind,*xx,*yy,*xxinfo,*yyinfo,bias,pxyprod,ib,jb,xspaceDim(),isXConsistent() && istrv(ib) && istrv(jb),MLid(),NULL,0,iset,jset,assumeReal);
        }

        return;
    }

    else if ( xx && !yy )
    {
        const SparseVector<gentype> &xjb = xgetloc(jb);

        const vecInfo &xinfoj = xinfo(jb);

        d2K(xygrad,xnormgrad,xyxygrad,xyxnormgrad,xyynormgrad,xnormxnormgrad,xnormynormgrad,ynormynormgrad,minmaxind,ib,jb,bias,altK,pxyprod,xx,&xjb,xxinfo,&xinfoj,iset,jset);

        return;
    }

    else if ( !xx && yy )
    {
        const SparseVector<gentype> &xib = xgetloc(ib);

        const vecInfo &xinfoi = xinfo(ib);

        d2K(xygrad,xnormgrad,xyxygrad,xyxnormgrad,xyynormgrad,xnormxnormgrad,xnormynormgrad,ynormynormgrad,minmaxind,ib,jb,bias,altK,pxyprod,&xib,yy,&xinfoi,yyinfo,iset,jset);

        return;
    }

    else if ( !xx && !yy )
    {
        const SparseVector<gentype> &xib = xgetloc(ib);
        const SparseVector<gentype> &xjb = xgetloc(jb);

        const vecInfo &xinfoi = xinfo(ib);
        const vecInfo &xinfoj = xinfo(jb);

        d2K(xygrad,xnormgrad,xyxygrad,xyxnormgrad,xyynormgrad,xnormxnormgrad,xnormynormgrad,ynormynormgrad,minmaxind,ib,jb,bias,altK,pxyprod,&xib,&xjb,&xinfoi,&xinfoj,iset,jset);

        return;
    }

    const SparseVector<gentype> *xinear = NULL;
    const SparseVector<gentype> *xjnear = NULL;

    const SparseVector<gentype> *xifar = NULL;
    const SparseVector<gentype> *xjfar = NULL;

    const SparseVector<gentype> *xifarfar = NULL;
    const SparseVector<gentype> *xjfarfar = NULL;

    const vecInfo *xinearinfo = NULL;
    const vecInfo *xjnearinfo = NULL;

    const vecInfo *xifarinfo = NULL;
    const vecInfo *xjfarinfo = NULL;

    int i,ii,iokr,iok,idiagr,igradOrder,iplanr,iplan;
    int j,jj,jokr,jok,jdiagr,jgradOrder,jplanr,jplan;

    const gentype *itup = NULL;
    const gentype *iitup = NULL;

    const gentype *jtup = NULL;
    const gentype *jjtup = NULL;

    int imethodkey = detangle_x(xinear,xifar,xifarfar,xinearinfo,xifarinfo,i,ii,itup,iitup,iokr,iok,ib,idiagr,xx,xxinfo,igradOrder,iplanr,iplan,iset) & 7;
    int jmethodkey = detangle_x(xjnear,xjfar,xjfarfar,xjnearinfo,xjfarinfo,j,jj,jtup,jjtup,jokr,jok,jb,jdiagr,yy,yyinfo,jgradOrder,jplanr,jplan,jset) & 7;

    if ( idiagr || jdiagr )
    {
        xygrad         = 0.0;
        xnormgrad      = 0.0;
        xyxygrad       = 0.0;
        xyxnormgrad    = 0.0;
        xyynormgrad    = 0.0;
        xnormxnormgrad = 0.0;
        xnormynormgrad = 0.0;
        ynormynormgrad = 0.0;

        return;
    }

    NiceAssert( !itup && !iitup );
    NiceAssert( !jtup && !jjtup );

    NiceAssert( !imethodkey ); (void) imethodkey;
    NiceAssert( !jmethodkey ); (void) jmethodkey;

    d2K(xygrad,xnormgrad,xyxygrad,xyxnormgrad,xyynormgrad,xnormxnormgrad,xnormynormgrad,ynormynormgrad,minmaxind,i,j,bias,altK,pxyprod,xinear,xjnear,xinearinfo,xjnearinfo,iset,jset);

    NiceAssert( !iokr && !jokr );

    return;
}

template <class T>
void ML_Base::d2K2delxdelx(T &xxscaleres, T &yyscaleres, T &xyscaleres, T &yxscaleres, T &constres, int &minmaxind, 
                          int ib, int jb, 
                          const T &bias, const MercerKernel &altK, const gentype **pxyprod, 
                          const SparseVector<gentype> *xx, const SparseVector<gentype> *yy, 
                          const vecInfo *xxinfo, const vecInfo *yyinfo, int &iset, int &jset) const
{
//phantomx
    NiceAssert( ( xx && xxinfo ) || ( !xx && !xxinfo ) );
    NiceAssert( ( yy && yyinfo ) || ( !yy && !yyinfo ) );

    // Shortcut for speed

    if ( xx && yy &&  xx->isnofaroffindpresent() && yy->isnofaroffindpresent() )
    {
        NiceAssert( (*xx).nearupsize() == 1 );
        NiceAssert( (*yy).nearupsize() == 1 );

        NiceAssert( (*xx).farupsize() == 1 );
        NiceAssert( (*yy).farupsize() == 1 );

        if ( altK.suggestXYcache() && getxymat() && ( ib >= 0 ) && ( jb >= 0 ) )
        {
            const Matrix<double> &xy = *getxymat();
            Vector<int> xyderef(2);

            xyderef("&",zeroint()) = ib;
            xyderef("&",1) = jb;

            retMatrix<double> tmpma;

            altK.d2K2delxdelx(xxscaleres,yyscaleres,xyscaleres,yxscaleres,constres,minmaxind,*xx,*yy,*xxinfo,*yyinfo,bias,pxyprod,ib,jb,xspaceDim(),isXConsistent() && istrv(ib) && istrv(jb),MLid(),&(xy(xyderef,xyderef,tmpma)),0,iset,jset,assumeReal);
        }

        else
        {
            altK.d2K2delxdelx(xxscaleres,yyscaleres,xyscaleres,yxscaleres,constres,minmaxind,*xx,*yy,*xxinfo,*yyinfo,bias,pxyprod,ib,jb,xspaceDim(),isXConsistent() && istrv(ib) && istrv(jb),MLid(),NULL,0,iset,jset,assumeReal);
        }

        return;
    }

    else if ( xx && !yy )
    {
        const SparseVector<gentype> &xjb = xgetloc(jb);

        const vecInfo &xinfoj = xinfo(jb);

        d2K2delxdelx(xxscaleres,yyscaleres,xyscaleres,yxscaleres,constres,minmaxind,ib,jb,bias,altK,pxyprod,xx,&xjb,xxinfo,&xinfoj,iset,jset);

        return;
    }

    else if ( !xx && yy )
    {
        const SparseVector<gentype> &xib = xgetloc(ib);

        const vecInfo &xinfoi = xinfo(ib);

        d2K2delxdelx(xxscaleres,yyscaleres,xyscaleres,yxscaleres,constres,minmaxind,ib,jb,bias,altK,pxyprod,&xib,yy,&xinfoi,yyinfo,iset,jset);

        return;
    }

    else if ( !xx && !yy )
    {
        const SparseVector<gentype> &xib = xgetloc(ib);
        const SparseVector<gentype> &xjb = xgetloc(jb);

        const vecInfo &xinfoi = xinfo(ib);
        const vecInfo &xinfoj = xinfo(jb);

        d2K2delxdelx(xxscaleres,yyscaleres,xyscaleres,yxscaleres,constres,minmaxind,ib,jb,bias,altK,pxyprod,&xib,&xjb,&xinfoi,&xinfoj,iset,jset);

        return;
    }

    const SparseVector<gentype> *xinear = NULL;
    const SparseVector<gentype> *xjnear = NULL;

    const SparseVector<gentype> *xifar = NULL;
    const SparseVector<gentype> *xjfar = NULL;

    const SparseVector<gentype> *xifarfar = NULL;
    const SparseVector<gentype> *xjfarfar = NULL;

    const vecInfo *xinearinfo = NULL;
    const vecInfo *xjnearinfo = NULL;

    const vecInfo *xifarinfo = NULL;
    const vecInfo *xjfarinfo = NULL;

    int i,ii,iokr,iok,idiagr,igradOrder,iplanr,iplan;
    int j,jj,jokr,jok,jdiagr,jgradOrder,jplanr,jplan;

    const gentype *itup = NULL;
    const gentype *iitup = NULL;

    const gentype *jtup = NULL;
    const gentype *jjtup = NULL;

    int imethodkey = detangle_x(xinear,xifar,xifarfar,xinearinfo,xifarinfo,i,ii,itup,iitup,iokr,iok,ib,idiagr,xx,xxinfo,igradOrder,iplanr,iplan,iset) & 7;
    int jmethodkey = detangle_x(xjnear,xjfar,xjfarfar,xjnearinfo,xjfarinfo,j,jj,jtup,jjtup,jokr,jok,jb,jdiagr,yy,yyinfo,jgradOrder,jplanr,jplan,jset) & 7;

    if ( idiagr || jdiagr )
    {
        xxscaleres = 0.0;
        yyscaleres = 0.0;
        xyscaleres = 0.0;
        constres = 0.0;

        return;
    }

    NiceAssert( !itup && !iitup );
    NiceAssert( !jtup && !jjtup );

    NiceAssert( !imethodkey ); (void) imethodkey;
    NiceAssert( !jmethodkey ); (void) jmethodkey;

    d2K2delxdelx(xxscaleres,yyscaleres,xyscaleres,yxscaleres,constres,minmaxind,i,j,bias,altK,NULL,xinear,xjnear,xinearinfo,xjnearinfo,iset,jset);

    if ( iokr || jokr )
    {
        Vector<int> iiokr(2);
        Vector<int> iiok(2);
        Vector<int> iiset(2);
        Vector<const gentype *> xxalt(2);

        iiokr("&",0) = iokr;
        iiok("&",0)  = iok;
        iiset("&",0) = iset;
        xxalt("&",0) = (*xx).isfarfarfarindpresent(3) ? &((*xx).fff(3)) : &nullgentype();

        iiokr("&",1) = jokr;
        iiok("&",1)  = jok;
        iiset("&",1) = jset;
        xxalt("&",1) = (*yy).isfarfarfarindpresent(3) ? &((*yy).fff(3)) : &nullgentype();

        gentype UUres;

        (*UUcallback)(UUres,2,*this,iiokr,iiok,iiset,xxalt,defbasisUU);

        xxscaleres *= (T) UUres;
        yyscaleres *= (T) UUres;
        xyscaleres *= (T) UUres;
        yxscaleres *= (T) UUres;
        constres   *= (T) UUres;
    }

    if ( iplanr || jplanr )
    {
        Vector<int> iiplanr(2);
        Vector<int> iiplan(2);
        Vector<int> iiset(2);
        Vector<const gentype *> xxalt(2);

        iiplanr("&",0) = iplanr;
        iiplan("&",0)  = iplan;
        iiset("&",0)   = iset;
        xxalt("&",0)   = (*xx).isfarfarfarindpresent(7) ? &((*xx).fff(7)) : &nullgentype();

        iiplanr("&",1) = jplanr;
        iiplan("&",1)  = jplan;
        iiset("&",1)   = jset;
        xxalt("&",1)   = (*yy).isfarfarfarindpresent(7) ? &((*yy).fff(7)) : &nullgentype();

        gentype VVres;
        gentype kval(1.0);

        (*VVcallback)(VVres,2,kval,*this,iiplanr,iiplan,iiset,xxalt,defbasisVV);

        xxscaleres *= (T) VVres;
        yyscaleres *= (T) VVres;
        xyscaleres *= (T) VVres;
        yxscaleres *= (T) VVres;
        constres   *= (T) VVres;
    }

    return;
}


template <class T>
void ML_Base::d2K2delxdely(T &xxscaleres, T &yyscaleres, T &xyscaleres, T &yxscaleres, T &constres, int &minmaxind, 
                          int ib, int jb, 
                          const T &bias, const MercerKernel &altK, const gentype **pxyprod, 
                          const SparseVector<gentype> *xx, const SparseVector<gentype> *yy, 
                          const vecInfo *xxinfo, const vecInfo *yyinfo, int &iset, int &jset) const
{
//phantomx
    NiceAssert( ( xx && xxinfo ) || ( !xx && !xxinfo ) );
    NiceAssert( ( yy && yyinfo ) || ( !yy && !yyinfo ) );

    // Shortcut for speed

    if ( xx && yy &&  xx->isnofaroffindpresent() && yy->isnofaroffindpresent() )
    {
        NiceAssert( (*xx).nearupsize() == 1 );
        NiceAssert( (*yy).nearupsize() == 1 );

        NiceAssert( (*xx).farupsize() == 1 );
        NiceAssert( (*yy).farupsize() == 1 );

        if ( altK.suggestXYcache() && getxymat() && ( ib >= 0 ) && ( jb >= 0 ) )
        {
            const Matrix<double> &xy = *getxymat();
            Vector<int> xyderef(2);

            xyderef("&",zeroint()) = ib;
            xyderef("&",1) = jb;

            retMatrix<double> tmpma;

            altK.d2K2delxdely(xxscaleres,yyscaleres,xyscaleres,yxscaleres,constres,minmaxind,*xx,*yy,*xxinfo,*yyinfo,bias,pxyprod,ib,jb,xspaceDim(),isXConsistent() && istrv(ib) && istrv(jb),MLid(),&(xy(xyderef,xyderef,tmpma)),0,iset,jset,assumeReal);
        }

        else
        {
            altK.d2K2delxdely(xxscaleres,yyscaleres,xyscaleres,yxscaleres,constres,minmaxind,*xx,*yy,*xxinfo,*yyinfo,bias,pxyprod,ib,jb,xspaceDim(),isXConsistent() && istrv(ib) && istrv(jb),MLid(),NULL,0,iset,jset,assumeReal);
        }

        return;
    }

    else if ( xx && !yy )
    {
        const SparseVector<gentype> &xjb = xgetloc(jb);

        const vecInfo &xinfoj = xinfo(jb);

        d2K2delxdely(xxscaleres,yyscaleres,xyscaleres,yxscaleres,constres,minmaxind,ib,jb,bias,altK,pxyprod,xx,&xjb,xxinfo,&xinfoj,iset,jset);

        return;
    }

    else if ( !xx && yy )
    {
        const SparseVector<gentype> &xib = xgetloc(ib);

        const vecInfo &xinfoi = xinfo(ib);

        d2K2delxdely(xxscaleres,yyscaleres,xyscaleres,yxscaleres,constres,minmaxind,ib,jb,bias,altK,pxyprod,&xib,yy,&xinfoi,yyinfo,iset,jset);

        return;
    }

    else if ( !xx && !yy )
    {
        const SparseVector<gentype> &xib = xgetloc(ib);
        const SparseVector<gentype> &xjb = xgetloc(jb);

        const vecInfo &xinfoi = xinfo(ib);
        const vecInfo &xinfoj = xinfo(jb);

        d2K2delxdely(xxscaleres,yyscaleres,xyscaleres,yxscaleres,constres,minmaxind,ib,jb,bias,altK,pxyprod,&xib,&xjb,&xinfoi,&xinfoj,iset,jset);

        return;
    }

    const SparseVector<gentype> *xinear = NULL;
    const SparseVector<gentype> *xjnear = NULL;

    const SparseVector<gentype> *xifar = NULL;
    const SparseVector<gentype> *xjfar = NULL;

    const SparseVector<gentype> *xifarfar = NULL;
    const SparseVector<gentype> *xjfarfar = NULL;

    const vecInfo *xinearinfo = NULL;
    const vecInfo *xjnearinfo = NULL;

    const vecInfo *xifarinfo = NULL;
    const vecInfo *xjfarinfo = NULL;

    int i,ii,iokr,iok,idiagr,igradOrder,iplanr,iplan;
    int j,jj,jokr,jok,jdiagr,jgradOrder,jplanr,jplan;

    const gentype *itup = NULL;
    const gentype *iitup = NULL;

    const gentype *jtup = NULL;
    const gentype *jjtup = NULL;

    int imethodkey = detangle_x(xinear,xifar,xifarfar,xinearinfo,xifarinfo,i,ii,itup,iitup,iokr,iok,ib,idiagr,xx,xxinfo,igradOrder,iplanr,iplan,iset) & 7;
    int jmethodkey = detangle_x(xjnear,xjfar,xjfarfar,xjnearinfo,xjfarinfo,j,jj,jtup,jjtup,jokr,jok,jb,jdiagr,yy,yyinfo,jgradOrder,jplanr,jplan,jset) & 7;

    if ( idiagr || jdiagr )
    {
        xxscaleres = 0.0;
        yyscaleres = 0.0;
        xyscaleres = 0.0;
        constres = 0.0;

        return;
    }

    NiceAssert( !itup && !iitup );
    NiceAssert( !jtup && !jjtup );

    NiceAssert( !imethodkey ); (void) imethodkey;
    NiceAssert( !jmethodkey ); (void) jmethodkey;

    d2K2delxdely(xxscaleres,yyscaleres,xyscaleres,yxscaleres,constres,minmaxind,i,j,bias,altK,NULL,xinear,xjnear,xinearinfo,xjnearinfo,iset,jset);

    if ( iokr || jokr )
    {
        Vector<int> iiokr(2);
        Vector<int> iiok(2);
        Vector<int> iiset(2);
        Vector<const gentype *> xxalt(2);

        iiokr("&",0) = iokr;
        iiok("&",0)  = iok;
        iiset("&",0) = iset;
        xxalt("&",0) = (*xx).isfarfarfarindpresent(3) ? &((*xx).fff(3)) : &nullgentype();

        iiokr("&",1) = jokr;
        iiok("&",1)  = jok;
        iiset("&",1) = jset;
        xxalt("&",1) = (*yy).isfarfarfarindpresent(3) ? &((*yy).fff(3)) : &nullgentype();

        gentype UUres;

        (*UUcallback)(UUres,2,*this,iiokr,iiok,iiset,xxalt,defbasisUU);

        xxscaleres *= (T) UUres;
        yyscaleres *= (T) UUres;
        xyscaleres *= (T) UUres;
        yxscaleres *= (T) UUres;
        constres   *= (T) UUres;
    }

    if ( iplanr || jplanr )
    {
        Vector<int> iiplanr(2);
        Vector<int> iiplan(2);
        Vector<int> iiset(2);
        Vector<const gentype *> xxalt(2);

        iiplanr("&",0) = iplanr;
        iiplan("&",0)  = iplan;
        iiset("&",0)   = iset;
        xxalt("&",0)   = (*xx).isfarfarfarindpresent(7) ? &((*xx).fff(7)) : &nullgentype();

        iiplanr("&",1) = jplanr;
        iiplan("&",1)  = jplan;
        iiset("&",1)   = jset;
        xxalt("&",1)   = (*yy).isfarfarfarindpresent(7) ? &((*yy).fff(7)) : &nullgentype();

        gentype VVres;
        gentype kval(1.0);

        (*VVcallback)(VVres,2,kval,*this,iiplanr,iiplan,iiset,xxalt,defbasisVV);

        xxscaleres *= (T) VVres;
        yyscaleres *= (T) VVres;
        xyscaleres *= (T) VVres;
        yxscaleres *= (T) VVres;
        constres   *= (T) VVres;
    }

    return;
}

template <class T>
void ML_Base::dnK2del(Vector<T> &sc, Vector<Vector<int> > &n, int &minmaxind, 
                     const Vector<int> &q, 
                     int ib, int jb, 
                     const T &bias, const MercerKernel &altK, const gentype **pxyprod, 
                     const SparseVector<gentype> *xx, const SparseVector<gentype> *yy, 
                     const vecInfo *xxinfo, const vecInfo *yyinfo, int &iset, int &jset) const
{
//phantomx
    NiceAssert( ( xx && xxinfo ) || ( !xx && !xxinfo ) );
    NiceAssert( ( yy && yyinfo ) || ( !yy && !yyinfo ) );

    // Shortcut for speed

    if ( xx && yy &&  xx->isnofaroffindpresent() && yy->isnofaroffindpresent() )
    {
        NiceAssert( (*xx).nearupsize() == 1 );
        NiceAssert( (*yy).nearupsize() == 1 );

        NiceAssert( (*xx).farupsize() == 1 );
        NiceAssert( (*yy).farupsize() == 1 );

        if ( altK.suggestXYcache() && getxymat() && ( ib >= 0 ) && ( jb >= 0 ) )
        {
            const Matrix<double> &xy = *getxymat();
            Vector<int> xyderef(2);

            xyderef("&",zeroint()) = ib;
            xyderef("&",1) = jb;

            retMatrix<double> tmpma;

            altK.dnK2del(sc,n,minmaxind,q,*xx,*yy,*xxinfo,*yyinfo,bias,pxyprod,ib,jb,xspaceDim(),isXConsistent() && istrv(ib) && istrv(jb),MLid(),&(xy(xyderef,xyderef,tmpma)),0,iset,jset,assumeReal);
        }

        else
        {
            altK.dnK2del(sc,n,minmaxind,q,*xx,*yy,*xxinfo,*yyinfo,bias,pxyprod,ib,jb,xspaceDim(),isXConsistent() && istrv(ib) && istrv(jb),MLid(),NULL,0,iset,jset,assumeReal);
        }

        return;
    }

    else if ( xx && !yy )
    {
        const SparseVector<gentype> &xjb = xgetloc(jb);

        const vecInfo &xinfoj = xinfo(jb);

        dnK2del(sc,n,minmaxind,q,ib,jb,bias,altK,pxyprod,xx,&xjb,xxinfo,&xinfoj,iset,jset);

        return;
    }

    else if ( !xx && yy )
    {
        const SparseVector<gentype> &xib = xgetloc(ib);

        const vecInfo &xinfoi = xinfo(ib);

        dnK2del(sc,n,minmaxind,q,ib,jb,bias,altK,pxyprod,&xib,yy,&xinfoi,yyinfo,iset,jset);

        return;
    }

    else if ( !xx && !yy )
    {
        const SparseVector<gentype> &xib = xgetloc(ib);
        const SparseVector<gentype> &xjb = xgetloc(jb);

        const vecInfo &xinfoi = xinfo(ib);
        const vecInfo &xinfoj = xinfo(jb);

        dnK2del(sc,n,minmaxind,q,ib,jb,bias,altK,pxyprod,&xib,&xjb,&xinfoi,&xinfoj,iset,jset);

        return;
    }

    const SparseVector<gentype> *xinear = NULL;
    const SparseVector<gentype> *xjnear = NULL;

    const SparseVector<gentype> *xifar = NULL;
    const SparseVector<gentype> *xjfar = NULL;

    const SparseVector<gentype> *xifarfar = NULL;
    const SparseVector<gentype> *xjfarfar = NULL;

    const vecInfo *xinearinfo = NULL;
    const vecInfo *xjnearinfo = NULL;

    const vecInfo *xifarinfo = NULL;
    const vecInfo *xjfarinfo = NULL;

    int i,ii,iokr,iok,idiagr,igradOrder,iplanr,iplan;
    int j,jj,jokr,jok,jdiagr,jgradOrder,jplanr,jplan;

    const gentype *itup = NULL;
    const gentype *iitup = NULL;

    const gentype *jtup = NULL;
    const gentype *jjtup = NULL;

    int imethodkey = detangle_x(xinear,xifar,xifarfar,xinearinfo,xifarinfo,i,ii,itup,iitup,iokr,iok,ib,idiagr,xx,xxinfo,igradOrder,iplanr,iplan,iset) & 7;
    int jmethodkey = detangle_x(xjnear,xjfar,xjfarfar,xjnearinfo,xjfarinfo,j,jj,jtup,jjtup,jokr,jok,jb,jdiagr,yy,yyinfo,jgradOrder,jplanr,jplan,jset) & 7;

    if ( idiagr || jdiagr )
    {
        static T zeroval(0.0);

        sc = zeroval;

        return;
    }

    NiceAssert( !itup && !iitup );
    NiceAssert( !jtup && !jjtup );

    NiceAssert( !imethodkey ); (void) imethodkey;
    NiceAssert( !jmethodkey ); (void) jmethodkey;

    dnK2del(sc,n,minmaxind,q,i,j,bias,altK,NULL,xinear,xjnear,xinearinfo,xjnearinfo,iset,jset);

    if ( iokr || jokr )
    {
        Vector<int> iiokr(2);
        Vector<int> iiok(2);
        Vector<int> iiset(2);
        Vector<const gentype *> xxalt(2);

        iiokr("&",0) = iokr;
        iiok("&",0)  = iok;
        iiset("&",0) = iset;
        xxalt("&",0) = (*xx).isfarfarfarindpresent(3) ? &((*xx).fff(3)) : &nullgentype();

        iiokr("&",1) = jokr;
        iiok("&",1)  = jok;
        iiset("&",1) = jset;
        xxalt("&",1) = (*yy).isfarfarfarindpresent(3) ? &((*yy).fff(3)) : &nullgentype();

        gentype UUres;

        (*UUcallback)(UUres,2,*this,iiokr,iiok,iiset,xxalt,defbasisUU);

        sc *= (T) UUres;
    }

    if ( iplanr || jplanr )
    {
        Vector<int> iiplanr(2);
        Vector<int> iiplan(2);
        Vector<int> iiset(2);
        Vector<const gentype *> xxalt(2);

        iiplanr("&",0) = iplanr;
        iiplan("&",0)  = iplan;
        iiset("&",0)   = iset;
        xxalt("&",0)   = (*xx).isfarfarfarindpresent(7) ? &((*xx).fff(7)) : &nullgentype();

        iiplanr("&",1) = jplanr;
        iiplan("&",1)  = jplan;
        iiset("&",1)   = jset;
        xxalt("&",1)   = (*yy).isfarfarfarindpresent(7) ? &((*yy).fff(7)) : &nullgentype();

        gentype VVres;
        gentype kval(1.0);

        (*VVcallback)(VVres,2,kval,*this,iiplanr,iiplan,iiset,xxalt,defbasisVV);

        sc *= (T) VVres;
    }

    return;
}

template <class T>
void ML_Base::dK2delx(T &xscaleres, T &yscaleres, int &minmaxind, 
                     int ib, int jb, 
                     const T &bias, const MercerKernel &altK, const gentype **pxyprod, 
                     const SparseVector<gentype> *xx, const SparseVector<gentype> *yy, 
                     const vecInfo *xxinfo, const vecInfo *yyinfo, int &iset, int &jset) const
{
//phantomx
    NiceAssert( ( xx && xxinfo ) || ( !xx && !xxinfo ) );
    NiceAssert( ( yy && yyinfo ) || ( !yy && !yyinfo ) );

    // Shortcut for speed

    if ( xx && yy &&  xx->isnofaroffindpresent() && yy->isnofaroffindpresent() )
    {
        NiceAssert( (*xx).nearupsize() == 1 );
        NiceAssert( (*yy).nearupsize() == 1 );

        NiceAssert( (*xx).farupsize() == 1 );
        NiceAssert( (*yy).farupsize() == 1 );

        if ( altK.suggestXYcache() && getxymat() && ( ib >= 0 ) && ( jb >= 0 ) )
        {
            const Matrix<double> &xy = *getxymat();
            Vector<int> xyderef(2);

            xyderef("&",zeroint()) = ib;
            xyderef("&",1) = jb;

            retMatrix<double> tmpma;

            altK.dK2delx(xscaleres,yscaleres,minmaxind,*xx,*yy,*xxinfo,*yyinfo,bias,pxyprod,ib,jb,xspaceDim(),isXConsistent() && istrv(ib) && istrv(jb),MLid(),&(xy(xyderef,xyderef,tmpma)),iset,jset,assumeReal);
        }

        else
        {
            altK.dK2delx(xscaleres,yscaleres,minmaxind,*xx,*yy,*xxinfo,*yyinfo,bias,pxyprod,ib,jb,xspaceDim(),isXConsistent() && istrv(ib) && istrv(jb),MLid(),NULL,iset,jset,assumeReal);
        }

        return;
    }

    else if ( xx && !yy )
    {
        const SparseVector<gentype> &xjb = xgetloc(jb);

        const vecInfo &xinfoj = xinfo(jb);

        dK2delx(xscaleres,yscaleres,minmaxind,ib,jb,bias,altK,pxyprod,xx,&xjb,xxinfo,&xinfoj,iset,jset);

        return;
    }

    else if ( !xx && yy )
    {
        const SparseVector<gentype> &xib = xgetloc(ib);

        const vecInfo &xinfoi = xinfo(ib);

        dK2delx(xscaleres,yscaleres,minmaxind,ib,jb,bias,altK,pxyprod,&xib,yy,&xinfoi,yyinfo,iset,jset);

        return;
    }

    else if ( !xx && !yy )
    {
        const SparseVector<gentype> &xib = xgetloc(ib);
        const SparseVector<gentype> &xjb = xgetloc(jb);

        const vecInfo &xinfoi = xinfo(ib);
        const vecInfo &xinfoj = xinfo(jb);

        dK2delx(xscaleres,yscaleres,minmaxind,ib,jb,bias,altK,pxyprod,&xib,&xjb,&xinfoi,&xinfoj,iset,jset);

        return;
    }

    const SparseVector<gentype> *xinear = NULL;
    const SparseVector<gentype> *xjnear = NULL;

    const SparseVector<gentype> *xifar = NULL;
    const SparseVector<gentype> *xjfar = NULL;

    const SparseVector<gentype> *xifarfar = NULL;
    const SparseVector<gentype> *xjfarfar = NULL;

    const vecInfo *xinearinfo = NULL;
    const vecInfo *xjnearinfo = NULL;

    const vecInfo *xifarinfo = NULL;
    const vecInfo *xjfarinfo = NULL;

    int i,ii,iokr,iok,idiagr,igradOrder,iplanr,iplan;
    int j,jj,jokr,jok,jdiagr,jgradOrder,jplanr,jplan;

    const gentype *itup = NULL;
    const gentype *iitup = NULL;

    const gentype *jtup = NULL;
    const gentype *jjtup = NULL;

    int imethodkey = detangle_x(xinear,xifar,xifarfar,xinearinfo,xifarinfo,i,ii,itup,iitup,iokr,iok,ib,idiagr,xx,xxinfo,igradOrder,iplanr,iplan,iset) & 7;
    int jmethodkey = detangle_x(xjnear,xjfar,xjfarfar,xjnearinfo,xjfarinfo,j,jj,jtup,jjtup,jokr,jok,jb,jdiagr,yy,yyinfo,jgradOrder,jplanr,jplan,jset) & 7;

    if ( idiagr || jdiagr )
    {
        xscaleres = 0.0;
        yscaleres = 0.0;

        return;
    }

    NiceAssert( !itup && !iitup );
    NiceAssert( !jtup && !jjtup );

    T tmpab,tmpbb;

    switch ( imethodkey+(10*jmethodkey) )
    {
        case  0:
        {
            dK2delx(xscaleres,yscaleres,minmaxind,i, j, bias,altK,NULL,xinear,xjnear,xinearinfo,xjnearinfo,iset,jset);

            break;
        }

        case 10:
        {
            dK2delx(xscaleres,yscaleres,minmaxind,i, j, bias,altK,NULL,xinear,xjnear,xinearinfo,xjnearinfo,iset,jset);
            dK2delx(tmpab,    tmpbb,    minmaxind,i, jj,bias,altK,NULL,xinear,xjfar, xinearinfo,xjfarinfo ,iset,jset);

            xscaleres -= tmpab;
            yscaleres -= tmpbb;

            break;
        }

        default:
        {
            throw("Illegal attempt to mix gradient constraints and rank constraints.");

            break;
        }
    }

    if ( iokr || jokr )
    {
        Vector<int> iiokr(2);
        Vector<int> iiok(2);
        Vector<int> iiset(2);
        Vector<const gentype *> xxalt(2);

        iiokr("&",0) = iokr;
        iiok("&",0)  = iok;
        iiset("&",0) = iset;
        xxalt("&",0) = (*xx).isfarfarfarindpresent(3) ? &((*xx).fff(3)) : &nullgentype();

        iiokr("&",1) = jokr;
        iiok("&",1)  = jok;
        iiset("&",1) = jset;
        xxalt("&",1) = (*yy).isfarfarfarindpresent(3) ? &((*yy).fff(3)) : &nullgentype();

        gentype UUres;

        (*UUcallback)(UUres,2,*this,iiokr,iiok,iiset,xxalt,defbasisUU);

        xscaleres *= (T) UUres;
        yscaleres *= (T) UUres;
    }

    if ( iplanr || jplanr )
    {
        Vector<int> iiplanr(2);
        Vector<int> iiplan(2);
        Vector<int> iiset(2);
        Vector<const gentype *> xxalt(2);

        iiplanr("&",0) = iplanr;
        iiplan("&",0)  = iplan;
        iiset("&",0)   = iset;
        xxalt("&",0)   = (*xx).isfarfarfarindpresent(7) ? &((*xx).fff(7)) : &nullgentype();

        iiplanr("&",1) = jplanr;
        iiplan("&",1)  = jplan;
        iiset("&",1)   = jset;
        xxalt("&",1)   = (*yy).isfarfarfarindpresent(7) ? &((*yy).fff(7)) : &nullgentype();

        gentype VVres;
        gentype kval(1.0);

        (*VVcallback)(VVres,2,kval,*this,iiplanr,iiplan,iiset,xxalt,defbasisVV);

        xscaleres *= (T) VVres;
        yscaleres *= (T) VVres;
    }

    return;
}

void ML_Base::densedKdx(double &res, int i, int j, const double &bias) const
{
//phantomx
    //FIXME: implement gradients and rank

    getKernel().densedKdx(res,xgetloc(i),xgetloc(j),xinfo(i),xinfo(j),bias,i,j,xspaceDim(),isXConsistent() && istrv(i) && istrv(j),MLid(),assumeReal);

    return;
}

void ML_Base::denseintK(double &res, int i, int j, const double &bias) const
{
//phantomx
    //FIXME: implement gradients and rank

    getKernel().denseintK(res,xgetloc(i),xgetloc(j),xinfo(i),xinfo(j),bias,i,j,xspaceDim(),isXConsistent() && istrv(i) && istrv(j),MLid(),assumeReal);

    return;
}

double ML_Base::distK(int i, int j) const
{
//phantomx
    //FIXME: implement gradients and rank

    return getKernel().distK(xgetloc(i),xgetloc(j),xinfo(i),xinfo(j),i,j,xspaceDim(),isXConsistent() && istrv(i) && istrv(j),MLid(),NULL,0,0,assumeReal);
}

void ML_Base::ddistKdx(double &xscaleres, double &yscaleres, int &minmaxind, int i, int j) const
{
//phantomx
    //FIXME: implement gradients and rank

    getKernel().ddistKdx(xscaleres,yscaleres,minmaxind,xgetloc(i),xgetloc(j),xinfo(i),xinfo(j),i,j,xspaceDim(),isXConsistent() && istrv(i) && istrv(j),MLid(),NULL,0,0,assumeReal);

    return;
}

































const gentype &ML_Base::xelm(gentype &res, int i, int j) const
{
//phantomx
    //FIXME: implement gradients and rank

    return getKernel().xelm(res,xgetloc(i),i,j);
}

int ML_Base::xindsize(int i) const
{
//phantomx
    //FIXME: implement gradients and rank

    return getKernel().xindsize(xgetloc(i),i);
}

const vecInfo &ML_Base::xinfo(int i) const
{
    return locxinfo(i);
}

int ML_Base::xtang(int i) const
{
    return locxtang(i);
}


























void ML_Base::xferx(const ML_Base &xsrc)
{
    incxvernum();
    incgvernum();

    NiceAssert( xsrc.N() == N() );

    allxdatagent = xsrc.allxdatagent;

    getKernel_unsafe().setIPdiffered(1);

    resetKernel(1,-1);

    return;
}




























/*
int ML_Base::dcov(double &resv, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, int aaxis, int baxis, const vecInfo *xainf, const vecInfo *xbinf) const
{
    // Note that all gradients are in the relevant parts of the kernel evaluations, so we 
    // can cheat and convert to x :: axis format.

    (void) xainf;
    (void) xbinf;

    SparseVector<gentype> xaa(xa);
    SparseVector<gentype> xbb(xb);

    xaa.ff("&",aaxis) = 1;
    xbb.ff("&",baxis) = 1;

    return cov(resv,xaa,xbb);
}

int ML_Base::dvar(double &resv, const SparseVector<gentype> &xa, int aaxis, const vecInfo *xainf) const
{
    (void) xainf;

    SparseVector<gentype> xaa(xa);

    xaa.ff("&",aaxis) = 1;

    return var(resv,xaa);
}

int ML_Base::dcov(SparseVector<SparseVector<double> > &resv, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo *xainf, const vecInfo *xbinf) const
{
    (void) xainf;
    (void) xbinf;

    SparseVector<gentype> xaa(xa);
    SparseVector<gentype> xbb(xb);

    int i,j;
    int res = 0;

    resv.zero();

    for ( i = 0 ; i < indKey().size() ; i++ )
    {
        resv("&",indKey()(i)).zero();

        for ( j = 0 ; j < indKey().size() ; j++ )
        {
            xaa.ff("&",indKey()(i)) = 1;
            xbb.ff("&",indKey()(j)) = 1;

            res |= cov(resv("&",indKey()(i))("&",indKey()(j)),xaa,xbb);

            xaa.zero(indKey()(i)+INDFARFAROFFSTART);
            xbb.zero(indKey()(j)+INDFARFAROFFSTART);
        }
    }

    return res;
}

int ML_Base::dvar(SparseVector<double> &resv, const SparseVector<gentype> &xa, const vecInfo *xainf) const
{
    (void) xainf;

    SparseVector<gentype> xaa(xa);

    int i;
    int res = 0;

    resv.zero();

    for ( i = 0 ; i < indKey().size() ; i++ )
    {
        xaa.ff("&",indKey()(i)) = 1;

        res |= var(resv("&",indKey()(i)),xaa);

        xaa.zero(indKey()(i)+INDFARFAROFFSTART);
    }

    return res;
}

int ML_Base::dcov(Matrix<double> &resv, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo *xainf, const vecInfo *xbinf) const
{
    (void) xainf;
    (void) xbinf;

    SparseVector<gentype> xaa(xa);
    SparseVector<gentype> xbb(xb);

    int i,j;
    int res = 0;

    resv.resize(indKey().size(),indKey().size());

    for ( i = 0 ; i < indKey().size() ; i++ )
    {
        for ( j = 0 ; j < indKey().size() ; j++ )
        {
            xaa.ff("&",indKey()(i)) = 1;
            xbb.ff("&",indKey()(j)) = 1;

            res |= cov(resv("&",i,j),xaa,xbb);

            xaa.zero(indKey()(i)+INDFARFAROFFSTART);
            xbb.zero(indKey()(j)+INDFARFAROFFSTART);
        }
    }

    return res;
}

int ML_Base::dvar(Vector<double> &resv, const SparseVector<gentype> &xa, const vecInfo *xainf) const
{
    (void) xainf;

    SparseVector<gentype> xaa(xa);

    int i;
    int res = 0;

    resv.resize(indKey().size());

    for ( i = 0 ; i < indKey().size() ; i++ )
    {
        xaa.ff("&",indKey()(i)) = 1;

        res |= var(resv("&",i),xaa);

        xaa.zero(indKey()(i)+INDFARFAROFFSTART);
    }

    return res;
}
*/























void ML_Base::dgTrainingVector(Vector<double> &resx, int i) const
{
    Vector<gentype> res;
    gentype resn;

    NiceAssert( i >= 0 );

    dgTrainingVector(res,resn,i);

    resx.resize(xspaceDim()) = 0.0;

    int j,k;

    for ( k = 0 ; k < x(i).nearindsize() ; k++ )
    {
        resx("&",x(i).ind(k)) += (double) ((resn*(x(i)).direcref(k)));
    }

    if ( ML_Base::N() )
    {
        for ( j = 0 ; j < ML_Base::N() ; j++ )
        {
            if ( x(j).nearindsize() )
            {
                for ( k = 0 ; k < x(j).nearindsize() ; k++ )
                {
                    resx("&",x(j).ind(k)) += (double) ((res(j)*(x(j)).direcref(k)));
                }
            }
        }
    }

    return;
}

void ML_Base::dgTrainingVector(Vector<gentype> &resx, int i) const
{
    Vector<gentype> res;
    gentype resn;
    gentype zv(0.0);

    NiceAssert( i >= 0 );

    dgTrainingVector(res,resn,i);

    resx.resize(xspaceDim()) = zv;

    int j,k;

    for ( k = 0 ; k < x(i).nearindsize() ; k++ )
    {
        resx("&",x(i).ind(k)) += ((resn*(x(i)).direcref(k)));
    }

    if ( ML_Base::N() )
    {
        for ( j = 0 ; j < ML_Base::N() ; j++ )
        {
            if ( x(j).nearindsize() )
            {
                for ( k = 0 ; k < x(j).nearindsize() ; k++ )
                {
                    resx("&",x(j).ind(k)) += ((res(j)*(x(j)).direcref(k)));
                }
            }
        }
    }

    return;
}

int ML_Base::ggTrainingVector(Vector<double> &resg, int i, int retaltg, gentype ***pxyprodi) const
{
    int resi = 0;

    if ( gOutType() == 'R' )
    {
        resg.resize(1);

        resi = ggTrainingVector(resg("&",zeroint()),i,retaltg,pxyprodi);
    }

    else
    {
        gentype res;

        resi = ggTrainingVector(res,i,retaltg,pxyprodi);

        const Vector<gentype> &tempres = (const Vector<gentype> &) res;

        resg.resize(tempres.size());

        if ( resg.size() )
        {
            int i;

            for ( i = 0 ; i < resg.size() ; i++ )
            {
                resg("&",i) = (double) tempres(i);
            }
        }
    }

    return resi;
}

int ML_Base::covarTrainingVector(Matrix<gentype> &resv, const Vector<int> &i) const
{
    NiceAssert( ( i.size() == 0 ) || ( i >= zeroint() ) );

    int m = i.size();
    int res = 0;

    resv.resize(m,m);

    gentype dummy;

    if ( m )
    {
        int ii,jj;

        for ( ii = 0 ; ii < m ; ii++ )
        {
            for ( jj = 0 ; jj < m ; jj++ )
            {
                res |= covTrainingVector(resv("&",ii,jj),dummy,i(ii),i(jj));
            }
        }
    }

    return 0;
}

int ML_Base::covar(Matrix<gentype> &resv, const Vector<SparseVector<gentype> > &xx) const
{
    int m = xx.size();
    int res = 0;

    resv.resize(m,m);

    gentype dummy;

    if ( m )
    {
        int ii,jj;

        for ( ii = 0 ; ii < m ; ii++ )
        {
            for ( jj = 0 ; jj < m ; jj++ )
            {
                res |= cov(resv("&",ii,jj),dummy,xx(ii),xx(jj));
            }
        }
    }

    return 0;
}


















// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================

gentype &UUcallbacknon(gentype &res, int m, const ML_Base &caller, Vector<int> &iokr, Vector<int> &iok, Vector<int> &iset,Vector<const gentype *> xalt, int defbasis)
{
    (void) m;
    (void) caller;
    (void) iokr;
    (void) iok;
    (void) iset;
    (void) xalt;
    (void) defbasis;

    res = 1.0;

    return res;
}

gentype &UUcallbackdef(gentype &res, int mm, const ML_Base &caller, Vector<int> &iokr, Vector<int> &iok, Vector<int> &iset, Vector<const gentype *> xalt, int defbasis)
{
    NiceAssert( iokr.size() == mm );
    NiceAssert( iok.size()  == mm );
    NiceAssert( iset.size() == mm );
    NiceAssert( xalt.size() == mm );

    int m = mm;
    int i;

    if ( ( sum(iokr) > 0 ) || ( defbasis >= 0 ) )
    {
        Vector<SparseVector<gentype> > x(m);
        Vector<vecInfo> xinfo(m);

        for ( i = m-1 ; i >= 0 ; i-- )
        {
            if ( !iokr(i) && ( defbasis >= 0 ) )
            {
                iokr("&",i) = 1;
                iok("&",i)  = defbasis;
            }

            if ( iokr(i) )
            {
                if ( iok(i) > 0 ) { x("&",i)("&",0) = (caller.VbasisUU())(iok(i)); caller.getUUOutputKernel().getvecInfo(xinfo("&",i),x(i)); }
                else              { x("&",i)("&",0) = (*(xalt(i)));                caller.getUUOutputKernel().getvecInfo(xinfo("&",i),x(i)); }
            }

            else
            {
                iok.remove(i);
                iokr.remove(i);
                iset.remove(i);
                xalt.remove(i);
                x.remove(i);
                xinfo.remove(i);

                m--;
            }
        }

        if ( m == mm )
        {
            int z = 0;

            if ( m == 0 )
            {
                caller.getUUOutputKernel().K0(res,zerointgentype(),NULL,0,0,0,caller.MLid());
            }

            else if ( m == 1 )
            {
                caller.getUUOutputKernel().K1(res,x(z),xinfo(z),zerointgentype(),NULL,iok(z),0,0,0,caller.MLid(),NULL,iset(z));
            }

            else if ( m == 2 )
            {
                caller.getUUOutputKernel().K2(res,x(z),x(1),xinfo(z),xinfo(1),zerointgentype(),NULL,iok(z),iok(1),0,0,0,caller.MLid(),NULL,iset(z),iset(1));
            }

            else if ( m == 3 )
            {
                caller.getUUOutputKernel().K3(res,x(z),x(1),x(2),xinfo(z),xinfo(1),xinfo(2),zerointgentype(),NULL,iok(z),iok(1),iok(2),0,0,0,caller.MLid(),NULL,iset(z),iset(1),iset(2));
            }

            else if ( m == 4 )
            {
                caller.getUUOutputKernel().K4(res,x(z),x(1),x(2),x(3),xinfo(z),xinfo(1),xinfo(2),xinfo(3),zerointgentype(),NULL,iok(z),iok(1),iok(2),iok(3),0,0,0,caller.MLid(),NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,iset(z),iset(1),iset(2),iset(3));
            }

            else if ( m >= 5 )
            {
                Vector<const SparseVector<gentype> *> xx(m);
                Vector<const vecInfo *> xxinfo(m);
                Vector<int> ii(m);

                for ( i = 0 ; i < m ; i++ )
                {
                    xx("&",i) = &(x(i));
                    xxinfo("&",i) = &(xinfo(i));
                    ii("&",i) = iok(i);
                }

                caller.getUUOutputKernel().Km(m,res,xx,xxinfo,zerointgentype(),ii,NULL,0,0,0,caller.MLid(),NULL,&iset);
            }
        }

        else
        {
            NiceAssert( caller.getUUOutputKernel().isSimpleLinearKernel() );

            if ( m == 0 )
            {
                res = 1.0;
            }

            else if ( m == 1 )
            {
                res = x(zeroint())(zeroint());
            }

            else if ( m == 2 )
            {
                res = x(zeroint())(zeroint());
                res = emul(res,x(1)(zeroint()));
            }

            else if ( m == 3 )
            {
                res = x(zeroint())(zeroint());
                res = emul(res,x(1)(zeroint()));
                res = emul(res,x(2)(zeroint()));
            }

            else if ( m == 4 )
            {
                res = x(zeroint())(zeroint());
                res = emul(res,x(1)(zeroint()));
                res = emul(res,x(2)(zeroint()));
                res = emul(res,x(3)(zeroint()));
            }

            else if ( m >= 5 )
            {
                res = x(zeroint())(zeroint());

                for ( i = 1 ; i < m ; i++ )
                {
                    res = emul(res,x(i)(zeroint()));
                }
            }
        }
    }

    else
    {
        res = 1.0; // This will never actually be reached.
    }

    return res;
}

const gentype &VVcallbacknon(gentype &res, int m, const gentype &kval, const ML_Base &caller, Vector<int> &iokr, Vector<int> &iok, Vector<int> &iset, Vector<const gentype *> xalt, int defbasis)
{
    (void) m;
    (void) kval;
    (void) caller;
    (void) iokr;
    (void) iok;
    (void) iset;
    (void) xalt;
    (void) defbasis;

    res = kval;

    return kval;
}

const gentype &VVcallbackdef(gentype &res, int mm, const gentype &kval, const ML_Base &caller, Vector<int> &iokr, Vector<int> &iok, Vector<int> &iset, Vector<const gentype *> xalt, int defbasis)
{
    NiceAssert( iokr.size() == mm );
    NiceAssert( iok.size()  == mm );
    NiceAssert( iset.size() == mm );
    NiceAssert( xalt.size() == mm );

    int m = mm;
    int i;

    res = kval;

    if ( mm && ( ( sum(iokr) > 0 ) || ( defbasis >= 0 ) ) )
    {
        Vector<gentype> s(m);

        for ( i = m-1 ; i >= 0 ; i-- )
        {
            if ( !iokr(i) && ( defbasis >= 0 ) )
            {
                iokr("&",i) = 1;
                iok("&",i)  = defbasis;
            }

            if ( iokr(i) )
            {
                if ( iok(i) > 0 ) { s("&",i) = (caller.VbasisVV())(iok(i)); }
                else              { s("&",i) = (*(xalt(i))); }
            }

            else
            {
                iok.remove(i);
                iokr.remove(i);
                iset.remove(i);
                xalt.remove(i);
                s.remove(i);

                m--;
            }
        }

        if ( m )
        {
            for ( i = 0 ; i < m ; i++ )
            {
                res *= s(i);
            }
        }
    }

    return res;
}


int ML_Base::setUUOutputKernel(const MercerKernel &xkernel, int modind)
{
    UUoutkernel = xkernel;

    return resetKernel(modind,-1,0);
}





int ML_Base::addToBasisUU(int i, const gentype &o)
{
    NiceAssert( !isBasisUserUU );
    NiceAssert( i >= 0 );
    NiceAssert( i <= NbasisUU() );

    // Add to basis set

    locbasisUU.add(i);
    locbasisUU("&",i) = o;

    return 1;
}

int ML_Base::removeFromBasisUU(int i)
{
    NiceAssert( !isBasisUserUU );
    NiceAssert( i >= 0 );
    NiceAssert( i < NbasisUU() );

    // Remove from basis set

    locbasisUU.remove(i);

    return 1;
}

int ML_Base::setBasisUU(int i, const gentype &o)
{
    NiceAssert( !isBasisUserUU );
    NiceAssert( i >= 0 );
    NiceAssert( i < NbasisUU() );

    locbasisUU("&",i) = o;
    resetKernel(1,-1,0);

    return 1;
}

int ML_Base::setBasisUU(const Vector<gentype> &o)
{
    NiceAssert( !isBasisUserUU );

    locbasisUU = o;
    resetKernel(1,-1,0);

    return 1;
}

int ML_Base::setBasisUU(int n, int d)
{
    NiceAssert( n >= 0 );
    NiceAssert( d >= 0 );

    Vector<gentype> o(n);
    Vector<double> ubase(d);

    int i;

    if ( n )
    {
        for ( i = 0 ; i < n ; i++ )
        {
            if ( d > 0 )
            {
                ubase = 0.0;

                while ( sum(ubase) == 0 )
                {
                    randfill(ubase);
                }

                ubase /= sum(ubase);
            }

            o("&",i) = ubase;
        }
    }

    return setBasisUU(o);
}

int ML_Base::setBasisYUU(void)
{
    int res = 0;

    if ( !isBasisUserUU )
    {
        setBasisUU(y());
        isBasisUserUU = 1;
        res = 1;
    }

    return res;
}

int ML_Base::setBasisUUU(void)
{
    int res = 0;

    if ( isBasisUserUU )
    {
        isBasisUserUU = 0;
        res = 1;
    }

    return res;
}







int ML_Base::addToBasisVV(int i, const gentype &o)
{
    NiceAssert( !isBasisUserVV );
    NiceAssert( i >= 0 );
    NiceAssert( i <= NbasisVV() );

    // Add to basis set

    locbasisVV.add(i);
    locbasisVV("&",i) = o;

    return 1;
}

int ML_Base::removeFromBasisVV(int i)
{
    NiceAssert( !isBasisUserVV );
    NiceAssert( i >= 0 );
    NiceAssert( i < NbasisVV() );

    // Remove from basis set

    locbasisVV.remove(i);

    return 1;
}

int ML_Base::setBasisVV(int i, const gentype &o)
{
    NiceAssert( !isBasisUserVV );
    NiceAssert( i >= 0 );
    NiceAssert( i < NbasisVV() );

    locbasisVV("&",i) = o;
    resetKernel(1,-1,0);

    return 1;
}

int ML_Base::setBasisVV(const Vector<gentype> &o)
{
    NiceAssert( !isBasisUserVV );

    locbasisVV = o;
    resetKernel(1,-1,0);

    return 1;
}

int ML_Base::setBasisVV(int n, int d)
{
    NiceAssert( n >= 0 );
    NiceAssert( d >= 0 );

    Vector<gentype> o(n);
    Vector<double> ubase(d);

    int i;

    if ( n )
    {
        for ( i = 0 ; i < n ; i++ )
        {
            if ( d > 0 )
            {
                ubase = 0.0;

                while ( sum(ubase) == 0 )
                {
                    randfill(ubase);
                }

                ubase /= sum(ubase);
            }

            o("&",i) = ubase;
        }
    }

    return setBasisVV(o);
}

int ML_Base::setBasisYVV(void)
{
    int res = 0;

    if ( !isBasisUserVV )
    {
        setBasisVV(y());
        isBasisUserVV = 1;
        res = 1;
    }

    return res;
}

int ML_Base::setBasisUVV(void)
{
    int res = 0;

    if ( isBasisUserVV )
    {
        isBasisUserVV = 0;
        res = 1;
    }

    return res;
}
































int ML_Base::egetparam(int ind, Vector<gentype> &val, const Vector<gentype> &xa, int ia, const Vector<gentype> &xb, int ib) const
{
    int k,res = 0;

    NiceAssert( xa.size() == xb.size() );

    val.resize(xa.size());

    for ( k = 0 ; k < xa.size() ; k++ )
    {
        res |= getparam(ind,val("&",k),xa(k),ia,xb(k),ib);
    }

    return res;
}


int ML_Base::getparam(int ind, gentype &val, const gentype &xa, int ia, const gentype &xb, int ib) const
{
    int res = 0;

    // Function to access parameters via indexing

    NiceAssert( ia >= 0 );
    NiceAssert( ib >= 0 );

    gentype dummy;

    if ( ( ind <= 499 ) && ( !ia && !ib ) )
    {
        SparseVector<gentype> res;

        switch ( ind )
        {
            case  0: { val = C();                       break; }
            case  1: { val = eps();                     break; }
            case  2: { val = sigma();                   break; }
            case  3: { val = betarank();                break; }
            case  4: { val = tspaceDim();               break; }
            case  5: { val = order();                   break; }
            case  6: { val = sparlvl();                 break; }
            case  7: { val = xspaceDim();               break; }
            case  8: { val = tspaceSparse();            break; }
            case  9: { val = xspaceSparse();            break; }
            case 10: { val = N();                       break; }
            case 11: { val = type();                    break; }
            case 12: { val = subtype();                 break; }
            case 13: { val = numClasses();              break; }
            case 14: { val = isTrained();               break; }
            case 15: { val = isMutable();               break; }
            case 16: { val = isPool();                  break; }
            case 17: { val = isUnderlyingScalar();      break; }
            case 18: { val = isUnderlyingVector();      break; }
            case 19: { val = isUnderlyingAnions();      break; }
            case 20: { val = isClassifier();            break; }
            case 21: { val = isRegression();            break; }
            case 22: { val = numInternalClasses();      break; }
            case 23: { val = NbasisUU();                break; }
            case 24: { val = basisTypeUU();             break; }
            case 25: { val = defProjUU();               break; }
            case 26: { val.force_string() = gOutType(); break; }
            case 27: { val.force_string() = hOutType(); break; }
            case 28: { val.force_string() = targType(); break; }
            case 29: { val = ClassLabels();             break; }
            case 30: { val = zerotol();                 break; }
            case 31: { val = Opttol();                  break; }
            case 32: { val = maxtraintime();            break; }
            case 33: { val = lrmvrank();                break; }
            case 34: { val = ztmvrank();                break; }
            case 35: { val = memsize();                 break; }
            case 36: { val = maxitcnt();                break; }
            case 37: { val = maxitermvrank();           break; }
            case 38: { val = y();                       break; }
            case 39: { val = d();                       break; }
            case 40: { val = Cweight();                 break; }
            case 41: { val = epsweight();               break; }
            case 42: { val = alphaState();              break; }
            case 43: { val = Cweightfuzz();             break; }
            case 44: { val = sigmaweight();             break; }
            case 45: { val = VbasisUU();                break; }
            case 46: { val = indKey();                  break; }
            case 47: { val = indKeyCount();             break; }
            case 48: { val = dattypeKey();              break; }
            case 49: { val = dattypeKeyBreak();         break; }
            case 50: { convertSparseToSet(val,xsum(res));       break; }
            case 51: { convertSparseToSet(val,xmean(res));      break; }
            case 52: { convertSparseToSet(val,xmeansq(res));    break; }
            case 53: { convertSparseToSet(val,xsqsum(res));     break; }
            case 54: { convertSparseToSet(val,xsqmean(res));    break; }
            case 55: { convertSparseToSet(val,xmedian(res));    break; }
            case 56: { convertSparseToSet(val,xvar(res));       break; }
            case 57: { convertSparseToSet(val,xstddev(res));    break; }
            case 59: { convertSparseToSet(val,xmax(res));       break; }
            case 60: { convertSparseToSet(val,xmin(res));       break; }
            case 61: { val = NbasisVV();                break; }
            case 62: { val = basisTypeVV();             break; }
            case 63: { val = defProjVV();               break; }
            case 65: { val = VbasisVV();                break; }

            case 100: { val = Cclass((int) xa);              break; }
            case 101: { val = epsclass((int) xa);            break; }
            case 102: { val = isenabled((int) xa);           break; }
            case 103: { val = NNC((int) xa);                 break; }
            case 104: { val = isenabled((int) xa);           break; }
            case 105: { val = getInternalClass(xa);          break; }
            case 106: { convertSparseToSet(val,x((int) xa)); break; } 

            case 200: { val = calcDist(xa,xb);  break; }

            case 300: { ggTrainingVector(val,(int) xa); break; }
            case 301: { hhTrainingVector(val,(int) xa); break; }
            case 302: { varTrainingVector(val,dummy,(int) xa); break; }
            case 303: { Vector<double> res; dgTrainingVector(res,(int) xa); val = res;  break; }
            case 304: { Vector<gentype> res; gentype resn; dgTrainingVector(res,resn,(int) xa); val = res; break; }
            case 305: { Vector<gentype> res; gentype resn; dgTrainingVector(res,resn,(int) xa); val = resn; break; }
            case 306: { double res; int z = 0; stabProbTrainingVector(res, (int) xa, (int) xb(z), (double) xb(1), (int) xb(2), (double) xb(3), (double) xb(4)); val = res;  break; }

            case 400: { covTrainingVector(val,dummy,(int) xa, (int) xb); break; }
            case 401: { K2(val,(int) xa, (int) xb);                      break; }
            case 402: { K2ip(val.force_double(),(int) xa, (int) xb,0.0); break; }
            case 403: { val = distK((int) xa, (int) xb);                 break; }

            case 499: { Keqn(val); break; }

            default:
            {
                val.force_null();
                break;
            }
        }
    }

    else if ( ind <= 499 )
    {
        val.force_null();
    }

    else if ( ( ind <= 599 ) && !ib )
    {
        SparseVector<gentype> xx;

        if ( convertSetToSparse(xx,xa,ia) )
        {
            res = 1;
        }

        else
        {
            gentype dummy;

            switch ( ind )
            {
                case 500: { gg(val,xx); break; }
                case 501: { hh(val,xx); break; }
                case 502: { var(val,dummy,xx); break; }
                case 503: { Vector<double> res; dg(res,xx); val = res;  break; }
                case 504: { Vector<gentype> res; gentype resn; dg(res,resn,xb,xx); val = res;  break; }
                case 505: { Vector<gentype> res; gentype resn; dg(res,resn,xb,xx); val = resn; break; }
                case 506: { double res; int z = 0; stabProb(res,xx,(int) xb(z),(double) xb(1),(int) xb(2),(double) xb(3),(double) xb(4)); val = res;  break; }

                default:
                {
                    val.force_null();
                    break;
                }
            }
        }
    }

    else if ( ind <= 599 )
    {
        val.force_null();
    }

    else if ( ind <= 699 )
    {
        SparseVector<gentype> xx;
        SparseVector<gentype> yy;

        convertSetToSparse(xx,xa,ia);
        convertSetToSparse(yy,xb,ib);

        gentype dummy;

        switch ( ind )
        {
            case 600: { cov(val,dummy,xx,yy);               break; }
            case 601: { K2(val,xx,yy);                      break; }
            case 602: { K2ip(val.force_double(),xx,yy,0,0); break; }
            case 603: { val = distK(xx,yy);                 break; }

            default:
            {
                val.force_null();
                break;
            }
        }
    }

    else if ( ind <= 799 )
    {
        NiceAssert( ia = 0 );
        NiceAssert( ib = 0 );

        int i,m = (int) xa;

        Vector<SparseVector<gentype> > xx(m);
        const Vector<gentype> &xxb = (const Vector<gentype> &) xb;

        NiceAssert( xxb.size() == m );

        for ( i = 0 ; i < m ; i++ )
        {
            convertSetToSparse(xx("&",i),xxb(i),0);
        }

        switch ( ind )
        {
            case 701: { Km(val,xx); break; }

            default:
            {
                val.force_null();
                break;
            }
        }
    }

    else if ( ind <= 899 )
    {
        res = getKernel().getparam(ind-800,val,xa,ia,xb,ib);
    }

    else if ( ind <= 999 )
    {
        res = getUUOutputKernel().getparam(ind-900,val,xa,ia,xb,ib);
    }

    else
    {
        val.force_null();
    }

    return res;
}

