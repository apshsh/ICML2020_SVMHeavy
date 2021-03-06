
//
// Functional and RKHS Vector class
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "FNVector.h"

// Get Func/RKHS part of vector

const FuncVector &getFuncpart(const Vector<gentype> &src);
const FuncVector &getFuncpart(const Vector<gentype> &src)
{
    NiceAssert( src.infsize() && ( src.type() == 1 ) );

    if ( src.imoverhere )
    {
        return dynamic_cast<const FuncVector &>(*(src.imoverhere));
    }

    return dynamic_cast<const FuncVector &>(src);
}

const RKHSVector &getRKHSpart(const Vector<gentype> &src);
const RKHSVector &getRKHSpart(const Vector<gentype> &src)
{
    NiceAssert( src.infsize() && ( src.type() == 2 ) );

    if ( src.imoverhere )
    {
        return dynamic_cast<const RKHSVector &>(*(src.imoverhere));
    }

    return dynamic_cast<const RKHSVector &>(src);
}

// Inner-product functions for RKHS
//
// conj = 0: noConj
//        1: normal
//        2: revConj

gentype &FuncVector::inner1(gentype &res) const
{
    NiceAssert( !ismixed() );

    int gran = samplesize() ? samplesize() : DEFAULT_SAMPLES_SAMPLE;
    double unitsize = pow(1.0/((double) gran),dim());
    int j;

    if ( dim() == 1 )
    {
        unsafesample(gran);

        gentype av;

        setzero(res);
        setzero(av);

        int totpts = 0;
        int nullpts = 0;

        for ( j = 0 ; j < gran ; j++ )
        {
            totpts++;

            if ( (*this)(j).isValNull() )
            {
                nullpts++;
            }

            else
            {
                av  = (*this)(j);
                av *= unitsize;

                res += av;
            }
        }

        res *= ( ( nullpts == totpts ) ? 1.0 : (((double) totpts)/(((double) totpts-nullpts))) );
    }

    else
    {
        gentype av;

        setzero(res);
        setzero(av);

        Vector<int> i(dim());
        SparseVector<gentype> xx;

        i = zeroint();

        int totpts = 0;
        int nullpts = 0;

        int done = dim() ? 0 : 1;

        while ( !done )
        {
            for ( j = 0 ; j < dim() ; j++ )
            {
                xx("&",j) = ((double) i(j))/((double) gran);
            }

            totpts++;

            (*this)(av,xx); av.finalise();

            if ( av.isValNull() )
            {
                nullpts++;
            }

            else
            {
                av *= unitsize;

                res += av;
            }

            done = 1;

            for ( j = 0 ; done && ( j < dim() ) ; j++ )
            {
                i("&",j)++;

                done     = i(j)/gran;
                i("&",j) = i(j)%gran;
            }
        }

        res *= ( ( nullpts == totpts ) ? 1.0 : (((double) totpts)/(((double) totpts-nullpts))) );
    }

    return res;
}

gentype &FuncVector::inner2(gentype &res, const Vector<gentype> &bb, int conj) const
{
    const FuncVector &b = getFuncpart(bb);

    NiceAssert(   !ismixed() );
    NiceAssert( !b.ismixed() );
    NiceAssert( ( b.type() == 1 ) || ( dim() == b.dim() ) );

    int gran = samplesize() ? samplesize() : DEFAULT_SAMPLES_SAMPLE;
    double unitsize = pow(1.0/((double) gran),dim());
    int j;

    if ( dim() == 1 )
    {
          unsafesample(gran);
        b.unsafesample(gran);

        gentype av;
        gentype bv;

        setzero(res);
        setzero(av);
        setzero(bv);

        int totpts = 0;
        int nullpts = 0;

        for ( j = 0 ; j < gran ; j++ )
        {
            totpts++;

            if ( (*this)(j).isValNull() || b(j).isValNull() )
            {
                nullpts++;
            }

            else
            {
                av = (*this)(j);
                bv = b(j);

                if ( conj & 1 ) { setconj(av); }
                if ( conj & 2 ) { setconj(bv); }

                av *= bv;
                av *= unitsize;

                res += av;
            }
        }

        res *= ( ( nullpts == totpts ) ? 1.0 : (((double) totpts)/(((double) totpts-nullpts))) );
    }

    else
    {
        gentype av;
        gentype bv;

        setzero(res);
        setzero(av);
        setzero(bv);

        Vector<int> i(dim());
        SparseVector<gentype> xx;

        i = zeroint();

        int totpts = 0;
        int nullpts = 0;

        int done = dim() ? 0 : 1;

        while ( !done )
        {
            for ( j = 0 ; j < dim() ; j++ )
            {
                xx("&",j) = ((double) i(j))/((double) gran);
            }

            totpts++;

            (*this)(av,xx); av.finalise();
            b(bv,xx);       bv.finalise();

            if ( av.isValNull() || bv.isValNull() )
            {
                nullpts++;
            }

            else
            {
                if ( conj & 1 ) { setconj(av); }
                if ( conj & 2 ) { setconj(bv); }

                av *= bv;
                av *= unitsize;

                res += av;
            }

            done = 1;

            for ( j = 0 ; done && ( j < dim() ) ; j++ )
            {
                i("&",j)++;

                done     = i(j)/gran;
                i("&",j) = i(j)%gran;
            }
        }

        res *= ( ( nullpts == totpts ) ? 1.0 : (((double) totpts)/(((double) totpts-nullpts))) );
    }

    return res;
}

gentype &FuncVector::inner3(gentype &res, const Vector<gentype> &bb, const Vector<gentype> &cc) const
{
    const FuncVector &b = getFuncpart(bb);
    const FuncVector &c = getFuncpart(cc);

    NiceAssert(   !ismixed() );
    NiceAssert( !b.ismixed() );
    NiceAssert( !c.ismixed() );
    NiceAssert( ( b.type() == 1 ) || ( dim() == b.dim() ) );
    NiceAssert( ( c.type() == 1 ) || ( dim() == c.dim() ) );

    int gran = samplesize() ? samplesize() : DEFAULT_SAMPLES_SAMPLE;
    double unitsize = pow(1.0/((double) gran),dim());
    int j;

    if ( dim() == 1 )
    {
          unsafesample(gran);
        b.unsafesample(gran);
        c.unsafesample(gran);

        gentype av;

        setzero(res);
        setzero(av);

        int totpts = 0;
        int nullpts = 0;

        for ( j = 0 ; j < gran ; j++ )
        {
            totpts++;

            if ( (*this)(j).isValNull() || b(j).isValNull() || c(j).isValNull() )
            {
                nullpts++;
            }

            else
            {
                av  = (*this)(j);
                av *= b(j);
                av *= c(j);
                av *= unitsize;

                res += av;
            }
        }

        res *= ( ( nullpts == totpts ) ? 1.0 : (((double) totpts)/(((double) totpts-nullpts))) );
    }

    else
    {
        gentype av;
        gentype bv;
        gentype cv;

        setzero(res);
        setzero(av);
        setzero(bv);
        setzero(cv);

        Vector<int> i(dim());
        SparseVector<gentype> xx;

        i = zeroint();

        int totpts = 0;
        int nullpts = 0;

        int done = dim() ? 0 : 1;

        while ( !done )
        {
            for ( j = 0 ; j < dim() ; j++ )
            {
                xx("&",j) = ((double) i(j))/((double) gran);
            }

            totpts++;

            (*this)(av,xx); av.finalise();
            b(bv,xx);       bv.finalise();
            c(cv,xx);       cv.finalise();

            if ( av.isValNull() || bv.isValNull() || cv.isValNull() )
            {
                nullpts++;
            }

            else
            {
                av *= bv;
                av *= cv;
                av *= unitsize;

                res += av;
            }

            done = 1;

            for ( j = 0 ; done && ( j < dim() ) ; j++ )
            {
                i("&",j)++;

                done     = i(j)/gran;
                i("&",j) = i(j)%gran;
            }
        }

        res *= ( ( nullpts == totpts ) ? 1.0 : (((double) totpts)/(((double) totpts-nullpts))) );
    }

    return res;
}

gentype &FuncVector::inner4(gentype &res, const Vector<gentype> &bb, const Vector<gentype> &cc, const Vector<gentype> &dd) const
{
    const FuncVector &b = getFuncpart(bb);
    const FuncVector &c = getFuncpart(cc);
    const FuncVector &d = getFuncpart(dd);

    NiceAssert(   !ismixed() );
    NiceAssert( !b.ismixed() );
    NiceAssert( !c.ismixed() );
    NiceAssert( !d.ismixed() );
    NiceAssert( ( b.type() == 1 ) || ( dim() == b.dim() ) );
    NiceAssert( ( c.type() == 1 ) || ( dim() == c.dim() ) );
    NiceAssert( ( d.type() == 1 ) || ( dim() == d.dim() ) );

    int gran = samplesize() ? samplesize() : DEFAULT_SAMPLES_SAMPLE;
    double unitsize = pow(1.0/((double) gran),dim());
    int j;

    if ( dim() == 1 )
    {
          unsafesample(gran);
        b.unsafesample(gran);
        c.unsafesample(gran);
        d.unsafesample(gran);

        gentype av;

        setzero(res);
        setzero(av);

        int totpts = 0;
        int nullpts = 0;

        for ( j = 0 ; j < gran ; j++ )
        {
            totpts++;

            if ( (*this)(j).isValNull() || b(j).isValNull() || c(j).isValNull() || d(j).isValNull() )
            {
                nullpts++;
            }

            else
            {
                av  = (*this)(j);
                av *= b(j);
                av *= c(j);
                av *= d(j);
                av *= unitsize;

                res += av;
            }
        }

        res *= ( ( nullpts == totpts ) ? 1.0 : (((double) totpts)/(((double) totpts-nullpts))) );
    }

    else
    {
        gentype av;
        gentype bv;
        gentype cv;
        gentype dv;

        setzero(res);
        setzero(av);
        setzero(bv);
        setzero(cv);
        setzero(dv);

        Vector<int> i(dim());
        SparseVector<gentype> xx;

        i = zeroint();

        int totpts = 0;
        int nullpts = 0;

        int done = dim() ? 0 : 1;

        while ( !done )
        {
            for ( j = 0 ; j < dim() ; j++ )
            { 
                xx("&",j) = ((double) i(j))/((double) gran);
            }

            totpts++;

            (*this)(av,xx); av.finalise();
            b(bv,xx);       bv.finalise();
            c(cv,xx);       cv.finalise();
            d(dv,xx);       dv.finalise();

            if ( av.isValNull() || bv.isValNull() || cv.isValNull() || dv.isValNull() )
            {
                nullpts++;
            }

            else
            {
                av *= bv;
                av *= cv;
                av *= dv;
                av *= unitsize;

                res += av;
            }

            done = 1;

            for ( j = 0 ; done && ( j < dim() ) ; j++ )
            {
                i("&",j)++;

                done     = i(j)/gran;
                i("&",j) = i(j)%gran;
            }
        }

        res *= ( ( nullpts == totpts ) ? 1.0 : (((double) totpts)/(((double) totpts-nullpts))) );
    }

    return res;
}

gentype &FuncVector::innerp(gentype &res, const Vector<const Vector<gentype> *> &bb) const
{
    int p = bb.size()+1;

    NiceAssert( !ismixed() );

    if ( p == 1 ) { return inner1(res); }
    if ( p == 2 ) { return inner2(res,*(bb(0))); }
    if ( p == 3 ) { return inner3(res,*(bb(0)),*(bb(1))); }
    if ( p == 4 ) { return inner4(res,*(bb(0)),*(bb(1)),*(bb(2))); }

    Vector<const FuncVector *> b(p-1);

    int gran = samplesize() ? samplesize() : DEFAULT_SAMPLES_SAMPLE;
    double unitsize = pow(1.0/((double) gran),dim());

    int j,k;

    for ( k = 1 ; k < p ; k++ )
    {
        b("&",k-1) = &getFuncpart(*(bb(k-1)));

        NiceAssert( !(*(b(k-1))).ismixed() );
        NiceAssert( ( (*(b(k-1))).type() == 1 ) || ( dim() == (*(b(k-1))).dim() ) );

        if ( dim() == 1 )
        {
            (*(b(k-1))).unsafesample(gran);
        }
    }

    if ( dim() == 1 )
    {
        unsafesample(gran);

        gentype av;

        setzero(res);
        setzero(av);

        int totpts = 0;
        int nullpts = 0;
        int isnull = 0;

        for ( j = 0 ; j < gran ; j++ )
        {
            totpts++;

            av = (*this)(j);

            isnull = av.isValNull();

            for ( k = 1 ; !isnull && ( k < p ) ; k++ )
            {
                isnull |= (*(b(k-1)))(j).isValNull();

                av *= (*(b(k-1)))(j);
            }

            if ( isnull )
            {
                nullpts++;
            }

            else
            {
                av *= unitsize;
                res += av;
            }
        }

        res *= ( ( nullpts == totpts ) ? 1.0 : (((double) totpts)/(((double) totpts-nullpts))) );
    }

    else
    {
        gentype av;
        gentype bv;

        setzero(res);
        setzero(av);
        setzero(bv);

        Vector<int> i(dim());
        SparseVector<gentype> xx;

        i = zeroint();

        int totpts = 0;
        int nullpts = 0;
        int isnull = 0;

        int done = dim() ? 0 : 1;

        while ( !done )
        {
            for ( j = 0 ; j < dim() ; j++ )
            {
                xx("&",j) = ((double) i(j))/((double) gran);
            }

            totpts++;

            (*this)(av,xx); av.finalise();

            isnull = av.isValNull();

            for ( k = 1 ; !isnull && ( k < p ) ; k++ )
            {
                (*(b(k-1)))(bv,xx); bv.finalise();

                isnull |= bv.isValNull();

                av *= bv;
            }

            av *= unitsize;

            if ( isnull ) { nullpts++; }
            else          { res += av; }

            done = 1;

            for ( j = 0 ; done && ( j < dim() ) ; j++ )
            {
                i("&",j)++;

                done     = i(j)/gran;
                i("&",j) = i(j)%gran;
            }
        }

        res *= ( ( nullpts == totpts ) ? 1.0 : (((double) totpts)/(((double) totpts-nullpts))) );
    }

    return res;
}

double &FuncVector::inner1Real(double &res) const
{
    NiceAssert( !ismixed() );

    int gran = samplesize() ? samplesize() : DEFAULT_SAMPLES_SAMPLE;
    double unitsize = pow(1.0/((double) gran),dim());
    int j;

    if ( dim() == 1 )
    {
        unsafesample(gran);

        setzero(res);

        int totpts = 0;
        int nullpts = 0;

        for ( j = 0 ; j < gran ; j++ )
        {
            totpts++;

            if ( (*this)(j).isValNull() )
            {
                nullpts++;
            }

            else
            {
                res += unitsize*((double) (*this)(j));
            }
        }

        res *= ( ( nullpts == totpts ) ? 1.0 : (((double) totpts)/(((double) totpts-nullpts))) );
    }

    else
    {
        gentype av;

        setzero(res);
        setzero(av);

        Vector<int> i(dim());
        SparseVector<gentype> xx;

        i = zeroint();

        int totpts = 0;
        int nullpts = 0;

        int done = dim() ? 0 : 1;

        while ( !done )
        {
            for ( j = 0 ; j < dim() ; j++ )
            {
                xx("&",j) = ((double) i(j))/((double) gran);
            }

            totpts++;

            (*this)(av,xx); av.finalise();

            if ( av.isValNull() )
            {
                nullpts++;
            }

            else
            {
                res += unitsize*((double) av);
            }

            done = 1;

            for ( j = 0 ; done && ( j < dim() ) ; j++ )
            {
                i("&",j)++;

                done     = i(j)/gran;
                i("&",j) = i(j)%gran;
            }
        }

        res *= ( ( nullpts == totpts ) ? 1.0 : (((double) totpts)/(((double) totpts-nullpts))) );
    }

    return res;
}

double &FuncVector::inner2Real(double &res, const Vector<gentype> &bb, int conj) const
{
    (void) conj;

    const FuncVector &b = getFuncpart(bb);

    NiceAssert(   !ismixed() );
    NiceAssert( !b.ismixed() );
    NiceAssert( ( b.type() == 1 ) || ( dim() == b.dim() ) );

    int gran = samplesize() ? samplesize() : DEFAULT_SAMPLES_SAMPLE;
    double unitsize = pow(1.0/((double) gran),dim());
    int j;

    if ( dim() == 1 )
    {
          unsafesample(gran);
        b.unsafesample(gran);

        setzero(res);

        int totpts = 0;
        int nullpts = 0;

        for ( j = 0 ; j < gran ; j++ )
        {
            totpts++;

            if ( (*this)(j).isValNull() || b(j).isValNull() )
            {
                nullpts++;
            }

            else
            {
                res += unitsize*((double) (*this)(j))*((double) b(j));
            }
        }

        res *= ( ( nullpts == totpts ) ? 1.0 : (((double) totpts)/(((double) totpts-nullpts))) );
    }

    else
    {
        gentype av;
        gentype bv;

        setzero(res);
        setzero(av);
        setzero(bv);

        Vector<int> i(dim());
        SparseVector<gentype> xx;

        i = zeroint();

        int totpts = 0;
        int nullpts = 0;

        int done = dim() ? 0 : 1;

        while ( !done )
        {
            for ( j = 0 ; j < dim() ; j++ )
            {
                xx("&",j) = ((double) i(j))/((double) gran);
            }

            totpts++;

            (*this)(av,xx); av.finalise();
            b(bv,xx);       bv.finalise();

            if ( av.isValNull() || bv.isValNull() )
            {
                nullpts++;
            }

            else
            {
                res += unitsize*((double) av)*((double) bv);
            }

            done = 1;

            for ( j = 0 ; done && ( j < dim() ) ; j++ )
            {
                i("&",j)++;

                done     = i(j)/gran;
                i("&",j) = i(j)%gran;
            }
        }

        res *= ( ( nullpts == totpts ) ? 1.0 : (((double) totpts)/(((double) totpts-nullpts))) );
    }

    return res;
}

double &FuncVector::inner3Real(double &res, const Vector<gentype> &bb, const Vector<gentype> &cc) const
{
    const FuncVector &b = getFuncpart(bb);
    const FuncVector &c = getFuncpart(cc);

    NiceAssert(   !ismixed() );
    NiceAssert( !b.ismixed() );
    NiceAssert( !c.ismixed() );
    NiceAssert( ( b.type() == 1 ) || ( dim() == b.dim() ) );
    NiceAssert( ( c.type() == 1 ) || ( dim() == c.dim() ) );

    int gran = samplesize() ? samplesize() : DEFAULT_SAMPLES_SAMPLE;
    double unitsize = pow(1.0/((double) gran),dim());
    int j;

    if ( dim() == 1 )
    {
          unsafesample(gran);
        b.unsafesample(gran);
        c.unsafesample(gran);

        setzero(res);

        int totpts = 0;
        int nullpts = 0;

        for ( j = 0 ; j < gran ; j++ )
        {
            totpts++;

            if ( (*this)(j).isValNull() || b(j).isValNull() || c(j).isValNull() )
            {
                nullpts++;
            }

            else
            {
                res += unitsize*((double) (*this)(j))*((double) b(j))*((double) c(j));
            }
        }

        res *= ( ( nullpts == totpts ) ? 1.0 : (((double) totpts)/(((double) totpts-nullpts))) );
    }

    else
    {
        gentype av;
        gentype bv;
        gentype cv;

        setzero(res);
        setzero(av);
        setzero(bv);
        setzero(cv);

        Vector<int> i(dim());
        SparseVector<gentype> xx;

        i = zeroint();

        int totpts = 0;
        int nullpts = 0;

        int done = dim() ? 0 : 1;

        while ( !done )
        {
            for ( j = 0 ; j < dim() ; j++ )
            {
                xx("&",j) = ((double) i(j))/((double) gran);
            }

            totpts++;

            (*this)(av,xx); av.finalise();
            b(bv,xx);       bv.finalise();
            c(cv,xx);       cv.finalise();

            if ( av.isValNull() || bv.isValNull() || cv.isValNull() )
            {
                nullpts++;
            }

            else
            {
                res += unitsize*((double) av)*((double) bv)*((double) cv);
            }

            done = 1;

            for ( j = 0 ; done && ( j < dim() ) ; j++ )
            {
                i("&",j)++;

                done     = i(j)/gran;
                i("&",j) = i(j)%gran;
            }
        }

        res *= ( ( nullpts == totpts ) ? 1.0 : (((double) totpts)/(((double) totpts-nullpts))) );
    }

    return res;
}

double &FuncVector::inner4Real(double &res, const Vector<gentype> &bb, const Vector<gentype> &cc, const Vector<gentype> &dd) const
{
    const FuncVector &b = getFuncpart(bb);
    const FuncVector &c = getFuncpart(cc);
    const FuncVector &d = getFuncpart(dd);

    NiceAssert(   !ismixed() );
    NiceAssert( !b.ismixed() );
    NiceAssert( !c.ismixed() );
    NiceAssert( !d.ismixed() );
    NiceAssert( ( b.type() == 1 ) || ( dim() == b.dim() ) );
    NiceAssert( ( c.type() == 1 ) || ( dim() == c.dim() ) );
    NiceAssert( ( d.type() == 1 ) || ( dim() == d.dim() ) );

    int gran = samplesize() ? samplesize() : DEFAULT_SAMPLES_SAMPLE;
    double unitsize = pow(1.0/((double) gran),dim());
    int j;

    if ( dim() == 1 )
    {
          unsafesample(gran);
        b.unsafesample(gran);
        c.unsafesample(gran);
        d.unsafesample(gran);

        setzero(res);

        int totpts = 0;
        int nullpts = 0;

        for ( j = 0 ; j < gran ; j++ )
        {
            totpts++;

            if ( (*this)(j).isValNull() || b(j).isValNull() || c(j).isValNull() || d(j).isValNull() )
            {
                nullpts++;
            }

            else
            {
                res += unitsize*((double) (*this)(j))*((double) b(j))*((double) c(j))*((double) d(j));
            }
        }

        res *= ( ( nullpts == totpts ) ? 1.0 : (((double) totpts)/(((double) totpts-nullpts))) );
    }

    else
    {
        gentype av;
        gentype bv;
        gentype cv;
        gentype dv;

        setzero(res);
        setzero(av);
        setzero(bv);
        setzero(cv);
        setzero(dv);

        Vector<int> i(dim());
        SparseVector<gentype> xx;

        i = zeroint();

        int totpts = 0;
        int nullpts = 0;

        int done = dim() ? 0 : 1;

        while ( !done )
        {
            for ( j = 0 ; j < dim() ; j++ )
            {
                xx("&",j) = ((double) i(j))/((double) gran);
            }

            totpts++;

            (*this)(av,xx); av.finalise();
            b(bv,xx);       bv.finalise();
            c(cv,xx);       cv.finalise();
            d(dv,xx);       dv.finalise();

            if ( av.isValNull() || bv.isValNull() || cv.isValNull() || dv.isValNull() )
            {
                nullpts++;
            }

            else
            {
                res += unitsize*((double) av)*((double) bv)*((double) cv)*((double) dv);
            }

            done = 1;

            for ( j = 0 ; done && ( j < dim() ) ; j++ )
            {
                i("&",j)++;

                done     = i(j)/gran;
                i("&",j) = i(j)%gran;
            }
        }

        res *= ( ( nullpts == totpts ) ? 1.0 : (((double) totpts)/(((double) totpts-nullpts))) );
    }

    return res;
}

double &FuncVector::innerpReal(double &res, const Vector<const Vector<gentype> *> &bb) const
{
    NiceAssert( !ismixed() );

    int p = bb.size()+1;

    if ( p == 1 ) { return inner1Real(res); }
    if ( p == 2 ) { return inner2Real(res,*(bb(0))); }
    if ( p == 3 ) { return inner3Real(res,*(bb(0)),*(bb(1))); }
    if ( p == 4 ) { return inner4Real(res,*(bb(0)),*(bb(1)),*(bb(2))); }

    Vector<const FuncVector *> b(p-1);

    int gran = samplesize() ? samplesize() : DEFAULT_SAMPLES_SAMPLE;
    double unitsize = pow(1.0/((double) gran),dim());

    int j,k;

    for ( k = 1 ; k < p ; k++ )
    {
        b("&",k-1) = &getFuncpart(*(bb(k-1)));

        NiceAssert( !(*(b(k-1))).ismixed() );
        NiceAssert( ( (*(b(k-1))).type() == 1 ) || ( dim() == (*(b(k-1))).dim() ) );

        if ( dim() == 1 )
        {
            (*(b(k-1))).unsafesample(gran);
        }
    }

    if ( dim() == 1 )
    {
        setzero(res);

        int totpts = 0;
        int nullpts = 0;
        int isnull = 0;

        double av;

        for ( j = 0 ; j < gran ; j++ )
        {
            totpts++;

            isnull = ((*this)(j)).isValNull();

            av = isnull ? 0.0 : ((double) (*this)(j));

            for ( k = 1 ; !isnull && ( k < p ) ; k++ )
            {
                isnull |= (*(b(k-1)))(j).isValNull();

                if ( !isnull )
                {
                    av *= ((double) (*(b(k-1)))(j));
                }
            }

            if ( isnull )
            {
                nullpts++;
            }

            else
            {
                res += unitsize*av;
            }
        }

        res *= ( ( nullpts == totpts ) ? 1.0 : (((double) totpts)/(((double) totpts-nullpts))) );
    }

    else
    {
        gentype av;
        gentype bv;

        setzero(res);
        setzero(av);
        setzero(bv);

        Vector<int> i(dim());
        SparseVector<gentype> xx;

        i = zeroint();

        int totpts = 0;
        int nullpts = 0;
        int isnull = 0;

        int done = dim() ? 0 : 1;

        while ( !done )
        {
            for ( j = 0 ; j < dim() ; j++ )
            {
                xx("&",j) = ((double) i(j))/((double) gran);
            }

            totpts++;

            (*this)(av,xx); av.finalise();

            isnull = av.isValNull();

            for ( k = 1 ; !isnull && ( k < p ) ; k++ )
            {
                (*(b(k-1)))(bv,xx); bv.finalise();

                isnull |= bv.isValNull();

                av *= bv;
            }

            if ( isnull )
            {
                nullpts++;
            }

            else
            {
                res += unitsize*((double) av);
            }

            done = 1;

            for ( j = 0 ; done && ( j < dim() ) ; j++ )
            {
                i("&",j)++;

                done     = i(j)/gran;
                i("&",j) = i(j)%gran;
            }
        }

        res *= ( ( nullpts == totpts ) ? 1.0 : (((double) totpts)/(((double) totpts-nullpts))) );
    }

    return res;
}

double FuncVector::absinf(void) const
{
    NiceAssert( !ismixed() );

    double res = 0.0;

    int gran = samplesize() ? samplesize() : DEFAULT_SAMPLES_SAMPLE;
    int j;

    if ( dim() == 1 )
    {
        unsafesample(gran);

        double avr = 0.0;

        for ( j = 0 ; j < gran ; j++ )
        {
            if ( !((*this)(j).isValNull()) )
            {
                avr = ::absinf((*this)(j));
                res = ( res > avr ) ? res : avr;
            }
        }
    }

    else
    {
        gentype av;
        double avr = 0.0;

        setzero(av);

        Vector<int> i(dim());
        SparseVector<gentype> xx;

        i = zeroint();

        int done = dim() ? 0 : 1;

        while ( !done )
        {
            for ( j = 0 ; j < dim() ; j++ )
            {
                xx("&",j) = ((double) i(j))/((double) gran);
            }

            (*this)(av,xx); av.finalise();

            if ( !(av.isValNull()) )
            {
                avr = ::absinf(av);
                res = ( res > avr ) ? res : avr;
            }

            done = 1;

            for ( j = 0 ; done && ( j < dim() ) ; j++ )
            {
                i("&",j)++;

                done     = i(j)/gran;
                i("&",j) = i(j)%gran;
            }
        }
    }

    return res;
}


Vector<gentype> &FuncVector::subit(const Vector<gentype> &b)
{ 
    NiceAssert( b.infsize() );

    unsample(); 

    if ( b.imoverhere ) 
    { 
        subit(*(b.imoverhere));
    } 

    else if ( ( b.type() == 1 ) && !b.ismixed() )
    {
        const FuncVector &bb = dynamic_cast<const FuncVector &>(b); 

        NiceAssert( fdim == bb.fdim ); 

        valfn -= bb.valfn; 
    } 

    else
    {
        int i = NE();

        extrapart.add(i);

        extrapart("&",i) = dynamic_cast<FuncVector *>(b.makeDup());
        (*extrapart("&",i)).negate();
    }

    return *this; 
}

Vector<gentype> &FuncVector::addit(const Vector<gentype> &b)
{ 
    NiceAssert( b.infsize() );

    unsample(); 

    if ( b.imoverhere ) 
    { 
        addit(*(b.imoverhere));
    } 

    else if ( ( b.type() == 1 ) && !b.ismixed() )
    {
        const FuncVector &bb = dynamic_cast<const FuncVector &>(b); 

        NiceAssert( fdim == bb.fdim ); 

        valfn += bb.valfn; 
    } 

    else
    {
        int i = NE();

        extrapart.add(i);

        extrapart("&",i) = dynamic_cast<FuncVector *>(b.makeDup());
    }

    return *this; 
}

Vector<gentype> &FuncVector::addit(const gentype &b)
{
    precalcVec += b;

    valfn += b;

    return *this;
}

Vector<gentype> &FuncVector::subit(const gentype &b)
{
    precalcVec -= b;

    valfn -= b;

    return *this;
}

Vector<gentype> &FuncVector::mulit(const gentype &b)
{
    precalcVec *= b;

    valfn *= b;

    if ( NE() )
    {
        int i;

        for ( i = 0 ; i < NE() ; i++ )
        {
            (*extrapart("&",i)).mulit(b);
        }
    }

    return *this;
}

Vector<gentype> &FuncVector::rmulit(const gentype &b)
{
    rightmult(b,precalcVec);

    rightmult(b,valfn);

    if ( NE() )
    {
        int i;

        for ( i = 0 ; i < NE() ; i++ )
        {
            (*extrapart("&",i)).rmulit(b);
        }
    }

    return *this;
}

Vector<gentype> &FuncVector::divit(const gentype &b)
{
    precalcVec /= b;

    valfn /= b;

    if ( NE() )
    {
        int i;

        for ( i = 0 ; i < NE() ; i++ )
        {
            (*extrapart("&",i)).divit(b);
        }
    }

    return *this;
}

Vector<gentype> &FuncVector::rdivit(const gentype &b)
{
    rightmult(inv(b),precalcVec);

    rightmult(inv(b),valfn);

    if ( NE() )
    {
        int i;

        for ( i = 0 ; i < NE() ; i++ )
        {
            (*extrapart("&",i)).rdivit(b);
        }
    }

    return *this;
}

Vector<gentype> &FuncVector::mulit(const Vector<gentype> &b)
{ 
    NiceAssert( b.infsize() );

    unsample(); 

    if ( b.imoverhere ) 
    { 
        mulit(*(b.imoverhere));
    } 

    else if ( ( b.type() == 1 ) && !b.ismixed() )
    {
        const FuncVector &bb = dynamic_cast<const FuncVector &>(b); 

        NiceAssert( fdim == bb.fdim ); 

        valfn *= bb.valfn; 
    } 

    else
    {
        throw("Mixed functional vector multiplication is ill-defined");
    } 

    return *this; 
}

Vector<gentype> &FuncVector::rmulit(const Vector<gentype> &b)
{ 
    NiceAssert( b.infsize() );

    unsample(); 

    if ( b.imoverhere ) 
    { 
        rmulit(*(b.imoverhere));
    } 

    else if ( ( b.type() == 1 ) && !b.ismixed() )
    {
        const FuncVector &bb = dynamic_cast<const FuncVector &>(b); 

        NiceAssert( fdim == bb.fdim ); 

        rightmult(bb.valfn,valfn); 
    } 

    else
    {
        throw("Mixed functional vector multiplication is ill-defined");
    } 

    return *this; 
}

Vector<gentype> &FuncVector::divit(const Vector<gentype> &b)
{ 
    NiceAssert( b.infsize() );

    unsample(); 

    if ( b.imoverhere ) 
    { 
        divit(*(b.imoverhere));
    } 

    else if ( ( b.type() == 1 ) && !b.ismixed() )
    {
        const FuncVector &bb = dynamic_cast<const FuncVector &>(b); 

        NiceAssert( fdim == bb.fdim ); 

        valfn /= bb.valfn; 
    } 

    else
    {
        throw("Mixed functional vector multiplication is ill-defined");
    } 

    return *this; 
}

Vector<gentype> &FuncVector::rdivit(const Vector<gentype> &b)
{ 
    NiceAssert( b.infsize() );

    unsample(); 

    if ( b.imoverhere ) 
    { 
        rdivit(*(b.imoverhere));
    } 

    else if ( ( b.type() == 1 ) && !b.ismixed() )
    {
        const FuncVector &bb = dynamic_cast<const FuncVector &>(b); 

        NiceAssert( fdim == bb.fdim ); 

        rightmult(inv(bb.valfn),valfn); 
    } 

    else
    {
        throw("Mixed functional vector multiplication is ill-defined");
    } 

    return *this; 
}

FuncVector &FuncVector::assign(const FuncVector &src) 
{ 
    //NONONO!!! Vector<gentype>::assign(static_cast<const Vector<gentype> &>(src));

    precalcVec = src.precalcVec;
    valfn      = src.valfn;
    fdim       = src.fdim;

    if ( NE() )
    {
        int i; 

        for ( i = 0 ; i < NE() ; i++ ) 
        { 
            MEMDEL(extrapart("&",i)); 
            extrapart("&",i) = NULL; 
        } 

        extrapart.resize(0); 
    }
 
    if ( src.NE() )
    {
        extrapart.resize(src.NE());

        int i; 

        for ( i = 0 ; i < src.NE() ; i++ ) 
        { 
            extrapart("&",i) = dynamic_cast<FuncVector *>((*(src.extrapart(i))).makeDup());
        }
    }

    return *this; 
}

gentype &FuncVector::operator()(gentype &res, const SparseVector<gentype> &i) const 
{ 
    SparseVector<SparseVector<gentype> > ii; 

    ii("&",zeroint()) = i; 

    res = valfn(ii);

    if ( NE() )
    {
        int i;

        for ( i = 0 ; i < NE() ; i++ )
        {
            res += (dynamic_cast<const FuncVector &>(*extrapart(i)))(i);
        }
    }

    return res; 
}

void FuncVector::sample(int Nsamp)
{
    NiceAssert( Nsamp > 0 );
    NiceAssert( dim() == 1 );

    if ( precalcVec.size() != Nsamp )
    {
        int i;

        // Take uniform samples on x

        SparseVector<gentype> x;

        precalcVec.resize(Nsamp);

        for ( i = 0 ; i < Nsamp ; i++ )
        {
            x("&",zeroint()) = (((double) i)+0.5)/(((double) Nsamp));

            ((*this)(precalcVec("&",i),x)).finalise();
        }
    }

    return;
}





















































gentype &RKHSVector::inner1(gentype &res) const
{
    if ( ismixed() ) { return FuncVector::inner1(res); }

    int i;

    setzero(res);

    gentype tmpa;
    gentype tmpb;
    gentype tmpc;

    setzero(tmpa);
    setzero(tmpb);
    setzero(tmpc);

    for ( i = 0 ; i < N() ; i++ )
    {
        vecInfo iinfo; spKern.getvecInfo(iinfo,x(i));

        tmpa = a(i);
        tmpa *= spKern.K1(tmpb,x(i),iinfo,tmpc);

        res += tmpa;
    }

    return res;
}

gentype &RKHSVector::inner2(gentype &res, const Vector<gentype> &bb, int conj) const
{
    if (                          ismixed() ) { return FuncVector::inner2(res,bb,conj); }
    if ( ( bb.type() != 2 ) || bb.ismixed() ) { return bb.inner2(res,*this,( ( conj == 1 ) ? 2 : ( conj == 2 ) ? 1 : conj ) ); }

    int i,j;

    const RKHSVector &b = getRKHSpart(bb);

    NiceAssert( kern() == b.kern() );

    setzero(res);

    gentype tmpa; 
    gentype tmpb; 
    gentype tmpc;
    gentype tmpd;

    setzero(tmpa);
    setzero(tmpb);
    setzero(tmpc);
    setzero(tmpd);

    for ( i = 0 ; i < N() ; i++ )
    {
        vecInfo iinfo; spKern.getvecInfo(iinfo,x(i));

        for ( j = 0 ; j < b.N() ; j++ )
        {
            vecInfo jinfo; spKern.getvecInfo(jinfo,b.x(j));

            tmpa = a(i);
            tmpd = b.a(j);

            if ( conj & 1 )
            {
                setconj(tmpa);
            }

            if ( conj & 2 )
            {
                setconj(tmpd);
            }

            tmpa *= spKern.K2(tmpb,x(i),b.x(j),iinfo,jinfo,tmpc);
            tmpa *= tmpd;

            res += tmpa;
        }
    }

    return res;
}

gentype &RKHSVector::inner3(gentype &res, const Vector<gentype> &bb, const Vector<gentype> &cc) const
{
    if (                          ismixed() ) { return FuncVector::inner3(res,bb,cc); }
    if ( ( bb.type() != 2 ) || bb.ismixed() ) { return bb.inner3(res,*this,cc); }
    if ( ( cc.type() != 2 ) || cc.ismixed() ) { return cc.inner3(res,bb,*this); }

    int i,j,k;

    const RKHSVector &b = getRKHSpart(bb);
    const RKHSVector &c = getRKHSpart(cc);

    NiceAssert( kern() == b.kern() );
    NiceAssert( kern() == c.kern() );

    setzero(res);

    gentype tmpa; 
    gentype tmpb; 
    gentype tmpc;
    gentype tmpd;
    gentype tmpe;

    setzero(tmpa);
    setzero(tmpb);
    setzero(tmpc);
    setzero(tmpd);
    setzero(tmpe);

    for ( i = 0 ; i < N() ; i++ )
    {
        vecInfo iinfo; spKern.getvecInfo(iinfo,x(i));

        for ( j = 0 ; j < b.N() ; j++ )
        {
            vecInfo jinfo; spKern.getvecInfo(jinfo,b.x(j));

            for ( k = 0 ; k < c.N() ; k++ )
            {
                vecInfo kinfo; spKern.getvecInfo(kinfo,c.x(k));

                tmpa = a(i);
                tmpd = b.a(j);
                tmpe = c.a(k);

                tmpa *= spKern.K3(tmpb,x(i),b.x(j),c.x(k),iinfo,jinfo,kinfo,tmpc);
                tmpa *= tmpd;
                tmpa *= tmpe;

                res += tmpa;
            }
        }
    }

    return res;
}

gentype &RKHSVector::inner4(gentype &res, const Vector<gentype> &bb, const Vector<gentype> &cc, const Vector<gentype> &dd) const
{
    if (                          ismixed() ) { return FuncVector::inner4(res,bb,cc,dd); }
    if ( ( bb.type() != 2 ) || bb.ismixed() ) { return bb.inner4(res,*this,cc,dd); }
    if ( ( cc.type() != 2 ) || cc.ismixed() ) { return cc.inner4(res,bb,*this,dd); }
    if ( ( dd.type() != 2 ) || dd.ismixed() ) { return dd.inner4(res,bb,cc,*this); }

    int i,j,k,l;

    const RKHSVector &b = getRKHSpart(bb);
    const RKHSVector &c = getRKHSpart(cc);
    const RKHSVector &d = getRKHSpart(dd);

    NiceAssert( kern() == b.kern() );
    NiceAssert( kern() == c.kern() );
    NiceAssert( kern() == d.kern() );

    setzero(res);

    gentype tmpa; 
    gentype tmpb; 
    gentype tmpc;
    gentype tmpd;
    gentype tmpe;
    gentype tmpf;

    setzero(tmpa);
    setzero(tmpb);
    setzero(tmpc);
    setzero(tmpd);
    setzero(tmpe);
    setzero(tmpf);

    for ( i = 0 ; i < N() ; i++ )
    {
        vecInfo iinfo; spKern.getvecInfo(iinfo,x(i));

        for ( j = 0 ; j < b.N() ; j++ )
        {
            vecInfo jinfo; spKern.getvecInfo(jinfo,b.x(j));

            for ( k = 0 ; k < c.N() ; k++ )
            {
                vecInfo kinfo; spKern.getvecInfo(kinfo,c.x(k));

                for ( l = 0 ; l < d.N() ; l++ )
                {
                    vecInfo linfo; spKern.getvecInfo(linfo,d.x(l));

                    tmpa = a(i);
                    tmpd = b.a(j);
                    tmpe = c.a(k);
                    tmpf = d.a(k);

                    tmpa *= spKern.K4(tmpb,x(i),b.x(j),c.x(k),d.x(l),iinfo,jinfo,kinfo,linfo,tmpc);
                    tmpa *= tmpd;
                    tmpa *= tmpe;
                    tmpa *= tmpf;

                    res += tmpa;
                }
            }
        }
    }

    return res;
}

gentype &RKHSVector::innerp(gentype &res, const Vector<const Vector<gentype> *> &bb) const
{
    int p = bb.size()+1;

    setzero(res);

    if ( p == 1 ) { return inner1(res); }
    if ( p == 2 ) { return inner2(res,*(bb(0))); }
    if ( p == 3 ) { return inner3(res,*(bb(0)),*(bb(1))); }
    if ( p == 4 ) { return inner4(res,*(bb(0)),*(bb(1)),*(bb(2))); }

    if ( ismixed() ) { return FuncVector::innerp(res,bb); }

    Vector<const RKHSVector *> b(p-1);

    int j,k;

    int zerosize = 0;

    for ( k = 1 ; k < p ; k++ )
    {
        if ( ( (*(bb(k-1))).type() != 2 ) || (*(bb(k-1))).ismixed() )
        {
            Vector<const Vector<gentype> *> bbb(bb);

            bbb("&",k-1) = this;

            return (*(bb(k-1))).innerp(res,bbb);
        }

        b("&",k-1) = &getRKHSpart(*(bb(k-1)));

        if ( !((*(b(k-1))).N()) )
        {
            zerosize = 1;
        }

        NiceAssert( kern() == (*(b(k-1))).kern() );
    }

    if ( !zerosize )
    {
        Vector<const SparseVector<gentype> *> xxx(p);
        Vector<vecInfo> xxxi(p);
        Vector<const vecInfo *> xxxii(p);
        Vector<int> i(p);
        int done = 0;
        gentype tmpa,tmpb,tmpc;

        setzero(tmpc);

        for ( j = 1 ; j < p ; j++ )
        {
            xxxii("&",j-1) = &(xxxi(j-1));
        }

        i = zeroint();

        while ( !done )
        {
            xxx("&",0) = &(x(i(0)));
            spKern.getvecInfo(xxxi("&",0),(*(xxx(0))));

            tmpa = alpha(i(0));

            for ( j = 1 ; j < p ; j++ )
            {
                xxx("&",j) = &((*(b(j-1))).x(i(j)));
                spKern.getvecInfo(xxxi("&",j),(*(xxx(j))));

                tmpa *= (*(b(j-1))).alpha(i(j));
            }

            tmpa *= spKern.Km(p,tmpb,xxx,xxxii,tmpc,i);

            res += tmpa;

            done = 1;

            i("&",0)++;

            done     = i(0)/N();
            i("&",0) = i(0)%N();

            for ( j = 1 ; done && ( j < p ) ; j++ )
            {
                i("&",j)++;

                done     = i(j)/((*(b(j-1))).N());
                i("&",j) = i(j)%((*(b(j-1))).N());
            }
        }
    }

    return res;
}

double &RKHSVector::inner1Real(double &res) const
{
    if ( ismixed() ) { return FuncVector::inner1Real(res); }

    int i;

    setzero(res);

    double tmpa;
    double tmpb;
    double tmpc;

    setzero(tmpa);
    setzero(tmpb);
    setzero(tmpc);

    for ( i = 0 ; i < N() ; i++ )
    {
        vecInfo iinfo; spKern.getvecInfo(iinfo,x(i));

        tmpa = (double) a(i);
        tmpa *= spKern.K1(tmpb,x(i),iinfo,tmpc);

        res += tmpa;
    }

    return res;
}

double &RKHSVector::inner2Real(double &res, const Vector<gentype> &bb, int conj) const
{
    if (                          ismixed() ) { return FuncVector::inner2Real(res,bb,conj); }
    if ( ( bb.type() != 2 ) || bb.ismixed() ) { return bb.inner2Real(res,*this,( ( conj == 1 ) ? 2 : ( ( conj == 2 ) ? 1 : conj ) )); }

    (void) conj;

    int i,j;

    const RKHSVector &b = getRKHSpart(bb);

    NiceAssert( kern() == b.kern() );

    setzero(res);

    double tmpa; 
    double tmpb; 
    double tmpc;
    double tmpd;

    setzero(tmpa);
    setzero(tmpb);
    setzero(tmpc);
    setzero(tmpd);

    for ( i = 0 ; i < N() ; i++ )
    {
        vecInfo iinfo; spKern.getvecInfo(iinfo,x(i));

        for ( j = 0 ; j < b.N() ; j++ )
        {
            vecInfo jinfo; spKern.getvecInfo(jinfo,b.x(j));

            tmpa = (double) a(i);
            tmpd = (double) b.a(j);

            tmpa *= spKern.K2(tmpb,x(i),b.x(j),iinfo,jinfo,tmpc);
            tmpa *= tmpd;

            res += tmpa;
        }
    }

    return res;
}

double &RKHSVector::inner3Real(double &res, const Vector<gentype> &bb, const Vector<gentype> &cc) const
{
    if (                          ismixed() ) { return FuncVector::inner3Real(res,bb,cc); }
    if ( ( bb.type() != 2 ) || bb.ismixed() ) { return bb.inner3Real(res,*this,cc); }
    if ( ( cc.type() != 2 ) || cc.ismixed() ) { return cc.inner3Real(res,bb,*this); }

    int i,j,k;

    const RKHSVector &b = getRKHSpart(bb);
    const RKHSVector &c = getRKHSpart(cc);

    NiceAssert( kern() == b.kern() );
    NiceAssert( kern() == c.kern() );

    setzero(res);

    double tmpa; 
    double tmpb; 
    double tmpc;
    double tmpd;
    double tmpe;

    setzero(tmpa);
    setzero(tmpb);
    setzero(tmpc);
    setzero(tmpd);
    setzero(tmpe);

    for ( i = 0 ; i < N() ; i++ )
    {
        vecInfo iinfo; spKern.getvecInfo(iinfo,x(i));

        for ( j = 0 ; j < b.N() ; j++ )
        {
            vecInfo jinfo; spKern.getvecInfo(jinfo,b.x(j));

            for ( k = 0 ; k < c.N() ; k++ )
            {
                vecInfo kinfo; spKern.getvecInfo(kinfo,c.x(k));

                tmpa = (double) a(i);
                tmpd = (double) b.a(j);
                tmpe = (double) c.a(k);

                tmpa *= spKern.K3(tmpb,x(i),b.x(j),c.x(k),iinfo,jinfo,kinfo,tmpc);
                tmpa *= tmpd;
                tmpa *= tmpe;

                res += tmpa;
            }
        }
    }

    return res;
}

double &RKHSVector::inner4Real(double &res, const Vector<gentype> &bb, const Vector<gentype> &cc, const Vector<gentype> &dd) const
{
    if (                          ismixed() ) { return FuncVector::inner4Real(res,bb,cc,dd); }
    if ( ( bb.type() != 2 ) || bb.ismixed() ) { return bb.inner4Real(res,*this,cc,dd); }
    if ( ( cc.type() != 2 ) || cc.ismixed() ) { return cc.inner4Real(res,bb,*this,dd); }
    if ( ( dd.type() != 2 ) || dd.ismixed() ) { return dd.inner4Real(res,bb,cc,*this); }

    int i,j,k,l;

    const RKHSVector &b = getRKHSpart(bb);
    const RKHSVector &c = getRKHSpart(cc);
    const RKHSVector &d = getRKHSpart(dd);

    NiceAssert( kern() == b.kern() );
    NiceAssert( kern() == c.kern() );
    NiceAssert( kern() == d.kern() );

    setzero(res);

    double tmpa; 
    double tmpb; 
    double tmpc;
    double tmpd;
    double tmpe;
    double tmpf;

    setzero(tmpa);
    setzero(tmpb);
    setzero(tmpc);
    setzero(tmpd);
    setzero(tmpe);
    setzero(tmpf);

    for ( i = 0 ; i < N() ; i++ )
    {
        vecInfo iinfo; spKern.getvecInfo(iinfo,x(i));

        for ( j = 0 ; j < b.N() ; j++ )
        {
            vecInfo jinfo; spKern.getvecInfo(jinfo,b.x(j));

            for ( k = 0 ; k < c.N() ; k++ )
            {
                vecInfo kinfo; spKern.getvecInfo(kinfo,c.x(k));

                for ( l = 0 ; l < d.N() ; l++ )
                {
                    vecInfo linfo; spKern.getvecInfo(linfo,d.x(l));

                    tmpa = (double) a(i);
                    tmpd = (double) b.a(j);
                    tmpe = (double) c.a(k);
                    tmpf = (double) d.a(k);

                    tmpa *= spKern.K4(tmpb,x(i),b.x(j),c.x(k),d.x(l),iinfo,jinfo,kinfo,linfo,tmpc);
                    tmpa *= tmpd;
                    tmpa *= tmpe;
                    tmpa *= tmpf;

                    res += tmpa;
                }
            }
        }
    }

    return res;
}

double &RKHSVector::innerpReal(double &res, const Vector<const Vector<gentype> *> &bb) const
{
    int p = bb.size()+1;

    setzero(res);

    if ( p == 1 ) { return inner1Real(res); }
    if ( p == 2 ) { return inner2Real(res,*(bb(0))); }
    if ( p == 3 ) { return inner3Real(res,*(bb(0)),*(bb(1))); }
    if ( p == 4 ) { return inner4Real(res,*(bb(0)),*(bb(1)),*(bb(2))); }

    if ( ismixed() ) { return FuncVector::innerpReal(res,bb); }

    Vector<const RKHSVector *> b(p-1);

    int j,k;

    int zerosize = 0;

    for ( k = 1 ; k < p ; k++ )
    {
        if ( ( (*(bb(k-1))).type() != 2 ) || (*(bb(k-1))).ismixed() )
        {
            Vector<const Vector<gentype> *> bbb(bb);

            bbb("&",k-1) = this;

            return (*(bb(k-1))).innerpReal(res,bbb);
        }

        b("&",k-1) = &getRKHSpart(*(bb(k-1)));

        if ( !((*(b(k-1))).N()) )
        {
            zerosize = 1;
        }

        NiceAssert( kern() == (*(b(k-1))).kern() );
    }

    if ( !zerosize )
    {
        Vector<const SparseVector<gentype> *> xxx(p);
        Vector<vecInfo> xxxi(p);
        Vector<const vecInfo *> xxxii(p);
        Vector<int> i(p);
        int done = 0;
        gentype tmpa,tmpb,tmpc;

        setzero(tmpc);

        for ( j = 1 ; j < p ; j++ )
        {
            xxxii("&",j-1) = &(xxxi(j-1));
        }

        i = zeroint();

        while ( !done )
        {
            xxx("&",0) = &(x(i(0)));
            spKern.getvecInfo(xxxi("&",0),(*(xxx(0))));

            tmpa = alpha(i(0));

            for ( j = 1 ; j < p ; j++ )
            {
                xxx("&",j) = &((*(b(j-1))).x(i(j)));
                spKern.getvecInfo(xxxi("&",j),(*(xxx(j))));

                tmpa *= (*(b(j-1))).alpha(i(j));
            }

            tmpa *= spKern.Km(p,tmpb,xxx,xxxii,tmpc,i);

            res += (double) tmpa;

            done = 1;

            i("&",0)++;

            done     = i(0)/N();
            i("&",0) = i(0)%N();

            for ( j = 1 ; done && ( j < p ) ; j++ )
            {
                i("&",j)++;

                done     = i(j)/((*(b(j-1))).N());
                i("&",j) = i(j)%((*(b(j-1))).N());
            }
        }
    }

    return res;
}

Vector<gentype> &RKHSVector::subit(const Vector<gentype> &b)
{ 
    unsample(); 

    if ( ismixed() )
    {
        FuncVector::subit(b);
    }

    else if ( b.imoverhere ) 
    { 
        subit(*(b.imoverhere));
    } 

    else if ( ( b.type() == 2 ) && !b.ismixed() )
    {
        const RKHSVector &bb = dynamic_cast<const RKHSVector &>(b); 

        NiceAssert( spKern == bb.spKern ); 

        xx.append(xx.size(),bb.xx); 

        alpha.negate(); 
        alpha.append(alpha.size(),bb.alpha); 
        alpha.negate(); 
    } 

    else
    {
        const FuncVector &bb = dynamic_cast<const FuncVector &>(b); 

        FuncVector::fdim  = bb.fdim;
        FuncVector::valfn = bb.valfn;

        (FuncVector::extrapart).resize(1);
        (FuncVector::extrapart)("&",0) = dynamic_cast<FuncVector *>(makeDup());

        (FuncVector::valfn).negate();

        resizeN(0);
        revertToFunc = 1;
    } 

    return *this; 
}

Vector<gentype> &RKHSVector::addit(const Vector<gentype> &b)
{ 
    unsample(); 

    if ( ismixed() )
    {
        FuncVector::addit(b);
    }

    else if ( b.imoverhere ) 
    { 
        addit(*(b.imoverhere));
    } 

    else if ( ( b.type() == 2 ) && !b.ismixed() )
    {
        const RKHSVector &bb = dynamic_cast<const RKHSVector &>(b); 

        NiceAssert( spKern == bb.spKern ); 

        xx.append(xx.size(),bb.xx); 

        alpha.append(alpha.size(),bb.alpha); 
    } 

    else
    {
        const FuncVector &bb = dynamic_cast<const FuncVector &>(b); 

        FuncVector::fdim  = bb.fdim;
        FuncVector::valfn = bb.valfn;

        (FuncVector::extrapart).resize(1);
        (FuncVector::extrapart)("&",0) = dynamic_cast<FuncVector *>(makeDup());

        resizeN(0);
        revertToFunc = 1;
    } 

    return *this; 
}

Vector<gentype> &RKHSVector::subit(const gentype &b)
{
    unsample();

    FuncVector::fdim  = fdim;
    FuncVector::valfn = b;

    (FuncVector::extrapart).resize(1);
    (FuncVector::extrapart)("&",0) = dynamic_cast<FuncVector *>(makeDup());

    (FuncVector::valfn).negate();

    resizeN(0);
    revertToFunc = 1;

    return *this; 
}

Vector<gentype> &RKHSVector::addit(const gentype &b)
{
    unsample();

    FuncVector::fdim  = fdim;
    FuncVector::valfn = b;

    (FuncVector::extrapart).resize(1);
    (FuncVector::extrapart)("&",0) = dynamic_cast<FuncVector *>(makeDup());

    resizeN(0);
    revertToFunc = 1;

    return *this; 
}

Vector<gentype> &RKHSVector::mulit(const gentype &b)
{
    precalcVec *= b;

    if ( ismixed() )
    {
        FuncVector::valfn *= b;
    }

    alpha *= b;              

    return *this; 
}

Vector<gentype> &RKHSVector::rmulit(const gentype &b) 
{ 
    rightmult(b,precalcVec);

    if ( ismixed() )
    {
        rightmult(b,FuncVector::valfn);
    }

    rightmult(b,alpha);      

    return *this; 
}

Vector<gentype> &RKHSVector::divit(const gentype &b) 
{ 
    precalcVec /= b;

    if ( ismixed() )
    {
        FuncVector::valfn /= b;
    }

    alpha /= b;              

    return *this; 
}

Vector<gentype> &RKHSVector::rdivit(const gentype &b) 
{ 
    rightmult(inv(b),precalcVec);

    if ( ismixed() )
    {
        rightmult(inv(b),FuncVector::valfn);
    }

    rightmult(inv(b),alpha); 

    return *this;
}

gentype &RKHSVector::operator()(gentype &res, const SparseVector<gentype> &xx) const
{
    if ( revertToFunc )
    {
        return FuncVector::operator()(res,xx);
    }

    setzero(res);

    if ( N() )
    {
        int i;
        gentype tmp,zerobias;
        vecInfo xinfo;
        vecInfo xxinfo;

        spKern.getvecInfo(xxinfo,xx);

        setzero(tmp);
        setzero(zerobias);

        for ( i = 0 ; i < N() ; i++ )
        {
            spKern.getvecInfo(xinfo,x(i));
            spKern.K2(tmp,x(i),xx,xinfo,xxinfo,zerobias);

            rightmult(a(i),tmp);

            res += tmp;
        }
    }     

    return res;
}




































Vector<gentype> &BernVector::subit(const Vector<gentype> &b)
{ 
    unsample(); 

    if ( ismixed() )
    {
        FuncVector::subit(b);
    }

    else if ( b.imoverhere ) 
    { 
        subit(*(b.imoverhere));
    } 

    else if ( ( b.type() == 3 ) && !b.ismixed() && ( dynamic_cast<const BernVector &>(b).Nw() == Nw() ) )
    {
        const BernVector &bb = dynamic_cast<const BernVector &>(b); 

        ww.subit(bb.ww);
    } 

    else
    {
        const FuncVector &bb = dynamic_cast<const FuncVector &>(b); 

        FuncVector::fdim  = bb.fdim;
        FuncVector::valfn = bb.valfn;

        (FuncVector::extrapart).resize(1);
        (FuncVector::extrapart)("&",0) = dynamic_cast<FuncVector *>(makeDup());

        (FuncVector::valfn).negate();

        ww.resize(0);
        revertToFunc = 1;
    } 

    return *this; 
}

Vector<gentype> &BernVector::addit(const Vector<gentype> &b)
{ 
    unsample(); 

    if ( ismixed() )
    {
        FuncVector::addit(b);
    }

    else if ( b.imoverhere ) 
    { 
        addit(*(b.imoverhere));
    } 

    else if ( ( b.type() == 3 ) && !b.ismixed() && ( dynamic_cast<const BernVector &>(b).Nw() == Nw() ) )
    {
        const BernVector &bb = dynamic_cast<const BernVector &>(b); 

        ww.addit(bb.ww);
    } 

    else
    {
        const FuncVector &bb = dynamic_cast<const FuncVector &>(b); 

        FuncVector::fdim  = bb.fdim;
        FuncVector::valfn = bb.valfn;

        (FuncVector::extrapart).resize(1);
        (FuncVector::extrapart)("&",0) = dynamic_cast<FuncVector *>(makeDup());

        ww.resize(0);
        revertToFunc = 1;
    } 

    return *this; 
}

Vector<gentype> &BernVector::subit(const gentype &b)
{
    unsample();

    FuncVector::fdim  = fdim;
    FuncVector::valfn = b;

    (FuncVector::extrapart).resize(1);
    (FuncVector::extrapart)("&",0) = dynamic_cast<FuncVector *>(makeDup());

    (FuncVector::valfn).negate();

    ww.resize(0);
    revertToFunc = 1;

    return *this; 
}

Vector<gentype> &BernVector::addit(const gentype &b)
{
    unsample();

    FuncVector::fdim  = fdim;
    FuncVector::valfn = b;

    (FuncVector::extrapart).resize(1);
    (FuncVector::extrapart)("&",0) = dynamic_cast<FuncVector *>(makeDup());

    ww.resize(0);
    revertToFunc = 1;

    return *this; 
}

Vector<gentype> &BernVector::mulit(const gentype &b)
{
    precalcVec *= b;

    if ( ismixed() )
    {
        FuncVector::valfn *= b;
    }

    ww *= b;              

    return *this; 
}

Vector<gentype> &BernVector::rmulit(const gentype &b) 
{ 
    rightmult(b,precalcVec);

    if ( ismixed() )
    {
        rightmult(b,FuncVector::valfn);
    }

    rightmult(b,ww);      

    return *this; 
}

Vector<gentype> &BernVector::divit(const gentype &b) 
{ 
    precalcVec /= b;

    if ( ismixed() )
    {
        FuncVector::valfn /= b;
    }

    ww /= b;              

    return *this; 
}

Vector<gentype> &BernVector::rdivit(const gentype &b) 
{ 
    rightmult(inv(b),precalcVec);

    if ( ismixed() )
    {
        rightmult(inv(b),FuncVector::valfn);
    }

    rightmult(inv(b),ww); 

    return *this;
}

gentype &BernVector::operator()(gentype &res, const SparseVector<gentype> &xx) const
{
    if ( revertToFunc )
    {
        return FuncVector::operator()(res,xx);
    }

    NiceAssert( xx.size() <= dim() );

    setzero(res);

    gentype nn(Nw());
    gentype ov(1.0);


    if ( dim() == 0 )
    {
        res = 1.0; // Based on empty-product-=-1 assumption
    }

    else if ( dim() == 1 )
    {
        if ( Nw() == -1 )
        {
            res = 1.0;
        }

        else
        {
            int j;

            const gentype &x = xx(0);

            for ( j = 0 ; j <= Nw() ; j++ )
            {
                gentype jj(j);
                gentype tmp;

                tmp = ww(j)*pow(x,jj)*pow(ov-x,nn-jj)*((double) xnCr(Nw(),j));

                if ( !j )
                {
                    res = tmp;
                }

                else
                {
                    res += tmp;
                }
            }
        }
    }

    else
    {
        throw("Multi-dimensional Bernstein polynomials require multiple weight vectors, and I don't have time to implement that right now - see http://www.iue.tuwien.ac.at/phd/heitzinger/node17.html");
    }

    return res;
}

































































// Calculate L2 distance from RKHSVector to function of given dimension,
// assuming a function of var(0,0), var(0,1), ..., var(0,dim-1)
//
// It is assume functions are over [0,1]^dim with gran steps per dimension

double calcL2distsq(const Vector<gentype> &ff, gentype &g, int dim, int scaleit, int gran)
{
    NiceAssert( dim  >= 0 );
    NiceAssert( gran >= 1 );
    NiceAssert( ff.infsize() );

    retVector<gentype> tmp;

    const FuncVector &f = dynamic_cast<const FuncVector &>(ff(tmp));

    dim = ( dim >= 0 ) ? dim : f.dim(); // dimension override

    double res = 0.0;
    double unitsize = sqrt( scaleit ? pow(1.0/((double) gran),dim) : 1.0 );
    int j;

    if ( dim == 1 )
    {
        f.unsafesample(gran);

        SparseVector<SparseVector<gentype> > xx;
        SparseVector<gentype> &x = xx("&",0);
        gentype fv,gv;

        int totpts = 0;
        int nullpts = 0;

        for ( j = 0 ; j < gran ; j++ )
        {
            x("&",0) = ((double) j)/((double) gran);

            fv = f(j);
            gv = g(xx); gv.finalise();

            totpts++;

            if ( fv.isValNull() || fv.isValNull() )
            {
                nullpts++;
            }

            else
            {
                fv -= gv;
                fv *= unitsize;

                res += (double) norm2(fv);
            }
        }

        res *= ( ( nullpts == totpts ) ? 1.0 : (((double) totpts)/(((double) totpts-nullpts))) );
    }

    else
    {
        Vector<int> i(dim);
        SparseVector<SparseVector<gentype> > xx;
        SparseVector<gentype> &x = xx("&",0);
        gentype fv,gv;

        i = zeroint();

        int totpts = 0;
        int nullpts = 0;

        int done = dim ? 0 : 1;

        while ( !done )
        {
            for ( j = 0 ; j < dim ; j++ )
            {
                x("&",j) = ((double) i(j))/((double) gran);
            }

            f(fv,x).finalise();
            gv = g(xx); gv.finalise();

            fv -= gv;
            fv *= unitsize;

            totpts++;

            if ( fv.isValNull() )
            {
                nullpts++;
            }

            else
            {
                res += (double) norm2(fv);
            }

            done = 1;

            for ( j = 0 ; done && ( j < dim ) ; j++ )
            {
                i("&",j)++;

                done     = i(j)/gran;
                i("&",j) = i(j)%gran;
            }
        }

        res *= ( ( nullpts == totpts ) ? 1.0 : (((double) totpts)/(((double) totpts-nullpts))) );
    }

    return res;
}

double calcL2distsq(const gentype &f, gentype &g, int dim, int scaleit, int gran)
{
errstream() << "dim = " << dim << "\n";
    NiceAssert( dim  >= 0 );
    NiceAssert( gran >= 1 );

    retVector<gentype> tmp;

    double res = 0.0;
    double unitsize = sqrt( scaleit ? pow(1.0/((double) gran),dim) : 1.0 );
    int j;

    {
        Vector<int> i(dim);
        SparseVector<SparseVector<gentype> > xx;
        SparseVector<gentype> &x = xx("&",0);
        gentype fv,gv;

        i = zeroint();

        int totpts = 0;
        int nullpts = 0;

        int done = dim ? 0 : 1;

        while ( !done )
        {
            for ( j = 0 ; j < dim ; j++ )
            {
                x("&",j) = ((double) i(j))/((double) gran);
            }

            fv = f(xx); fv.finalise();
            gv = g(xx); gv.finalise();

            fv -= gv;
            fv *= unitsize;

            totpts++;

            if ( fv.isValNull() )
            {
                nullpts++;
            }

            else
            {
                res += (double) norm2(fv);
            }

            done = 1;

            for ( j = 0 ; done && ( j < dim ) ; j++ )
            {
                i("&",j)++;

                done     = i(j)/gran;
                i("&",j) = i(j)%gran;
            }
        }

        res *= ( ( nullpts == totpts ) ? 1.0 : (((double) totpts)/(((double) totpts-nullpts))) );
    }

    return res;
}





































std::ostream &operator<<(std::ostream &output, const FuncVector &src)
{
    return src.outstream(output);
}

std::istream &operator>>(std::istream &input, FuncVector &dest)
{
    return dest.instream(input);
}

std::istream &streamItIn(std::istream &input, FuncVector &dest, int processxyzvw)
{
    return dest.streamItIn(input,processxyzvw);
}

std::ostream &FuncVector::outstream(std::ostream &output) const
{
    if ( !NE() )
    {
        output << "[[ FN f: " << valfn << " : " << fdim << " ]]";
    }

    else
    {
        output << "[[ FN f: " << valfn << " : " << fdim << " : " << NE();

        int i;

        for ( i = 0 ; i < NE() ; i++ )
        {
            output << " : " << *(extrapart(i));
        }

        output << " ]]";
    }

    return output;
}

std::istream &FuncVector::instream(std::istream &input)
{
    unsample();

    if ( NE() )
    {
        int i; 

        for ( i = 0 ; i < NE() ; i++ ) 
        { 
            MEMDEL(extrapart("&",i)); 
            extrapart("&",i) = NULL; 
        } 

        extrapart.resize(0); 
    }
 
    wait_dummy dummy;
    char tt;

    while ( input.peek() == '[' )
    {
        input.get(tt);
    }

    while ( isspace(input.peek()) )
    {
        input.get(tt);
    }

    input >> dummy; input >> valfn; input >> dummy; input >> fdim;

    while ( isspace(input.peek()) )
    {
        input.get(tt);
    }

    if ( input.peek() == ':' )
    {
        input.get(tt); NiceAssert( tt == ':' );

        while ( isspace(input.peek()) )
        {
            input.get(tt);
        }

        int qNE;

        input >> qNE;

        extrapart.resize(qNE);

        int i;

        for ( i = 0 ; i < NE() ; i++ )
        {
            while ( isspace(input.peek()) )
            {
                input.get(tt);
            }

            input.get(tt); NiceAssert( tt == '[' );
            input.get(tt); NiceAssert( tt == '[' );

            while ( isspace(input.peek()) )
            {
                input.get(tt);
            }

            std::string typestring;

            input >> typestring;

            Vector<gentype> *tmp = NULL;

            makeFuncVector(typestring,tmp,input);

            extrapart("&",i) = dynamic_cast<FuncVector *>(tmp);
        }

        while ( isspace(input.peek()) )
        {
            input.get(tt);
        }
    }

    input.get(tt); NiceAssert( tt = ']' );
    input.get(tt); NiceAssert( tt = ']' );

    return input;
}

std::istream &FuncVector::streamItIn(std::istream &input, int processxyzvw)
{
    (void) processxyzvw;

    return instream(input);
}


































std::ostream &operator<<(std::ostream &output, const RKHSVector &src)
{
    return src.outstream(output);
}

std::istream &operator>>(std::istream &input, RKHSVector &dest)
{
    return dest.instream(input);
}

std::istream &streamItIn(std::istream &input, RKHSVector &dest, int processxyzvw)
{
    return dest.streamItIn(input,processxyzvw);
}

std::ostream &RKHSVector::outstream(std::ostream &output) const
{
    if ( revertToFunc )
    {
        FuncVector::outstream(output);
    }

    else
    {
        output << "[[ RKHS kernel: " << spKern << "\n";
        output << "   RKHS x:      " << xx     << "\n";
        output << "   RKHS a:      " << alpha  << " ]]";
    }

    return output;
}

std::istream &RKHSVector::instream(std::istream &input)
{
    unsample();

    wait_dummy dummy;
    char tt;

    input >> dummy; input >> spKern;
    input >> dummy; input >> xx;
    input >> dummy; input >> alpha;

    while ( isspace(input.peek()) )
    {
        input.get(tt);
    }

    input.get(tt); NiceAssert( tt = ']' );
    input.get(tt); NiceAssert( tt = ']' );

    return input;
}

std::istream &RKHSVector::streamItIn(std::istream &input, int processxyzvw)
{
    (void) processxyzvw;

    return instream(input);
}




























std::ostream &operator<<(std::ostream &output, const BernVector &src)
{
    return src.outstream(output);
}

std::istream &operator>>(std::istream &input, BernVector &dest)
{
    return dest.instream(input);
}

std::istream &streamItIn(std::istream &input, BernVector &dest, int processxyzvw)
{
    return dest.streamItIn(input,processxyzvw);
}

std::ostream &BernVector::outstream(std::ostream &output) const
{
    if ( revertToFunc )
    {
        FuncVector::outstream(output);
    }

    else
    {
        output << "[[ Bern w: " << ww  << " ]]";
    }

    return output;
}

std::istream &BernVector::instream(std::istream &input)
{
    unsample();

    wait_dummy dummy;
    char tt;

    input >> dummy; input >> ww;

    while ( isspace(input.peek()) )
    {
        input.get(tt);
    }

    input.get(tt); NiceAssert( tt = ']' );
    input.get(tt); NiceAssert( tt = ']' );

    return input;
}

std::istream &BernVector::streamItIn(std::istream &input, int processxyzvw)
{
    (void) processxyzvw;

    return instream(input);
}



























// Make Func and RKHS Vectors

void makeFuncVector(const std::string &typestring, Vector<gentype> *&res, std::istream &src)
{
    if ( typestring == "FN" )
    {
        FuncVector *altres;

        MEMNEW(altres,FuncVector());

        src >> *altres;

        res = altres;
    }

    else if ( typestring == "RKHS" )
    {
        RKHSVector *altres;

        MEMNEW(altres,RKHSVector());

        src >> *altres;

        res = altres;
    }

    else if ( typestring == "Bern" )
    {
        BernVector *altres;

        MEMNEW(altres,BernVector());

        src >> *altres;

        res = altres;
    }

    else
    {
        throw("Unknown typestring");
    }

    return;
}

void makeFuncVector(const std::string &typestring, Vector<gentype> *&res, std::istream &src, int processxyzvw)
{
    if ( typestring == "FN" )
    {
        FuncVector *altres;

        MEMNEW(altres,FuncVector());

        streamItIn(src,*altres,processxyzvw);

        res = altres;
    }

    else if ( typestring == "RKHS" )
    {
        RKHSVector *altres;

        MEMNEW(altres,RKHSVector());

        streamItIn(src,*altres,processxyzvw);

        res = altres;
    }

    else if ( typestring == "Bern" )
    {
        BernVector *altres;

        MEMNEW(altres,BernVector());

        streamItIn(src,*altres,processxyzvw);

        res = altres;
    }

    else
    {
        throw("Unknown typestring");
    }

    return;
}





















Vector<gentype> &makeanRKHSVector(Vector<gentype> &res, const MercerKernel &kern, const SparseVector<gentype> &x, const gentype &a)
{
    RKHSVector tmp;

    tmp.resizeN(1);
    tmp.kern("&") = kern;
    tmp.x("&",0) = x;
    tmp.a("&",0) = a;

    res = tmp;

    return res;
}
