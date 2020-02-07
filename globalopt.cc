
//
// Global optimisation options base-class
//
// Date: 29/09/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "globalopt.h"
#include "hyper_base.h"

int godebug = 0; // to debug vector.h

void overfn(gentype &res, Vector<gentype> &x, void *arg)
{
    void *fnarg                       = ((void *)                      ((void **) arg)[0]);
    int &dim                          = *((int *)                      ((void **) arg)[1]);
    Vector<gentype> &xmod             = *((Vector<gentype> *)          ((void **) arg)[2]);
    GlobalOptions &gopts              = *((GlobalOptions *)            ((void **) arg)[3]);
    Vector<Vector<gentype> > &allxres = *((Vector<Vector<gentype> > *) ((void **) arg)[4]);
    gentype &penterm                  = *((gentype *)                  ((void **) arg)[5]);

    void (*fn)(gentype &, Vector<gentype> &, void *) = ((void (*)(gentype &, Vector<gentype> &, void *)) ((void **) arg)[6]);

    gopts.convertx(dim,xmod,x,0,1);
//errstream() << "phantomxxxxx 0 convert\n";
//gopts.convertx(dim,xmod,x,0,1);
//errstream() << "phantomxxxxx 1 model_convert\n";
//SparseVector<gentype> xalt(x);
//SparseVector<gentype> xmoddummy(x);
//gopts.model_convertx(xmoddummy,xalt,0,0,1);
//errstream() << "phantomxxxxx 2 done\n";

    (*fn)(res,xmod,fnarg);
//errstream() << "f(g(" << x << ")) = f(" << xmod << ") = " << res << "\n";
errstream() << "Global optimiser: f(g(.))" << xmod << ") = " << res << "\n";

    if ( !(penterm.isCastableToRealWithoutLoss()) || ( ( (double) penterm ) != 0.0 ) )
    {
        gentype xmodvecform(xmod);

        res += penterm(xmodvecform);
errstream() << "Global optimiser: f(g(.)) + penterm(" << xmodvecform << ") = " << res << "\n";
    }

    // Save to allxres if this is an actual evaluation ( x.size() == 0 if intermediate operation )

    if ( x.size() )
    {
        allxres.append(allxres.size(),xmod);

        if ( ( gopts.isProjection == 4 ) && ( gopts.stopearly == 0 ) && ( gopts.berndim > 1 ) && ( gopts.firsttest || ( res < gopts.bestyet ) ) )
        {
            int i,j;
            int effdim = gopts.berndim;

            gopts.stopearly = 0;
            gopts.firsttest = 0;
            gopts.bestyet   = res;

            for ( i = 0 ; i < effdim-1 ; i++ )
            {
                for ( j = i+1 ; j < effdim ; j++ )
                {
                    if ( (double) abs2(x(i)-x(j)) > 0.95 )
                    {
                        gopts.stopearly = 1;
                        break;
                    }
                }
            }
        }
    }

    errstream() << ".";

    return;
}

double calcabc(int dim, 
               Vector<gentype> &fakexmin, Vector<gentype> &fakexmax, 
               Vector<double> &a, Vector<double> &b, Vector<double> &c,
               const Vector<gentype> &xmin, const Vector<gentype> &xmax, 
               const Vector<int> &distMode)
{
    double xwidth = 1;

    a.resize(dim+1);
    b.resize(dim+1);
    c.resize(dim+1);

    if ( dim )
    {
        int i;

        for ( i = 0 ; i < dim ; i++ )
        {
            double lb = (double) xmin(i);
            double ub = (double) xmax(i);

            int zerowidth = ( lb == ub ) ? 1 : 0;

            xwidth = ( !i || ( (ub-lb) > xwidth ) ) ? (ub-lb) : xwidth;

            fakexmin("&",i) = 0.0;
            fakexmax("&",i) = zerowidth ? 0.0 : 1.0;

            if ( distMode(i) == 1 )
            {
                // Logarithmic grid
                //
                // v = a + e^(b+c.t)
                // 
                // c = log(10)
                // 
                // t = 0 => v = lb
                //       => a+e^b = lb
                //       => b = log(lb-a)
                // t = 1 => v = ub
                //       => a+e^(b+c) = ub
                //       => e^(b+c) = ub-a
                //       => b+c = log(ub-a)
                //       => b = log(ub-a)-c
                // 
                // => log(lb-a) = lob(ub-a)-c
                // => log((ub-a)/(lb-a)) = log(e^c)
                // => (ub-a)/(lb-a) = e^c
                // => ub - a = (e^c)lb - (e^c)a
                // => ((e^c)-1).a = (e^c)lb - ub
                // => a = ((e^c)lb - ub)/((e^c)-1)

                c("&",i) = log(10.0);
                a("&",i) = ( (exp(c(i))*lb) - ub ) / ( exp(c(i)) - 1.0 );
                b("&",i) = log( lb - a(i) );
            }

            else if ( distMode(i) == 2 )
            {
                // Anti-logarithmic grid (inverse of logarithmic grid)
                //
                // v = (1/c) log(t-a) - (b/c)
                //
                // c = log(10)
                // 
                // t = 0 => v = lb
                //       => 0 = a + e^(b+c.lb)
                //       => -a = e^(b+c.lb)
                //       => b+c.lb = log(-a)
                //       => b = log(-a) - c.lb
                // t = 0 => t = ub
                //       => 1 = a + e^(b+c.ub)
                //       => 1-a = e^(b+c.ub)
                //       => b+c.ub = log(1-a)
                //       => b = log(1-a) - c.ub
                //
                // => log(-a) - c.lb = log(1-a) - c.ub
                // => c.(ub-lb) = log((a-1)/a)
                // => (a-1)/a = exp(c.(ub-lb))
                // => 1 - 1/a = exp(c.(ub-lb))
                // => 1/a = 1 - exp(c.(ub-lb))
                // => a = 1/(1 - exp(c.(ub-lb)))

                c("&",i) = log(10);
                a("&",i) = 1/(1-exp(c(i)*(ub-lb)));
                b("&",i) = log(-a(i))-(c(i)*lb);
            }

            else if ( distMode(i) == 4 )
            {
                // Anti-logistic grid
                //
                // v = a - (1/b) log( 1/(0.5+(c*(t-0.5))) - 1 )
                //   = a - (1/b) log(0.5-(c*(t-0.5))) + (1/b) log(0.5+(c*(t-0.5)))
                //   = a - (1/b) log(1/(0.5+(c*(t-0.5)))) + (1/b) log(1/(0.5-(c*(t-0.5))))
                //
                // a = lb+((ub-lb)/2)
                // a = (ub+lb)/2
                //
                // m = small number < 1/2
                // B = big number
                // D = (ub-lb)/2
                //
                // t = 0 => v = a-B
                //       => a - (1/b) log( 1/(0.5*(1-c)) - 1 ) = a-B
                //       => a - (1/b) log( 2/(1-c) - 1 ) = a-B
                //       => a - (1/b) log( (1+c)/(1-c) ) = a-B
                //       => a - (1/b) log(1+c) + (1/b) log(1-c) = a-B
                //       => (1/b) log(1+c) - (1/b) log(1-c) = B
                //       => log(1+c) - log(1-c) = b.B
                // t = 1 => v >= a+B
                //       => a - (1/b) log( 1/(0.5*(1+c)) - 1 ) = a+B
                //       => a - (1/b) log( 2/(1+c) - 1 ) = a+B
                //       => a - (1/b) log( (1-c)/(1+c) ) = a+B
                //       => a - (1/b) log(1-c) + (1/b) log(1+c) = a+B
                //       => (1/b) log(1-c) - (1/b) log(1+c) = -B
                //       => (1/b) log(1+c) - (1/b) log(1-c) = B
                //       => log(1+c) - log(1-c) = b.B
                // t = m => v = lb
                //       => a - (1/b) log( 1/(0.5+(c*(m-0.5))) - 1 ) = lb
                //       => a - (1/b) log( 1/(0.5*(1-c)+cm) - 1 ) = lb
                //       => (1/b) log( 1/(0.5*(1-c)+cm) - 1 ) = a-lb
                //       => (1/b) log( 1/(0.5*(1-c)+cm) - 1 ) = (ub-lb)/2
                //       => (1/b) log( 2/(1-c+2cm) - 1 ) = (ub-lb)/2
                //       => (1/b) log( (2-(1-c+2cm))/(1-c+2cm) ) = (ub-lb)/2
                //       => (1/b) log( (1+c-2cm)/(1-c+2cm) ) = (ub-lb)/2
                //       => log(1+c-2cm) - log(1-c+2cm) = b.(ub-lb)/2
                // t = 1-m => v = ub
                //       => a - (1/b) log( 1/(0.5+(c*(1-m-0.5))) - 1 ) = lb
                //       => a - (1/b) log( 1/(0.5*(1+c)-cm) - 1 ) = ub
                //       => (1/b) log( 1/(0.5*(1+c)-cm) - 1 ) = a-ub
                //       => (1/b) log( 1/(0.5*(1+c)-cm) - 1 ) = -(ub-lb)/2
                //       => (1/b) log( 2/(1+c-2cm) - 1 ) = -(ub-lb)/2
                //       => (1/b) log( (2-(1+c-2cm))/(1+c-2cm) ) = -(ub-lb)/2
                //       => (1/b) log( (1-c+2cm)/(1+c-2cm) ) = -(ub-lb)/2
                //       => log(1-c+2cm) - log(1+c-2cm) = -b.(ub-lb)/2
                //       => log(1+c-2cm) - log(1-c+2cm) = b.(ub-lb)/2
                //
                // NB: - symmetry means we need only consider one of t=m,1-m 
                //       (note that the final two expressions are identical)
                //       where both imply that:
                //
                //       b = (log(1+c-2cm)-log(1-c+2cm))/D
                //       b = (log(1+(c*(1-2m)))-log(1-(c*(1-2m))))/D   (*)
                //
                //       where we have defined D = (ub-lb)/2
                //
                //     - likewise symmetry means we only need consider one
                //       of t=0,1 (see final expressions), which using (*) imply:
                //
                //       q = B.(log(1+(c*(1-2m)))-log(1-(c*(1-2m)))) - D.(log(1+c)-log(1-c)) = 0    (**)
                //
                //     - note the derivative of (**) wrt c
                //
                //       B.(1-2m).( 1/(1+(c*(1-2m))) + 1/(1-(c*(1-2m))) ) - D.( 1/(1+c) + 1/(1-c) ) = 0
                //
                //       Assuming B sufficiently large this will be increasing.

                double aval = (ub+lb)/2;
                double qval,bval,cval;

                double cmin = 1e-6;
                double cmax = 0.5;

                double m = NEARZEROVAL;
                double B = NEARINFVAL;
                double D = (ub-lb)/2;
                double ratmax = RATIOMAX;

                cmin = 0;
                cmax = 1;

                while ( 2*(cmax-cmin)/(cmax+cmin) > ratmax )
                {
                    cval  = (cmin+cmax)/2;
                    //bval  = 2*((1/(1+(cval*(1-(2*m)))))-(1/(1-(cval*(1-(2*m))))))/D;
                    bval  = 2*((1/(1-(cval*(1-(2*m)))))-(1/(1+(cval*(1-(2*m))))))/D;
                    qval  = 2*((1.0/(1-cval))-(1.0/(1+cval)))/bval;
                    if ( qval == B )
                    {
                        cmax = cval;
                        cmin = cval;
                    }

                    else if ( qval > B )
                    {
                        cmax = cval;
                    }

                    else
                    {
                        cmin = cval;
                    }
                }

                //cval = cmax;
                //bval = 2*((1/(1+(cval*(1-(2*m)))))-(1/(1-(cval*(1-(2*m))))))/D;
                bval = 2*((1/(1-(cval*(1-(2*m)))))-(1/(1+(cval*(1-(2*m))))))/D;

                a("&",i) = aval;
                b("&",i) = bval;
                c("&",i) = cval;
            }

            else
            {
                // Linear (potentially random) grid
                //
                // v = a + b.t
                //
                // t = 0 => v = lb
                //       => a = lb
                // t = 1 => v = ub
                //       => lb+b = ub
                //       => b = ub-lb

                a("&",i) = lb;
                b("&",i) = ub-a(i);
                c("&",i) = 0;
            }
        }
    }

    return xwidth;
}





int GlobalOptions::analyse(const Vector<Vector<gentype> > &allxres,
                           const Vector<gentype> &allfresmod,
                           Vector<double> &hypervol,
                           Vector<int> &parind,
                           int calchypervol) const
{
    NiceAssert( allxres.size() == allfresmod.size() );

    int N = allxres.size();

    parind.resize(N);
    hypervol.resize(N);

    hypervol = 0.0;

    if ( N )
    {
        // Work out hypervolume sequence

        if ( calchypervol )
        {
            if ( allfresmod(zeroint()).isValReal() )
            {
                int i;

                for ( i = 0 ; i < N ; i++ )
                {
                    if ( !i || ( (double) allfresmod(i) < -(hypervol(i-1)) ) )
                    {
                        hypervol("&",i) = -((double) allfresmod(i));
                    }

                    else
                    {
                        hypervol("&",i) = hypervol(i-1);
                    }
                }
            }

            else
            {
                int i,j;

                int M = (allfresmod(zeroint())).size();
                double **X;

                MEMNEWARRAY(X,double *,N);

                NiceAssert(X);

                for ( i = 0 ; i < N ; i++ )
                {
                    MEMNEWARRAY(X[i],double,M);

                    NiceAssert( X[i] );

                    for ( j = 0 ; j < M ; j++ )
                    {
                        X[i][j] = -((double) ((allfresmod(i))(j)));
                    }

                    hypervol("&",i) = h(X,i+1,M);
                }

                for ( i = 0 ; i < N ; i++ )
                {
                    MEMDELARRAY(X[i]);
                }

                MEMDELARRAY(X);
            }
        }

        // Work out Pareto set

        int m = allfresmod(zeroint()).size();

        retVector<int> tmpva;

        parind = cntintvec(N,tmpva);

        int pos,i,j,isdom;

        for ( pos = parind.size()-1 ; pos >= 0 ; pos-- )
        {
            NiceAssert( allfresmod(pos).size() == m );

            // Test if allfres(parind(pos)) is dominated by any points
            // in parind != pos.  If it is then remove it from parind.

            isdom = 0;

            for ( i = 0 ; i < parind.size() ; i++ )
            {
                if ( parind(i) != parind(pos) )
                {
                    // Test if allfres(i) dominates allfres(pos) - that is,
                    // if allfres(pos)(j) >= allfres(i)(j) for all j

                    isdom = 1;

                    for ( j = 0 ; j < m ; j++ )
                    {
                        if ( allfresmod(parind(pos))(j) < allfresmod(parind(i))(j) )
                        {
                            isdom = 0;
                            break;
                        }
                    }

                    if ( isdom )
                    {
                        break;
                    }
                }
            }

            if ( isdom )
            {
                parind.remove(pos);
            }
        }
    }

    return parind.size();
}
