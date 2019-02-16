#include <cmath>
#include <limits>
#include "nr.h"
using namespace std;

namespace {
	inline void shft3(DP &a, DP &b, DP &c, const DP d)
	{
		a=b;
		b=c;
		c=d;
	}
}
/*p404. Here ITMAX is the maximum allowed number of iterations;
 * CGOLD is the golden ratio;
 * ZEPS is a small number that protects against trying to achieve fractional accuracy for a
 * minimum that happens to be exactly zero.
 */ 
DP NR::brent(const DP ax, const DP bx, const DP cx, DP f(const DP),
	const DP tol, DP &xmin)
/*Given a function f, and given a bracketing triplet of abscissas ax, bx, cx (such that bx is
 * between ax and cx, and f(bx) is less than both f(ax) and f(cx)), this routine isolates the 
 * minimum to a fractional precision of about tol using Brent's method. The abscissa of the 
 * minimum is returned as xmin, and the minimum function value is returned as brent, the 
 * returned function value. 
 */
{
	const int ITMAX=100;
	const DP CGOLD=0.3819660;
	const DP ZEPS=numeric_limits<DP>::epsilon()*1.0e-3;
	int iter;
	DP a,b,d=0.0,etemp,fu,fv,fw,fx;
	DP p,q,r,tol1,tol2,u,v,w,x,xm;
	DP e=0.0;		//This will be the distance moved on the step before last.

	a=(ax < cx ? ax : cx);		//a and b must be in ascending order,
	b=(ax > cx ? ax : cx);		//but input abscissas need not be
	x=w=v=bx;					//Initializations...
	fw=fv=fx=f(x);
	for (iter=0;iter<ITMAX;iter++) {	//Main program loop.
		xm=0.5*(a+b);
		tol2=2.0*(tol1=tol*fabs(x)+ZEPS);
		if (fabs(x-xm) <= (tol2-0.5*(b-a))) {	//Test for done here
			xmin=x;
			return fx;
		}
		if (fabs(e) > tol1) {		//Construct a trial parabolic fit
			r=(x-w)*(fx-fv);
			q=(x-v)*(fx-fw);
			p=(x-v)*q-(x-w)*r;
			q=2.0*(q-r);
			if (q > 0.0) p = -p;
			q=fabs(q);
			etemp=e;
			e=d;
			if (fabs(p) >= fabs(0.5*q*etemp) || p <= q*(a-x) || p >= q*(b-x))
				d=CGOLD*(e=(x >= xm ? a-x : b-x));
			//The above conditions determine the acceptability of the parabolic fit. Here we
			//take the golden section step into the larger of the two segments.
			else {
				d=p/q;		//Take the parabolic step
				u=x+d;
				if (u-a < tol2 || b-u < tol2)
					d=SIGN(tol1,xm-x);
			}
		} else {
			d=CGOLD*(e=(x >= xm ? a-x : b-x));
		}
		u=(fabs(d) >= tol1 ? x+d : x+SIGN(tol1,d));
		fu=f(u);
		//This is the one function evaluation per iteration
		if (fu <= fx) {		//Now decide what to do with our function evaluation
			if (u >= x) a=x; else b=x;
			shft3(v,w,x,u);	//Housekeeping follows
			shft3(fv,fw,fx,fu);
		} else {
			if (u < x) a=u; else b=u;
			if (fu <= fw || w == x) {
				v=w;
				w=u;
				fv=fw;
				fw=fu;
			} else if (fu <= fv || v == x || v == w) {
				v=u;
				fv=fu;
			}
		}		//Done with housekeeping. Back for another iteration.
	}
	nrerror("Too many iterations in brent");
	xmin=x;		//Never get here
	return fx;
}
