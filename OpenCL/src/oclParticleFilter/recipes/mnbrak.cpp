#include <cmath>
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
/*p.400. Here GOLD is the default ratio by which successive intervals are magnified;
 * GLIMIT is the maximum magnification allowed for a parabolic-fit step
 * 
 */ 
void NR::mnbrak(DP &ax, DP &bx, DP &cx, DP &fa, DP &fb, DP &fc,
	DP func(const DP))
/*Given a function func, and given distinct initial points ax and bx, this routine searches in
 * the downhill direction (defined by the function as evaluated at the initial points) and returns
 * new points ax, bx, cx that bracket a minimum of the function. Also returned are the function 
 * values at the three points, fa, fb, and fc. 
 */ 
{
	const DP GOLD=1.618034,GLIMIT=100.0,TINY=1.0e-20;
	DP ulim,u,r,q,fu;

	fa=func(ax);
	fb=func(bx);
	if (fb > fa) {		//Switch roles of a and b so that we can go downhill in the
		SWAP(ax,bx);    //direction from a to b
		SWAP(fb,fa);
	}
	cx=bx+GOLD*(bx-ax);	//First guess for c
	fc=func(cx);
	while (fb > fc) {	//Keep returning here until we bracket.
		r=(bx-ax)*(fb-fc);	//Compute \(u\) by parabolic extrapolation from  
		q=(bx-cx)*(fb-fa);  //a,b,c. TINY is used to prevent any posible division by zero
		u=bx-((bx-cx)*q-(bx-ax)*r)/
			(2.0*SIGN(MAX(fabs(q-r),TINY),q-r));
		ulim=bx+GLIMIT*(cx-bx);
		//We won't go farther than this. Test various possibilities:
		if ((bx-u)*(u-cx) > 0.0) {		//Parabolic u is between b and c: try it
			fu=func(u);
			if (fu < fc) {		//Got a minimum between b and c
				ax=bx;
				bx=u;
				fa=fb;
				fb=fu;
				return;
			} else if (fu > fb) {	//Got a minimum between between a and u
				cx=u;
				fc=fu;
				return;
			}
			u=cx+GOLD*(cx-bx);		//Parabolic fit was no use. Use default magnification.
			fu=func(u);
		} else if ((cx-u)*(u-ulim) > 0.0) {		//Parabolic fit is between c and its allowed limit
			fu=func(u);
			if (fu < fc) {
				shft3(bx,cx,u,cx+GOLD*(cx-bx));		
				shft3(fb,fc,fu,func(u));
			}
		} else if ((u-ulim)*(ulim-cx) >= 0.0) {		//Limit parabolic u to maximum allowed value
			u=ulim;
			fu=func(u);
		} else {		//Reject parabolic u, use default magnification.	
			u=cx+GOLD*(cx-bx);
			fu=func(u);
		}
		shft3(ax,bx,cx,u);		//Eliminate oldest point and continue
		shft3(fa,fb,fc,fu);
	}
}
