/*
 * A bottom-up adaptation of the minimalist divide-and-conquer 
 * algorithm for 3D convex hulls using OpenCL.
 *
 * File: convexHull3d.cl
 *
 * Last Modified: June 12, 2012
 * 
 * Jeffrey M. White
 * Kevin A. Wortman
 */

constant int NIL = -1;
constant float INF = 1e30;

typedef struct 
{
	float x;
	float y;
	float z;
	int prev;
	int next;
} Point;

__kernel void convexHull3dKernel(__global Point *P, __global int *A, __global int *B, __global int *M) 
{
	int id = get_global_id(0);
	int m = M[0];
	int leftGroupIndex = id*m;
	int rightGroupIndex = (leftGroupIndex + ((id+1)*m))/2;
	int eventListOffset = leftGroupIndex*2;
	int point;	

	// u: end of left hull based on x coordinate
	// v: beginning of right hull based on x coordinate
	int u = leftGroupIndex;
	int v = rightGroupIndex;
	int i = leftGroupIndex*2;
	int j = rightGroupIndex*2;
	
	// find end of list for u
	for ( ; P[u].next != NIL; u = P[u].next) ;
	
	// FIND INITIAL BRIDGE for u and v
	for ( ; ; ) {
		int p, q, r;
		// calculate turn1 value
		float turn1;
		p = u; q = v; r = P[v].next;
		if (p == NIL || q == NIL || r == NIL)
	        turn1 = 1.0;
	    else
	 		turn1 = (P[q].x - P[p].x)*(P[r].y - P[p].y) - (P[r].x - P[p].x)*(P[q].y - P[p].y);

		// calculate turn2 value
		float turn2;
		p = P[u].prev; q = u; r = v;
		if (p == NIL || q == NIL || r == NIL)
	        turn2 = 1.0;
	    else
	 		turn2 = (P[q].x - P[p].x)*(P[r].y - P[p].y) - (P[r].x - P[p].x)*(P[q].y - P[p].y);

		if (turn1 < 0)
			v = P[v].next;
		else if (turn2 < 0)
			u = P[u].prev;
		else
			break;
	}

	int k = eventListOffset;	// starting index of event list
	float oldTime = -INF;       // initialize oldTime to negative inifinity
 	float newTime;
	float t[6];                 // array to hold 6 times for each set of time calculations
    int minimumTime;

	// TRACK BRIDGE FROM U TO V OVER TIME
	// progress through time in an infinite loop until 
	// no more insertion/deletion events occur
	for ( ; ; oldTime = newTime) {		
		// calculate each moment in time: (if time is < 0 , then clockwise)
		int p, q, r;

		// calculate time 0 value
		// t[0] = time(P[ B[i] ].prev, B[i], P[ B[i] ].next);	// B[i]
		p = P[ B[i] ].prev; q = B[i]; r = P[ B[i] ].next;
		if (p == NIL || q == NIL || r == NIL)
	        t[0] = INF;
	    else
	 		t[0] = ((P[q].x - P[p].x)*(P[r].z - P[p].z) - (P[r].x - P[p].x)*(P[q].z - P[p].z))/((P[q].x - P[p].x)*(P[r].y - P[p].y) - (P[r].x - P[p].x)*(P[q].y - P[p].y));

		// calculate time 1 value
		// t[1] = time(P[ B[j] ].prev, B[j], P[ B[j] ].next);	// B[j]
		p = P[ B[j] ].prev; q = B[j]; r = P[ B[j] ].next;
		if (p == NIL || q == NIL || r == NIL)
	        t[1] = INF;
	    else
	 		t[1] = ((P[q].x - P[p].x)*(P[r].z - P[p].z) - (P[r].x - P[p].x)*(P[q].z - P[p].z))/((P[q].x - P[p].x)*(P[r].y - P[p].y) - (P[r].x - P[p].x)*(P[q].y - P[p].y));

		// calculate time 2 value
		// t[2] = time(P[u].prev, u, v); 			// -u u v 
		p = P[u].prev; q = u; r = v;
		if (p == NIL || q == NIL || r == NIL)
	        t[2] = INF;
	    else
	 		t[2] = ((P[q].x - P[p].x)*(P[r].z - P[p].z) - (P[r].x - P[p].x)*(P[q].z - P[p].z))/((P[q].x - P[p].x)*(P[r].y - P[p].y) - (P[r].x - P[p].x)*(P[q].y - P[p].y));

		// calculate time 3 value
		// t[3] = time(u, P[u].next, v); 			// u +u v
		p = u; q = P[u].next; r = v;
		if (p == NIL || q == NIL || r == NIL)
	        t[3] = INF;
	    else
	 		t[3] = ((P[q].x - P[p].x)*(P[r].z - P[p].z) - (P[r].x - P[p].x)*(P[q].z - P[p].z))/((P[q].x - P[p].x)*(P[r].y - P[p].y) - (P[r].x - P[p].x)*(P[q].y - P[p].y));

		// calculate time 4 value
		// t[4] = time(u, P[v].prev, v); 			// u -v v
		p = u; q = P[v].prev; r = v;
		if (p == NIL || q == NIL || r == NIL)
	        t[4] = INF;
	    else
	 		t[4] = ((P[q].x - P[p].x)*(P[r].z - P[p].z) - (P[r].x - P[p].x)*(P[q].z - P[p].z))/((P[q].x - P[p].x)*(P[r].y - P[p].y) - (P[r].x - P[p].x)*(P[q].y - P[p].y));

        // calculate time 5 value
		// t[5] = time(u, v, P[v].next); 			// u v v+
		p = u; q = v; r = P[v].next;
		if (p == NIL || q == NIL || r == NIL)
	        t[5] = INF;
	    else
	 		t[5] = ((P[q].x - P[p].x)*(P[r].z - P[p].z) - (P[r].x - P[p].x)*(P[q].z - P[p].z))/((P[q].x - P[p].x)*(P[r].y - P[p].y) - (P[r].x - P[p].x)*(P[q].y - P[p].y));

		newTime = INF;	// initialize newTime to no new events in time

		// find the scenario with the minimum moment in time calculation
		for (int x=0; x<6; x++) {
			if (t[x] > oldTime && t[x] < newTime) {
				minimumTime = x;
				newTime = t[x];
			}
		}

		// check to see if no new events occured
		if (newTime == INF)
			break;	// break out of infinite moment in time loop

		// new events have occured, thus, act on the 
		// event with the lowest time calculation
		switch (minimumTime) {
			case 0:
				if (P[ B[i] ].x < P[u].x)
					A[k++] = B[i];
				// ACT
				point = B[i++];
				if ( P[ P[point].prev ].next != point) {   // insert
					P[ P[point].prev ].next = P[ P[point].next ].prev = point;
				}
				else { // delete
					P[ P[point].prev ].next = P[point].next;
					P[ P[point].next ].prev = P[point].prev;
				}
				// END ACT
				break;
			case 1:
				if (P[ B[j] ].x > P[v].x)
					A[k++] = B[j];
				// ACT
				point = B[j++];
				if ( P[ P[point].prev ].next != point) {   // insert
					P[ P[point].prev ].next = P[ P[point].next ].prev = point;
				}
				else { // delete
					P[ P[point].prev ].next = P[point].next;
					P[ P[point].next ].prev = P[point].prev;
				}
				// END ACT
				break;
			case 2: // -u u v
				A[k++] = u;
				u = P[u].prev;
				break;
			case 3: // u +u v
				u = P[u].next;
				A[k++] = u;
				break;
			case 4: // u -v v
				v = P[v].prev;
				A[k++] = v;
				break;
			case 5: // u v +v
				A[k++] = v;
				v = P[v].next;
				break;
		}
	}

	A[k] = NIL;

	// connect both groups
	P[u].next = v; 
	P[v].prev = u;

	for (k--; k >= eventListOffset; k--) {
		if (P[ A[k] ].x <= P[ u ].x || P[ A[k] ].x >= P[ v ].x) {
			// pass current point to act funtion
			// ACT
			point = A[k];
			if ( P[ P[point].prev ].next != point) {   // insert
				P[ P[point].prev ].next = P[ P[point].next ].prev = point;
			}
			else { // delete
				P[ P[point].prev ].next = P[point].next;
				P[ P[point].next ].prev = P[point].prev;
			}			
			// END ACT
			// get new u or v pointers
			if (A[k] == u) 
				u = P[u].prev; 
			else if (A[k] == v) 
				v = P[v].next;
		}
		else {
			P[u].next = A[k];
			P[ A[k] ].prev = u;
			P[v].prev = A[k];
			P[ A[k] ].next = v;

			// get new u or v pointers
			if ( P[ A[k] ].x < P[rightGroupIndex].x )
				u = A[k];
			else
				v = A[k];
		}
	}
}
