                  
               // N-particles LJ system in 3D                      //
               // Basic Verlet                    //
               // interaction: 6-12                        //
               // No cutoffs, no PBCs           //
               // No minimum image convention   //
               // MD simulations are performed for up to ~ 100 LJ particles   //

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#define Npartmax 100    // It used to create the lattice ( Npart < Npartmax )
 double M_PI=3.1416;

typedef struct{
  double x, y, z;
} position;

typedef struct{
  double vx, vy, vz;
} velocity;

typedef struct{
  double px, py, pz;
} momentum;

typedef struct{
  double ax, ay, az;
} acceleration;

int Npart;                  // No of particles
double *mass;      // Array for storing the mass of each of the "Npartmax" particles
position *R;       // Positions: R[i].x, R[i].y, R[i].z
position *Rcorr;   // Corrected positions: R[i].x, R[i].y, R[i].z
velocity *V;       // Velocities: V[i].vx, V[i].vy, V[i].vz
acceleration *acc; // Accelerations: acc[i].ax, acc[i].ay, acc[i].az
double tot_mass;            // Total mass of the system
FILE *fp_traj;              // Create a file to store the trajectories
position *Rp, *Rnew; // Array which is used to store the positions of the previous timestep
double boxsize;                        // Length of a box
double dt, tmin, tmax;      // Time grid
double rcut, rcut2, rcut4, rcut6, rcut8, rcut12, rcut14, frcut, potrcut; // cut off distance and powers of cut off distance and the force factor at the cut off distance and potential at the cut off distance
int cutoff; // use velocity verlet, use cutoff
clock_t start, diff;


// Gaussian distributed random numbers - Koonin & Meredith 1986
// It is used in the function " mass_setup() ".
void GAUSS( double *gauss1, double *gauss2 )
{
  double x1, x2, twou, radius, theta;

  x1 = (double)rand()/(double)RAND_MAX;
  x2 = (double)rand()/(double)RAND_MAX;
  twou = -2*log( 1.0-x1 );
  radius = sqrt( twou);
  theta = 2*M_PI*x2;
  *gauss1 = radius*cos(theta);
  *gauss2 = radius*sin(theta);
}


// Random gaussian mass distribution 
// special != 1 : Assign random mass for every particle which are sampled from a gaussian distribution.
// special == 1 : All masses are equal to 1.
// iprint  == 1 : Print masses at command prompt
void mass_setup( int iprint, int special )
{
  int i;
  double g1, dumb;

  for ( i=0 ; i<Npart ; i++ )
       mass[i] = 1.0;

  if ( special == 1 ) return;

  if ( iprint == 1 )
         fprintf( stderr, "Mass matrix\n" );

  for ( i=0 ; i<Npart ; i++ )
    {
      GAUSS( &g1, &dumb );
      mass[i] = 1.0 + 0.1 * g1;
      if ( mass[i] < 0.0 )
        mass[i] = - mass[i];

      if ( iprint == 1 )
        {
          fprintf( stderr, " %f", mass[i] );
	  if ( i>0 && 5*(i/5) == i ) printf( "\n" );
	}
    }
  if ( iprint == 1 ) 
         fprintf( stderr, "\n" );
}


// Print, to the command prompt, positions and velocities for all the particles.
void output_x_v()
{
  int i;

  for ( i=0 ; i<Npart ; i++ )
    {
      fprintf( stderr, "Position & velocity:  %d %f %f %f %f %f %f\n", 
           i, R[i].x, R[i].y, R[i].z, V[i].vx, V[i].vy, V[i].vz ); 
    }
}


// Write the positions of the particles in the file " fp_traj ".
void record_trajectories( )
{
  int i;

  for ( i=0 ; i<Npart ; i++ )
    {
      fprintf( fp_traj, " %f %f %f %f %f %f", R[i].x, R[i].y, R[i].z, Rcorr[i].x, Rcorr[i].y, Rcorr[i].z);
    }
  //fprintf( fp_traj, " %f %f %f %f %f %f", R[0].x, R[0].y, R[0].z, R[1].x, R[1].y, R[1].z);
}


// Place the particles in a compact square 3D grid with unit spacing and N vertices
void initial_grid( int N, int xyz[Npartmax][3] )
{
  int gen, i, j, k, ii, pass, n, nnewlist, noldlist, same;
  FILE *fp;

  n = 0;
  xyz[n][0] = 0;
  xyz[n][1] = 0;
  xyz[n][2] = 0;
  n++;

  for ( gen = 1; gen<5 ; gen++ )
    {
      nnewlist = 0;

  for ( pass=0 ; pass<3 ; pass++ )
    {
      noldlist = nnewlist;
      nnewlist = n;

      for ( i=noldlist ; i<nnewlist ; i++ )
        {
          for ( j=0 ; j<3 ; j++ )
            {
              if ( xyz[i][j] == gen - 1 )
                {
                  xyz[n][0] = xyz[i][0];
                  xyz[n][1] = xyz[i][1];
                  xyz[n][2] = xyz[i][2];
                  xyz[n][j] = gen;
                  n++;

                  for ( k=0; k<n-1 ; k++ )
                    {
                      same = 0;
                      if ( xyz[n-1][0] == xyz[k][0] &&
                           xyz[n-1][1] == xyz[k][1] &&
                           xyz[n-1][2] == xyz[k][2] )
                        {
                          n--;
                          same = 1;
                        }
                      if ( same == 1 ) break;
                    } // k loop

                  if ( n == N )
                    {
                        fp = fopen( "list_initial", "w" );
                        for ( ii=0 ; ii<N ; ii++ )
                           fprintf( fp, "%d %d %d %d\n", ii,
                                   xyz[ii][0], xyz[ii][1], xyz[ii][2] );
                        fclose(fp);
                        return;
                    }

                }

            }  // j loop

        }  // i loop

    }  // pass loop

    }  // gen loop

}

// Initial positions on square lattice
// Initial gaussian random velocities
// Velocities are adjusted for center of mass with 0 momentum (stationary)
// Center of mass is located close to (0,0) because gaussian random centered at zero
void initial_conditions( )
{
  int i;
  double g1, size, speed_scale, r1, r2, r3;
  double sq3, v;
  momentum P;
  int xyz[Npartmax][3];

                        // 3D lattice positions
  initial_grid( Npart, xyz );

  size = 1.2;
  for ( i=0 ; i<Npart ; i++ )
    {
      R[i].x = size*xyz[i][0];
      R[i].y = size*xyz[i][1];
      R[i].z = size*xyz[i][2];
	  Rcorr[i].x = size*xyz[i][0];
      Rcorr[i].y = size*xyz[i][1];
      Rcorr[i].z = size*xyz[i][2];
    }

                        // random velocities
  speed_scale = 1.0;
  for ( i=0 ; i<Npart ; i++ )
    {
      r1 = (double)rand()/(double)RAND_MAX;
      r2 = (double)rand()/(double)RAND_MAX;
      r3 = (double)rand()/(double)RAND_MAX;
      V[i].vx = speed_scale * ( 2*r1 - 1.0 );
      V[i].vy = speed_scale * ( 2*r2 - 1.0 );
      V[i].vz = speed_scale * ( 2*r3 - 1.0 );
    }

                        // remove CM motion
  P.px = 0.0;
  P.py = 0.0;
  P.pz = 0.0;
  tot_mass = 0.0;
  for ( i=0 ; i<Npart ; i++ )
    {
      tot_mass = tot_mass + mass[i];
      P.px = P.px + mass[i] * V[i].vx;
      P.py = P.py + mass[i] * V[i].vy;
      P.pz = P.pz + mass[i] * V[i].vz;
    }
  for ( i=0 ; i<Npart ; i++ )
    {
      V[i].vx = V[i].vx - P.px/tot_mass;
      V[i].vy = V[i].vy - P.py/tot_mass;
      V[i].vz = V[i].vz - P.pz/tot_mass;
    }

                        // previous value (needed in Verlet step)
                        // Euler step
  for ( i=0 ; i<Npart ; i++ )
    {
      Rp[i].x = R[i].x - dt*V[i].vx;
      Rp[i].y = R[i].y - dt*V[i].vy;
      Rp[i].z = R[i].z - dt*V[i].vz;
    }
}


// Calculate & print the total force of the system
// Print mass & accelerations for each particle
void output_force()
{
  int i;
  double Ftotx, Ftoty, Ftotz, fx, fy, fz;

  Ftotx = 0.0;
  Ftoty = 0.0;
  Ftotz = 0.0;
  for ( i=0 ; i<Npart ; i++ )
    {
      fx =  mass[i]*acc[i].ax;
      fy =  mass[i]*acc[i].ay; 
      fz =  mass[i]*acc[i].az;
      printf("Mass & acceleration:  %d %f %f %f %f -- %f %f %f\n", 
	     i, mass[i], acc[i].ax, acc[i].ay, acc[i].az, fx, fy, fz );
      Ftotx = Ftotx + fx;
      Ftoty = Ftoty + fy;
      Ftotz = Ftotz + fz;
    }
  printf("Total force (x & y): %f %f %f\n", Ftotx, Ftoty, Ftotz );
}


// Compute forces & accelerations due to the " 6-12 LJ potential ".
// Choose if you want to print them in the command prompt (iprint).
void accelerations( int iprint )
{
  int i, j;
  double xij, yij, zij, rij2, rij4, rij8, rij14;
  double Fijx, Fijy, Fijz;
  double Fx[Npartmax], Fy[Npartmax], Fz[Npartmax];
  double factor;


   for ( i=0 ; i<Npart ; i++ )
    {
      Fx[i] = 0.0;
      Fy[i] = 0.0;
      Fz[i] = 0.0;
    }

  for ( i=0 ; i<Npart-1 ; i++ )
    {
      for ( j= i+1 ; j<Npart ; j++ )
	{
              xij = R[i].x - R[j].x;
              yij = R[i].y - R[j].y;
	          zij = R[i].z - R[j].z;
			  
			  // Periodic boundary conditions
			  xij -= floor(xij/boxsize)*boxsize + boxsize*0.5;
			  yij -= floor(yij/boxsize)*boxsize + boxsize*0.5;
			  zij -= floor(zij/boxsize)*boxsize + boxsize*0.5;
			  
              rij2 = xij*xij + yij*yij + zij*zij ;
			  
			  //Apply cut-off distance
				  if (rij2 < rcut2 || cutoff != 1)
				  {
					  rij4 = rij2*rij2;
					  rij8 = rij4*rij4;	   
					  rij14 = rij2*rij4*rij8;   
						  
					  factor = 4.0 * ( 12.0/rij14 - 6.0/rij8 ) - frcut;
					  
					  Fijx = factor * xij;
              		  Fijy = factor * yij;
              		  Fijz = factor * zij;

              		  Fx[i] = Fx[i] + Fijx;
	    	  		  Fy[i] = Fy[i] + Fijy;
	    	  		  Fz[i] = Fz[i] + Fijz;
              		  Fx[j] = Fx[j] - Fijx;
	    	  		  Fy[j] = Fy[j] - Fijy;
	    	  		  Fz[j] = Fz[j] - Fijz;
				  }

             

	} // end j loop
    } // i loop

   for ( i=0 ; i<Npart ; i++ )
    {
      acc[i].ax = Fx[i] / mass[i];
      acc[i].ay = Fy[i] / mass[i];
      acc[i].az = Fz[i] / mass[i];
    }

  if ( iprint == 1 )
       output_force();
}


/*
// (Basic) Verlet time step
void time_step_verlet( int iprint )
{
  int i, j;
  
  accelerations( iprint );  // Calculate accelerations at timestep "t"
  
  // (Basic) Verlet Algorithm.
  // Rnew : positions at current timestep (t).
  // R    : positions at the previous timestep (t - dt)
  // Rp   : positions 2 timesteps back (t - 2*dt)
  
  // Old Verlet scheme
  
  for ( i=0 ; i<Npart ; i++ )
    {
      Rnew[i].x = 2*R[i].x - Rp[i].x + acc[i].ax*dt*dt;
      Rnew[i].y = 2*R[i].y - Rp[i].y + acc[i].ay*dt*dt;
      Rnew[i].z = 2*R[i].z - Rp[i].z + acc[i].az*dt*dt;

      V[i].vx = ( Rnew[i].x - Rp[i].x ) / ( 2*dt );
      V[i].vy = ( Rnew[i].y - Rp[i].y ) / ( 2*dt );
      V[i].vz = ( Rnew[i].z - Rp[i].z ) / ( 2*dt );
    }
	
  // Re-set arrays.
  // Rp   : positions at t-dt
  // R    : positions at t
  // Rnew : will be re-computed at the next timestep (t+dt)
  for ( i=0 ; i<Npart ; i++ )
    {
      Rp[i].x = R[i].x;
      Rp[i].y = R[i].y;
      Rp[i].z = R[i].z;
      R[i].x = Rnew[i].x;
      R[i].y = Rnew[i].y;
      R[i].z = Rnew[i].z;
    }
}   */

// Velocity Verlet scheme
void time_step_verlet( int iprint )
{
  int i, j;
  
  for ( i = 0 ; i<Npart ; i++ )
  	{
		R[i].x += dt*V[i].vx + dt*dt*acc[i].ax*0.5;
		R[i].y += dt*V[i].vy + dt*dt*acc[i].ay*0.5;
		R[i].z += dt*V[i].vz + dt*dt*acc[i].az*0.5;
		
		V[i].vx += dt*acc[i].ax*0.5;
		V[i].vy += dt*acc[i].ay*0.5;
		V[i].vz += dt*acc[i].az*0.5;
		
		Rcorr[i].x = R[i].x - floor(R[i].x/boxsize)*boxsize; 
		Rcorr[i].y = R[i].y - floor(R[i].y/boxsize)*boxsize;
		Rcorr[i].z = R[i].z - floor(R[i].z/boxsize)*boxsize;
	}
	
  accelerations( iprint );  // Calculate accelerations at timestep "t+dt"

  for ( i = 0 ; i<Npart ; i++ )
    {
		V[i].vx += dt*acc[i].ax*0.5;
		V[i].vy += dt*acc[i].ay*0.5;
		V[i].vz += dt*acc[i].az*0.5;
	}
}


// Kinetic, Potential and Total Energy & Center of Mass momentum (compute & print).
// It recomputes the potential. This function could be merged with "accelerations()"!
void energy_momentum( int info, double time, clock_t diff )
{
  double kinetic, potential, pot, etotal;
  momentum P;
  double xij, yij, zij, rij2, rij4, rij6, rij12;
  int i, j;

  // Kinetic energy of the system
  kinetic = 0.0;
  for ( i=0 ; i<Npart ; i++ )
    {
      kinetic = kinetic + 0.5 * mass[i] * 
              ( V[i].vx*V[i].vx + V[i].vy*V[i].vy +  V[i].vz*V[i].vz );
    }

  // Potential Energy of the system
  potential = 0.0;
  for ( i=0 ; i<Npart ; i++ )
    {
      for ( j=i+1 ; j<Npart ; j++ )
        {
            xij = R[i].x - R[j].x;
            yij = R[i].y - R[j].y;
            zij = R[i].z - R[j].z;
			
			// Periodic boundary conditions
			xij -= floor(xij/boxsize)*boxsize + boxsize*0.5;
			yij -= floor(yij/boxsize)*boxsize + boxsize*0.5;
			zij -= floor(zij/boxsize)*boxsize + boxsize*0.5;
			
			rij2 = xij*xij + yij*yij + zij*zij;
			
			if (rij2 < rcut2 || cutoff != 1)
			{
				rij4 = rij2*rij2;
				rij6 = rij4*rij2;	   
				rij12 = rij6*rij6;;   
					  
				pot = 4.0 * ( 1.0/rij12 - 1.0/rij6 ) - potrcut;
            	potential = potential + pot;
			}
            
		}
    }

  // Total energy of the system
  etotal = kinetic + potential;
  fprintf( fp_traj, "\n%f %f %f %f %f", time, (double)diff, kinetic, potential, etotal ); 

  // Print the energies and current timestep
  if ( info == 0 )
    {
       printf( "%f %f %f %f\n", time, kinetic, potential, etotal );
       return;
    }
  
  // Center of mass momentum
  P.px = 0.0;
  P.py = 0.0;
  P.pz = 0.0;
  for ( i=0 ; i<Npart ; i++ )
    {
       P.px = P.px + mass[i] * V[i].vx;
       P.py = P.py + mass[i] * V[i].vy;
       P.pz = P.pz + mass[i] * V[i].vz;
    }

  // Velocity of the system (it should be zero)
  P.px = P.px / tot_mass;
  P.py = P.py / tot_mass;
  P.pz = P.pz / tot_mass;

  // Print energies and center of mass momentum
  fprintf( stderr, "Kin + pot = energy :  %f %f %f\n", 
	  kinetic, potential, etotal ); 
  fprintf( stderr, 
          "Center of Mass P   :    %f %f %f\n", P.px, P.py, P.pz );

}

// Allocate arrays. If number of particles is very large, it automatically doesn't print trajectories.
void allocateArrays(int *print_trajectory)
{

    mass = (double*)malloc(Npart*sizeof(double));
    R    = (position*)malloc(Npart*sizeof(position));
	Rcorr= (position*)malloc(Npart*sizeof(position));
    V    = (velocity*)malloc(Npart*sizeof(velocity));
    acc  = (acceleration*)malloc(Npart*sizeof(acceleration));
    Rp   = (position*)malloc(Npart*sizeof(position));
    Rnew = (position*)malloc(Npart*sizeof(position));

    if (Npart >= 90)
        *print_trajectory = 0;

}

// MAIN FUNCTION
int main( int argn, char * argv[] )
{
  int dbg_print, special, print_energy, print_trajectory, i;
  double t;

  // Extra debug print
  dbg_print = 0;
  // Number of particles
  Npart = 12;
  // Boxsize (larger than 1.2)
  boxsize = 4.0;
  //
  cutoff = 0;
  // Cut off distance
  rcut = pow(2.0,1/6);
  rcut2 = rcut*rcut;
  rcut4 = rcut2*rcut2;
  rcut6 = rcut2*rcut4;
  rcut8 = rcut4*rcut4;
  rcut12 = rcut6*rcut6;
  rcut14 = rcut2*rcut12;
  
  if (cutoff == 1)
	{
	// Force-factor at cut-off distance
  	frcut = 4.0 * ( 12.0/rcut14 - 6.0/rcut8 );
  	// Potential at cut-off distance
  	potrcut = 4.0 * ( 1.0/rcut12 - 1.0/rcut6 );
	}
  else
	{
	frcut = 0;
	potrcut = 0;
	}
  // Time parameters
  tmin = 0.0;
  tmax = 10.0;
  dt = 0.001;
  // special != 1 : Assign random mass for every particle which are sampled from a gaussian distribution.
  // special == 1 : All masses are equal to 1.
  special = 1;
  
  // Record energy at every step (ONLY if print_energy = 1)
  print_energy = 1;
  // Print trajectories in file (ONLY if print_trajectory = 1)
  print_trajectory = 1;

  // Check user input
  for ( i=1 ; i<argn ; i++ )
    {
      if ( argv[i][0] == '-' )
	{
	  switch( argv[i][1] )
	  {
	  case( 'N' ):
            Npart = atoi( argv[++i] );
	    break;
	  case( 'e' ):
	    print_energy =1;
	    break;
	  case( 's' ):
	    special =1;
	    break;
	  case( 't' ):
	    tmax = atof( argv[++i] );
	    break;
	  default:
	    printf("Syntax: ./Nbody_basic_verlet <-N> <-e> <-s> <-t>\n");
	    exit(1);
	  }
	}
    }

  // Allocate arrays
  allocateArrays(&print_trajectory);
  // Information
  printf( "\n Npart: %d tmax %f\n", Npart, tmax ); 
  // Create the file to store the trajectories
  fp_traj = fopen( "trajectories", "w" );
  // Assign mass to the particles
  mass_setup( dbg_print, special );
  // Initial time
  t = tmin;
  // Initial positions & velocities
  initial_conditions( special );
  // Print positions & velocities
  fprintf( stderr,
   "\nInitial positions, velocities & energy-momentum (time %f)\n", t );
  output_x_v();
  //fprintf( fp_traj, "%f %f", t, 0.0);
  // Compute system energies and center of mass momentum
  energy_momentum( 1, t, diff );
  // Print the positions of the particles in the file "fp_traj"
  if (print_trajectory == 1)
                 record_trajectories( );
  
  // MD Iteration
  start = clock();
  while ( t < tmax )
    {
      // Compute positions and velocities at current timestep
      time_step_verlet( dbg_print );
      // Go to the next timestep
      t = t + dt;
	  diff = clock() - start;
	  //fprintf( fp_traj, "\n%f %f", t, (double)diff); 
      // Compute system energies and center of mass momentum
      if ( print_energy == 1 )
                     energy_momentum( 0, t, diff );
      // Write the positions of the particles in the file "fp_traj".
      if (print_trajectory == 1)
                     record_trajectories( );
    }
  

  // Print, to the command prompt, final positions & velocities.
  fprintf( stderr, 
   "\nFinal positions, velocities & energy-momentum (time %f)\n", t );
  output_x_v();
  // Compute final system energies and center of mass momentum
  //energy_momentum( 1, t );
  // Close trajectories file
  fclose( fp_traj );
  // Free all the dynamically allocated pointers
  free(Rnew);
  free(acc);
  free(Rp);
  free(V);
  free(R);
  free(mass); 
     
}
