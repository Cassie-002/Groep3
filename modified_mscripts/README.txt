/*-----------------------------------------------------------------------------------------*\
|  	/=====										    |
|      /										    |
|      |   =====    Classical Trajectory Calculations | Eindhoven university of Technology  |
|      \     |      MATLAB Code for MD simulation of binary collisions of diatomics	    |
|       \====|      Developed by MSc Benjamin Vollebregt 				    |
|          /=|===   Energy Technology Group, faculty of Mechanical Engineering		    |
|         /  |   	     								    |	
|         |										    |
|         \										    |
|          \=====									    |
|											    |
\*-----------------------------------------------------------------------------------------*/

This file includes all information concerning the MATLAB CTC code within this folder.

The folder contains all functions required to run a CTC simulation:
- getFij.m
- getM.m
- getRandRotMat.m
- getRdot.m
- getVdot.m
- getWdot.m
- LJ.m
- LJ_e.m
- dscatter.m

All of these functions files include clear comments and a description of the corresponding purpose, required inputs and computed outputs.
If not specified differently, all variables are defined in SI units.
The current version of the code is set for two colliding hydrogen molecules, interacting via the Lennard-Jones potential of Dondi. If one wished to run using a
different potential or molecule, one must change the parameters in the LJ.m and LJ_e.m functions, as well as the molecular information in the execution 
file (lines 14-20 & 58-70).

Furthermore, this folder contains 4 execution files from which one can run one or more CTC collisions. Each of these execution files are shortly discussed here below.

/*--------------------------------------------------------------------------------------------------------------------------------------------------------------*\
| CTC_H2_Single_Collision.m 																	 |
|  																				 |
| This code runs a single binary hydrogen collisions with random initial orientation and random initial energies. Energies are within the following ranges:	 |
| - Relative translational energy is between 100 and 6000 Kelvin (can be converted to Joules with Boltzmann's constant)						 |
| - Rotational energies of molecules A and B are between 0 and 3000 Kelvin											 |
|																				 |
| Note that the time-step size should be set small enough to ensure conservation of total energy throughout the simulation (0.1E-15 is recommended)		 |
| Once finished, the script visualizes the time-evolution of all energies within the system, as well as the time-evolution of the intermolecular distance, i.e., |
| the distance between the two molecules centers of mass.													 |
\*--------------------------------------------------------------------------------------------------------------------------------------------------------------*/


/*--------------------------------------------------------------------------------------------------------------------------------------------------------------*\
| CTC_H2_Single_Collision_Animation.m 																 |	
|  																				 |
| This code runs a single binary hydrogen collisions in a similar fashion to CTC_H2_Single_Collision.m. This script shows the animation of the binary collision  |
| in a 3D plot, which is updated each time-step. Once finished, the script saves the animation to CTC_animation.gif.						 |
| Note that this script can take a while to finish, increasing the time-step size decreases the total simulation time, but can cause an unstable simulation.	 |
\*--------------------------------------------------------------------------------------------------------------------------------------------------------------*/



/*--------------------------------------------------------------------------------------------------------------------------------------------------------------*\
| CTC_H2_Multiple_Collisions.m																	 |
|				 																 |
| This code generates a database of N collisions that are initially randomly oriented and take on initial energies similar to CTC_H2_Single_Collision.m		 |
| The number of collisions can be specified in line 6. For each collision, the impact parameter, pre- and postcollisional energies are saved to the variable     |
| "Table", which is saved to 'collision_dataset.txt'. Once finished, the script also plots the energy correlation graphs for relative translational energy Etr,  |
| rotational energy of molecule A ErA and rotational energy of molecule B ErB. Moreover, the script plots the scattering of the change in these energies as a    |
| function of the impact parameter b.																 |
\*--------------------------------------------------------------------------------------------------------------------------------------------------------------*/	

/*--------------------------------------------------------------------------------------------------------------------------------------------------------------*\
| CTC_H2_Multiple_Collisions_Parallel.m																 |
|				 																 |
| This code does the same as CTC_H2_Multiple_Collisions.m, but can be run in parallel using MATLAB's parpool feature. This script works best when the parpool 	 |
| with the desired number of CPUs is already active.														 |
\*--------------------------------------------------------------------------------------------------------------------------------------------------------------*/