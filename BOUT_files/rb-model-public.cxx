#include <bout/physicsmodel.hxx>
#include <smoothing.hxx>
#include <invert_laplace.hxx>
#include <initialprofiles.hxx>
#include <derivs.hxx>
#include <bout/constants.hxx>

class rbmodel : public PhysicsModel {
protected:
  int init(bool restarting);
  int rhs(BoutReal t);
private:
  Field3D n, vort; 
  Field3D phi;
  FieldGroup comms;
  BRACKET_METHOD bm;
  BoutReal kappa, mu;
  bool initial_noise ;
  class Laplacian* phiSolver;
};

int rbmodel::init(bool restart) {
  
  Options *options = Options::getRoot()->getSection("rbmodel");
  
  OPTION(options, kappa,                  -1.0) ;
  OPTION(options, mu,                     -1.0) ;
  int bracket; 
  OPTION(options, bracket,                   2) ;
  OPTION(options, initial_noise,         false) ;

  bm = BRACKET_ARAKAWA; 
  mesh->getCoordinates()->geometry();
  SOLVE_FOR2(n,vort) ;
  SAVE_REPEAT(phi);
  phiSolver = Laplacian::create();
  initial_profile("n", n);

  comms.add(n) ;
  comms.add(vort) ; 

  if (initial_noise){
    output << "\tSeeding random noise for triggering turbulent instabilities\n";  
    srand (time(NULL));
    for(int i=0; i < mesh->LocalNx ; i++){
      for(int k=0; k < mesh->LocalNz; k++){   
         n(i,0,k)    += 2.*(((double) rand()/(RAND_MAX)) - 0.5)*0.001;
         vort(i,0,k) += 2.*(((double) rand()/(RAND_MAX)) - 0.5)*0.001;
         
      }
    }
  }

  n.setBoundary("n") ;

  return 0;
}

int rbmodel::rhs(BoutReal time) {
  mesh->communicate(comms);
  phi = phiSolver->solve(vort, phi);
  mesh->communicate(phi);

  ddt(n) = - bracket(phi,n,bm)  + kappa * Delp2(n) ; 
  ddt(vort) = - bracket(phi, vort, bm) + DDZ(n) + mu * Delp2(vort); 

  return 0;
}

BOUTMAIN(rbmodel);
