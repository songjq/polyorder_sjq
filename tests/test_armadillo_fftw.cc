#include <iostream>
#include "armadillo"
#include "fftw3.h"

using namespace std;
using namespace arma;

void test_fftw_1d(){
    /** Test correctness **/
    colvec v = linspace<colvec>(0, 9, 10);
    v.print("v =");

    cx_colvec vk = fft(v);
    real(vk).print("real vk =");
    imag(vk).print("imag vk =");
    cx_colvec vk_arma = vk;

    vk.fill(0);
    vk.set_real(v);
    real(vk).print("real vk =");
    imag(vk).print("imag vk =");
    fftw_complex *in = reinterpret_cast<fftw_complex*>(vk.memptr());
    fftw_plan plan = fftw_plan_dft_1d(vk.n_elem, in, in,
                                      FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan);
    real(vk).print("real vk =");
    imag(vk).print("imag vk =");
    cx_colvec vk_fftw = vk;

    colvec diffr = abs(real(vk_arma) - real(vk_fftw));
    colvec diffc = abs(imag(vk_arma) - imag(vk_fftw));
    cout<<"Max difference btween arma::fft and fftw: "<<diffr.max();
    cout<<" and "<<diffc.max()<<endl;
}

void test_fftw_speed_1d(){
    /** Test speed **/
    uword N = 128;
    uword N_test = 1000000;
    colvec v = randu<colvec>(N);

    wall_clock timer;
    cx_colvec vk;
    timer.tic();
    for(uword i=0; i<N_test; i++)
        vk = fft(v);
    double n_secs = timer.toc();
    // N=64, N_test=1000000, 3.38597 secs.
    // N=128, N_test=1000000, 7.16354 secs.
    cout<<N_test<<" runs took "<<n_secs<<" seconds for arma::fft."<<endl;

    vk.fill(0);
    vk.set_real(v);
    fftw_complex *in = reinterpret_cast<fftw_complex*>(vk.memptr());
    fftw_plan plan = fftw_plan_dft_1d(vk.n_elem, in, in,
                                      FFTW_FORWARD, FFTW_MEASURE);
    timer.tic();
    for(uword i=0; i<N_test; i++)
        fftw_execute(plan);
    n_secs = timer.toc();
    // N=64, N_test=1000000, 0.284679 secs.
    // N=128, N_test=1000000, 0.778956 secs.
    cout<<N_test<<" runs took "<<n_secs<<" seconds for FFTW."<<endl;
}

void test_fftw_2d(){
    /** Test correctness **/
    mat v = linspace<mat>(0, 15, 16);
    v.reshape(4, 4);
    v.print("v =");

    cx_mat vk = fft2(v);
    real(vk).print("real vk =");
    imag(vk).print("imag vk =");
    cx_mat vk_arma = vk;

    vk.fill(0);
    vk.set_real(v);
    real(vk).print("real vk =");
    imag(vk).print("imag vk =");
    fftw_complex *in = reinterpret_cast<fftw_complex*>(vk.memptr());
    fftw_plan plan = fftw_plan_dft_2d(vk.n_rows, vk.n_cols, in, in,
                                      FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan);
    real(vk).print("real vk =");
    imag(vk).print("imag vk =");
    cx_mat vk_fftw = vk;

    mat diffr = abs(real(vk_arma) - real(vk_fftw));
    mat diffc = abs(imag(vk_arma) - imag(vk_fftw));
    cout<<"Max difference btween arma::fft and fftw: "<<diffr.max();
    cout<<" and "<<diffc.max()<<endl;
}

void test_fftw_speed_2d(){
    /** Test speed **/
    uword Nx = 64;
    uword Ny = 64;
    uword N_test = 100000;
    mat v = randu<mat>(Nx, Ny);

    wall_clock timer;
    cx_mat vk;
    timer.tic();
    for(uword i=0; i<N_test; i++)
        vk = fft2(v);
    double n_secs = timer.toc();
    // Nx=64, Ny=64, N_test=1000000, 178.636 secs.
    // Nx=128, Ny=64, N_test=1000000, 407.754 secs.
    cout<<N_test<<" runs took "<<n_secs<<" seconds for arma::fft."<<endl;

    vk.fill(0);
    vk.set_real(v);
    timer.tic();
    for(uword i=0; i<N_test; i++){
        fftw_complex *in = reinterpret_cast<fftw_complex*>(vk.memptr());
        fftw_plan plan = fftw_plan_dft_2d(vk.n_rows, vk.n_cols, in, in,
                                          FFTW_FORWARD,
                                          FFTW_ESTIMATE/*FFTW_MEASURE*/);
        fftw_execute(plan);
    }
    n_secs = timer.toc();
    // Nx=64, Ny=64, N_test=1000000, 42.200 secs.
    // Nx=128, Ny=64, N_test=1000000, 89.853 secs.
    cout<<N_test<<" runs took "<<n_secs<<" seconds for arma::fft."<<endl;
}

int main(){
    //test_fftw_1d();
    //test_fftw_speed_1d();
    //test_fftw_2d();
    test_fftw_speed_2d();

    return 0;
}
