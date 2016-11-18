#include <iostream>
#include "armadillo"

using namespace std;
using namespace arma;

void test_cx_mat(){
    uword N = 4;
    cx_mat cA(N, N);
    cx_double ci = cx_double(0, 1);

    cA.fill(ci);
    //cA.fill(fill::zeros);
    cA.print("cA =");

    mat A = real(cA);
    A.print("Real part of cA =");
    mat B = imag(cA);
    B.print("Imaginary part of cA =");

    mat I = eye<mat>(N, N);
    I.print("I = ");

    cx_mat cX = I + cA;
    cX.print("cX = I + cA =");

    cx_double z = cx_double(1, 1);
    cx_mat cZ = z * I + B;
    cZ.print("cZ = z * I + B =");
}

void change_subview(mat &A){
    A(1, 1) = 9;
}

void test_subview(){
    uword N = 5;
    mat A = eye<mat>(N, N);
    A.print("A =");
    mat B = A.submat(span(1, N-1), span(1, N-1));
    B.print("B =");
    B(1, 1) = 9;
    B.print("B =");
    A.print("A =");

    change_subview(B);
    B.print("B =");
    A(span(1, N-1), span(1, N-1)) = B;
    A.print("A =");

    mat C = A(span(1, N-2), span::all);
    C.print("C =");

    change_subview(C);
    C.print("C =");
    A(span(1, N-2), span::all) = C;
    A.print("A =");
}

void test_flip(){
    uword N = 8;
    colvec u = linspace(0, N, N+1);
    u.print("u =");
    colvec ru = flipud(u);
    ru.print("ru =");
}

void test_join(){
    uword N = 4;
    colvec u = linspace(0, N, N+1);
    u.print("u =");
    colvec ru = flipud(u);
    ru.print("ru =");

    colvec ju = join_vert(u, ru);
    ju.print("ju =");

    colvec ju2 = join_vert(u, flipud(u.subvec(1, N-1)));
    ju2.print("ju2 =");
}

void test_subcube_view(){
    mat A;
    A << 1 << 2 << 3 << endr
      << 4 << 5 << 6 << endr;
    cube C = zeros<cube>(2, 3, 4);
    C.slice(0) = A;
    C.slice(1) = 2 * A;
    C.slice(2) = 3 * A;
    C.slice(3) = 4 * A;

    C.print("C =");

    colvec v = C.tube(0, 0);
    v.print("v =");

    mat D;
    D << 1 << 2 << 3 << 4 << endr
      << 5 << 6 << 7 << 8 << endr
      << 9 << 0 << 1 << 2 << endr
      << 3 << 4 << 5 << 6 << endr;
    //colvec E = D * C.tube(0, 0); // This line fails to compile
    colvec E = D * v;  // This line will compile and give correct result.
    E.print("E =");

    C.tube(1, 1) = 2 * C.tube(0, 0);  // Correct.
    C.print("C =");
    //C.tube(1, 1) = 2 * v;  // Compile OK, runtime error for incompatible size
    for(uword i=0; i<v.n_elem; i++)
        C(1, 1, i) = 2 * v(i);
    C.print("C =");
}

void test_fft(){
    mat A;
    A << 1 << 2 << 3 << endr
      << 4 << 5 << 6 << endr;
    cube C = zeros<cube>(2, 3, 4);
    C.slice(0) = A;
    C.slice(1) = 2 * A;
    C.slice(2) = 3 * A;
    C.slice(3) = 4 * A;
    C.print("C =");

    cx_cube Ck(2, 3, 4);
    for(uword i=0; i<4; i++)
        Ck.slice(i) = fft2(C.slice(i));
    cube Ckr = real(Ck);
    cube Cki = imag(Ck);
    Ckr.print("Real Ck =");
    Cki.print("Imaginary Ck =");

    cube Cb(2, 3, 4);
    for(uword i=0; i<4; i++)
        Cb.slice(i) = real(ifft2(Ck.slice(i)));
    Cb.print("Cb =");

    cube E(1, 1, 4);
    for(uword i=0; i<4; i++)
        E(0,0,i) = i + 1;
    E.print("E =");
    cx_cube Ek(1, 1, 4);
    for(uword i=0; i<4; i++)
        Ek.slice(i) = fft2(E.slice(i));
    Ek.print("Ek =");
}

int main(){
    //test_cx_mat();
    //test_subview();
    //test_flip();
    //test_join();
    test_subcube_view();
    //test_fft();

    return 0;
}
