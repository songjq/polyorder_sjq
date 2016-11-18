#include <iostream>
#include <blitz/array.h>
//#include <blitz/tinyvec-et.h>
//#include <blitz/tinymat2.h>

using namespace std;

void test_tinymatrix(){
    blitz::TinyMatrix<double,3,3> h;
    blitz::TinyVector<double,3> x, y, z;
    //blitz::Array<double,2> h(3,3);
    //blitz::Array<double,1> x(3), y(3), z(3);

    h = 0, 1, 2,
        3, 4, 5,
        6, 7, 8;
    x = 1, 2, 3;
    y = 3, 2, 1;
    z = blitz::product(h, x);
    cout<<h<<endl;
    cout<<x<<endl;
    cout<<y<<endl;
    cout<<z<<endl;
}

void test_mean_1d(){
    cout<<"Test blitz::mean 1d: "<<endl;
    blitz::Array<double, 1> data(8);
    blitz::firstIndex i;
    data = i;
    cout<<data<<endl;
    cout<<"data mean: "<<mean(data)<<endl;
    cout<<endl;
}

void test_mean_2d(){
    cout<<"Test blitz::mean 2d: "<<endl;
    int Lx = 3;
    int Ly = 4;
    int Lz = 1;

    blitz::Array<double, 3> w(Lx, Ly, Lz);
    blitz::firstIndex i;
    blitz::secondIndex j;
    blitz::thirdIndex k;
    w = Ly * i + j;
    blitz::Range all = blitz::Range::all();
    cout<<"w(:, :, 0) ="<<w(all, all, 0)<<endl;  // Using blitz++ output

    blitz::Array<double, 2> dataz(Lx, Ly);
    dataz = blitz::mean(w, k);
    cout<<"data mean along z: "<<dataz<<endl;
    blitz::Array<double, 1> dataxz(Ly);
    dataxz = blitz::mean(dataz(j, i), j);
    cout<<"data mean along x and z: "<<dataxz<<endl;
    cout<<endl;
}

void test_mean_3d(){
    cout<<"Test blitz::mean 3d: "<<endl;
    int Lx = 3;
    int Ly = 4;
    int Lz = 2;

    blitz::Array<double, 3> w(Lx, Ly, Lz);
    blitz::firstIndex i;
    blitz::secondIndex j;
    blitz::thirdIndex k;
    w = Ly * Lz * i + Lz * j + k;
    blitz::Range all = blitz::Range::all();
    cout<<"w(:, :, 0) ="<<w(all, all, 0)<<endl;  // Using blitz++ output
    cout<<"w(:, :, 1) ="<<w(all, all, 1)<<endl;

    blitz::Array<double, 2> datay(Lx, Lz);
    datay = blitz::mean(w(i, k, j), k);
    cout<<"data mean along y: "<<datay<<endl;
    blitz::Array<double, 1> dataxy(Lz);
    dataxy = blitz::mean(datay(j, i), j);
    cout<<"data mean along x and y: "<<dataxy<<endl;
    cout<<endl;
}

void test_complex_double(){
    cout<<"Test blitz mixing complex and double operations: "<<endl;
    int Lx = 3;
    int Ly = 4;
    int Lz = 2;

    blitz::firstIndex i;
    blitz::secondIndex j;
    blitz::thirdIndex k;
    blitz::Range all = blitz::Range::all();

    complex<double> ic(0, 1);

    blitz::Array<double, 3> w(Lx, Ly, Lz);
    w = Ly * Lz * i + Lz * j + k;
    cout<<"w(:, :, 0) ="<<w(all, all, 0)<<endl;
    cout<<"w(:, :, 1) ="<<w(all, all, 1)<<endl;

    blitz::Array<complex<double>, 3> c(Lx, Ly, Lz);
    c = ic;
    cout<<"c(:, :, 0) ="<<c(all, all, 0)<<endl;
    cout<<"c(:, :, 1) ="<<c(all, all, 1)<<endl;

    blitz::Array<complex<double>, 3> r(Lx, Ly, Lz);

    /*
    // Following codes fail to compile
    cout<<"3D complex = 3D double"<<endl;
    r = w;
    cout<<"r(:, :, 0) ="<<r(all, all, 0)<<endl;
    cout<<"r(:, :, 1) ="<<r(all, all, 1)<<endl;
    */

    cout<<"3D complex + 3D double"<<endl;
    r = c + w;
    cout<<"r(:, :, 0) ="<<r(all, all, 0)<<endl;
    cout<<"r(:, :, 1) ="<<r(all, all, 1)<<endl;

    cout<<"3D double + 3D complex"<<endl;
    r = w + c;
    cout<<"r(:, :, 0) ="<<r(all, all, 0)<<endl;
    cout<<"r(:, :, 1) ="<<r(all, all, 1)<<endl;

    cout<<"3D complex + double scalar"<<endl;
    r = c + 1.0;
    cout<<"r(:, :, 0) ="<<r(all, all, 0)<<endl;
    cout<<"r(:, :, 1) ="<<r(all, all, 1)<<endl;

    cout<<"double scalar + 3D complex"<<endl;
    r = 1.0 + c;
    cout<<"r(:, :, 0) ="<<r(all, all, 0)<<endl;
    cout<<"r(:, :, 1) ="<<r(all, all, 1)<<endl;

    cout<<"3D complex + complex scalar"<<endl;
    r = c + ic;
    cout<<"r(:, :, 0) ="<<r(all, all, 0)<<endl;
    cout<<"r(:, :, 1) ="<<r(all, all, 1)<<endl;

    cout<<"complex scalar + 3D scalar"<<endl;
    r = ic + c;
    cout<<"r(:, :, 0) ="<<r(all, all, 0)<<endl;
    cout<<"r(:, :, 1) ="<<r(all, all, 1)<<endl;

    cout<<"3D complex - double scalar"<<endl;
    r = c - 1.0;
    cout<<"r(:, :, 0) ="<<r(all, all, 0)<<endl;
    cout<<"r(:, :, 1) ="<<r(all, all, 1)<<endl;

    cout<<"double scalar - 3D complex"<<endl;
    r = 1.0 - c;
    cout<<"r(:, :, 0) ="<<r(all, all, 0)<<endl;
    cout<<"r(:, :, 1) ="<<r(all, all, 1)<<endl;

    cout<<"3D double * 3D complex"<<endl;
    r = w * c;
    cout<<"r(:, :, 0) ="<<r(all, all, 0)<<endl;
    cout<<"r(:, :, 1) ="<<r(all, all, 1)<<endl;

    cout<<"3D complex * 3D double"<<endl;
    r = c * w;
    cout<<"r(:, :, 0) ="<<r(all, all, 0)<<endl;
    cout<<"r(:, :, 1) ="<<r(all, all, 1)<<endl;
}

void test_zip(){
    cout<<"Test blitz zip operations: "<<endl;
    int Lx = 3;
    int Ly = 4;
    int Lz = 2;

    blitz::firstIndex i;
    blitz::secondIndex j;
    blitz::thirdIndex k;
    blitz::Range all = blitz::Range::all();

    complex<double> ic(0, 1);

    blitz::Array<double, 3> w(Lx, Ly, Lz);
    w = Ly * Lz * i + Lz * j + k;
    cout<<"w(:, :, 0) ="<<w(all, all, 0)<<endl;
    cout<<"w(:, :, 1) ="<<w(all, all, 1)<<endl;

    cout<<"blitz::zip with 3D doulbe + double scalar"<<endl;
    blitz::Array<complex<double>, 3> c(Lx, Ly, Lz);
    c = blitz::zip(w, 1.0, complex<double>());
    cout<<"c(:, :, 0) ="<<c(all, all, 0)<<endl;
    cout<<"c(:, :, 1) ="<<c(all, all, 1)<<endl;

    cout<<"blitz::zip with 3D double + blitz::index expr"<<endl;
    c = blitz::zip(w, Ly * Lz * i + Lz * j + k, complex<double>());
    cout<<"c(:, :, 0) ="<<c(all, all, 0)<<endl;
    cout<<"c(:, :, 1) ="<<c(all, all, 1)<<endl;
}

int main(){
    //test_tinymatrix()
    //test_mean_1d();
    //test_mean_2d();
    //test_mean_3d();
    test_complex_double();
    //test_zip();
}
