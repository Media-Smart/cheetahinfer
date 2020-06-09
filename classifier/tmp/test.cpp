#include <iostream>
#include <vector>
//#include <memory>

using namespace std;

/*
unique_ptr<int> demo()
{
    unique_ptr<int> a = unique_ptr<int>(new int(4));
    return a;
}
*/


class A
{
public:
    A(int c)
    {
        cout << "int " << c << endl;
    }
    A()
    {
        cout << "construct" << endl;
    }

    A(const A&)
    {
        cout << "copy construct" << endl;
    }

	A& operator=(const A&)
    {
        cout << "copy assignment" << endl;
        return *this;
    }
};


A test()
{
	A a(1);
	return a;
}

int main()
{
    //INT ss;
    //a.a[0];
	A a(test());
    //A a;
    A b = a;
    A c(a);
    c = b;
    //auto aa = demo();
    //*aa = 123;
    //cout << *aa << endl;
    return 0;
}
