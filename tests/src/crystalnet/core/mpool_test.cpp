#include <cassert>

#include <crystalnet/core/mpool.hpp>

void test_1()
{
    MP mp;
    for (int i = 0; i < 10; ++i) {
        auto mb = mp.get(10);
        mp.put(mb);
    }
    assert(mp.allocated == 10);
    assert(mp.available == 10);
    for (int i = 0; i < 20; ++i) {
        auto mb = mp.get(20);
        mp.put(mb);
    }
    assert(mp.allocated == 30);
    assert(mp.available == 30);
}

int main()
{
    // test_1();
    return 0;
}