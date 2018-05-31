#!/usr/bin/env python3
import sys

excludes = ['crystalnet/linag/cblas_impl.hpp']


def gen_test_prog(src):
    include_name = src[len('../../include/'):]
    if include_name in excludes:
        return
    test_prog = include_name.replace('/', '_').replace('.', '_') + '.cpp'
    print(test_prog)
    with open(test_prog, 'w') as f:
        f.write('#include <%s>\n' % include_name)
        f.write('int main(){ return 0; }')


def main(args):
    for src in args:
        if src.endswith('.hpp'):
            gen_test_prog(src)


main(sys.argv[1:])
