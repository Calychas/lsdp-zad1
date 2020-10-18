#!/usr/bin/env bash

from_path=$1
shift
to_path=$1
shift

for ARG in "$@"
do
    scp "$from_path/$ARG" "$to_path"
done

#./copy.sh calychas@192.168.0.64:lsdp_test/one calychas@192.168.0.64:lsdp_test/two file1 file2 file3