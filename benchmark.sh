#!/bin/sh

# User defined.
SZ=512
RUNS=3
PROG="sum"

FILE="$PROG.csv"

rm -f $FILE

echo "optimization,runtime,array_size,init_wall,init_cpu,ref_wall,ref_cpu" >>$FILE

for optimization in {0..3}; do
	for runtime in $(seq 1 $RUNS); do
		out=$(make -e OPT=$optimization clean $PROG && ./$PROG $SZ)
		[[ "$?" -ne 0 ]] && exit 1
		parsed=$(echo $out | awk -F 'gcc' '{print $2}')
		array_size=$(echo $parsed | cut -d " " -f 12)
		init_wall=$(echo $parsed | cut -d " " -f 15 | cut -d "=" -f 2)
		init_cpu=$(echo $parsed | cut -d " " -f 17 | cut -d "=" -f 2)
		ref_wall=$(echo $parsed | cut -d " " -f 23 | cut -d "=" -f 2)
		ref_cpu=$(echo $parsed | cut -d " " -f 25 | cut -d "=" -f 2)
		echo "$optimization,$runtime,$array_size,$init_wall,$init_cpu,$ref_wall,$ref_cpu" >>$FILE
	done
done
