
# bash measure_inference_time.sh inf_time_rgb.py inf_result_rgb.txt

for var in {1..100}
do
	start=`date +%s.%N`
	python $1
	finish=`date +%s.%N`
	diff=$( echo "$finish - $start" | bc -l )
	line="${diff},"
	echo "$line" >> $2
done

