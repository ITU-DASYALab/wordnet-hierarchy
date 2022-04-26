#!/bin/bash

escape_gnuplot(){
	sed -e 's/[@{}^_]/\\\\\0/g'
}

mkplot_cmd(){
	local metadata="$1"
	local col="$2"
	echo 'plot \'
	jq -r '.stats[] as $run | [$run.filename,$run.description] | @tsv' "$metadata" | while read line ; do
		#echo "line:$line"
		local file="$(echo "$line" | cut -f1)"
		local desc="$(echo "$line" | cut -f2 | escape_gnuplot)"
		printf '\t"%s" using 1:%d title "%s" with linespoints,\\\n' "$file" "$col" "$desc"
	done | head --bytes=-3; echo
}

gnuplot_cmds(){
	local metadata="$1"
	local basename="$(jq -r '.basename' "$metadata")"
	local hierarchy="$(jq -r '.input.hierarchy_f' "$metadata" | escape_gnuplot)"
	cat <<GNUPLOT
set encoding utf8
set key right
set terminal png enhanced
set output "${basename}_nodes.png"

set title sprintf("File: %s", "$hierarchy")
set ylabel "# nodes"
set xlabel "Level in tree"
set grid
set xrange [0:]
set yrange [0:]

GNUPLOT
mkplot_cmd "$metadata" 2
cat <<GNUPLOT
	
set output "${basename}_taggings.png"
set ylabel "# associated taggings"

GNUPLOT
mkplot_cmd "$metadata" 3
}

generate_plot(){
	local metadata="$1"
	gnuplot_cmds "$metadata" | gnuplot
}

bail(){
	echo "$@" >&2
	exit 1
}

main(){
	local metadata="$1"
	if [ ! -r "$metadata" ] ; then
		bail "metadata doesnt exist"
	fi
	cd "$(dirname "$metadata")"
	generate_plot "$(basename "$metadata")"
}

main spotify/out/wn_hierarchy_t25_max20_metadata.json
