set encoding utf8
set key right
set terminal png enhanced
set output "tree_nodes.png"

set title sprintf("File: tree\\\_t25.json")
set ylabel "# nodes"
set xlabel "Level in tree"
set grid
set xrange [0:]
set yrange [0:]

plot \
	"stats_before_collapse.tsv" using 1:2 title "before collapse" with linespoints,\
	"stats_after_collapse.tsv" using 1:2 title "after collapse" with linespoints
	
	
set output "tree_taggings.png"
set ylabel "# associated taggings"

plot \
	"stats_before_collapse.tsv" using 1:3 title "before collapse" with linespoints,\
	"stats_after_collapse.tsv" using 1:3 title "after collapse" with linespoints
