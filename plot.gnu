set encoding utf8
set key right
set terminal png enhanced
set output "tree_nodes.png"
#set output "order:org-budget-singular.png"
#set output "order:org-singular-budget.png"

set title sprintf("File: tree\\\_t25.json")
set ylabel "# nodes"
set xlabel "Level in tree"
set grid
set xrange [0:]
set yrange [0:]

plot \
	"stats_initial.tsv" using 1:2 title "initial" with linespoints,\
	"stats_after_collapse.tsv" using 1:2 title "budget" with linespoints,\
	"stats_remove_singular_chain.tsv" using 1:2 title "singular chain" with linespoints
	
set output "tree_taggings.png"
set ylabel "# associated taggings"

plot \
	"stats_initial.tsv" using 1:3 title "initial" with linespoints,\
	"stats_after_collapse.tsv" using 1:3 title "budget" with linespoints,\
	"stats_remove_singular_chain.tsv" using 1:3 title "singular chain" with linespoints
