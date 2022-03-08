#set terminal gif animate delay 2 size 768,768
#set output 'animation.gif' 
stats 'output.txt' nooutput
#set xrange [0:1]
#set yrange [0:1]
set style line 2

#do for [ii=1:int(STATS_blocks)] {
#    plot 'output.txt' index (ii - 1) with lines lw 2
    plot 'output.txt' index (STATS_blocks - 5) with lines lw 2
#}