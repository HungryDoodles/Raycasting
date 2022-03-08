set terminal gif animate delay 20 size 1401,700
set output 'animationLarge.gif' 
stats 'output.txt' nooutput
set xrange [0:7005]
set yrange [0:3500]
set style line 2

do for [ii=1:int(STATS_blocks)] {
    plot 'output.txt' index (ii - 1) with lines lw 1

}