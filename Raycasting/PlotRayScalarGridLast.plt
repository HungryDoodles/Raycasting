#set terminal gif animate delay 59 size 1401,700
#set output 'animationLarge.gif' 
stats 'output.txt' nooutput
set xrange [0:7005]
set yrange [0:3500]
set style line 2

#plot 'output.txt' index (int(STATS_blocks) - 5) with lines lw 1
plot 'output.txt' index (0) with lines lw 1