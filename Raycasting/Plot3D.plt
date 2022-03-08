set terminal gif animate delay 5 size 1024,1024
set output 'animation3D.gif' 
stats 'output3D.txt' nooutput
#set hidden3d
set xrange [0.25:0.75]
set yrange [0.25:0.75]
set zrange [0.0:0.7]
set style line 2
set view 87,10

do for [ii=int(STATS_blocks*0.3):int(STATS_blocks)] {
    splot 'output3D.txt' index (ii - 1) with points pt 7 ps 0.4 lc palette

}