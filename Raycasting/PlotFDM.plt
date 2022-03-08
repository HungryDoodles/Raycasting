set terminal gif animate delay 25 size 1024,1024
set output 'FDManimation.gif' 
stats 'output.txt' nooutput
numRay = STATS_blocks
stats 'FDM.out.txt' nooutput
numFDM = STATS_blocks
set xrange [0:1]
set yrange [0:1]
set cbrange [-0.015:0.015]
set style line 2
#set pm3d map
#set pm3d interpolate 0,0
set pm3d map explicit interpolate 0,0

do for [ii=1:int(45)] {
    splot 'FDM.out.txt' index ((ii - 1) * 5) using ($1/100):($2/100):($3/100) matrix with pm3d,\
	'output.txt' index ((ii - 1) * 5 + 5) using 1:2:(0) with lines lw 3 lt 2 notitle
    #splot 'FDM.out.txt' index (ii - 1) using ($1/125):($2/125):($3/125) matrix
    #plot 'output.txt' index (ii - 1) with lines
}