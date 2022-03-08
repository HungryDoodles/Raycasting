#set terminal png size 1024,1024
#set output 'FDM.png' 
#stats 'output.txt' nooutput
#numRay = STATS_blocks
stats 'FDM.out.txt' nooutput
numFDM = STATS_blocks
set xrange [0:1]
set yrange [0:1]

#set table "temp_Rays.dat"
#set style line 2
#plot 'output.txt' index (1) with lines
#unset table

#set cbrange [-0.1:0.1]
set pm3d map explicit interpolate 0,0

splot 'FDM.out.txt' index (numFDM - 2) using ($1/100):($2/100):($3/100) matrix with pm3d
#splot 'FDM.out.txt' index (numFDM - 20) using ($1/100):($2/100):($3/100) matrix with pm3d,\
#	'output.txt' index (numFDM - 20 + 5) using 1:2:(0) with lines lw 3 lt 2 notitle
#splot 'output.txt' index (numFDM - 52) using 1:2:(1) with lines lt -1 lw 2,\
#	'FDM.out.txt' index (numFDM - 52) using ($1/125):($2/125):($3/125) matrix