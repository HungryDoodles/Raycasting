#set terminal png size 1024,1024
#set output 'SGT.png'

plot 'BSplineTest.txt' with lines,\
	'BSplineTestDeriv.txt' u 1:2 with lines title "dx",\
	'BSplineTestDeriv.txt' u 1:3 with lines title "ddx"