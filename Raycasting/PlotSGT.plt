#set terminal png size 1024,1024
#set output 'SGT.png'

plot 'ScalarGridTest.txt' with lines,\
	'ScalarGridDif0Test.txt' with lines,\
	'ScalarGridDif1Test.txt' with lines