set terminal epslatex size 5.9,3.5
set output 'MDSLP3N4.tex'
input = 'MDSLP3N4'
set datafile separator " "

set multiplot layout 1,2

set origin 0,0.15
set size 0.48,0.7
set lmargin at screen 0.12
set rmargin at screen 0.4
set xlabel 'x' offset 0,-1
set ylabel 'y' offset 0,-1
set zlabel 'z' offset 2,0
set xtics offset -0.05,-0.3 1
set ytics offset 0.1,-0.3 1
set ztics 1
unset key
set datafile separator " "
set ticslevel 0
set view 70,30
splot [-2:5][-1:3][-2.5:2] input using 5:6:7 with lines, '' using 8:9:10 with lines, '' using 11:12:13 with lines, '' using 14:15:16 with lines
# [-0.2:1.4][-0.6:0.6][-0.15:0.15]

set size ratio 0.7
set lmargin at screen 0.55
set rmargin at screen 0.99
unset xtics
unset ytics
set xtics 2
set ytics 1
set xlabel 'time t' offset 0,0.75
set ylabel 'energy' offset 1.5,0
set key at 9.8,0.55
set datafile separator " "
plot [0:10][-4.7:3] input using 1:2 t 'Kinetic energy' with lines, '' using 1:3 t 'Potential energy' with lines, '' using 1:4 t 'Total energy' with lines
# [0:10][-1.05:0.8]

unset multiplot
