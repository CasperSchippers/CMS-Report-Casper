set terminal epslatex size 5.9,2.45#3.5
set output 'MDSLP2.tex'
input = 'MDSLP2'
set datafile separator " "

set multiplot layout 1,2

# set origin 0,0.15
set size 0.48,1#0.7
set lmargin at screen 0.12
set rmargin at screen 0.4
set xlabel 'x' offset 0,-1
set ylabel 'y' offset 0,-1
set zlabel 'z' offset 2,0
set xtics offset -0.05,-0.3 0.5
set ytics offset 0.1,-0.3 0.5
set ztics 0.2
unset key
set datafile separator " "
set ticslevel 0
set view 70,30
splot [-0.7:1.5][-0.7:1.5][-0.6:0.8] input using 5:6:7 with lines, '' using 8:9:10 with lines, '' using 11:12:13 with lines
# [-0.2:1.4][-0.6:0.6][-0.15:0.15]

set size ratio 0.7
set lmargin at screen 0.55
set rmargin at screen 0.99
unset xtics
unset ytics
set xtics 2
set ytics 0.5
set xlabel 'time t' offset 0,0.75
set ylabel 'energy' offset 1.5,0
set key at 9.8,0.3
set datafile separator " "
plot [0:10] input using 1:2 t 'Kinetic energy' with lines, '' using 1:3 t 'Potential energy' with lines, '' using 1:4 t 'Total energy' with lines
# [0:10][-1.05:0.8]

unset multiplot
