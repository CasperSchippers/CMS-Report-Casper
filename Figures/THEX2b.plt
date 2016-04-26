set terminal epslatex size 5.9,3.5
set output 'THEX2b.tex'

vMorse(x) = (1-exp(-x))**2
vLin(x) = x**2

set xlabel '$a(l-l_0)$'
set ylabel '$v/D_e$'
# set y2tics ('$v=D_e$' 1)
set y2range [0:3]

plot [-2:8][0:3] vMorse(x) with lines t 'Morse potential', vLin(x) with lines t 'Taylor expanded Morse potential'
