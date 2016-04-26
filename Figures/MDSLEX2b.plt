set terminal epslatex size 5.9,3.5
set output 'MDSLEX2b.tex'

vLJ(x) = 4*((x**-12) - (x**-6))

set samples 400
set isosamples 400

set xlabel '$r/\sigma$'
set ylabel '$U\sub{LJ}/\varepsilon$'
unset key

plot [0:3][-1.5:5] vLJ(x)
