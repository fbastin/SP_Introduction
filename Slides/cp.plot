set grid
plot [0:10][0:8]7-x,4-x
replot 7-2*x lw 3,4-5.0/9*x lw 3
replot 7-3*x lw 3,4-7.0/9*x lw 3

set terminal jpg
set output "graph.jpg"

replot 7-4*x,4-1.0/3*x