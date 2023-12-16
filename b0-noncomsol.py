#!/usr/bin/python3

# based on
# https://github.com/lukepolson/youtube_channel/blob/main/Python%20Metaphysics%20Series/vid8.ipynb

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import numba
from numba import jit

gridsize=220
physical_size=1.1 # m
stepsize=physical_size/gridsize # m
physical_boundary=1.0 # m
grid_boundary=round(gridsize*physical_boundary/physical_size)-1

edge=np.linspace(0,physical_size,gridsize)
upper_side=1+0*edge
lower_side=0*edge
right_side=0*edge
left_side=0*edge

xv,yv=np.meshgrid(edge,edge)

# this tries to solve for u2

@numba.jit("f8[:,:](f8[:,:],i8)",nopython=True,nogil=True)
def compute_potential_u2(potential,n_iter):
    length=len(potential[0])
    for n in range(n_iter):
        for i in range(1,length-1):
            for j in range(grid_boundary+1,length-1):
                potential[j][i]=(potential[j+1][i]+potential[j-1][i]+potential[j][i+1]+potential[j][i-1])/4
        for i in range(grid_boundary+1,length-1):
            for j in range(1,length-1):
                potential[j][i]=(potential[j+1][i]+potential[j-1][i]+potential[j][i+1]+potential[j][i-1])/4
        # Neumann along x=0, derivative is zero
        i=0
        for j in range(grid_boundary,length):
            potential[j][i]=potential[j][i+1]
        # Neumann along y=grid_boundary, derivative is 1
        j=grid_boundary
        for i in range(0,grid_boundary+1):
            potential[j][i]=potential[j+1][i]-1*stepsize
        # Neumann along x=grid_boundary, derivative is zero
        i=grid_boundary
        for j in range(0,grid_boundary+1):
            potential[j][i]=potential[j][i+1]
        # Neumann along x=-1, derivative is zero
        i=-1
        for j in range(0,length):
            potential[j][i]=potential[j][i-1]
        # Neumann along y=-1, derivative is zero
        j=-1
        for i in range(0,length):
            potential[j][i]=potential[j-1][i]
        # Dirichlet along y=0, value is zero
        j=0
        for i in range(grid_boundary,length):
            potential[j][i]=0
    return potential

@numba.jit("f8[:,:](f8[:,:],i8)",nopython=True,nogil=True)
def compute_potential_u3(potential,n_iter):
    length=len(potential[0])
    for n in range(n_iter):
        for i in range(1,length-1):
            for j in range(grid_boundary+1,length-1):
                potential[j][i]=(potential[j+1][i]+potential[j-1][i]+potential[j][i+1]+potential[j][i-1])/4
        for i in range(grid_boundary+1,length-1):
            for j in range(1,length-1):
                potential[j][i]=(potential[j+1][i]+potential[j-1][i]+potential[j][i+1]+potential[j][i-1])/4
        # Neumann along x=0, derivative is zero
        i=0
        for j in range(grid_boundary,length):
            potential[j][i]=potential[j][i+1]
        # Dirichlet along y=grid_boundary, value is u1
        j=grid_boundary
        for i in range(0,grid_boundary+1):
            potential[j][i]=j*stepsize
        # Dirichlet along x=grid_boundary, value is u1
        i=grid_boundary
        for j in range(0,grid_boundary+1):
            potential[j][i]=j*stepsize
        # Dirichlet along x=-1, value is zero
        i=-1
        for j in range(0,length):
            potential[j][i]=0
        # Dirichlet along y=-1, value is zero
        j=-1
        for i in range(0,length):
            potential[j][i]=0
        # Dirichlet along y=0, value is zero
        j=0
        for i in range(grid_boundary,length):
            potential[j][i]=0
    return potential

@numba.jit("f8[:,:](f8[:,:],i8)",nopython=True,nogil=True)
def compute_potential_u1(potential,n_iter):
    length=len(potential[0])
    for n in range(n_iter):
        # Neumann along x=0, derivative is zero
        i=0
        for j in range(0,grid_boundary+1):
            potential[j][i]=potential[j][i+1]
        # Neumann along y=grid_boundary, value is 1
        j=grid_boundary
        for i in range(0,grid_boundary+1):
            potential[j][i]=potential[j-1][i]+1*stepsize
        # Neumann along x=grid_boundary, value is 0
        i=grid_boundary
        for j in range(0,grid_boundary+1):
            potential[j][i]=potential[j][i-1]
        # Dirichlet along y=0, value is zero
        j=0
        for i in range(0,grid_boundary+1):
            potential[j][i]=0
        for i in range(1,grid_boundary):
            for j in range(grid_boundary):
                potential[j][i]=(potential[j+1][i]+potential[j-1][i]+potential[j][i+1]+potential[j][i-1])/4
    return potential


u3=np.zeros((gridsize,gridsize))
u3=compute_potential_u3(u3,n_iter=100000)

u1=np.zeros((gridsize,gridsize))
u1=compute_potential_u1(u1,n_iter=100000)

u2=np.zeros((gridsize,gridsize))
u2=compute_potential_u2(u2,n_iter=100000)


current = 0.175 # amperes; design current = step in scalar potential
maxphi = 14 # amperes; biggest you can imagine the scalar potential to be
num = round(maxphi/current) # half the number of equipotentials
maxlevel = (2*num-1)*current/2
minlevel = -maxlevel
levels = np.arange(minlevel,maxlevel,current)

fig,(ax1,ax2)=plt.subplots(2,1)
u13=u1+u3
for i in range(grid_boundary+1):
    u13[i][grid_boundary]=u1[i][grid_boundary]
    u13[grid_boundary][i]=u1[grid_boundary][i]

mask=(xv<physical_boundary)&(yv<physical_boundary)
#u13=np.ma.masked_array(u13,mask=mask)
u13_contours=ax1.contour(xv,yv,u13,13,levels=levels)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
fig.colorbar(u13_contours,ax=ax1,label='$u$')

u23=u2-u3
u23=np.ma.masked_array(u23,mask=mask)
u23_contours=ax2.contour(xv,yv,u23,13,levels=levels)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
fig.colorbar(u23_contours,ax=ax2,label='$u$')
plt.show()

print("There are %d outer coils."%len(u23_contours.allsegs))
