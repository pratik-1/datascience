# Various methods for generating random numbers

import numpy as np
import random
import os

def random_range(a, b, n):
    x = []
    for i in range(n):
        x.append(np.random.randint(a,b))
    return x

a = np.random.randint(1, 100)           # Random integer between 1 and 100
b = np.random.random(10)                # Random integer between 0 and 1
print(random_range(45, 87, 10))         # similar to np.random.randint(low, high, size)
print(random.randrange(0, 100, 5))      # randrange(start, stop, step)


c = np.arange(100)                      # create list of 100 numbers
np.random.shuffle(c)                    # Shuffle list
print(c)
print(np.random.choice(c, 4))           # Choose number at random choice(list, size)

# ***********************************************************************************
import matplotlib.pyplot as plt


# Uniform distrbution
rand_un = np.random.uniform(0,1,10000)  # Get 100000 number between 0 and 1.
plt.hist(rand_un, bins=50)
plt.ylabel('frequency')
plt.title('Uniform distribution')
plt.show()

# Normal distrbution
rand_n = np.random.randn(10000)            # Get 100000 number between 0(mean) and 1(std dev).
rand_n = 5*np.random.randn(10000) + 15    # Get 100000 number between 15(mean) and 5(std dev).
plt.hist(rand_n, bins=50)
plt.ylabel('frequency')
plt.title('Normal distribution')
plt.show()

# # 2nd method to plot
# rdn = np.random.uniform(0, 1, 1000)
# rand = range(1000)
# plt.plot(rand, rdn)
# plt.show()

h = np.random.randn(2, 4)     # gets 2 by 4 array from standard  normal distribution

# ****************************************************************************************
# Using the Monte-Carlo simulation to find the value of pi
"""

Geometry and mathematics behind the calculation of pi:

Consider a circle of radius r unit circumscribed inside a square of side 2r units such that
the circle’s diameter and the square’s sides have the same dimensions

What is the probability that a point chosen at random would lie inside the circle? This
probability would be given by the following formulae:Thus, we find out that the probability of 
a point lying inside the circle is pi/4. The purpose
of the simulation is to calculate this probability and use this to estimate the value of pi.

The following are the steps to be implemented to run this simulation:
1.Generate points with both x and y coordinates lying between  0  and  1 .
2.Calculate x*x + y*y. If it is less than  1 , it lies inside the circle. If it is greater than 1, it
    lies outside the circle.
3.Calculate the total number of points that lie inside the circle. Divide it by the total
    number of points generated to get the probability of a point lying inside the circle.
4.Use this probability to calculate the value of pi.
5.Repeat the process for a sufficient number of times, say, 1,000 times and generate
    1,000 different values of pi.
6.Take an average of all the 1,000 values of pi to arrive at the final value of pi.

"""
pi = []
for i in range(1000):
    x = np.random.uniform(0,1,1000)
    y = np.random.uniform(0,1,1000)
    z = np.sqrt(x ** 2 + y ** 2)
    prob = len([zee for zee in z if zee <= 1]) / len(z)
    pi.append(4*prob)

print(np.mean(pi))



b = np.random.uniform(0,1,1000)
plt.hist(pi, bins=25)
plt.ylabel('frequency')
plt.title('Normal distribution')
plt.show()



