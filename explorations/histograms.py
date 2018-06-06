import numpy as np

import matplotlib.pyplot as plt

me = [16, 28, 19, 6, 17, 29, 37, 59, 12, 11, 12, 66, 21, 32, 17, 87, 110, 63, 27, 23, 31, 18, 186, 21, 19, 19, 184, 28, 79, 29, 127, 16, 37, 152, 90, 19, 12, 27, 24, 26, 117, 16, 25, 16, 64, 27, 17, 18, 51, 25]
rand = [17, 5, 4, 6, 6, 19, 13, 8, 10, 5, 4, 7, 7, 7, 13, 6, 5, 7, 6, 7, 6, 11, 11, 9, 10, 6, 7, 12, 9, 7, 8, 6, 7, 5, 8, 5, 5, 5, 8, 7, 10, 9, 5, 5, 8, 15, 9, 9, 10, 8]
e2_950 = [9, 13, 15, 10, 9, 13, 12, 14, 26, 9, 12, 12, 12, 8, 11, 12, 11, 16, 10, 11, 6, 16, 15, 16, 6, 15, 11, 9, 19, 13, 8, 7, 8, 13, 9, 8, 8, 13, 10, 15, 7, 17, 8, 5, 12, 9, 13, 6, 12, 15]
e1_900 = [9, 8, 10, 9, 11, 10, 9, 11, 6, 11, 12, 13, 11, 12, 7, 8, 11, 6, 6, 10, 7, 10, 13, 11, 11, 12, 11, 10, 8, 7, 10, 6, 11, 11, 9, 13, 6, 7, 12, 7, 10, 9, 6, 10, 15, 10, 14, 10, 8, 10]
e1_975 = [13, 14, 12, 17, 21, 9, 21, 11, 11, 17, 13, 18, 15, 10, 10, 17, 12, 13, 9, 18, 11, 12, 12, 15, 10, 11, 1, 17, 12, 13, 8, 18, 9, 18, 20, 6, 14, 20, 13, 11, 21, 13, 21, 12, 7, 15, 9, 12, 10, 15]

bins = [i for i in range(2, 30, 4)]

path = 'data/histograms/'

plt.figure(0)
plt.hist(me, bins=bins)
plt.title("Human - Frames per trial")
plt.savefig(path + 'me.png')

plt.figure(1)
plt.hist(rand, bins=bins)
plt.title("Random Actor - Frames per trial")
plt.savefig(path + 'rand.png')
plt.clf()

plt.figure(2)
plt.hist(e2_950, bins=bins)
plt.title("Gamma=0.950 - Frames per trial")
plt.savefig(path + 'e2_950.png')
plt.clf()

plt.figure(3)
plt.hist(e1_900, bins=bins)
plt.title("Gamma=0.900 - Frames per trial")
plt.savefig(path + 'e1_900.png')
plt.clf()

plt.figure(4)
plt.hist(e1_975, bins=bins)
plt.title("Gamma=0.975 - Frames per trial")
plt.savefig(path + 'e1_975.png')
plt.clf()
