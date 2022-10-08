from mpl_toolkits.mplot3d import Axes3D # pip install mpl_toolkits.clifford
import matplotlib.pyplot as plt # pip install --upgrade matplotlib
import numpy as np
import struct

from pathlib import Path
data = Path('f:\\UserData\\Projects\\dpcpp-mcml\\build\\output.bin').read_bytes()

print(len(data))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

size = 10
m = np.zeros(shape = (size * 7, size, size))

for l in range(0,7):
    for z in range(0,10):
        for y in range(0,10):
            for x in range(0,10):
                
                index = (x + y * 10 + z * 100 + l * 1000) * 4

                i = int.from_bytes(data[index:index + 4], byteorder='little', signed=False)

                if i != 0:
                    
                    print((x,y,z))

                    m[(x,y,z)] = 1

#random_location_1 = (1,1,2)
#random_location_2 = (3,5,8)
#m[random_location_1] = 1
#m[random_location_2] = 1

pos = np.where(m==1)
ax.scatter(pos[0], pos[1], pos[2], c='black')
plt.show()