
MIN_N_POW_2 = 1
MAX_N_POW_2 = 23
MIN_COORD = 0
MAX_COORD = 1000
AVG_COORD = (MAX_COORD - MIN_COORD) / 2.0
STD_DEV = MAX_COORD / 4.0
NUM_CLUSTERS = 3
CLUSTER_STD_DEV = STD_DEV / 2.0
SURFACE_STD_DEV = MAX_COORD * .01 # 1%
SEED = 0xDEADFACE

import itertools
import random

class Distribution:
    def __init__(self, name):
        self.name = name

    # should return a list of 3 floats
    def make_point(self):
        raise NotImplementedError

    def make_file(self, n):
        assert n > 0
        filename = self.name + '.' + str(n) + '.in'
        f = open(filename, 'w')

        f.write(str(n) + '\n')

        for i in range(n):
            p = self.make_point()
            assert len(p) == 3
            s = ' '.join([str(coord) for coord in p]) + '\n'
            f.write(s)

        f.close()

class UniformCube(Distribution):
    def __init__(self):
        Distribution.__init__(self, "uniform_cube")

    def make_point(self):
        return [ random.uniform(MIN_COORD, MAX_COORD)
                 for i in range(3) ]

class NormalOrigin(Distribution):
    def __init__(self):
        Distribution.__init__(self, "normal_origin")

    def make_point(self):
        return [ random.normalvariate(AVG_COORD, STD_DEV)
                 for i in range(3) ]

class NormalCluster(Distribution):
    def __init__(self):
        Distribution.__init__(self, str(NUM_CLUSTERS) + "_clusters")
        norm = NormalOrigin()
        self.clusters = [ norm.make_point()
                          for i in range(NUM_CLUSTERS) ]
        assert len(self.clusters) == NUM_CLUSTERS

    def make_point(self):
        center = random.choice(self.clusters)
        return [ random.normalvariate(center[i], CLUSTER_STD_DEV)
                 for i in range(NUM_CLUSTERS) ]

class CubeSurface(Distribution):
    def __init__(self):
        Distribution.__init__(self, "cube_surface")

    def make_point(self):
        i = random.uniform(MIN_COORD, MAX_COORD)
        j = random.uniform(MIN_COORD, MAX_COORD)
        k = random.normalvariate(0, SURFACE_STD_DEV)
        points_on_face = list(itertools.permutations([i, j, k]))
        face_index = random.randint(0, 5)
        return points_on_face[face_index]

def main():
    for e in range(MIN_N_POW_2, MAX_N_POW_2 + 1):
        n = 1 << e
        print 'n = '+str(n)+' (' + str(e)+'/'+str(MAX_N_POW_2)+')...'
        UniformCube().make_file(n)
        NormalOrigin().make_file(n)
        NormalCluster().make_file(n)
        CubeSurface().make_file(n)

main()
