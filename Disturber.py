import numpy as np


class Disturber:

    # Class with all the methods to disturb a cluster

    def __init__(self, local_optimizer):
        self.local_optimizer = local_optimizer

    def random_setup(self, cluster):
        pass

    def angular_movement(self, cluster):
        pass

    def md(self, cluster):
        pass

    def twist(self, cluster):
        pass

    def etching(self, cluster):
        pass

    @staticmethod
    def split_cluster(cluster, p1=np.random.rand(3), p2=np.random.rand(3), p3=np.random.rand(3)):
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        d = - np.dot(normal, p1)
        group1 = []
        group2 = []
        for atom in cluster:
            val = np.dot(normal, atom.position) + d
            if val > 0:
                group1.append(atom)
            else:
                group2.append(atom)
        return group1, group2
