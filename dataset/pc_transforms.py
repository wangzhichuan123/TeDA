import numpy as np 
import collections
import torch 
from scipy.linalg import expm, norm 

class PointsToTensor(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, pc):
        if not torch.is_tensor(pc):
            if str(pc.dtype) == 'float64':
                pc = pc.astype(np.float32)
            pc = torch.from_numpy(np.array(pc))
        return pc 

# ony suits the gpu 
class PointCloudScaling(object):
    def __init__(self,
                 scale=[2. / 3, 3. / 2],
                 anisotropic=True,
                 scale_xyz=[True, True, True],
                 mirror=[0, 0, 0],  # the possibility of mirroring. set to a negative value to not mirror
                 **kwargs):
        self.scale_min, self.scale_max = np.array(scale).astype(np.float32)
        self.anisotropic = anisotropic
        self.scale_xyz = scale_xyz
        self.mirror = torch.from_numpy(np.array(mirror))
        self.use_mirroring = torch.sum(torch.tensor(self.mirror)>0) != 0

    def __call__(self, pc):
        device = pc.device 
        scale = torch.rand(3 if self.anisotropic else 1, dtype=torch.float32, device=device) * (
                self.scale_max - self.scale_min) + self.scale_min
        if self.use_mirroring:
            assert self.anisotropic==True
            self.mirror = self.mirror.to(device)
            mirror = (torch.rand(3, device=device) > self.mirror).to(torch.float32) * 2 - 1
            scale *= mirror
        for i, s in enumerate(self.scale_xyz):
            if not s: scale[i] = 1

        pc *= scale
        
        return pc
    

class PointCloudCenterAndNormalize(object):
    def __init__(self, centering=True,
                 normalize=True,
                 gravity_dim=2,
                 append_xyz=False,
                 **kwargs):
        self.centering = centering
        self.normalize = normalize
        self.gravity_dim = gravity_dim
        self.append_xyz = append_xyz

    def __call__(self, pc):
        if self.centering:
            pc = pc - torch.mean(pc, axis=-1, keepdims=True)
        if self.normalize:
            m = torch.max(torch.sqrt(torch.sum(pc ** 2, axis=-1, keepdims=True)), axis=0, keepdims=True)[0]
            pc = pc / m
        return pc


class PointCloudRotation(object):
    def __init__(self, angle=[0, 0, 0], **kwargs):
        self.angle = np.array(angle) * np.pi

    @staticmethod
    def M(axis, theta):
        return expm(np.cross(np.eye(3), axis / norm(axis) * theta))

    def __call__(self, pc):
        device = pc.device

        if isinstance(self.angle, collections.Iterable):
            rot_mats = []
            for axis_ind, rot_bound in enumerate(self.angle):
                theta = 0
                axis = np.zeros(3)
                axis[axis_ind] = 1
                if rot_bound is not None:
                    theta = np.random.uniform(-rot_bound, rot_bound)
                rot_mats.append(self.M(axis, theta))
            # Use random order
            np.random.shuffle(rot_mats)
            rot_mat = torch.tensor(rot_mats[0] @ rot_mats[1] @ rot_mats[2], dtype=torch.float32, device=device)
        else:
            raise ValueError()

        """ DEBUG
        from openpoints.dataset import vis_multi_points
        old_points = data.cpu().numpy()
        # old_points = data['pos'].numpy()
        # new_points = (data['pos'] @ rot_mat.T).numpy()
        new_points = (data @ rot_mat.T).cpu().numpy()
        vis_multi_points([old_points, new_points])
        End of DEBUG"""
        
        pc = pc @ rot_mat.T
        return pc
