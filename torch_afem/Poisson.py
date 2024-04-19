import torch
import torch.nn as nn
import torch.nn.functional as F
from .mesh2d import *

class Poisson(nn.Module):
    """
    A lightweight Poisson equation solver

    - Linear Lagrange element on triangulations

    Reference: Long Chen's iFEM library
    https://github.com/lyc102/ifem/blob/master/equation/Poisson.m
    """

    def __init__(
        self,
        domain=((0, 1), (0, 1)),
        h=1 / 4,
        quadrature_order=1,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        # self.quadpts = quadpts
        self.domain = domain
        self.h = h
        self.mesh_size = int(1/h)+1
        self.quadrature = quadpts(order=quadrature_order, dtype=dtype)
        self.dtype = dtype
        self._initialize()

    def _initialize(self) -> None:
        # TODO update default TriMesh2D options (done)

        node, elem = rectangleMesh(
            x_range=self.domain[0], y_range=self.domain[1], h=self.h
        )
        self.trimesh = TriMesh2D(node, elem, dtype=self.dtype)
        # TODO set u to be parameters (done)
        self.freeNode = freeNode = self.trimesh.freeNode
        self.nDof = nDof = int(freeNode.sum())
        self.nNode = nNode = node.size(0)
        self.nElem = elem.size(0)

        self.register_parameter(
            "u", nn.Parameter(torch.zeros((nDof, 1), dtype=self.dtype))
        )
        nn.init.zeros_(self.u)
        self.register_buffer("uh", torch.zeros((nNode, 1), dtype=self.dtype))

    def _assemble(self, pde):
        node = self.trimesh.node
        elem = self.trimesh.elem
        gradPhi = self.trimesh.gradLambda
        area = self.trimesh.area
        nElem = self.nElem
        nNode = self.nNode

        phi, weight = self.quadrature

        # quadrature points
        quadPts = torch.einsum("qp, npd->qnd", phi, node[elem])

        # diffusion coefficient
        Kp = torch.stack([pde.diffusion_coeff(p) for p in quadPts], dim=0)
        K = torch.einsum("q, qn->n", weight, Kp)

        intgradPhiAgradPhi = torch.einsum(
            "n,n,ndi,ndj->nij", K, area, gradPhi, gradPhi)

        I = elem[:, :, None].expand_as(intgradPhiAgradPhi)
        J = elem[:, None, :].expand_as(intgradPhiAgradPhi)
        IJ = torch.stack([I, J])
        A = torch.sparse_coo_tensor(
            IJ.view(2, -1),
            intgradPhiAgradPhi.contiguous().view(-1),
            size=(nNode, nNode),
        )

        # right hand side
        b = torch.zeros((nNode, 1), dtype=self.dtype)

        if callable(pde.source):
            fK = torch.stack([pde.source(p) for p in quadPts], dim=0)
            bt = torch.einsum("q, qn, qp, n->np", weight, fK, phi, area)
        elif torch.is_tensor(pde.source):
            
            if pde.source.size(-1) == 1:  # (bsz, nNode, 1)
                pass
            else:  # (bsz, n, n)
                f =  F.interpolate(pde.source,
                                  size=(self.mesh_size+1, self.mesh_size+1),
                                    mode='bilinear',
                                    align_corners=True)
                fK = f.view(-1)[elem].mean(-1)
                bt = torch.einsum("q, n, qp, n->np", weight, fK, phi, area)

        b.scatter_(0, index=elem.view(-1, 1), src=bt.view(-1, 1), reduce="add")

        isBdNode = self.trimesh.isBdNode
        freeNode = self.trimesh.freeNode

        self.uh.scatter_(
            0,
            index=torch.where(isBdNode)[0].unsqueeze(-1),
            src=pde.g_D(node[isBdNode]).unsqueeze(-1),
        )
        b -= torch.sparse.mm(A, self.uh)

        self.A = A
        self.b = b

        A = A.to_dense()
        maskFreeNode = torch.outer(freeNode, freeNode)
        nDof = self.nDof
        A_int = A[maskFreeNode].view(nDof, nDof).to_sparse()
        A_int = A_int.coalesce()
        b_int = b[freeNode]
        self.b_int = b_int
        self.A_int = A_int

    def forward(self, u):
        return torch.sparse.mm(self.A_int, u)

    def solve(self, f=None) -> None:
        """
        direct solver, not working in sparse only
        """
        freeNode = self.trimesh.freeNode

        if self.A.is_sparse:
            A = self.A.to_dense()
        else:
            A = self.A.copy()
        b = self.b if f is None else f

        self.u.detach_()
        self.u = nn.Parameter(
            torch.linalg.solve(A[freeNode, :][:, freeNode], b[freeNode])
        )

    def get_u(self):
        """
        assemble u and u_g back into 1
        """
        self.uh[self.trimesh.freeNode] = self.u.detach()
        return self.uh.squeeze()

    def energy(self, u, b):
        """
        0.5*u^T A u - f*u
        """
        Au = self.forward(u)
        return 0.5 * (u.T).mm(Au) - (b.T).mm(u)
