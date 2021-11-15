import torch
import torch.nn as nn
from ..mesh import *


def quadpts(order=2):
    '''
    ported from iFEM's quadpts
    '''

    if order == 1:     # Order 1, nQuad 1
        baryCoords = [1/3, 1/3, 1/3]
        weight = 1
    elif order == 2:    # Order 2, nQuad 3
        baryCoords = [[2/3, 1/6, 1/6],
                      [1/6, 2/3, 1/6],
                      [1/6, 1/6, 2/3]]
        weight = [1/3, 1/3, 1/3]
    elif order == 3:     # Order 3, nQuad 4
        baryCoords = [[1/3, 1/3, 1/3],
                      [0.6, 0.2, 0.2],
                      [0.2, 0.6, 0.2],
                      [0.2, 0.2, 0.6]]
        weight = [-27/48, 25/48, 25/48, 25/48]
    elif order == 4:     # Order 4, nQuad 6
        baryCoords = [[0.108103018168070, 0.445948490915965, 0.445948490915965],
                      [0.445948490915965, 0.108103018168070, 0.445948490915965],
                      [0.445948490915965, 0.445948490915965, 0.108103018168070],
                      [0.816847572980459, 0.091576213509771, 0.091576213509771],
                      [0.091576213509771, 0.816847572980459, 0.091576213509771],
                      [0.091576213509771, 0.091576213509771, 0.816847572980459], ]
        weight = [0.223381589678011, 0.223381589678011, 0.223381589678011,
                  0.109951743655322, 0.109951743655322, 0.109951743655322]
    return torch.Tensor(baryCoords), torch.Tensor(weight)

class DataSinCos:
    '''
    Trigonometric data for Poisson equation

        f = 2*pi^2*np.cos(pi*x)*np.cos(pi*y);
        u = cos(pi*x)*np.cos(pi*y);
        Du = (-pi*np.sin(pi*x)*np.cos(pi*y), -pi*np.cos(pi*x)*np.sin(pi*y));

    The u satisfies the zero flux condition du/dn = 0 on boundary of [0,1]^2
    and thus g_N is not assigned.
    
    Ported from Long Chen's iFEM package to Python
    '''
    def __init__(self):
        self.pi = torch.tensor(np.pi, dtype=float)

    def f(self, p):
        x = p[:,0]; y = p[:,1]
        return 2*self.pi**2*np.cos(self.pi*x)*np.cos(self.pi*y)

    def exactu(self, p):
        x = p[:,0]; y = p[:,1]
        return np.cos(self.pi*x)*np.cos(self.pi*y)
    
    def g_D(self,p):
        return self.exactu(p)
    
    def Du(self,p):
        x = p[:,0]; y = p[:,1]
        Dux = -self.pi*torch.sin(self.pi*x)*torch.cos(self.pi*y)
        Duy = -self.pi*torch.cos(self.pi*x)*torch.sin(self.pi*y)
        return torch.stack([Dux, Duy], dim=-1)

    def d(self,p):
        return torch.ones(p.shape[0], dtype=float)


class Poisson(nn.Module):
    '''
    A lightweight port of the Poisson
    from Long Chen's iFEM library

    Linear Lagrange element on triangulations
    '''
    def __init__(self) -> None:
        super().__init__()
        self.quadpts = quadpts()

    def forward(self, pde, mesh: TriMesh2D) -> torch.Tensor:
        node = mesh.node
        elem = mesh.elem
        isBdNode = mesh.isBdNode
        Dphi = mesh.Dlambda
        area = mesh.area

        N = len(node)
        NT = len(elem)

        phi, weight = self.quadpts
        nQuad = len(phi)

        # diffusion coeff
        K = torch.zeros(NT)
        for p in range(nQuad):
            # quadrature points in the x-y coordinate
            pxy = phi[p,0]*node[elem[:,0]] + phi[p,1]*node[elem[:,1]] + phi[p,2]*node[elem[:,2]]
            K += weight[p]*pde.d(pxy)

        # stiffness matrix
        A = torch.sparse_coo_tensor(size=(N, N))
        for i in range(3):
            for j in range(3):
                # $A_{ij}|_{\tau} = \int_{\tau}K\nabla \phi_i\cdot \nabla \phi_j dxdy$ 
                Aij = area*K*(Dphi[...,i]*Dphi[...,j]).sum(axis=-1)
                A += torch.sparse_coo_tensor([elem[:,i],elem[:,j]], Aij, (N,N))     

        # right hand side
        b = torch.zeros(N)
        bt = torch.zeros((NT,3))

        for p in range(nQuad):
            # quadrature points in the x-y coordinate
            pxy = phi[p,0]*node[elem[:,0]] + phi[p,1]*node[elem[:,1]] + phi[p,2]*node[elem[:,2]]
            fp = pde.f(pxy)
            for i in range(3):
                bt[:,i] += weight[p]*phi[p,i]*fp

        bt *= area.reshape(-1,1)
        b = torch.bincount(elem.view(-1), weights=bt.view(-1))

        # Dirichlet
        u = torch.zeros(N)
        u[isBdNode] = pde.g_D(node[isBdNode])
        b -= A.dot(u) 

        # Direct solve
        freeNode = ~isBdNode
        u[freeNode] = torch.solve(A[freeNode,:][:,freeNode], b[freeNode])

        # compute Du
        dudx =  (u[elem]*Dphi[:,0,:]).sum(axis=-1)
        dudy =  (u[elem]*Dphi[:,1,:]).sum(axis=-1)      
        Du = torch.stack([dudx, dudy], dim=-1)

        soln = {'u': u,
                'Du': Du}
        eqn = {'A': A,
            'b': b,
            'freeNode': freeNode}

        return soln, eqn
   
    