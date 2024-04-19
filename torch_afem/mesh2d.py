import torch
import torch.nn as nn
from .utils import *

def rectangleMesh(x_range=(0,1), y_range=(0,1), h=0.25):
    """ 
    Input: 
    - x's range, (x_min, x_max)
    - y's range, (y_min, y_max)
    - h, mesh size, can be a tuple
    Return the element matrix (NT, 3)
    of the mesh a torch.meshgrid 
    """
    try:
        hx, hy = h[0], h[1]
    except:
        hx, hy = h, h

    # need to add h because arange is not inclusive
    xp = torch.arange(x_range[0], x_range[1]+hx, hx)
    yp = torch.arange(y_range[0], y_range[1]+hy, hy)
    nx, ny = len(xp), len(yp)

    x, y = torch.meshgrid(xp, yp)
    
    elem = []
    for j in range(ny-1):
        for i in range(nx-1):      
            a = i + j*nx
            b = (i+1) + j*nx
            d = i + (j+1)*nx
            c = (i+1) + (j+1)*nx
            elem += [[a, c, d], [b, c, a]]

    node = torch.stack([x.flatten(), y.flatten()], dim=-1)
    elem = torch.tensor(elem, dtype=torch.long)
    return node, elem

def quadpts(order=2, dtype=torch.float64):
    '''
    ported from iFEM's quadpts
    '''

    if order == 1:     # Order 1, nQuad 1
        baryCoords = [[1/3, 1/3, 1/3]]
        weight = [1]
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
    return torch.tensor(baryCoords, dtype=dtype), torch.tensor(weight, dtype=dtype)

class TriMesh2D(nn.Module):
    '''
    Set up auxiliary data structures for Dirichlet boundary condition


    Combined the following routine from Long Chen's iFEM 
        - setboundary: get a boundary bool matrix according to elem
        - delmesh: delete mesh by eval()
        - auxstructure: edge-based auxiliary data structure
        - gradbasis: compute the gradient of local barycentric coords

    Input:
        - node: (N, 2)
        - elem: (NT, 3)

    Outputs:
        - edge: (NE, 2) global indexing of edges
        - elem2edge: (NT, 3) local to global indexing
        - edge2edge: (NE, 4)
          edge2elem[e,:2] are the global indexes of two elements sharing the e-th edge
          edge2elem[e,-2:] are the local indices of e to edge2elem[e,:2]
        - neighbor: (NT, 3) the local to global indices map of neighbor of elements
          neighbor[t,i] is the global index of the element opposite to the i-th vertex of the t-th element. 

    Example: the following routine gets all ifem similar data structures
        node, elem = rectangleMesh(x_range=(0,1), y_range=(0,1), h=1/16)
        T = TriMesh2D(node,elem)
        T.delete_mesh('(x>0) & (y<0)')
        T.update_auxstructure()
        T.update_gradbasis()
        node, elem = T.node, T.elem
        Dphi = T.Dlambda
        area = T.area
        elem2edgeSign = T.elem2edgeSign
        edge2elem = T.edge2elem

    Notes: 
        1. Python assigns the first appeared entry's index in unique; Matlab assigns the last appeared entry's index in unique.
        2. Matlab uses columns as natural indexing, reshape(NT, 3) in Matlab should be changed to
        reshape(3, -1).T in Python if initially the data is concatenated along axis=0 using torch.r_[].

    TODO:
        - Add Neumann boundary.
        - Change torch.bincount to torch.scatter

    '''

    def __init__(self,
                 node=None,
                 elem=None,
                 bdFlag=None,
                 dtype: torch.dtype = torch.float64,
                 ) -> None:
        super().__init__()

        self.dtype = dtype
        self.node = node.to(dtype)
        self.elem = elem
        self.bdFlag = bdFlag
        self._init_auxstruct()
        self._init_grad()

    def _init_auxstruct(self):
        elem = self.elem
        numElem = self.elem.size(0)
        numNode = self.node.size(0)

        # every edge's sign
        allEdge = torch.cat(
            [elem[:, [1, 2]], elem[:, [2, 0]], elem[:, [0, 1]]], dim=0)
        elem2edgeSign = torch.ones(3*numElem, dtype=int)
        elem2edgeSign[allEdge[:, 0] > allEdge[:, 1]] = -1
        self.elem2edgeSign = elem2edgeSign.view(3, -1).T
        allEdge, _ = torch.sort(allEdge, axis=1)
        # TODO indices in sort obj is dummy, can be used

        # edge structures
        self.edge, E2e, e2E, counts = unique(allEdge,
                                             return_counts=True,
                                             dim=0)
        self.elem2edge = e2E.view(3, -1).T

        # neighbor structures
        E2e_reverse = torch.zeros_like(E2e)
        E2e_reverse[e2E] = torch.arange(3*numElem)

        k1 = torch.div(E2e, numElem, rounding_mode='floor')
        k2 = torch.div(E2e_reverse, numElem, rounding_mode='floor')
        t1 = E2e - numElem*k1
        t2 = E2e_reverse - numElem*k2
        ix = self.isIntEdge = (counts == 2)  # interior edge indicator
        # edge to elem
        self.edge2elem = torch.stack([t1, t2, k1, k2], dim=-1)

        self.neighbor = torch.zeros((numElem, 3), dtype=int)
        ixElemLocalEdge1 = torch.stack([t1[ix], k1[ix]], dim=-1)
        ixElemLocalEdge2 = torch.stack([t2, k2], dim=-1)
        ixElemLocalEdge = torch.cat(
            [ixElemLocalEdge1, ixElemLocalEdge2], dim=0)
        ixElem = torch.cat([t2[ix], t1], dim=0)
        for i in range(3):
            ix = (ixElemLocalEdge[:, 1] == i)  # i-th edge's neighbor
            # TODO: check if bincount is necessary here
            self.neighbor[:, i] = torch.bincount(ixElemLocalEdge[ix, 0],
                                                 weights=ixElem[ix],
                                                 minlength=numElem)

        isBdEdge = (counts == 1)  # boundary edge indicator
        if self.bdFlag is None:
            self.bdFlag = isBdEdge[e2E].view(3, -1).T
        Dirichlet = self.edge[isBdEdge]
        self.isBdNode = torch.zeros(numNode, dtype=bool)
        self.isBdNode[Dirichlet.ravel()] = True
        self.freeNode = ~self.isBdNode

    def _init_grad(self):
        node, elem = self.node, self.elem

        ve1 = node[elem[:, 2]]-node[elem[:, 1]]
        ve2 = node[elem[:, 0]]-node[elem[:, 2]]
        ve3 = node[elem[:, 1]]-node[elem[:, 0]]
        area = torch.abs(0.5*(-ve3[:, 0]*ve2[:, 1] + ve3[:, 1]*ve2[:, 0]))
        gradLambda = torch.zeros((len(elem), 2, 3), dtype=self.dtype)
        # (# elem, 2-dim vector, 3 vertices)

        gradLambda[..., 2] = torch.stack(
            [-ve3[:, 1]/(2*area), ve3[:, 0]/(2*area)], dim=-1)
        gradLambda[..., 0] = torch.stack(
            [-ve1[:, 1]/(2*area), ve1[:, 0]/(2*area)], dim=-1)
        gradLambda[..., 1] = torch.stack(
            [-ve2[:, 1]/(2*area), ve2[:, 0]/(2*area)], dim=-1)
        # torch.stack with dim=-1 is equivalent to np.c_[]

        self.area = area
        self.gradLambda = gradLambda

    def get_elem2edge(self):
        return self.elem2edge

    def get_bdFlag(self):
        return self.bdFlag

    def get_edge(self):
        return self.edge

    def forward(self, x):
        return None

    def delete_mesh(self, expr=None):
        '''
        Update the mesh by deleting the eval(expr)
        '''
        assert expr is not None
        node, elem = self.node, self.elem
        center = node[elem].mean(axis=1)
        x, y = center[:, 0], center[:, 1]

        # delete element
        idx = eval(expr)
        mask = torch.ones(len(elem), dtype=bool)
        mask[idx] = False
        elem = elem[mask]

        # re-mapping the indices of vertices
        # to remove the unused ones
        isValidNode = torch.zeros(len(node), dtype=bool)
        indexMap = torch.zeros(len(node), dtype=int)

        isValidNode[elem.ravel()] = True
        self.node = node[isValidNode]

        indexMap[isValidNode] = torch.arange(len(self.node))
        self.elem = indexMap[elem]