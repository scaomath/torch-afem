import numpy as np
import torch
try:
    import plotly.figure_factory as ff
    import plotly.io as pio
    import plotly.graph_objects as go
except ImportError as e:
    print('Please install Plotly for showing mesh and solutions.')
import matplotlib.pyplot as plt
import matplotlib.tri as tri

def showmesh(node,elem, **kwargs):
    triangulation = tri.Triangulation(node[:,0], node[:,1], elem)
    markersize = 3000/len(node)
    if kwargs.items():
        h = plt.triplot(triangulation, 'b-h', **kwargs)
    else:
        h = plt.triplot(triangulation, 'b-h', linewidth=0.5, alpha=0.5, markersize=markersize)
    return h

def showsolution(node,elem,u,**kwargs):
    '''
    show 2D solution either of a scalar function or a vector field
    '''
    markersize = 3000/len(node)
    
    if u.ndim == 1:
        uplot = ff.create_trisurf(x=node[:,0], y=node[:,1], z=u,
                            simplices=elem,
                            colormap="Viridis", # similar to matlab's default colormap
                            showbackground=False,
                            aspectratio=dict(x=1, y=1, z=1),
                            )
        fig = go.Figure(data=uplot)

    elif u.ndim == 2 and u.shape[-1] == 2:
        assert u.shape[0] == elem.shape[0]
        u /= (np.abs(u)).max()
        center = node[elem].mean(axis=1)
        uplot = ff.create_quiver(x=center[:,0], y=center[:,1], 
                            u=u[:,0], v=u[:,1],
                            scale=.05,
                            arrow_scale=.5,
                            name='gradient of u',
                            line_width=1,
                            )

    fig = go.Figure(data=uplot)
    
    fig.update_layout(template='plotly_dark',
                    margin=dict(l=5, r=5, t=5, b=5),
                    **kwargs)
    fig.show()

def unique(x,
           sorted=False,
           return_counts=False,
           dim=None):
    """
    modified from
    https://github.com/pytorch/pytorch/issues/36748#issuecomment-619514810

    Args:
        input (Tensor): the input tensor
        sorted (bool): Whether to sort the unique elements in ascending order
            before returning as output.
        return_indices (bool): If True, also return the indices of ar (along the specified axis, if provided, or in the flattened array) that result in the unique array (added in this script).
        return_inverse (bool): Whether to also return the indices for where
            elements in the original input ended up in the returned unique list.
        return_counts (bool): Whether to also return the counts for each unique
            element.
        dim (int): the dimension to apply unique. If ``None``, the unique of the
            flattened input is returned. default: ``None``

    Returns:
        (Tensor, Tensor (optional), Tensor (optional)): A tensor or a tuple of tensors containing

            - **output** (*Tensor*): the output list of unique scalar elements.
            - **indices**: (optional) if :attr:`return_indices` is True, the indices of the first occurrences of the unique values in the original array.
            - **inverse_indices** (*Tensor*): (optional) if
              :attr:`return_inverse` is True, there will be an additional
              returned tensor (same shape as input) representing the indices
              for where elements in the original input map to in the output;
              otherwise, this function will only return a single tensor.
            - **counts** (*Tensor*): (optional) if
              :attr:`return_counts` is True, there will be an additional
              returned tensor (same shape as output or output.size(dim),
              if dim was specified) representing the number of occurrences
              for each unique value or tensor.
    """
    if return_counts:
        out, inverse, counts = torch.unique(x, 
                                    sorted=sorted, 
                                    return_inverse=True, 
                                    return_counts=True,
                                    dim=dim)
    else:
        out, inverse = torch.unique(x, 
                                    sorted=sorted, 
                                    return_inverse=False, 
                                    dim=dim)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype,
                        device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    # scatter_(dim, index, src)
    # Writes all values from the tensor src into self at the indices specified in the index tensor.
    indices = inverse.new_empty(out.size(0)).scatter_(0, inverse, perm)
    if return_counts:
        return out, indices, inverse, counts
    
    else:
        return out, indices, inverse