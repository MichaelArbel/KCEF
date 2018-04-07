import numpy as np
import matplotlib.pyplot as plt
from itertools import product


def visualise_array(Xs, Ys, A, samples=None):
    im = plt.imshow(A, origin='lower')
    im.set_extent([Xs.min(), Xs.max(), Ys.min(), Ys.max()])
    im.set_interpolation('nearest')
    im.set_cmap('gray')
    if samples is not None:
        plt.plot(samples[:, 0], samples[:, 1], 'bx')
    plt.ylim([Ys.min(), Ys.max()])
    plt.xlim([Xs.min(), Xs.max()])


def pdf_grid(Xs, Ys, est):
    D = np.zeros((len(Xs), len(Ys)))
    G = np.zeros(D.shape)
    
    for i, x in enumerate(Xs):
        for j, y in enumerate(Ys):
            point = np.array([x, y])
            D[j, i] = est.log_pdf(point)
            G[j, i] = np.linalg.norm(est.grad(point))
    
    return D, G


def get_project_path(project_name):
    import os
    import re

    cwd = os.getcwd()
    path = re.split( project_name, cwd)
    return os.path.join(path[0], project_name)


def visualise_fit_2d_cond(est, X=None, nodes=None, normalize = False,center=None , width=None, gen=None, N=None, num=None ):
    if num==None:
        num = 50
    if N is None:
        N=100
    if center is None:
        center = [0,0]
    if width is None:
        width = [20, 20]
    Y =None
    if gen is not None:
        if N is None:
            N = 100
        Y = gen.generate_cond(X , N)
    if nodes is None:
        nodes = [1]
    else:
        u = 0
        for node in nodes:
            u +=   len(est.graph[node][0])
        if len(nodes)>2 | (u!=2 & nodes[0]>0):
            return 0

    x = np.linspace(center[0] - width[0]/2, center[0] + width[0]/2, num=num)
    y = np.linspace(center[1] - width[1]/2, center[1] + width[1]/2, num=num)
    xx, yy = np.meshgrid(x, y)
    y_map = np.array([np.reshape(xx, [-1,1]), np.reshape(yy, [-1,1])])
    y_map = y_map[:,:,0].T

    if len(nodes)==1:
        if nodes[0]==0:
            y_map = np.concatenate([np.reshape(np.repeat(X,  y_map.shape[0]), [-1, 1]  ), y_map], axis=1)
        else:
            est.set_cond(X,nodes[0])
        D   = est.log_pdf(y_map)
        G   = est.grad(y_map,as_array = False)
        G   = np.linalg.norm(G, axis=1)
        D = np.reshape(D, [num, num])
        G = np.reshape(G, [num, num])
    else:
        est.set_cond(X,nodes[0])
        xx = np.reshape(xx, [-1, 1])
        yy = np.reshape(yy, [-1, 1])
        rep_x = np.reshape(np.repeat(X,  xx.shape[0]), [-1, 1]  )
        D_x     = est.log_pdf(xx)
        X_cond  = np.array( [rep_x, xx])
        X_cond  = X_cond[:,:,0].T
        D_y     =  est.log_pdf(yy, x = X_cond, node = nodes[1])
        if normalize:

            x_cond = np.array([np.reshape(np.repeat(X,  x.shape[0]), [-1, 1]  ), np.reshape(x, [-1, 1])])
            x_cond = x_cond[:,:,0].T
            log_Z = est.log_partition(yy, x_cond, node = nodes[1])
            D_y     -= np.tile(log_Z, num) 
        D = np.reshape(D_x+D_y, [num, num])
    plt.subplot(1, 2, 1)
    visualise_array(x, y, D, Y)
    plt.title("log pdf")

    if len(nodes)==1:
        plt.subplot(1 ,2,2)
        visualise_array(x, y, G, Y)
        plt.title("gradient norm")

    plt.tight_layout()
    plt.show()


def intercat_vis( est, gen, x_1, x_2, N=None, Y_cond =None, width =None, node=None, ):
    X_cond = np.array([x_1,x_2])
    Y = gen.generate_cond(X_cond , N)
    Y_cond = [0,0]
    width = [20, 20]
    visualise_fit_2d_cond(est, node, Y_cond, width, X =np.reshape(X_cond,[1,-1]), samples=Y )
    plt.show()




def visualise_array_2d(Xs, Ys, A, samples=None, ax=None):
    # visualise found fit
    if ax is None:
        fig=plt.figure()
        ax = fig.add_subplot(111)

    vmin=np.nanmin(A)
    vmax=np.nanmax(A)
    heatmap = ax.pcolor(Xs, Ys, A.T, cmap='viridis', vmin=vmin, vmax=vmax)
    heatmap.cmap.set_under('white')
    
    colorbar=plt.colorbar(heatmap, ax=ax)
    colorbar.set_clim(vmin=vmin, vmax=vmax)
    
    if samples is not None:
        ax.scatter(samples[:, 0], samples[:, 1], c='r', s=1);

def pdf_grid(Xs, Ys, est, ref_vec=None, x_ind=0, y_ind=1, kind="grad_norm"):
    n_x = len(Xs)
    n_y = len(Ys)
    
    if ref_vec is None or len(ref_vec)==2:
        X_test = np.array(list(product(Xs, Ys)))
    else:
        assert len(ref_vec)>=2
        X_test = np.tile(ref_vec, (n_x * n_y,1))
        X_test[:,np.array([x_ind, y_ind])]=np.array(list(product(Xs, Ys)))

    est.set_data(X_test.T)
    
    if kind=="grad_norm":
        gradients = est.grad_multiple()
        if not ref_vec is None:
            gradients = gradients[np.array([x_ind, y_ind]),:]

        gradient_norms = np.sum(gradients**2, axis=0)
        return gradient_norms.reshape(n_x, n_y)
    elif kind=="log_pdf":
        log_pdfs = est.log_pdf_multiple()
        return log_pdfs.reshape(n_x, n_y)
    else:
        raise ValueError("Wrong kind: %s" % kind)


def visualise_fit_2d(est, X=None, Xs=None, Ys=None, res=50, ref_vec=None,
                     x_ind=0, y_ind=1, ax=None, kind="grad_norm"):
    
    # visualise found fit
    if ax is None:
        fig=plt.figure()
        ax = fig.add_subplot(111)
    if Xs is None:
        x_min = -5
        x_max = 5
        if not X is None:
            x_min=np.min(X[:,x_ind])
            x_max=np.max(X[:,x_ind])
            delta = x_max - x_min
            x_min-=delta/10.
            x_max+=delta/10.


    if Ys is None:
        y_min = -5
        y_max = 5
        if not X is None:
            y_min=np.min(X[:,y_ind])
            y_max=np.max(X[:,y_ind])
            delta = y_max - y_min
            y_min-=delta/10.
            y_max+=delta/10.
            
    xy_max = np.max([x_max, y_max])
    xy_min = np.min([x_min, y_min])
    Xs = np.linspace(xy_min, xy_max, res)
    Ys = np.linspace(xy_min, xy_max, res)

    G = pdf_grid(Xs, Ys, est, ref_vec, x_ind, y_ind, kind)
     
    """
    plt.subplot(121)
    visualise_array_2d(Xs, Ys, D, X[:,np.array([x_ind,y_ind])])
    plt.axes().set_aspect('equal')
    plt.title("log pdf")
    #plt.colorbar()
    
    plt.subplot(122)
    """
    
    visualise_array_2d(Xs, Ys, G, X[:,np.array([x_ind,y_ind])], ax=ax)
    #plt.colorbar()

def heatmap(df, x_label, y_label, log_scale = False):
    #fig=plt.figure()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    fig, ax = plt.subplots()
    vmin = np.nanmin(df)
    vmax = np.nanmax(df)
    heatmap = ax.pcolor(df, cmap='viridis',vmin=vmin, vmax=vmax)
    heatmap.cmap.set_under('white')
    num_ticks = 5
    ax.set_ylabel(r'\textit{log} '+y_label)
    ax.set_xlabel(r'\textit{log} '+x_label)
    plt.title(r'\textit{negative log-likelihood}')
    step_y = len(df.index)/4
    step_x = len(df.columns)/4
    if log_scale:
        plt.yticks(np.arange(0.5, len(df.index),step_y), [int(round(np.log(df.index[step_y*w])/np.log(10))) for w in range(len(df.index)/step_y) ])
        plt.xticks(np.arange(0.5, len(df.columns),step_x), [int(round(np.log(df.columns[step_x*w])/np.log(10))) for w in range(len(df.columns)/step_x) ])
    else:
        plt.yticks(np.arange(0.5, len(df.index),step_y), [int(round(df.index[step_y*w])) for w in range(len(df.index)/step_y) ])
        plt.xticks(np.arange(0.5, len(df.columns),step_x), [int(round(df.columns[step_x*w])) for w in range(len(df.columns)/step_x) ])
       
    colorbar=plt.colorbar(heatmap, ax=ax)
    colorbar.set_clim(vmin=vmin, vmax=vmax)
    return fig, ax
 