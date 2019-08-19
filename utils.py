# https://github.com/eamid/examples/blob/master/utils.py

import matplotlib.pyplot as plt

def plot_results(maps, labels, index=[], extension=''):
    fig = plt.figure()
    fig.set_figheight(5)
    fig.set_figwidth(15)
    ax1 = fig.add_subplot(2, 3, 1); ax1.axis('off')
    ax2 = fig.add_subplot(2, 3, 2); ax2.axis('off')
    ax3 = fig.add_subplot(2, 3, 3); ax3.axis('off')
    ax4 = fig.add_subplot(2, 3, 4); ax4.axis('off')
    ax5 = fig.add_subplot(2, 3, 5); ax5.axis('off')
    ax1.scatter(maps[0][:,0], maps[0][:,1], s=0.1, c=labels)
    ax2.scatter(maps[1][:,0], maps[1][:,1], s=0.1, c=labels)
    ax3.scatter(maps[2][:,0], maps[2][:,1], s=0.1, c=labels)
    ax4.scatter(maps[3][:,0], maps[3][:,1], s=0.1, c=labels)
    ax5.scatter(maps[4][:,0], maps[4][:,1], s=0.1, c=labels)
    if index:
        ax1.scatter(maps[0][index,0], maps[0][index,1], s=80, c='red', marker='x')
        ax2.scatter(maps[1][index,0], maps[1][index,1], s=80, c='red', marker='x')
        ax3.scatter(maps[2][index,0], maps[2][index,1], s=80, c='red', marker='x')
        ax4.scatter(maps[3][index,0], maps[3][index,1], s=80, c='red', marker='x')
        ax5.scatter(maps[4][index,0], maps[4][index,1], s=80, c='red', marker='x')
    ax1.title.set_text('t-SNE ' + extension)
    ax2.title.set_text('UMAP ' + extension)
    ax3.title.set_text('TriMap ' + extension)
    ax4.title.set_text('hub_Trimap ' + extension)
    ax5.title.set_text('PCA ' + extension)
    plt.show()
