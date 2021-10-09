import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    height, width, feat_stride = 128,128,1
        
    fig = plt.figure()
    ax  = fig.add_subplot(121)
    plt.ylim(-10,17)
    plt.xlim(-10,17)

    shift_x = np.arange(0, width * feat_stride, feat_stride)
    shift_y = np.arange(0, height * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    boxes   = np.stack([shift_x,shift_y,shift_x,shift_y],axis=-1).reshape([-1,4]).astype(np.float32)
    plt.scatter(boxes[3:,0],boxes[3:,1])
    plt.scatter(boxes[0:3,0],boxes[0:3,1],c="r")
    ax.invert_yaxis()

    ax = fig.add_subplot(122)
    plt.ylim(-10,17)
    plt.xlim(-10,17)

    shift_x = np.arange(0, width * feat_stride, feat_stride)
    shift_y = np.arange(0, height * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    boxes = np.stack([shift_x,shift_y,shift_x,shift_y],axis=-1).reshape([-1,4]).astype(np.float32)
    plt.scatter(shift_x,shift_y)

    heatmap = np.random.uniform(0,1,[128,128,80]).reshape([-1,80])
    reg     = np.random.uniform(0,1,[128,128,2]).reshape([-1,2])
    wh      = np.random.uniform(5,20,[128,128,2]).reshape([-1,2])

    boxes[:,:2] = boxes[:,:2] + reg
    boxes[:,2:] = boxes[:,2:] + reg
    plt.scatter(boxes[0:3,0],boxes[0:3,1])
    boxes[:,:2] = boxes[:,:2] - wh/2
    boxes[:,2:] = boxes[:,2:] + wh/2
    for i in [0,1,2]:
        rect = plt.Rectangle([boxes[i, 0],boxes[i, 1]], wh[i,0], wh[i,1], color="r",fill=False)
        ax.add_patch(rect)
    
    ax.invert_yaxis()

    plt.show()
