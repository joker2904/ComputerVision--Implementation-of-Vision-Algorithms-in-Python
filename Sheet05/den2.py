import cv2
import numpy as np
import maxflow
import networkx
import matplotlib.pyplot as plt



smoothing = 100

# Load the image and convert it to grayscale image 
#image_path = 'your_image.png'
img = cv2.imread('./images/noise.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = 255 * (img > 128).astype(np.uint8)

# Create the graph.
g = maxflow.Graph[float]()
# Add the nodes. nodeids has the identifiers of the nodes in the grid.
nodeids = g.add_grid_nodes(img.shape)
# Add non-terminal edges with the same capacity.
#g.add_grid_edges(nodeids,150)
# Add the terminal edges. The image pixels are the capacities
# of the edges from the source node. The inverted image pixels
# are the capacities of the edges to the sink node.
#g.add_grid_tedges(nodeids, img -np.log(0.6), 255 - img +np.log(1-0.6) )
rho = 0.6
uc0 = - np.log( np.power(rho,img) ) 
uc1 = - np.log( np.power(1-rho,255-img) )  
pairwise_cost_same=0.01
pairwise_cost_diff=0.15
    
### 4) Add terminal edges
#g.add_grid_edges(nodeids, 255 - img + smoothing)
# Add the terminal edges. The image pixels are the capacities
# of the edges from the source node. The inverted image pixels
# are the capacities of the edges to the sink node.
g.add_grid_tedges(nodeids, uc0,  uc1)
#g.add_grid_edges(nodeids, 30)



for i in range(0,nodeids.shape[0]):
    for j in range(0,nodeids.shape[1]):
        #print (nodeids[i][j])
        if j > 0:
           if img[i][j] == img[i][j-1]:
              g.add_edge(nodeids[i][j], nodeids[i][j-1], img[i][j]*pairwise_cost_same, img[i][j]*pairwise_cost_same)
           else:
              g.add_edge(nodeids[i][j], nodeids[i][j-1], img[i][j]* pairwise_cost_diff, img[i][j]*pairwise_cost_diff)
           
        if j < nodeids.shape[1] - 1:
              if img[i][j] == img[i][j+1]:
                 g.add_edge(nodeids[i][j], nodeids[i][j+1],img[i][j]*pairwise_cost_same ,img[i][j]*pairwise_cost_same )
              else:
                 g.add_edge(nodeids[i][j], nodeids[i][j+1],img[i][j]*pairwise_cost_diff ,img[i][j]*pairwise_cost_diff )
              
        if i > 0:
              if img[i][j] == img[i-1][j]:
                 g.add_edge(nodeids[i][j], nodeids[i-1][j], img[i][j]*pairwise_cost_same,img[i][j]*pairwise_cost_same)
              else:
                 g.add_edge(nodeids[i][j], nodeids[i-1][j], img[i][j]*pairwise_cost_diff,img[i][j]*pairwise_cost_diff)
              
        if i < nodeids.shape[0] - 1:
              if img[i][j] == img[i+1][j]:
                 g.add_edge(nodeids[i][j], nodeids[i+1][j], img[i][j]*pairwise_cost_same ,img[i][j]*pairwise_cost_same)
              else:
                 g.add_edge(nodeids[i][j], nodeids[i+1][j], img[i][j]*pairwise_cost_diff ,img[i][j]*pairwise_cost_diff)
              

# Find the maximum flow.
g.maxflow()
# Get the segments of the nodes in the grid.
sgm = g.get_grid_segments(nodeids)

# The labels should be 1 where sgm is False and 0 otherwise.
img_denoised = np.logical_not(sgm).astype(np.uint8) * 255

# Show the result.
plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.title('Binary image')
plt.subplot(122)
plt.title('Denoised binary image')
plt.imshow(img_denoised, cmap='gray')
plt.show()

# Save denoised image
cv2.imwrite('img_denoised.png', img_denoised)
