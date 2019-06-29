import cv2
import numpy as np
import maxflow
import networkx
import matplotlib.pyplot as plt




def question_3(I,rho=0.6,pairwise_cost_same=0.01,pairwise_cost_diff=0.5):

    smoothness = 100
    ### 1) Define Graph
    g = maxflow.Graph[float]()
    I = 255 * (I > 128).astype(np.uint8)
    ### 2) Add pixels as nodes
    nodeids = g.add_grid_nodes(I.shape)
    print ( I.shape, nodeids.shape)
    #g.add_grid_edges(nodeids, weights = pairwise_cost_diff - pairwise_cost_same, structure=structure,symmetric=True)
    
    ### 3) Compute Unary cost
    uc0 = - np.log( np.power(rho,I) ) 
    uc1 = - np.log( np.power(1-rho,255-I) )  
   
    
    ### 4) Add terminal edges
    g.add_grid_tedges(nodeids, uc0, uc1)
    
    pairwise_cost_same = pairwise_cost_same * smoothness
    pairwise_cost_diff = pairwise_cost_diff * smoothness
    
     ### 5) Add Node edges
    ### Vertical Edges
    
    ### Horizontal edges
    # (Keep in mind the stucture of neighbourhood and set the weights according to the pairwise potential)
    

    for i in range(0,nodeids.shape[0]):
        for j in range(0,nodeids.shape[1]):
         
           #Horizontal edges
           if j > 0:
              if I[i][j] == I[i][j-1]:
                 g.add_edge(nodeids[i][j], nodeids[i][j-1], pairwise_cost_same, pairwise_cost_same)
              else:
                 g.add_edge(nodeids[i][j], nodeids[i][j-1], pairwise_cost_diff, pairwise_cost_diff)
           
           if j < nodeids.shape[1] - 1:
              if I[i][j] == I[i][j+1]:
                 g.add_edge(nodeids[i][j], nodeids[i][j+1],pairwise_cost_same ,pairwise_cost_same )
              else:
                 g.add_edge(nodeids[i][j], nodeids[i][j+1],pairwise_cost_diff ,pairwise_cost_diff )
           
           #Vertical edges
           if i > 0:
              if I[i][j] == I[i-1][j]:
                 g.add_edge(nodeids[i][j], nodeids[i-1][j], pairwise_cost_same,pairwise_cost_same)
              else:
                 g.add_edge(nodeids[i][j], nodeids[i-1][j], pairwise_cost_diff,pairwise_cost_diff)
              
           if i < nodeids.shape[0] - 1:
              if I[i][j] == I[i+1][j]:
                 g.add_edge(nodeids[i][j], nodeids[i+1][j], pairwise_cost_same ,pairwise_cost_same)
              else:
                 g.add_edge(nodeids[i][j], nodeids[i+1][j], pairwise_cost_diff ,pairwise_cost_diff)
              
   

    ### 6) Maxflow
    g.maxflow()
    sgm = g.get_grid_segments(nodeids)
    #print(I)
    
    #graph = g.get_nx_graph()
    #networkx.draw(graph)
    #plt.show()
    # The labels should be 1 where sgm is False and 0 otherwise.
    Denoised_I = np.int_(np.logical_not(sgm)).astype(np.uint8) * 255
    cv2.imshow('Original Img', I), \
    cv2.imshow('Denoised Img', Denoised_I), cv2.waitKey(0), cv2.destroyAllWindows()
    return

def question_4(I,rho=0.6):

    labels = np.unique(I).tolist()

    Denoised_I = np.zeros_like(I)
    ### Use Alpha expansion binary image for each label

    ### 1) Define Graph

    ### 2) Add pixels as nodes

    ### 3) Compute Unary cost

    ### 4) Add terminal edges

    ### 5) Add Node edges
    ### Vertical Edges

    ### Horizontal edges
    # (Keep in mind the stucture of neighbourhood and set the weights according to the pairwise potential)

    ### 6) Maxflow


    cv2.imshow('Original Img', I), \
    cv2.imshow('Denoised Img', Denoised_I), cv2.waitKey(0), cv2.destroyAllWindows()

    return

def main():
    image_q3 = cv2.imread('./images/noise.png', cv2.IMREAD_GRAYSCALE)
    image_q4 = cv2.imread('./images/noise2.png', cv2.IMREAD_GRAYSCALE)
   
    ### Call solution for question 3
    question_3(image_q3, rho=0.6, pairwise_cost_same=0.01, pairwise_cost_diff=0.15)
    question_3(image_q3, rho=0.6, pairwise_cost_same=0.01, pairwise_cost_diff=0.3)
    question_3(image_q3, rho=0.6, pairwise_cost_same=0.01, pairwise_cost_diff=0.6)

    ### Call solution for question 4
    #question_4(image_q4, rho=0.8)
    return

if __name__ == "__main__":
    main()



