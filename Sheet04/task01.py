import matplotlib.pyplot as plt
import numpy.linalg as la
import numpy as np
import cv2
import sys


def plot_snake(ax, V, fill='green', line='red', alpha=1, with_txt=False):
    """ plots the snake onto a sub-plot
    :param ax: subplot (fig.add_subplot(abc))
    :param V: point locations ( [ (x0, y0), (x1, y1), ... (xn, yn)]
    :param fill: point color
    :param line: line color
    :param alpha: [0 .. 1]
    :param with_txt: if True plot numbers as well
    :return:
    """
    V_plt = np.append(V.reshape(-1), V[0,:]).reshape((-1, 2))
    ax.plot(V_plt[:,0], V_plt[:,1], color=line, alpha=alpha)
    ax.scatter(V[:,0], V[:,1], color=fill,
               edgecolors='black',
               linewidth=2, s=50, alpha=alpha)
    if with_txt:
        for i, (x, y) in enumerate(V):
            ax.text(x, y, str(i))


def load_data(fpath, radius):
    """
    :param fpath:
    :param radius:
    :return:
    """
    Im = cv2.imread(fpath, 0)
    
    image = cv2.imread(fpath)
    #cv2.imshow('abcd',Im)
    h, w = Im.shape
    n = 20  # number of points
    u = lambda i: radius * np.cos(i) + w / 2
    v = lambda i: radius * np.sin(i) + h / 2
    V = np.array(
        [(u(i), v(i)) for i in np.linspace(0, 2 * np.pi, n + 1)][0:-1],
        'int64')

    return Im, V, image


# ===========================================
# RUNNING
# ===========================================

# FUNCTIONS
# ------------------------
# your implementation here

# ------------------------

#function to find the length of the snake (euclidean distances)
def get_length( snake ):
    n_points = len(snake)
    sum_of_npoints = 0

    for i in range(0,n_points):
        sum_of_npoints += np.sqrt( np.power(snake[(i+1)%n_points][0] - snake[i][0], 2) + np.power(snake[(i+1)%n_points][1] - snake[i][1], 2) )
        sum_of_npoints /= float(n_points)
    return sum_of_npoints


#contains the relative coordinates of the allowable positions of the neighbors. This is typically a 3x3 grid centered at the current snake vertex position. In this case, the kernel size is 9.
def get_neighborPosition( neighbour): #, int& neighborPos_Y, int& neighborPos_X ):
    #1 2 3
    #8 0 4
    #7 6 5
    if neighbour==0:         
       neighborPos_Y = 0
       neighborPos_X = 0
    if neighbour==1:         
        neighborPos_Y = -1
        neighborPos_X = -1
    if neighbour==2:         
        neighborPos_Y = -1
        neighborPos_X = 0
    if neighbour==3:         
        neighborPos_Y = -1
        neighborPos_X = +1
    if neighbour==4:         
        neighborPos_Y = 0
        neighborPos_X = +1
    if neighbour==5:         
        neighborPos_Y = +1
        neighborPos_X = +1
    if neighbour==6:         
        neighborPos_Y = +1
        neighborPos_X =  0
    if neighbour==7:         
        neighborPos_Y = +1
        neighborPos_X = -1
    if neighbour==8:         
        neighborPos_Y = 0
        neighborPos_X = -1
 
    return neighborPos_Y,neighborPos_X


        
def compute_curvatureEnergy(  currPoint,prevPoint, nextPoint,beta):
    #The Curvature energy
    #param currPoint: The point being analysed
    #param prevPoint: The previous point in the curve
    #param nextPoint: The next point in the curve
    #return: The curvature energy for the given points
    return   beta * ( np.power( nextPoint[0] - 2 * currPoint[0] + prevPoint[0], 2)+ np.power( nextPoint[1] - 2 * currPoint[1] + prevPoint[1], 2)) 


def compute_elasticityEnergy( currPoint,prevPoint, alpha, dist):
    #The Curvature energy
    #param currPoint: The point being analysed
    #param prevPoint: The previous point in the curve
    #param nextPoint: The next point in the curve
    #return: The curvature energy for the given points
    return   alpha * np.power(np.sqrt( np.power(prevPoint[0] - currPoint[0], 2) + np.power(prevPoint[1] - currPoint[1], 2))-dist,2)  


def compute_externalEnergy( p_curr ,gradientX,gradientY):
    #The edge energy (The tendency to move the curve towards edges). Using the sobel gradient.
    #param p_curr: The point being analysed
    #return: The edge energy for the given point
    return -( np.power( gradientX[int(p_curr[1])][int(p_curr[0])], 2 ) + np.power( gradientY[int(p_curr[1])][int(p_curr[0])], 2 ) )


# finds the minimum element in the specified column of a matrix
# and returns the row-ID of this minimum element

def find_columnmin( mat, colID ):
    curr_Val = 999999
    curr_IDD = 999999

    for i in range(0,mat.shape[0]):
        if curr_Val > mat[i][colID]:
           curr_Val = mat[i][colID]
           curr_IDD = i
    return curr_IDD




def run(fpath, radius,alpha,beta,lambdap):
    """ run experiment
    :param fpath:
    :param radius:
    :return:
    """
    Im, V ,image = load_data(fpath, radius)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    n_steps = 200
    snake = V
    NoOfVertices = len(snake)
    #for p in V:
    #    print(p)
    
    
    # ------------------------
    # your implementation here

    # ------------------------
   
    # Gradient Image variations used by the snake
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #gray_image2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #gray_image = cv2.GaussianBlur(gray_image,(5,5),1/8)
    #gradientY = cv2.GaussianBlur(gradientY,(5,5),1/8)
    
    gradientX = cv2.Sobel(np.float64(gray_image),cv2.CV_8U,1,0,ksize=3)
    gradientY = cv2.Sobel(np.float64(gray_image),cv2.CV_8U,0,1,ksize=3)

    print ( gradientY.shape )
    print ( gradientX.shape)
    # blur the edges using gaussian
    gradientX = cv2.GaussianBlur(gradientX,(5,5),0)#1/8)
    gradientY = cv2.GaussianBlur(gradientY,(5,5),0)#1/8)
   
    #cv2.imshow("gradienty",gradientY)
    #cv2.imshow("gradientx",gradientX)
    #alpha = 0.001       # The weight of the uniformity energy.
    #beta  = 0.000000007        # The weight of the curvature energy.
    #lambdap = 0.00005 
    KERNEL_SIZE = 9;
    #print(V)
 
    for t in range(n_steps):
        # ------------------------
        # your implementation here

        # ------------------------
        snake_length = get_length(snake)
        #this will hold the costs for each node and each state
        energyMatrix = np.zeros( (KERNEL_SIZE, NoOfVertices), dtype = 'float64' ); 
        #this will hold information about the minimum cost route to reach each node
        positnMatrix = np.zeros( (KERNEL_SIZE, NoOfVertices), dtype = 'int64' );
        for currVertex in range(0,NoOfVertices):
            # for all neighbors of the current node
            #print ( ' point - ',NoOfVertices, currVertex, snake[currVertex] )
            
            for i in range(0,KERNEL_SIZE):
                
                currNode = np.zeros((2,1))
                
                # find coordinate position of the neighbor in 3x3 neighborhood window              
                neighborPos_Y, neighborPos_X = get_neighborPosition( i )
                currNode[0] = snake[currVertex][0]+neighborPos_X
                currNode[1] = snake[currVertex][1]+neighborPos_Y
                
                #print ( currNode) 
                #find external energy at the current Node
                extEnergy = compute_externalEnergy(currNode,gradientX,gradientY) * lambdap
                
                # fill the total energy matrix with maximum floating point value
                energyMatrix[i][currVertex] = np.finfo('float64').max #float(999999999999999999999999.99)
                positnMatrix[i][currVertex] = -1
                
                # for all neighbors of the previous node
                for j in range(0,KERNEL_SIZE):
                     # find coordinate position of the neighbor in 3x3 neighborhood window              
                     neighPrevPos_Y, neighPrevPos_X = get_neighborPosition( j )
                     totalEnergy = 0 #extEnergy
                     # for all neighbors of the next node
                     for k in range(0,KERNEL_SIZE):
                        # find coordinate position of the neighbor in 3x3 neighborhood window              
                        neighborPos_Y, neighborPos_X = get_neighborPosition( k )  
                        initVertex = 0
                        nthVertex = NoOfVertices-1
                        
                        currentVertex = np.zeros((2,1))
                        prevVertex = np.zeros((2,1))
                        nextVertex = np.zeros((2,1))
                        
                        if(currVertex==0):
                            currentVertex[0] = snake[0][0]+neighborPos_X
                            currentVertex[1] = snake[0][1]+neighborPos_Y
                            prevVertex[0] = snake[nthVertex][0]+neighPrevPos_X
                            prevVertex[1] = snake[nthVertex][1]+neighPrevPos_Y 
                            nextVertex[0] = snake[1][0]+neighPrevPos_X
                            nextVertex[1] = snake[1][1]+neighPrevPos_Y 
                            
                            totalEnergy = compute_curvatureEnergy(currentVertex,prevVertex,nextVertex,beta)+compute_elasticityEnergy(currentVertex,prevVertex,alpha,snake_length)+extEnergy  
                        

                        elif(currVertex==nthVertex):
                            currentVertex[0] = snake[nthVertex][0]+neighborPos_X 
                            currentVertex[1] = snake[nthVertex][1]+neighborPos_Y
                            prevVertex[0] = snake[nthVertex-1][0]+neighPrevPos_X
                            prevVertex[1] = snake[nthVertex-1][1]+neighPrevPos_Y 
                            nextVertex[0] = snake[0][0]+neighPrevPos_X
                            nextVertex[1] = snake[0][1]+neighPrevPos_Y 
                            totalEnergy += energyMatrix[j][nthVertex-1] +  compute_curvatureEnergy(currentVertex,prevVertex,nextVertex,beta) +          compute_elasticityEnergy(currentVertex,prevVertex,alpha,snake_length)+ extEnergy                    
                        

                        else:
                            currentVertex[0] = snake[currVertex][0]+neighborPos_X
                            currentVertex[1] = snake[currVertex][1]+neighborPos_Y
                            prevVertex[0] = snake[currVertex-1][0]+neighPrevPos_X
                            prevVertex[1] = snake[currVertex-1][1]+neighPrevPos_Y 
                            nextVertex[0] = snake[currVertex+1][0]+neighPrevPos_X
                            nextVertex[1] = snake[currVertex+1][1]+neighPrevPos_Y 
                            totalEnergy += energyMatrix[j][currVertex-1] + compute_curvatureEnergy(currentVertex,prevVertex,nextVertex,beta)+                          compute_elasticityEnergy(currentVertex,prevVertex,alpha,snake_length)+extEnergy
                        
                        # Store minimum energy into table 
                        if energyMatrix[i][currVertex] > totalEnergy:
                           energyMatrix[i][currVertex] = totalEnergy
                           positnMatrix[i][currVertex] = j
                               
        posID = find_columnmin( energyMatrix, energyMatrix.shape[1]-1 )
        # Trace back the route that arrived in the above state
        for currVertex in range(NoOfVertices-1,-1,-1):
            posID = positnMatrix[posID][currVertex]
            #find coordinate position of the neighbor in 3x3 neighborhood window              
            neighPos_Y, neighPos_X = get_neighborPosition( posID ) 
            if (currVertex>0):  
               prevID = currVertex-1
            else:        
               prevID = NoOfVertices-1
            snake[prevID][0] += neighPos_X
            snake[prevID][1] += neighPos_Y
               
       
            
        ax.clear()
        ax.imshow(Im, cmap='gray')
        ax.set_title('frame ' + str(t))
        plot_snake(ax, snake)
        plt.pause(0.01)
        
    plt.pause(2)
    

if __name__ == '__main__':
    run('images/ball.png', radius=120,alpha = 0.001 ,beta = 0.00000007, lambdap = 0.000073 )
    run('images/coffee.png', radius=100,alpha = 0.001 ,beta = 0.000000007, lambdap = 0.00005)
