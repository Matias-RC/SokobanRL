import numpy as np

import random

def CreateEmpty(width, height):
    return  np.zeros((width, height))

def templateFits(template, grid, BoxIndex, BoxesInGrid):
    x = BoxIndex[1]*3-1
    y = BoxIndex[0]*3-1
    if y == -1:
        if not np.all(template[0, :]== -1):
            return False
        template = np.delete(template, 0, axis=0)
        y = y + 1
    elif x == -1:
        if not np.all(template[:, 0] == -1):
            return False
        template = np.delete(template, 0, axis=1)
        x =  x + 1
    elif y == grid.shape[0] and x == grid.shape[1]:
        if not np.all(template[-1, :]== -1) or np.all(template[:, -1] == -1):
            return False
        template = np.delete(template,-1,axis=0)
        template = np.delete(template,-1,axis=1)
    elif y == grid.shape[0]:
        if not np.all(template[-1, :]== -1):
            return False
        template = np.delete(template,-1,axis=0)
    elif x == grid.shape[1]:
        if not np.all(template[:, -1] == -1):
            return False
        template = np.delete(template,-1,axis=1)
    for i in range(template.shape[0]):
        for j in range(template.shape[1]):
            if template[i, j] != -1 and grid[y + i, x + j] != 0:
                return False  
    return True

def PlaceTemplate(template, grid, BoxIndex):
    x = BoxIndex[1]*3-1
    y = BoxIndex[0]*3-1
    if y == -1:
        template = np.delete(template, 0, axis=0)
        y = y+1
    elif x == -1:
        template = np.delete(template, 0, axis=1)
        x =  x+1
    elif y == grid.shape[0] and x == grid.shape[1]:
        template = np.delete(template,-1,axis=0)
        template = np.delete(template,-1,axis=1)
    elif y == grid.shape[0]:
        template = np.delete(template,-1,axis=0)
    elif x == grid.shape[1]:
        template = np.delete(template,-1,axis=1)
    for i in range(template.shape[0]):
        for  j in  range(template.shape[1]):
            if template[i,j] != -1:
                grid[y+i,x+j] = template[i,j]
    return grid

def fitTemplate(BoxIndex, BoxesInGrid , grid, templates):
    """
        template = RandomlySelectAndRotateTemplate(templates)
        if TemplateFits(block, template):
            PlaceTemplate(block, template)
        else:
            RetryRoomPartition()
    """
    i, j = BoxIndex
    m, n = BoxesInGrid
    if i and j == 0:
        template = random.choice(templates[0])
        grid = PlaceTemplate(template, grid, BoxIndex, BoxesInGrid)
    else:
        template = random.choice(templates[1])
        while not templateFits(template, grid, BoxIndex, BoxesInGrid):
            template = random.choice(templates[1])
        grid = PlaceTemplate(template, grid, BoxIndex, BoxesInGrid)
    
    return grid


def PartitionGrid3x3(grid, templates):
    if grid.shape[0] <= 3:
        pass
    n = (grid.shape[0]-grid.shape[1]%3)/3
    m  = (grid.shape[1]-grid.shape[0]%3)/3
    if n and m == 1:
        pass

    for i in range(m):
        for j in range(n):
            grid = fitTemplate(BoxIndex=[i,j], BoxesInGrid=[m,n], grid=grid, templates=templates)