import os
import sys
from typing import Container
import numpy as np
import time

W, H = 1200, 900 # dimensions of result image
w, h = 0, 0 # dimensions of each tile
CHANNEL_NUM = 3  # we work with rgb images
MAX_VALUE = 255  # max pixel value, required by ppm header
is_square = True # if the tiles have blanks or tabs
is_vertical, is_horizontal = False, False

class EdgeInfo: # description of tile's edge
    def __init__(
        self,
        is_female = True,
        width_of_piece = w, # piece = blank/tab
        height_of_piece = h,
        first_nonzero = 0,
        last_nonzero = w,
        delta = w
    ):
        self.is_female = is_female
        self.width_of_piece = width_of_piece
        self.height_of_piece = height_of_piece
        self.first_nonzero = first_nonzero
        self.last_nonzero = last_nonzero
        self.delta = delta

def read_image(path):
    # second line of header contains image dimensions
    w, h = np.loadtxt(path, skiprows=1, max_rows=1, dtype=np.int32)
    # skip 3 lines reserved for header and read image
    image = np.loadtxt(path, skiprows=3, dtype=np.int32).reshape((h, w, CHANNEL_NUM))
    return image

def write_image(path, img):
    h, w = img.shape[:2]
    # ppm format requires header in special format
    header = f'P3\n{w} {h}\n{MAX_VALUE}\n'
    with open(path, 'w') as f:
        f.write(header)
        for r, g, b in img.reshape((-1, CHANNEL_NUM)):
            f.write(f'{r} {g} {b} ')

def sum_of_tuples(ans, crd, dim='row'):
    if dim == 'row':
        return sum(map(lambda x: type(x) is list, ans[crd, :]))
    elif dim == 'col':
        return sum(map(lambda x: type(x) is list, ans[:, crd]))

def make_candidates(y, x): # neighbours around the main tile - [down, up, right, left] coordinates
    return [(y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)]

def edge(tile, edge_type, square=True): # vector edge of the tile
    # rotate the tile to move 'edge_type' edge to the top
    type_to_rot = {'top': 0, 'bottom': 2, 'right': 1, 'left': 3}
    tile = np.rot90(tile, type_to_rot[edge_type])

    # horizontally flip the tile after rotation to keep start and end points correct
    if edge_type == 'left' or edge_type == 'bottom':
        tile = tile[:, ::-1]
    tile_edge = np.copy(tile[0])
    
    if not square: # return pixels of the contour around the blank/tab
        for i in range(len(tile_edge)):
            j = np.where(tile[:, i] != [0, 0, 0])[0][0]
            tile_edge[i] = tile[j][i]
    
    return tile_edge

def find_edge_type(y0, x0, y, x): # main tile's edge type next to the tile-candidate
    if x0 - x == 0:
        if y0 - y == 1:
            edge_type = 'top'
        else:
            edge_type = 'bottom'
    elif x0 - x == 1:
        edge_type = 'left'
    else:
        edge_type = 'right'
    
    return edge_type

def find_dist(tiles, tiles_edge_info, t_indexes, main_inds, main_rots, edge_types, flat_edges): # [distance, rotations num, tile's ind]
    # finding minimal distance between result tiles and other tiles-candidates
    # returns the distance, rotation coefficient and index of the tile that fits the best
    # main_inds are indexes of the tiles that already in the result matrix
    eps, neigh_rot, neigh_ind = 1e6, -1, -1
    opposite_edge = {'right': 'left', 'left': 'right', 'top': 'bottom', 'bottom': 'top'}

    main_infos, main_edges = [], [] # edges descriptions for 'edge_type' side of main tile
    for i in range(len(main_inds)):
        main_tile = np.rot90(tiles[main_inds[i]], main_rots[i])
        if is_square:
            main_edges.append(edge(main_tile, edge_types[i]))
        else:
            main_infos.append(tiles_edge_info[main_inds[i]][main_rots[i]][edge_types[i]])
            contour = edge(main_tile, edge_types[i], square=False)
            main_edges.append(contour[main_infos[i].first_nonzero:main_infos[i].last_nonzero + 1])
    
    for t in t_indexes: # check all tiles that weren't used yet
        for rot in range(4): # rotate the tile 4 times
            # if main tile has a flat edge (no blanks or tabs) then the neighbour also 
            # should have it on the same side when we stack them along this side
            if not is_square and any([tiles_edge_info[t][rot][e].width_of_piece != w for e in flat_edges]):
                continue
            
            dist = []
            cur_tile = np.rot90(tiles[t], rot)
            for i in range(len(main_inds)):
                if is_square:
                    cur_edge = edge(cur_tile, opposite_edge[edge_types[i]])
                else:
                    cur_info = tiles_edge_info[t][rot][opposite_edge[edge_types[i]]]
                    if (main_infos[i].is_female == cur_info.is_female or             # we can only join different types of tiles
                        main_infos[i].width_of_piece != cur_info.width_of_piece or   # width of blank/tab should be the same
                        main_infos[i].height_of_piece != cur_info.height_of_piece or # height of blank/tab should be the same
                        main_infos[i].delta != cur_info.delta):                      # starting point should be the same
                        continue
                    
                    contour = edge(cur_tile, opposite_edge[edge_types[i]], square=False)
                    cur_edge = contour[cur_info.first_nonzero:cur_info.last_nonzero + 1]

                dist.append(np.linalg.norm(main_edges[i] - cur_edge))
            
            if len(dist) != len(main_inds):
                continue
            
            mean_dist = sum(dist) / len(dist) # mean distance from candidate-tile to result-tiles
            if mean_dist < eps:
                eps, neigh_rot, neigh_ind = mean_dist, rot, t
    
    return [eps, neigh_rot, neigh_ind]

def check_all_cands(result, tiles, tiles_edge_info, t_indexes, cands): # returns distances
    dists = []    
    for y_cand, x_cand in cands:
        if result[y_cand, x_cand] != -2: # can't overwrite the boarder
            neighbours = make_candidates(y_cand, x_cand)
            main_inds, main_rots, edge_types, flat_edges = [], [], [], []
            for y, x in neighbours:     
                if result[y, x] != -1:
                    if result[y, x] != -2:
                        edge_types.append(find_edge_type(y, x, y_cand, x_cand))
                        main_inds.append(result[y, x][0])
                        main_rots.append(result[y, x][1])
                    else:
                        # it's the boarder so this side is flat
                        flat_edges.append(find_edge_type(y_cand, x_cand, y, x))

            dist = find_dist(tiles, tiles_edge_info, t_indexes, main_inds, main_rots, edge_types, flat_edges)
            dists.append(dist)
        else:
            dists.append([1e6, -1, -1]) # return 1e6 as a distance so it won't be picked by argmin()

    return dists

def get_edge_info(tile, edge_type): # find blank/tab and it's meassurements
    info = EdgeInfo()
    # rotate the tile to move 'edge_type' edge to the top
    type_to_rot = {'top': 0, 'bottom': 2, 'right': 1, 'left': 3}
    tile = np.rot90(tile, type_to_rot[edge_type])
    
    # horizontally flip the tile after rotation to keep start and end points correct
    if edge_type == 'left' or edge_type == 'bottom':
        tile = tile[:, ::-1]
    tile_edge = tile[0]
    
    nz = np.nonzero(tile_edge)[0]
    info.first_nonzero, info.last_nonzero = nz[0], nz[-1]
    
    if info.last_nonzero - info.first_nonzero < w - 1: # the side has a tab
        info.is_female = False
    
    blank_start, blank_end = info.first_nonzero, info.last_nonzero
    if info.is_female:
        female_edge = np.where(tile_edge[info.first_nonzero:info.last_nonzero + 1] == 0)[0]
        info.width_of_piece = len(female_edge) // 3 # each pixel is described by 3 numbers (RGB)
        if info.width_of_piece == 0: # no zero pixels in the edge vector
            info.width_of_piece = w  # the side is flat
        else:
            blank_start = female_edge[0] - 1 + info.first_nonzero
            blank_end = female_edge[-1] + 2 + info.first_nonzero
    else:
        info.width_of_piece = info.last_nonzero - info.first_nonzero + 1

    # if the side has blank or tab we need to know it's size
    if info.width_of_piece != w:
        depth = 0
        if info.is_female:
            while [0, 0, 0] in tile[depth, blank_start:blank_end, :]:
                depth += 1
            nz1 = np.where(tile[0, info.first_nonzero:info.last_nonzero, :] == [0, 0, 0])[0]
            info.delta = nz1[0]
        else:
            while [0, 0, 0] in tile[depth, info.first_nonzero - 1:info.last_nonzero + 2, :]:
                depth += 1
            nz1 = np.nonzero(tile[depth + 2])[0]
            info.delta = info.first_nonzero - nz1[0]
            # re-write first and last nonzero elements so they are now start and end 
            # points of the tile's side, not the tab
            info.first_nonzero = nz1[0]
            info.last_nonzero = nz1[-1]

        info.height_of_piece = depth

    return info

def assemble(result, tiles, tiles_edge_info, is_vertical): # assembling based on result matrix
    # delete rows and columns that don't contain tuples
    xs = [sum_of_tuples(result, i, dim='row') for i in range(result.shape[0])]
    ys = [sum_of_tuples(result, i, dim='col') for i in range(result.shape[1])]
    
    xs1, xs2 = np.nonzero(xs)[0][0], np.nonzero(xs)[0][-1]
    ys1, ys2 = np.nonzero(ys)[0][0], np.nonzero(ys)[0][-1]
    
    result = result[xs1:xs2 + 1, ys1:ys2 + 1]
    
    y_nodes, x_nodes = np.arange(0, result.shape[0] * h, h), np.arange(0, result.shape[1] * w, w)
    xx, yy = np.meshgrid(x_nodes, y_nodes)
    nodes = np.vstack((xx.flatten(), yy.flatten())).T

    img = np.zeros((result.shape[0] * h, result.shape[1] * w, CHANNEL_NUM), dtype=np.uint8)
    
    # fill grid with tiles
    for (x, y), elem in zip(nodes, result.flatten()):
        if elem == -1 or elem == -2 or elem == list([-1, -1]): # didn't find fitting tile
            continue
        
        tile = np.rot90(tiles[elem[0]], elem[1])
        # calculate the offset for each side of tile
        tabs = dict()
        for edge_type in ['left', 'right', 'top', 'bottom']:
            edge_info = tiles_edge_info[elem[0]][elem[1]][edge_type]
            if edge_info.is_female or edge_info.height_of_piece == h:
                # no offset for the side with a blank or flat side
                edge_info.height_of_piece = 0
            tabs[edge_type] = edge_info.height_of_piece
        try:
            img[y - tabs['top']: y + tabs['bottom'] + h, x - tabs['left']: x + tabs['right'] + w] += tile.astype(np.uint8)
        except ValueError:
            continue
    if is_vertical: img = np.rot90(img)
    
    output_path = "image.ppm"
    write_image(output_path, img)

def add_boarders(x_max, x_min, y_max, y_min, img_width, img_height, result):  # add boarders to result matrix
    # tracking width and height of image in the result matrix

    global is_horizontal, is_vertical
    
    if x_max - x_min == img_height:
        is_horizontal = True
    if y_max - y_min == img_height:
        is_vertical = True        
    if x_max - x_min == img_width - 1:
        result[:, x_min - 1] = -2
        result[:, x_max + 1] = -2        
    if y_max - y_min == img_width - 1:
        result[y_min - 1, :] = -2
        result[y_max + 1, :] = -2            
    if is_vertical and x_max - x_min == img_height - 1:
        result[:, x_min - 1] = -2
        result[:, x_max + 1] = -2        
    if is_horizontal and y_max - y_min == img_height - 1:
        result[y_min - 1, :] = -2
        result[y_max + 1, :] = -2

def add_boarder_for_flat_edge(result, y0, x0, tiles_edge_info): # add boarders to result matrix
    # if the tile has flat edge(s) we can't place any more tiles next to it
    
    if is_square:
        return
    ind, rot = result[y0, x0]
    if tiles_edge_info[ind][rot]['bottom'].width_of_piece == w:
        result[y0 + 1, :] = -2
    elif tiles_edge_info[ind][rot]['top'].width_of_piece == w:
        result[y0 - 1, :] = -2
    elif tiles_edge_info[ind][rot]['left'].height_of_piece == h:
        result[:, x0 - 1] = -2
    elif tiles_edge_info[ind][rot]['right'].height_of_piece == h:
        result[:, x0 + 1] = -2

def get_tiles_edge_info(tiles): # description of each side for all the tiles
    tiles_edge_info = [] 
    for tile in tiles:
        info_rot = []        
        for rot in range(4):
            tile_rotated = np.copy(tile)
            tile_rotated = np.rot90(tile_rotated, rot)
            edges_info = dict()
            for edge_type in ['left', 'right', 'top', 'bottom']:
                edges_info[edge_type] = get_edge_info(tile_rotated, edge_type)
            info_rot.append(edges_info)
        
        tiles_edge_info.append(info_rot)

    return tiles_edge_info

def solve_puzzle(tiles_folder):
    tic = time.perf_counter()
    tiles = [read_image(os.path.join(tiles_folder, t)) for t in sorted(os.listdir(tiles_folder))]
    
    if np.array([0, 0, 0]) in tiles[0]: # if tile has blanks/tabs pixels
        global is_square
        is_square = False

    t_indexes = set(np.arange(len(tiles))) # indexes of tiles that weren't used yet
    img_height = int(3 * (len(tiles) / 12) ** 0.5) # number of tiles along y axis
    img_width = int(4 * (len(tiles) / 12) ** 0.5) # number of tiles along x axis
    
    global h, w
    h, w = H // img_height, W // img_width
    max_hw = max(img_height, img_width)
    
    tiles_edge_info = get_tiles_edge_info(tiles)
 
    # result matrix keeps the position of each tile and it's rotation coefficient
    # filled with -1 at the begining
    result = np.ones((max_hw * 2 + 1, max_hw * 2 + 1), dtype=tuple) * -1
    
    # place the first tile in tiles with zero rotation in the center of result matrix
    center = result.shape[0] // 2
    result[center, center] = [0, 0]
    t_indexes.remove(0)
    y0, x0 = center, center
    
    add_boarder_for_flat_edge(result, y0, x0, tiles_edge_info) # if tile has flat edge
    
    # neighbours of the tile
    cands = make_candidates(y0, x0)
    # best tile for each side of main tile
    dists = check_all_cands(result, tiles, tiles_edge_info, t_indexes, cands)
    
    # variables for tracking width and height of image in the result matrix
    x_min, x_max, y_min, y_max = x0, x0, y0, y0

    while len(t_indexes) > 0:
        # index of the tile that fits the best to main tile
        min_dist_index = np.argmin([dist[0] for dist in dists])
        tile_rot, next_tile = dists[min_dist_index][1:]
        
        # now this is the main tile, add it to result matrix
        y0, x0 = cands[min_dist_index]
        x_min, x_max = min(x0, x_min), max(x0, x_max)
        y_min, y_max = min(y0, y_min), max(y0, y_max)
        result[y0, x0] = [next_tile, tile_rot]

        add_boarder_for_flat_edge(result, y0, x0, tiles_edge_info)
        add_boarders(x_max, x_min, y_max, y_min, img_width, img_height, result)
        
        # already in the result matrix, not a candidate anymore
        cands.pop(min_dist_index)
        dists.pop(min_dist_index)
        
        if next_tile == -1: break # no matching tile found
        
        t_indexes.remove(next_tile)
        
        # correct the distances list in case we added boarder or new tile
        for i in range(len(cands)):
            y_cand, x_cand = cands[i]
            if result[y_cand, x_cand] == -2: # can't place a tile on the boarder
                dists[i] = [1e6, -1, -1]     # re-write it's distance so it won't be picked in argmin
            elif dists[i][2] == next_tile: # already calculated dist for this tile before
                # re-calculate distance between 'next_tile' and surrounding tiles
                dists[i] = check_all_cands(result, tiles, tiles_edge_info, t_indexes, [cands[i]])[0]
        
        new_cands = [] # candidates of new main tile
        for y_cand, x_cand in make_candidates(y0, x0):
            if result[y_cand, x_cand] == -1:
                if (y_cand, x_cand) in cands: # checked it for previous main tile, distance should be re-calculated
                    ind = cands.index((y_cand, x_cand))
                    cands.pop(ind)
                    dists.pop(ind)
                new_cands.append((y_cand, x_cand))
        
        if len(new_cands) > 0:
            new_dists = check_all_cands(result, tiles, tiles_edge_info, t_indexes, new_cands)
            cands.extend(new_cands)
            dists.extend(new_dists)

        toc = time.perf_counter()
        if (toc - tic) > 25:
            break
    
    assemble(result, tiles, tiles_edge_info, is_vertical)

if __name__ == "__main__":
    directory = sys.argv[1]
    solve_puzzle(directory)
