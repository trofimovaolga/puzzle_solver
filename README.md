# puzzle_solver

Algorithm for assembling a puzzle from fragments of images using only numpy library. 
The cutting can be in different number of fragments (from 12 to 432). 
The cut lines have different shapes: straight lines or rectangular protrusions. 
Each fragment can be randomly rotated by one of the angles (0, 90, 180, 270). 
The resolution of the original image is 1200x900.
It is guaranteed that there are no completely black pixels in the original image (RGB - 0, 0, 0).

Syntax for running a program (command line): python3 solver.py path_to_directory

The image format is ppm (simple "text" format, see Netpbm). 
As a result of the program's execution, an image.ppm file (in the same format as the fragment files) with the generated image should appear in the current working directory (cwd). The size of the output image -- 1200x900 (3 channels).
