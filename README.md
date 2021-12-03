# puzzle_solver

Algorithm for assembling a puzzle from fragments of images using only numpy library. 
The cutting can be in different number of fragments (from 12 to 432). 
The cut lines have different shapes: straight lines or rectangular protrusions. 
Each fragment can be randomly rotated by one of the angles (0, 90, 180, 270). 
The resolution of the original image is 1200x900.
It is guaranteed that there are no completely black pixels in the original image (RGB - 0, 0, 0).

<img src="https://user-images.githubusercontent.com/11677412/144580825-25e8b187-e8d3-4f69-ac50-d1518ef2a607.png" width="512">

<img src="https://user-images.githubusercontent.com/11677412/144580833-f1331462-ac09-4cf8-b54f-c2046faf2150.png" width="523">

Public dataset of puzzles: https://drive.google.com/file/d/1iiq2AqW9QK9DWElmPmxNIclXlhOCUnK-/view

Syntax for running a program (command line): python3 solver.py path_to_directory

The image format is ppm (simple "text" format, see Netpbm). 
As a result of the program's execution, an image.ppm file (in the same format as the fragment files) with the generated image should appear in the current working directory (cwd). The size of the output image -- 1200x900 (3 channels).

