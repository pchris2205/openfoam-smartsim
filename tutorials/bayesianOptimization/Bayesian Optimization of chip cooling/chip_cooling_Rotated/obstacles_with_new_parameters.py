FREECADPATH = '/home/pchris2205/freecad-python3/lib/'
import sys
sys.path.append(FREECADPATH)
import FreeCAD
import math
a1 = 100/100                      								#width1 cube [mm]
a2 = 100/100										            #width2 cube [mm]
h1 = 3                                                           #heigth of cubes when m odd [mm]
h1 = 3                                                           #heigth of cubes when m even [mm]
n1 = 10                      	    					    #number of cubes in x-direction when m odd
n2 = 10                                                    #number of cubes in x-direction when m even
m = 15                       								#number of cubes in y-direction
c = 15/10                    	    							#distance between cubes in y-direction, [mm]
y_0 = 20                	    						#distance of first row of cubes from inlet, [mm]
phi1 = 45                                                   #angle of even m rows, [°]
phi2 = 45                                                   #angle of odd m rows, [°]
angles =[phi1, phi2]

phi1_radians = math.radians(phi1)
phi2_radians = math.radians(phi2)
f1 = a1*(math.cos(phi1_radians)+math.sin(phi1_radians))
f2 = a2*(math.cos(phi2_radians)+math.sin(phi2_radians))
b = (38 - n1*a1)/(n1 + 1)                                       #distance between cubes in x-direction when m odd, [mm]
d = (38 - n2*a2)/(n2 + 1)                                       #distance between cubes in x-direction when m even, [mm]

import Part
cube = Part.makeBox(1,1,1)
cube2 = Part.makeBox(1,1,1)
doc = cube.cut(cube2)
stop_outer_loop = False
for i in range (0, m, 1):
    if stop_outer_loop:  
        break
    if (i % 2) == 0: 
        for j in range (0, n1, 1):
            x1 = b*(j + 1) + j*a1
            y1 = y_0 + c*i + a1*i + a2*i
            x_m1 = x1 + 0.5*a1
            y_m1 = y1 + 0.5*a1
            if y_m1 < 88:
                box = Part.makeBox(a1,a1,3)
                box.Placement.Base = FreeCAD.Vector(x1, y1, 0)
                box.rotate(FreeCAD.Vector(x_m1, y_m1, 0),FreeCAD.Vector(0, 0, 1), angles[1])
                doc = doc.fuse(box)
            else:
                print(f"obstacles on outlet, m = {i} not m = {m}")
                stop_outer_loop = True
                break
    else:
        for l in range (0, n2, 1):
            x2 = d*(l + 1) + a2*l
            y2 = y_0 + c*i + a1*i + a2*i
            x_m2 = x2 + 0.5*a2
            y_m2 = y2 + 0.5*a2
            if y_m2 < 88:
                box = Part.makeBox(a2,a2,3)
                box.Placement.Base = FreeCAD.Vector(x2, y2, 0)
                box.rotate(FreeCAD.Vector(x_m2, y_m2, 0),FreeCAD.Vector(0, 0, 1), angles[1])
                doc = doc.fuse(box)
            else:
                print(f"obstacles on outlet, m = {i} not m = {m}")
                stop_outer_loop = True
                break

doc.exportStl("obstacles.stl")