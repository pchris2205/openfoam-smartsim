# Eventually sourcing the FreeCAD library needed in Python is necessary
import FreeCAD
import math
a1 = !a1!/100                           
a2 = !a2!/100                           
h = 3                                   
n1 = !n1!                               
n2 = !n2!                               
m = !m!                                 
c = !c!/10                              
y_0 = !y_0!                             
phi1 = !phi1!                           
phi2 = !phi2!                           
angles =[phi1, phi2]
b = (38 - n1*a1)/(n1 + 1)               
d = (38 - n2*a2)/(n2 + 1)               

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
