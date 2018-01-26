### Display 3DFSC and coloring original map by angular resolution
### Written by Tom Goddard
### Modified by Yong Zi Tan

# Open lineplot.py
open lineplot.py

# Open both volumes and hide the 3DFSC volume.
set bg_color white
open #0 #===3DFSC====#
volume #0 originIndex #==Origin3DFSC==#
volume #0 voxelSize #==apix==#
open #1 #===FullMap====#
volume #1 originIndex #==OriginFullMap==#
volume #1 voxelSize #==apix==#
volume #0 hide
focus
colorkey 0.2,0.14 0.8,0.1 tickMarks True tickThickness 3 "Poorer" red "Better" blue ; 2dlabels create Label text "Relative XY-plane resolution (AU)" color black xpos 0.2 ypos 0.16

# Execute lineplot.py
fscplot #0

