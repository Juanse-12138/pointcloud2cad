import dxfwrite
from dxfwrite import DXFEngine as dxf

# Dimlines are separated from the core library.
# Dimension lines will not generated by the DXFEngine.
from dxfwrite.dimlines import dimstyles, LinearDimension

# create a new drawing
dwg = dxf.drawing('dimlines.dxf')

# dimensionline setup:
# add block and layer definition to drawing
dimstyles.setup(dwg)

# create a dimension line for following points
points = [ (1.7,2.5), (0,0), (3.3,6.9), (8,12)]

# define new dimstyles, for predefined ticks see dimlines.py
dimstyles.new("dots", tick="DIMTICK_DOT", scale=1., roundval=2, textabove=.5)
dimstyles.new("arrow", tick="DIMTICK_ARROW", tick2x=True, dimlineext=0.)
dimstyles.new('dots2', tick="DIMTICK_DOT", tickfactor=.5)

#add linear dimension lines
dwg.add(LinearDimension((3,3), points, dimstyle='dots', angle=15.))
dwg.add(LinearDimension((0,3), points, angle=90.))
dwg.add(LinearDimension((-2,14), points, dimstyle='arrow', angle=-10))

# next dimline is added as anonymous block
dimline = LinearDimension((-2,3), points, dimstyle='dots2', angle=90.)

# override dimension text
dimline.set_text(1, 'CATCH')
block = dxf.block(name='DOOR-01',basepoint=(500,0))
block.add()
# add dimline as anonymous block
dwg.add_anonymous_block(dimline, layer='DIMENSIONS')


block = dxf.block(name='DOOR-01',basepoint=(500,0))
block.add(dxf.line((0,0),(0,-900)))
block.add(dxf.line((0,-900),(100,-900)))
block.add(dxf.line((100,-900),(100,0)))
block.add(dxf.arc(900,(100,0),270,360))
drawing.blocks.add(block)
blockref = dxf.insert(blockname='DOOR-01', insert=(door2[0][0], door2[0][1])) # create a block-reference
drawing.add(blockref) # add block-reference to drawing
drawing.save()