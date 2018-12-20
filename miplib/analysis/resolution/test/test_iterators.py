

import numpy as np
from mayavi import mlab

from miplib.data.iterators.fourier_shell_iterators import ConicalFourierShellIterator
from miplib.data.containers.image import Image

dataset = Image(np.ones((255,255,255), dtype=np.uint8)*20, (0.05,0.05,0.05))

iterator = ConicalFourierShellIterator(dataset.shape, 6, 20)

section = iterator[(100, 106, 30, 60)]

dataset[section] = 255

mlab.contour3d(dataset)
mlab.show()
