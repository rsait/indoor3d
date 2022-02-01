# indoor3d

Python package (indoor3d) for processing of indoors 3D. This package is built on top of the Open3D package, with the aim of making easier to perform common tasks that arise in indoor data processing. 

Open3D can load `.pcd` or `.ply` files, so let's suppose your file is `file.ply`.

Firt let's see the data with Open3D functionalities.

```bash
import open3d as o3d
import numpy as np

pcd_filename = `file.ply`
pcd = o3d.io.read_point_cloud(pcd_filename)
o3d.visualization.draw_geometries([pcd])
```

![View of the pointcloud](images/pcd_view.png)


## Creating Sphinx documentation

First of all, you have to install `sphinx`.

`pip3 install sphinx`

To create the documentation, got to the directory with the *index.rst* and *conf.py* and there just type

`make html`

and you will find the file `index.html` in the directory `_build/html`.



