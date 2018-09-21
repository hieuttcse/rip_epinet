import pytools.file_io as file_io
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# data_folder = "some_path/training/dino"

# LF = file_io.read_lightfield(data_folder)
# param_dict = file_io.read_parameters(data_folder)
# depth_map = file_io.read_depth(data_folder, highres=True)
# disparity_map = file_io.read_disparity(data_folder, highres=False)
dmap = file_io.read_pfm('./my_output/bicycle.pfm')

cb_shrink = 0.7
cc = plt.imshow(dmap, cmap=cm.viridis, interpolation="none")
plt.colorbar(cc, shrink=cb_shrink)

plt.show()
