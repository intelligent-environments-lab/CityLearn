import math
import os
from PIL import Image, ImageDraw
import numpy as np

def get_plots(values, limits):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(nrows=3, sharex=True)
    fig.set_size_inches((10, 10))
    fig.set_dpi(72)

    fig.suptitle('District level energy consumption')

    names = ['Electricty consumption', 
            'Electricy consumption \n without Storage', 
            'Electricty without \n Storage & PV']
    colors = ['green',  'orange', 'blue']
    for ax, color, data, limit, name in zip(axs, colors, values, limits, names):
        ax.margins(0)
        ax.set_xlim(24)
        ax.grid('on')
        ax.set_ylabel(name)
        ax.plot(np.arange(1, len(data)+1), data, '-o', ms=5, color=color)
    ax.set_xlabel('Value before (x) hours')

    fig.set_frameon(False)
    fig.canvas.draw()
    img = np.array(fig.canvas.buffer_rgba())[..., :3]
    plt.close(fig)
    return img

def get_background():
    canvas_size = 720
    grid_size = canvas_size//5
    folder_path = os.path.dirname(__file__)
    grid = Image.open(os.path.join(folder_path, 'assets/grid.png'))
    grid = grid.resize((grid_size, grid_size))
    grid_coord = canvas_size//2-grid_size//2

    canvas = Image.new('RGB', (canvas_size, canvas_size), (255, 255, 255))
    canvas.paste(grid, (grid_coord, grid_coord), mask = grid)

    draw_obj = ImageDraw.Draw(canvas)
    cx, cy = canvas_size//2, canvas_size//2
    r = grid_size//4
    circle_coords = cx - r, cy - r, cx + r,  cy + r
    color = (100, 100, 100)
    draw_obj.ellipse(circle_coords, fill = color, outline = color)
    return canvas, canvas_size, draw_obj, color
    
    
class RenderBuilding:
    def __init__(self, index, canvas_size, num_buildings, line_color):
        self.canvas_size = canvas_size
        
        self.building_radius = canvas_size//3 # Can change this for size
        self.building_size = canvas_size//7 # Can change this for size
        
        self.max_glow_size = self.building_size//2
        self.glow_scales = {0: 0.4, 1: 0.6, 2: 0.8, 3: 1}
        
        self.building_type = index%2
        self.index = index
        self.num_buildings = num_buildings
        self.line_color = line_color
        
        self.angle = index*360//num_buildings
        
    def read_image(self, charge):
        charge_quartile = int(min(charge*100//25, 3))
#         assert charge_quartile in range(4)
        folder_path = os.path.dirname(__file__)
        im_name = f'assets/building-{self.building_type}-charge-{charge_quartile}.png'
        im_name = os.path.join(folder_path, im_name)
        img = Image.open(im_name)
        img = img.resize((self.building_size, self.building_size))
#         img = img.rotate(-self.angle) # Looks weird and has cropping issue
        return img
    
    def get_coords(self):
        x, y = 0, -self.building_radius
        angle = self.angle * math.pi / 180
        xd = int(math.cos(angle)*x - math.sin(angle)*y)
        yd = int(math.sin(angle)*x + math.cos(angle)*y)
        return xd, yd
    
    def read_glow_image(self, energy):
        energy_quartile = int(min(energy*100//25, 3))
        folder_path = os.path.dirname(__file__)
        glow_image = Image.open(os.path.join(folder_path, 'assets/glow.png'))
        glow_size = int(self.max_glow_size * self.glow_scales[energy_quartile])
        glow_image = glow_image.resize((glow_size, glow_size))
        return glow_image, glow_size
    
    def draw_line(self, canvas, draw_obj, energy, color):
        coords = self.get_coords()
        offset = self.canvas_size//2
        line_coords = coords[0] + offset, coords[1] + offset
        center = self.canvas_size//2, self.canvas_size//2
        draw_obj.line((center, line_coords), fill=color, width=2)
        
        glow_image, glow_size = self.read_glow_image(energy)
        glow_coords = (center[0] + line_coords[0])//2 - glow_size//2, \
                      (center[1] + line_coords[1])//2 - glow_size//2
        canvas.paste(glow_image, glow_coords, mask=glow_image)
    
    def draw_building(self, canvas, charge):
        building_image = self.read_image(charge)
        coords = self.get_coords()
        offset = self.canvas_size//2 - self.building_size//2
        coords = coords[0] + offset, coords[1] + offset
        canvas.paste(building_image, coords, mask=building_image)