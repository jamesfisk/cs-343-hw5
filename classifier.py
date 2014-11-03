from __future__ import division
import numpy as np
import random
import Image
#Student added
import os

"""
This is your object classifier. You should implement the train and
classify methods for this assignment.
"""
class ObjectClassifier():
    labels = ['Tree', 'Sydney', 'Steve', 'Cube']
    
    """
    Everytime a snapshot is taken, this method is called and
    the result is displayed on top of the four-image panel.
    """
    def classify(self, edge_pixels, orientations):
        features = self.get_features(edge_pixels, orientations)
        return features
        #return random.choice(self.labels)


    """
		Returns a list of features for a given image.
		f[0] -> Upward oriented edge pixels in top half of image
		f[1] -> Upward oriented edge pixels in bottom half of image
		f[2] -> Total upward oriented edge pixels
		f[3] -> Horizontally oriented edge pixels
		f[4] -> Vertically oriented edge pixels
		f[5] -> Total number of edge pixels
		f[6] -> Percentage of edge pixels
		f[7] -> Proportion of horizontal edge pixels to vertical edge pixels
		f[8] -> Average pixel value
		f[9] -> Average nonblack pixel value
		f[10] -> Number of white pixels
		"""
    def get_features (self, edge_pixels, orientations):
			top_half = orientations[:len(orientations) / 2]
			bottom_half = orientations[len(orientations) / 2:]

			f = [self.count_upward_oriented(edge_pixels, top_half), 
					 self.count_upward_oriented(edge_pixels, bottom_half),
					 self.count_upward_oriented(edge_pixels, orientations),
					 self.count_horizontal_pixels(edge_pixels, orientations),
					 self.count_vertical_pixels(edge_pixels, orientations),
					 self.count_edge_pixels(edge_pixels, orientations),
					 self.count_edge_pixels(edge_pixels, orientations) / (600 * 800),
					 self.count_horizontal_pixels(edge_pixels, orientations) / self.count_vertical_pixels(edge_pixels, orientations),
					 self.average_edge_pixel_value(edge_pixels, orientations),
					 self.average_nonzero_pixel_value(edge_pixels, orientations),
					 self.num_white_pixels(edge_pixels, orientations)]

			return f
    """
    This is your training method. Feel free to change the
    definition to take a directory name or whatever else you
    like. The load_image (below) function may be helpful in
    reading in each image from your datasets.
    """
    def train(self):
				sydney  = "Sydney/"
				cube = "Cube/"
				tree = "Tree/"
				steve = "Steve/"
				objects = [sydney, cube, tree, steve]
				path = "/v/filer4b/v38q001/jamesrf/Desktop/OpenNERO-2014-09-30-x86_64/Hw5/snapshots/edges/Training/"
				for item in objects:
					print item
					for elt in os.listdir(path + item):
						(np_edges, orientations) = load_image(path + item + elt)
						print self.classify(np_edges, orientations)
        #pass
    """
		IMAGE FEATURE FUNCTIONS
		"""

    def average_edge_pixel_value(self, np_edges, orientations):
				pix_val = 0
				for i in range(len(np_edges)):
					for j in range(len(np_edges[0])):
						pix_val += np_edges[i][j]
				return (pix_val / (600 * 800))
    
    def average_nonzero_pixel_value(self, np_edges, orientations):
				pix_val = 0
				nonzero_pix = 0
				for i in range(len(np_edges)):
					for j in range(len(np_edges[0])):
						pixel = np_edges[i][j]
						if pixel != 0:
							pix_val += pixel
							nonzero_pix += 1
				return (pix_val /nonzero_pix)
   
    def num_white_pixels(self, np_edges, orientations):
        sum_pix = 0
        for i in range(len(np_edges)):
					for j in range(len(np_edges[0])):
						if np_edges[i][j] == 255:
							sum_pix += 1
        return sum_pix

    def num_black_pixels(self, np_edges, orientations):
        sum_pix = 0
        for i in range(len(np_edges)):
					for j in range(len(np_edges[0])):
						if np_edges[i][j] == 0:
							sum_pix += 1
        return sum_pix


    def count_horizontal_pixels(self, np_edges, orientations):
				sum_pix = 0
				for i in range(len(orientations)):
					for j in range(len(orientations[0])):
							pixel = orientations[i][j]
							if ((pixel == 0 or pixel == 180) and np_edges[i][j] != 0):
								sum_pix += 1
				return sum_pix

    def count_vertical_pixels(self, np_edges, orientations):
				sum_pix = 0
				for i in range(len(orientations)):
					for j in range(len(orientations[0])):
							pixel = orientations[i][j]
							if (pixel == 270 or pixel == 90):
								sum_pix += 1
				return sum_pix



    def count_upward_oriented(self, np_edges, orientations):
				sum_pix = 0
				for i in range(len(orientations)):
					for j in range(len(orientations[0])):
							pixel = orientations[i][j]
							if ((pixel == 315 or pixel == 0 or pixel == 45) and np_edges[i][j] != 0):
								sum_pix += 1
				return sum_pix

    def count_edge_pixels(self, np_edges, orientations):
				sum_pix = 0
				for i in range(len(np_edges)):
					for j in range(len(np_edges[0])):
						if (np_edges[i][j] != 0):
							sum_pix += 1
				return sum_pix
        

"""
Loads an image from file and calculates the edge pixel orientations.
Returns a tuple of (edge pixels, pixel orientations).
"""
def load_image(filename):
    im = Image.open(filename)
    np_edges = np.array(im)
    upper_left = push(np_edges, 1, 1)
    upper_center = push(np_edges, 1, 0)
    upper_right = push(np_edges, 1, -1)
    mid_left = push(np_edges, 0, 1)
    mid_right = push(np_edges, 0, -1)
    lower_left = push(np_edges, -1, 1)
    lower_center = push(np_edges, -1, 0)
    lower_right = push(np_edges, -1, -1)
    vfunc = np.vectorize(find_orientation)
    orientations = vfunc(upper_left, upper_center, upper_right, mid_left, mid_right, lower_left, lower_center, lower_right)
    return (np_edges, orientations)

        
"""
Shifts the rows and columns of an array, putting zeros in any empty spaces
and truncating any values that overflow
"""
def push(np_array, rows, columns):
    result = np.zeros((np_array.shape[0],np_array.shape[1]))
    if rows > 0:
        if columns > 0:
            result[rows:,columns:] = np_array[:-rows,:-columns]
        elif columns < 0:
            result[rows:,:columns] = np_array[:-rows,-columns:]
        else:
            result[rows:,:] = np_array[:-rows,:]
    elif rows < 0:
        if columns > 0:
            result[:rows,columns:] = np_array[-rows:,:-columns]
        elif columns < 0:
            result[:rows,:columns] = np_array[-rows:,-columns:]
        else:
            result[:rows,:] = np_array[-rows:,:]
    else:
        if columns > 0:
            result[:,columns:] = np_array[:,:-columns]
        elif columns < 0:
            result[:,:columns] = np_array[:,-columns:]
        else:
            result[:,:] = np_array[:,:]
    return result

# The orientations that an edge pixel may have.
np_orientation = np.array([0,315,45,270,90,225,180,135])

"""
Finds the (approximate) orientation of an edge pixel.
"""
def find_orientation(upper_left, upper_center, upper_right, mid_left, mid_right, lower_left, lower_center, lower_right):
    a = np.array([upper_center, upper_left, upper_right, mid_left, mid_right, lower_left, lower_center, lower_right])
    return np_orientation[a.argmax()]


def main():
	classifier = ObjectClassifier()
	classifier.train()
main()
