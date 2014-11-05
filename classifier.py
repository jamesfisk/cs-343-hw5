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
    labels = ['Sydney', 'Cube', 'Tree', 'Steve']
    sydney_features = [13047.2, 28359.5, 40250.3, 26016.5, 69391.8, 117493.3, 0.29373325, 0.3746875567, 5.2645383333, 21.4338684044, 1674.5]
    cube_features = [11247.7777777778, 25055.4444444444, 34311.7777777778, 21849.7777777778, 62659.4444444444, 104580.555555556, 0.2614513889, 0.3504291513, 4.4927993056, 20.7388081088, 332.3333333333]
    tree_features = [16716.2222222222, 33044.3333333333, 42624.8888888889, 28237.1111111111, 74676.1111111111, 127227.888888889, 0.3180697222, 0.3780615109, 5.9146527778, 22.1799279468, 1391]
    steve_features = [11460.4166666667, 26481.4166666667, 39154, 24642.75, 69852.8333333333, 118387.583333333, 0.2959689583, 0.3519500759, 5.7145130208, 22.9063745015, 1425.25] 
    
    """
    Everytime a snapshot is taken, this method is called and
    the result is displayed on top of the four-image panel.
    """
    def classify(self, edge_pixels, orientations):
       				features = self.get_features(edge_pixels, orientations)
				#features = self.sydney_features
				#print(features)
				object_features = [self.sydney_features, self.cube_features, self.tree_features, self.steve_features]
				scores = [.25, .25, .25, .25]
				for i in range(len(features)):
					test_features = [self.sydney_features[i], self.cube_features[i], self.tree_features[i], self.steve_features[i]]
					feature_probs = self.probability_from_feature(features[i], test_features)
					weight = 1 - min(test_features)/max(test_features) 
					for j in range(len(feature_probs)):
						scores[j] *= feature_probs[j] * weight
				winner = scores.index(max(scores))
      				return self.labels[winner]

    """
    Assigns a probability to each object based upon a single feature
    """
    def probability_from_feature(self, feature, test_features):
				sum_abs = 0
				feature_probs = []
				for item in test_features:
					sum_abs += abs(feature - item)
				for item in test_features:
					feature_probs.append(1-(abs(feature - item)/sum_abs))
				return feature_probs



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


if __name__ == "__main__":
	classifier = ObjectClassifier()
	classifier.train()
