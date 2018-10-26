import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering
import cv2
import os
from tkinter import *
#from tkFileDialog import askopenfilename
from PIL import Image, ImageTk

seed_count = 0
# function to show an image
def showImage(img_to_show):
	img_to_show = img_to_show.resize((200, 200))
	img_to_show = np.asarray(img_to_show)
	#img_to_show = img_to_show[:, :]
	img_to_show = img_to_show.astype(float)
	plt.imshow(img_to_show, cmap='gray')
	plt.show()

# Region Growing
class Queue:
	def __init__(self):
		self.items = []

	def isEmpty(self):
		return self.items==[]

	def enque(self,item):
		self.items.insert(0,item)

	def deque(self):
		return self.items.pop()

	def qsize(self):
		return len(self.items)

	def isInside(self, item):
		return (item in self.items)

def regiongrow(src_image, dest_image, epsilon, start_point):
	Q = Queue()
	s = []

	x = start_point[0]
	y = start_point[1]

	image = src_image.convert("L")
	Q.enque((x,y))

	while not Q.isEmpty():
		t = Q.deque()
		x = t[0]
		y = t[1]

		if x < image.size[0]-1 and abs(image.getpixel( (x + 1 , y) ) - image.getpixel( (x , y) )  ) <= epsilon :
			if not Q.isInside( (x + 1 , y) ) and not (x + 1 , y) in s:
				Q.enque( (x + 1 , y) )
		
		if x > 0 and abs(image.getpixel( (x - 1 , y) ) - image.getpixel( (x , y) )  ) <= epsilon:
			if not Q.isInside( (x - 1 , y) ) and not (x - 1 , y) in s:
				Q.enque( (x - 1 , y) )

		if y < (image.size[1] - 1) and abs(image.getpixel( (x , y + 1) ) - image.getpixel( (x , y) )  ) <= epsilon:
			if not Q.isInside( (x, y + 1) ) and not (x , y + 1) in s:
				Q.enque( (x , y + 1) )

		if y > 0 and abs(  image.getpixel( (x , y - 1) ) - image.getpixel( (x , y) )  ) <= epsilon:
			if not Q.isInside( (x , y - 1) ) and not (x , y - 1) in s:
				Q.enque( (x , y - 1) )

		if t not in s:
			s.append( t )

	dest_image.load()
	putpixel = dest_image.im.putpixel

	for i in s:
		putpixel(i , 255)

	# showImage(dest_image)
	# showImage(src_image)

def inc_seed_count():
	global seed_count
	seed_count += 1

if __name__ == "__main__":
	root = Tk()

	#setting up a tkinter canvas with scrollbars
	frame = Frame(root, bd=2, relief=SUNKEN)
	frame.grid_rowconfigure(0, weight=1)
	frame.grid_columnconfigure(0, weight=1)
	xscroll = Scrollbar(frame, orient=HORIZONTAL)
	xscroll.grid(row=1, column=0, sticky=E+W)
	yscroll = Scrollbar(frame)
	yscroll.grid(row=0, column=1, sticky=N+S)
	canvas = Canvas(frame, bd=0, xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
	canvas.grid(row=0, column=0, sticky=N+S+E+W)
	xscroll.config(command=canvas.xview)
	yscroll.config(command=canvas.yview)
	frame.pack(fill=BOTH,expand=1)

	#adding the image
	#File = askopenfilename(parent=root, initialdir="C:/",title='Choose an image.')
	pixelVal = 10
	img1 = Image.open('sar5.png').convert('L').resize((200, 200))
	img1_copy = img1.resize((200, 200))
	
	for i in range (img1_copy.size[0] ):
		for j in range (img1_copy.size[1] ):
			img1_copy.im.putpixel( (i , j) , 0 )

	img2 = ImageTk.PhotoImage(img1)
	canvas.create_image(0,0,image=img2,anchor="nw")
	canvas.config(scrollregion=canvas.bbox(ALL))

	#function to be called when mouse is clicked
	def printcoords(event):
		inc_seed_count()
		print (event.x,event.y)
		regiongrow(img1, img1_copy, 20, [event.x, event.y])

		
	# mouseclick event
	canvas.bind("<Button 1>", printcoords)

	root.mainloop()

	# Spectral Clustering
	#print("Checking...")
	#showImage(img1_copy)
	final_img = np.asarray(img1_copy)
	print(final_img.shape)
	#final_img = final_img[:, :]

	final_img = final_img.astype(float)
	mask = final_img.astype(bool)
	print(final_img.shape)

	final_img += 1 + 0.2 * np.random.randn(*final_img.shape)

	graph = image.img_to_graph(final_img, mask=mask)
	graph.data = np.exp(-graph.data / graph.data.std())

	labels = spectral_clustering(graph, n_clusters=seed_count, eigen_solver='arpack')
	label_im = np.full(mask.shape, -1.)
	label_im[mask] = labels

	showImage(img1)
	showImage(img1_copy)
	plt.imshow(label_im, cmap='gray')

	plt.show()
