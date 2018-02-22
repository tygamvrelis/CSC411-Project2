from pylab import *
import urllib
import os
import urllib.request

def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result


def downloadImages(dataset):
    '''NOTE: WE BASICALLY COMBINED OUR UNCROPPED IMAGE REPOSITORIES FROM PROJECT 1
    SO THIS CODE IS JUST AN AMALGAMATION OF OUR PROJECT 1 DOWNLOAD CODE. HASHING
    IS DONE IN image_processing.py.


    Downloads the images from the dataset into os.getcwd()/../Data/Faces/uncropped.

    Returns:
    none

    Arguments:
    dataset -- a string representing the relative path of the dataset
    '''
    if not os.path.exists("../Data/Faces/uncropped"):
        os.makedirs("../Data/Faces/uncropped")

    act = list(set([a.split("\t")[0] for a in open(dataset).readlines()]))
    for a in act:
        name = a.split()[1].lower()
        i = 0
        for line in open(dataset):
            if a in line:
                filename = name + str(i) + '.' + line.split()[4].split('.')[-1]

                if not os.path.isfile("../Data/Faces/uncropped/" + filename):
                    if (timeout(urllib.request.urlretrieve, (line.split()[4], "../Data/Faces/uncropped/" + filename), {},
                                45) == False):
                        print("FAILED to downloaded image: " + filename)
                    else:
                        print("Downloaded image: " + filename)
                else:
                    print("Skipped (already downloaded): " + filename)

                i += 1


    testfile = urllib.URLopener()
    #First attempt at downloading all images

    act = ['Gerard Butler', 'Fran Drescher', 'Michael Vartan', 'America Ferrera', 'Daniel Radcliffe', 'Kristin Chenoweth']
    for a in act:
       name = a.split()[1].lower()
       i = 0
       for line in open(dataset):
           if a in line:
              filename = name + str(i) + '.' + line.split()[4].split('.')[-1]
              if not os.path.isfile("../Data/Faces/uncropped/" + filename):
                   timeout(testfile.retrieve, (line.split()[4], "../Data/Faces/uncropped/" + filename), {}, 30)
                   print filename

              if not os.path.isfile("../Data/Faces/uncropped/" + filename):
                   i+=1     #I want the index of the file aligned to its position in the text file for when I crop it
                   continue
              i += 1


    #I had insufficient images of Baldwin, Gilpin and Bracco so I used urllib.retrieve to download the images that
    #couldn't be obtained due to error 301, resource was moved. These images took far longer to download than normal.

    act = ['Alec Baldwin','Peri Gilpin', 'Lorraine Bracco']
    for a in act:
       name = a.split()[1].lower()
       i = 0
       for line in open(dataset):
           if a in line:
              filename = name + str(i) + '.' + line.split()[4].split('.')[-1]
              if not os.path.isfile("../Data/Faces/uncropped/" + filename):
                  timeout(urllib.urlretrieve, (line.split()[4], "../Data/Faces/uncropped/" + filename), {}, 30)

              if not os.path.isfile("../Data/Faces/uncropped/" + filename):
                   i+=1
                   continue

              i += 1
downloadImages("../Data/Faces/actors.txt")