import os
directory = '/home/chiheem/workspace/miniScene/images/train'

list = []

# Get all the folders' names
for letter in [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]:
    subdir = directory+'/'+letter
    for item in [name for name in os.listdir(subdir) if os.path.isdir(os.path.join(subdir, name))]:
        list.append(item)

list.sort()

# Write labels
f = open('labels.txt', 'w')
for item in list:
    f.write("%s\n" % item)

print("generateLabels.py completed executing successfully\n")
