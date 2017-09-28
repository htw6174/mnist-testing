import os

def fill_feed_dict(image_path, images_placeholder, labels_placeholder):
    image = open(image_path, mode = 'rb')
    image.read(13) #strip out header
    #load pixels into tensor
    image_data = []
    for i in range(28):
        for j in range(28):
            value = int.from_bytes(image.read(1), byteorder = 'big') #read value of red byte
            image_data.append(float(value) / 255) #convert to normalized vector
            image.read(2) #skip green and blue bytes
    image_label = num_to_label(int(image_path[0])) #convert first character of file name to a label
    feed_dict = {
        images_placeholder: [image_data], #list shape of [batch, 784]
        labels_placeholder: [image_label] #list shape of [batch, 10]
    }
    return feed_dict

#only supports creating one example image of each digit, new images will be overwritten
def make_images(images, labels, path = "./MNIST_generated_data/"):
    assert len(images) == len(labels)
    #make sure path ends with a '/', because it should be a directory
    if path[-1] != '/':
        path.join('/')

    #check that path exists
    if not os.path.exists(path):
        os.makedirs(path)

    #make each image
    for i in range(len(images)):
        make_image(images[i], labels[i], path)

def make_image(image, label, path):
    #name file according to label
    name = "error_no_label"
    for i in range(10):
        if label[i] == 1:
            name = i
    path = path + "{}.pgm".format(name)

    #convert image array to bytes
    image_bytes = b''
    for i in image:
        b = int(i*255).to_bytes(1, byteorder = 'big')
        image_bytes += b

    #create and write to file
    with open(path, 'wb') as image_file:
        image_file.write(bytes('P5\n28\n28\n255\n', 'ascii')) #'magic number' for pgm files, width, height, maximum gray value
        image_file.write(image_bytes)

def print_to_console(image_path):
    image = open(image_path, mode = 'rb')
    magic_number = image.readline().decode("utf-8").strip()
    width = image.readline().decode("utf-8").strip()
    height = image.readline().decode("utf-8").strip()
    maxvalue = image.readline().decode("utf-8").strip()
    print("Magic Number: {}\nWidth: {}\nHeight: {}\nMax Pixel Value: {}".format(magic_number, width, height, maxvalue))
    print("Data: ")
    pixels = 0
    for i in range(int(height)):
        line = ""
        for j in range(int(width)):
            char = ''
            value = int.from_bytes(image.read(1), byteorder = 'big')
            if value == 0:
                char = '-'
            else:
                char = '0'
            line += char
            pixels += 1
            if magic_number == 'P6': #if using the P6 format which includes 2 extra bytes per pixel for color
                image.read(2)
        print(line)
    print("Data Length: " + str(pixels) + '\n')
    image.close

def num_to_label(num):
    assert num >= 0 and num <= 9
    labels = []
    for i in range(10):
        if num == i:
            labels.append(1)
        else:
            labels.append(0)
    return labels

if __name__ == "__main__":
    imagesVar = 0
    labelsVar = 1
    feed_dict = fill_feed_dict("8.pgm", imagesVar, labelsVar)
    print(feed_dict[labelsVar])
    print_to_console("8.pgm")
    make_images(feed_dict[imagesVar], feed_dict[labelsVar])
    print_to_console("./MNIST_generated_data/8.pgm")
