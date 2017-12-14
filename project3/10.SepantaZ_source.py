from math import sqrt
##import tensorflow as tf
import numpy
from scipy import misc
import numpy as np
from sklearn import mixture
from PIL import Image
from sklearn.decomposition import PCA, SparsePCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn import cluster
import sklearn
#from sklearn.neural_network import MLPClassifier
#from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessClassifier
#from sklearn.tree import DecisionTreeClassifier
import matplotlib.image as mpimg



#face = misc.face()
#misc.imsave('/home/sepanta/Downloads/Paintings/1.TIF', face) # First we need to create the PNG file
#image = imread('/home/sepanta/Downloads/Paintings/1.TIF')
#image.reshape(38886952)
#image = image[0:784]
#print type(image)
#print image.shape


#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 2
batch_size = 15
multiplier = 1
size = 784*multiplier*multiplier
n_split = 5


#size = 100000
R_min_size = 7246656
min_size = 100000#n_split*size/2#784*2

def get_images(img_size):
    s_images = []
    nsh_image = 3
    nsv_image = 3
    images = []
    test = []
    y= []
    s_y = []
    for i in range(11):
        if i == 7 or i == 8:
            #image = misc.imread('/home/sepanta/Downloads/Paintings/'+str(i+1)+'.jpg', flatten=True)
            image = Image.open('C:\\Users\\sepanta\\Documents\\University\\CSIC5011\\Project3\\Paintings\\'+str(i+1)+'.jpg').convert('LA')
        else:
            #image = misc.imread('/home/sepanta/Downloads/Paintings/'+str(i+1)+'.TIF', flatten=True)
            image = Image.open('C:\\Users\\sepanta\\Documents\\University\\CSIC5011\\Project3\\Paintings\\'+str(i+1)+'.TIF').convert('LA')

        rf = sqrt(image.size[0]*image.size[1]/img_size)
        new_size = (int)(image.size[0]/rf) +1, (int)(image.size[1]/rf)+1
        image = image.resize(new_size, Image.ANTIALIAS)

        imgwidth, imgheight = image.size
        im = image
        height = imgheight/nsh_image
        width = imgwidth/nsv_image
        for i in range(0,imgheight,height):
            for j in range(0,imgwidth,width):
                box = (j, i, j+width, i+height)
                a = im.crop(box)
                s_image = np.array(a.getdata()).reshape(-1)
                s_image =s_image[:2*img_size/(nsv_image*nsh_image+2)]
                s_images.append(s_image)
                s_y.append(0)

        image = np.array(image.getdata())
        image = image.reshape(-1)
        image = image[:2*img_size]
        #print image.shape
        #images.append(image)
        images.append(image)
        #epoch_x.append(image)
        y.append(0)

    for i in range(10):
        #image = misc.imread('/home/sepanta/Downloads/Paintings/N'+str(i+1)+'.jpg', flatten=True)

        if i == 9:
            image = Image.open('C:\\Users\\sepanta\\Documents\\University\\CSIC5011\\Project3\\Paintings\\N'+str(i+1)+'.TIF').convert('LA')
        else:
            image = Image.open('C:\\Users\\sepanta\\Documents\\University\\CSIC5011\\Project3\\Paintings\\N'+str(i+1)+'.jpg').convert('LA')
        rf = sqrt(image.size[0]*image.size[1]/img_size)
        new_size = (int)(image.size[0]/rf) +1, (int)(image.size[1]/rf)+1
        image = image.resize(new_size, Image.ANTIALIAS)

        imgwidth, imgheight = image.size
        im = image
        height = imgheight/nsh_image
        width = imgwidth/nsv_image
        for i in range(0,imgheight,height):
            for j in range(0,imgwidth,width):
                box = (j, i, j+width, i+height)
                a = im.crop(box)
                s_image = np.array(a.getdata()).reshape(-1)
                s_image =s_image[:2*img_size/(nsv_image*nsh_image+2)]
                s_images.append(s_image)
                s_y.append(1)

        image = np.array(image.getdata())
        image = image.reshape(-1)
        image = image[:2*img_size]
        #print image.shape

        images.append(image)
        #epoch_x.append(image)
        y.append(1)


    for i in range(6):
        #image = misc.imread('/home/sepanta/Downloads/Paintings/N'+str(i+1)+'.jpg', flatten=True)

        if i == 5:
            image = Image.open('C:\\Users\\sepanta\\Documents\\University\\CSIC5011\\Project3\\Paintings\\R'+str(i+1)+'.png').convert('LA')
        else:
            image = Image.open('C:\\Users\\sepanta\\Documents\\University\\CSIC5011\\Project3\\Paintings\\R'+str(i+1)+'.TIF').convert('LA')
        rf = sqrt(image.size[0]*image.size[1]/img_size)
        new_size = (int)(image.size[0]/rf) +1, (int)(image.size[1]/rf)+1
        image = image.resize(new_size, Image.ANTIALIAS)

        imgwidth, imgheight = image.size
        im = image
        height = imgheight/nsh_image
        width = imgwidth/nsv_image
        for i in range(0,imgheight,height):
            for j in range(0,imgwidth,width):
                box = (j, i, j+width, i+height)
                a = im.crop(box)
                s_image = np.array(a.getdata()).reshape(-1)
                s_image =s_image[:2*img_size/(nsv_image*nsh_image+2)]
                s_images.append(s_image)
                s_y.append(1)

        image = np.array(image.getdata())
        image = image.reshape(-1)
        image = image[:2*img_size]
        #print image.shape

        images.append(image)
        #epoch_x.append(image)
        y.append(2)

    return images, y


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
    #                        size of window         movement of window
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')



def convolutional_neural_network(x):

    keep_rate = 0.8
    keep_prob = tf.placeholder(tf.float32)

    weights = {'W_conv1':tf.Variable(tf.random_normal([5,5,1,32])),
               'W_conv2':tf.Variable(tf.random_normal([5,5,32,64])),
               'W_fc':tf.Variable(tf.random_normal([7*7*64*multiplier*multiplier,1024])),
               'out':tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
               'b_conv2':tf.Variable(tf.random_normal([64])),
               'b_fc':tf.Variable(tf.random_normal([1024])),
               'out':tf.Variable(tf.random_normal([n_classes]))}

    x = tf.reshape(x, shape=[-1, 28*multiplier, 28*multiplier, 1])

    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)
    
    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2,[-1, 7*7*multiplier*multiplier*64])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out'])+biases['out']

    return output

def accuracy(result, split_count = 1):
    count = 0
    real_count = 11*split_count
    for i in range(real_count):
        if result[i] == 0:
            count = count + 1

    fake_group = 0
    wrong_count = 0
    if count > real_count/2:
        fake_group = 1
        wrong_count = real_count - count
    else:
        wrong_count = count

    fake_count = 10*split_count
    for i in range(fake_count):
        if result[i+real_count] != fake_group:
            wrong_count = wrong_count + 1

    return 1-(wrong_count/float(real_count+fake_count))



def do_clustering(images, n_components, neibghbour, method):
    resultSVM = []
    resultnn = []
    resultRBF = []
    resultDT = []
    for i in range(21):
        clf= LinearSVC(random_state=0)
        nn = MLPClassifier()
        dt = DecisionTreeClassifier()
        rbf = GaussianProcessClassifier()

        temp = images.tolist()
        del temp[i]
        l_y = list(y)
        del l_y[i]

        clf.fit(temp, l_y)
        nn.fit(temp, l_y)
        dt.fit(temp, l_y)
        rbf.fit(temp, l_y)

        resultSVM.append(clf.predict(images[i:i+1])) 
        resultnn.append(nn.predict(images[i:i+1])) 
        resultDT.append(dt.predict(images[i:i+1])) 
        resultRBF.append(dt.predict(images[i:i+1])) 

        del clf
        del nn
        del dt
        del rbf

    svm = accuracy(resultSVM)
    nn = accuracy(resultnn)
    dt = accuracy(resultDT)
    rbf = accuracy(resultRBF)
    print(str(neighbour) + ", " + str(n_components) + ", " +  str(svm) + " " + method + "SVM")
    print(str(neighbour) + ", " + str(n_components) + ", " +  str(nn) + " " + method + "nn")
    print(str(neighbour) + ", " + str(n_components) + ", " +  str(dt) + " " + method + "DT")
    print(str(neighbour) + ", " + str(n_components) + ", " +  str(rbf) + " " + method + "RBF")


    kmeans = KMeans(n_clusters=2, random_state=0).fit(images)
    kmeans = accuracy(kmeans.labels_)
    print(str(neighbour) + ", " + str(n_components) + ", " +  str(kmeans) + " " + method + "KMeans")


    ag = cluster.AgglomerativeClustering(n_clusters=2).fit(images)
    ag = accuracy(ag.labels_)
    print(str(neighbour) + ", " + str(n_components) + ", " +  str(ag) + " " + method + "AG")

    gmm = mixture.GaussianMixture(n_components=2).fit(images)
    gmm = accuracy(gmm.predict(images))
    print(str(neighbour) + ", " + str(n_components) + ", " +  str(gmm) + " " + method + "GMM")

    bgmm = mixture.BayesianGaussianMixture(n_components=2).fit(images)
    bgmm = accuracy(bgmm.predict(images))
    print(str(neighbour) + ", " + str(n_components) + ", " +  str(bgmm)  + " " + method + "BGMM")

    return svm, nn, dt, rbf, kmeans, ag, gmm, bgmm



def train_neural_network(images, i):
    x = tf.placeholder('float', [None, size])
    y = tf.placeholder('float')

    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    s_split = 784#2*min_size/n_split
    hm_epochs = 21#min_size/size
    test_indx = [i]#np.random.randint(21, size=i)
    print test_indx
    #train_indx = [0, 1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16]
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(21):
            if epoch in test_indx:
                continue
            #if epoch == i:
                #continue

            epoch_loss = 0
            for _ in range(1):#int(mnist.train.num_examples/batch_size)):
                epoch_x = []
                epoch_y = []
                for indx in range(n_split):
                    epoch_x.append(images[epoch][indx*s_split:(indx+1)*s_split])
                    if epoch>10:
                        epoch_y.append([1, 0])
                    else:
                        epoch_y.append([0, 1])

                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            #print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        #image = misc.imread('/home/sepanta/Downloads/Paintings/N'+str(9+1)+'.TIF', flatten=True)
        #image = image.reshape(-1)


        #for i in range(min_size/size):
            #image_s = image[i*size:(i+1)*(size)]
            #test_l.append([0, 1])
            #test.append(image_s)
        #image = Image.open('/home/sepanta/Downloads/Paintings/N'+str(9+1)+'.TIF').convert('LA')
        #image = image.resize(new_size, Image.ANTIALIAS)
        #image = np.array(image.getdata())
        #image = image.reshape(-1)
        #image = image[:2*min_size]
        #print image.shape
        #images.append(image)
        test = []
        test_l = []
        for t_indx in test_indx:
            for indx in range(n_split):
                test.append(images[t_indx][indx*s_split:(indx+1)*s_split])
                if t_indx>10:
                    test_l.append([1, 0])
                else:
                    test_l.append([0, 1])

        #test.append(image)
        #test_l.append([0, 1])
        correct = accuracy.eval({x:test, y:test_l})
        print('Accuracy:',correct)
        del x
        del y
        del prediction
        return correct


method = 1
if method == 0:
    images, y = get_images()
    correct = 0
    repeat = 1
    for i in range(21):
        for j in range(repeat):
            correct = correct + train_neural_network(images, i)

    print correct/(21*repeat)

if method == 1:
    plot = 1
    if plot:
        images, y= get_images(10000)
        o_images= images
        

        images = PCA(2).fit_transform(o_images)
        x = images[:,0]
        y = images[:,1]
        colors = [1]*11 + [20]*10 + [10]*6
        plt.scatter(x, y, c=colors, alpha=0.7, s=80)

#        for i in range(11):
            #plt.annotate(str(i), (x[i],y[i]))
        #for i in range(10):
            #plt.annotate(str(i), (x[i+11],y[i+11]))
            
        plt.savefig('pcaP.png')
        plt.clf()

        
        images = sklearn.manifold.Isomap(15, 2).fit_transform(o_images)
        x = images[:,0]
        y = images[:,1]
        colors = [1]*11 + [20]*10 + [10]*6
        plt.scatter(x, y, c=colors, alpha=0.7, s=80)
        
            
        plt.savefig('IsomapP.png')
        plt.clf()
        #plt.show()

        images = sklearn.manifold.LocallyLinearEmbedding(10, 2, method='standard').fit_transform(o_images)
        x = images[:,0]
        y = images[:,1]
        colors = [1]*11 + [20]*10 + [10]*6
        plt.scatter(x, y, c=colors, alpha=0.7, s=80)

        plt.savefig('LLeP.png')
        plt.clf()
        #plt.show()

if 0:
    fh = open("result.txt", "w")
    repeat = 20
    for img_size in [10000, 100000, 1000000]:
        images, y= get_images(img_size)
        o_images= images

        for neighbour in [10, 15, 20]:
            for n_components in [2, 5, 10, 15, 30, 50, 100, 1000]:
                p_s = [0, 0, 0, 0, 0, 0, 0, 0]
                p_i = [0, 0, 0, 0, 0, 0, 0, 0]
                p_l = [0, 0, 0, 0, 0, 0, 0, 0]
                for _ in range(repeat):
                    test = []
                    train = []

                    images = PCA(n_components).fit_transform(o_images)
                    pca_t = do_clustering(images, n_components, neighbour, "PCA")


                    if n_components < 30:
                        images = sklearn.manifold.Isomap(neighbour, n_components).fit_transform(o_images)
                        isomap_t = do_clustering(images, n_components, neighbour, "Isomap")

                        images = sklearn.manifold.LocallyLinearEmbedding(neighbour, n_components, method='standard').fit_transform(o_images)
                        lle_t = do_clustering(images, n_components, neighbour, "LLE")

                    for i in range(8):
                        p_s[i] = p_s[i] + pca_t[i]
                        p_i[i] = p_i[i] + isomap_t[i]
                        p_l[i] = p_l[i] + lle_t[i]

                for i in range(8):
                    p_s[i] = p_s[i]/float(repeat)
                    fh.write(str(img_size) + ", " + str(neighbour) + ", " + str(n_components) + ", " +  str(p_s[i])  + " PCA " + str(i)+"\n")

                    p_i[i] = p_i[i]/float(repeat)
                    fh.write(str(img_size) + ", " + str(neighbour) + ", " + str(n_components) + ", " +  str(p_i[i])  + " Isomap " + str(i)+"\n")

                    p_l[i] = p_l[i] /float(repeat)
                    fh.write(str(img_size) + ", " + str(neighbour) + ", " + str(n_components) + ", " +  str(p_l[i])  + " LLE " + str(i)+"\n")

                fh.close()
                fh = open("result.txt", "a")


















