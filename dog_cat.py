from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json

##creating the convolutional network

classificador = Sequential()
classificador.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2)))
classificador.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2)))

classificador.add(Flatten())

##creating the dense network

classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 1, activation = 'sigmoid'))

##setting the compiler parameters
classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                      metrics = ['accuracy'])
#preprocessing the images from dataset
gerador_treinamento = ImageDataGenerator(rescale = 1./255, rotation_range = 7,
                                         horizontal_flip = True,
                                         shear_range = 0.2,
                                         height_shift_range = 0.07,
                                         zoom_range = 0.2)
gerador_teste = ImageDataGenerator(rescale = 1/255)

base_treinamento = gerador_treinamento.flow_from_directory('dataset/training_set',
                                                           target_size = (64, 64),
                                                           batch_size = 32,
                                                           class_mode = 'binary')
base_teste = gerador_teste.flow_from_directory('dataset/test_set',
                                                           target_size = (64, 64),
                                                           batch_size = 32,
                                                           class_mode = 'binary')
##training the model created above
classificador.fit_generator(base_treinamento, steps_per_epoch = 4000,   ## o numero de epocas neste caso
                            epochs = 5, validation_data = base_teste,   ## é o numero de imagens no dataset
                            validation_steps = 1000)                    ## treino, validação o numero de teste 
                                                                        ## pode ser dividido para diminuir o 
                                                                        ## tempo de treinamento, diminui precisão
## saving the neural network structure and weights
classificador_json = classificador.to_json()
with open('classificador_cat_dog.json', 'w') as json_file:
    json_file.write(classificador_json)
classificador.save_weights('classificador_cat_dog.h5')

#load the neural network structure

arquivo = open('classificador_cat_dog.json', 'r')
estrutura_rede = arquivo.read()
arquivo.close()

#load the neural network weights

classificador = model_from_json(estrutura_rede)
classificador.load_weights('classificador_cat_dog.h5')

#load the image to be classified
img = input('Digite o caminho até a imagem a se analisar\n')

#preprocess the image to the network standards
imagem_teste = image.load_img(img,
                              target_size = (64,64))
imagem_teste = image.img_to_array(imagem_teste)
imagem_teste /= 255
imagem_teste = np.expand_dims(imagem_teste, axis = 0)
previsao = classificador.predict(imagem_teste)

#as a binary classification with 1 == cat and 0 == dog
if previsao > 0.5:
    print ('cat')
if previsao < 0.5:
    print('dog')
if previsao == 0.5:
    print('undefined')