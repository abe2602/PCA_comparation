import sklearn.datasets as dt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

NEW_NUMBER = 8
ORI_NUMBER = 13

#################### REDE ADAPTATIVA PCA ##############################
class Adaptative_PCA:

    def __init__(self, num_input, num_output):
        self.num_input = num_input
        self.num_output = num_output

        #Pesos
        self.wi = np.random.uniform(0, 0.5, (self.num_output, self.num_input)) 
        self.wi = 1/((self.wi.max() - self.wi.min())*(self.wi - self.wi.max()) + 1) #normaliza entre 0 e 1
        norm = np.linalg.norm(self.wi, axis=1) #Encontra a norma
        self.wi /= norm[:,None] #Normaliza

        self.wo = np.random.uniform(0, 0.5, (self.num_output, self.num_output)) 
        self.wo = np.tril(self.wo, -1)

    def feed_forward(self, data):
        self.output = np.dot(self.wi, data.T)

        for i in range(self.num_output):
            aux = np.dot(self.wo[i], self.output)
            self.output[i] += aux

        return self.output

    def train(self, data, interations = 4500):
        print("PCA Adapativa")
        lr=0.1
        slr=0.05
        tam_data = len(data)
        
        i = 0
        while i < interations:
            for j in range(tam_data):
                self.feed_forward(data[j])

                delta = lr*np.dot(self.output, data[j])
                self.wi += delta

                #Normaliza por colunas
                norm = np.linalg.norm(self.wi, axis=1)
                self.wi /= norm[:,None]

                aux = np.tril(np.dot(self.output, self.output.T), -1)
                delta = -slr*aux
                self.wo += delta

                #Atualiza parâmetros
                #pesos
                if(lr*0.9 < 0.0001):
                    lr = 0.0001
                else:
                    lr = lr*0.9

                #Pesos laterais
                if(slr*0.9 < 0.0001):
                    slr = 0.0001
                else:
                    slr = slr*0.9

            i+=1

    def project_data(self, data):
        output = np.zeros((len(data), NEW_NUMBER))

        for i in range(0, len(data)):
          output[i] = self.feed_forward(data[i]).T

        return output

#################### PCA CLÁSSICA ##############################
def biblePCA(data, target):
    x = data 
    y = target

    pca = PCA(NEW_NUMBER)
    pca.fit(x)
    B = pca.transform(x)
    return B

################### MLP ##################################
#Usarei o tensorflow com Keras para criar uma MLP

def MLP(data, targetOri, DIM):
	train, test, target_train, target_test = train_test_split(data, targetOri, test_size=0.20) #Divide as bases em treino e teste
	# create model
	model = keras.Sequential()
	model.add(keras.layers.Dense(12, input_dim=DIM, activation='relu'))
	model.add(keras.layers.Dense(DIM, activation='relu'))
	model.add(keras.layers.Dense(3, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	model.fit(train, target_train, epochs=200, batch_size = 10)
	loss, acc = model.evaluate(test, target_test)

	return acc

#################### MAIN ##############################
def main():

    #Dados do wine
    data, target = dt.load_wine(return_X_y=True)
    data = np.matrix(data)

    #Normaliza os dados
    norm = (data - np.mean(data, axis=0))/np.std(data, axis=0)

    #Inicializa a rede e treina
    pca_n = Adaptative_PCA(ORI_NUMBER, NEW_NUMBER)
    pca_n.train(norm)

    #Dados da nova projeção
    output = pca_n.project_data(norm)
    #show(output, target, "PCA Network")


    print("\nPCA Clássica")
    classicPCA = biblePCA(norm, target)
    #show(norm, target, "PCA Clássica")
    print(output)
    print("\n\n", classicPCA)
    
    #Transforma os target em tipos binários: 0 0 1 (caso seja da classe 3, pro exemplo)
    targetNormalized= keras.utils.to_categorical(target, 3)

    #Acurácias nas MLPs de cada um
    acc0 = MLP(output, targetNormalized, NEW_NUMBER)
    acc1 = MLP(classicPCA, targetNormalized, NEW_NUMBER)
    acc2 = MLP(norm, targetNormalized, ORI_NUMBER)
    print("pcaN: ", acc0, "pcaC: ", acc1, "ori: ", acc2)

if __name__ == '__main__':
	main()