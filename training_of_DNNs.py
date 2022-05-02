''' comment if you can train networks on the GPU '''

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from numpy import *
from numpy import linalg as la
from scipy import linalg as sa
from scipy import special as ss
from scipy.special import factorial

import matplotlib.pyplot as plt

from scipy.optimize import minimize

import keras
from keras import optimizers
from keras import initializers
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Conv2D, Flatten, Reshape, AveragePooling2D,UpSampling1D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint,EarlyStopping

''' get a random state distributed according to the Hilbert-Schmidt measure'''
def randomHS(dim):
	matG = random.normal(0,1,[dim,dim]) + 1j*random.normal(0,1,[dim,dim])
	rho = (matG@matG.transpose().conjugate())/trace(matG@matG.transpose().conjugate())
	return rho
	
''' basis of the d**2-1 hermitian matrices on the d-dimensional Hilbert space'''
def herbasis(dim):
    pom1 = zeros([1,dim,dim])+1j*zeros([1,dim,dim])
    pom1[0] = identity(dim)
    arrays = [dot(transpose(pom1[0][[i]]),pom1[0][[i]]) for i in range(dim-1)]
    pom = stack(arrays,axis=0)
    her = concatenate((pom1,pom),axis=0)
    arrays = [dot(transpose(her[0][[i]]),her[0][[j]])+dot(transpose(her[0][[j]]),her[0][[i]]) for i in range(dim) for j in range(i+1,dim)]
    pom = stack(arrays,axis=0)
    her = concatenate((her,pom),axis=0)
    arrays = [-1j*dot(transpose(her[0][[i]]),her[0][[j]])+1j*dot(transpose(her[0][[j]]),her[0][[i]]) for i in range(dim) for j in range(i+1,dim)]
    pom = stack(arrays,axis=0)
    pom = concatenate((her,pom),axis=0)
    return pom

def gellmann(Q,dim):
    q = zeros([dim**2,dim,dim])+1j*zeros([dim**2,dim,dim])
    for i in range(dim**2):
        v = Q[i]
        for j in range(0,i):
            v = v-trace(v@q[j])*q[j]
        q[i] = v/sqrt(trace(v@v))
    q[0] = identity(dim)
    return q

''' real vector representation of quantum states and POVM'''
def bellsFromRho(rho,A):
    l = shape(A)[0]
    return array([real(trace(rho@A[i])) for i in range(l)])

def rhoFromBells(bells,A):
    global dim
    l = shape(A)[0]
    return 1/dim*(identity(dim)+sum([bells[i]*A[i] for i in range(l)],axis=0))

def rhoFromBellsG(bells,A):
    global dim
    l = shape(A)[0]
    return 1/dim*(sum([bells[i]*A[i] for i in range(l)],axis=0))

''' square root measurement '''
def srm(m):
    global dim
    mstates = randompure(dim,m)
    mstates = reshape(mstates, (m,dim,dim))
    Sop = sum(mstates,axis=0)
    Sop = sa.sqrtm(la.inv(Sop))
    pom = [(Sop@mstates[i])@Sop for i in range(m)]
    return pom

''' compute the probability distribution given a quantum state and measurement '''
def probdists(stav,povm):
    l = shape(povm)[0]
    probtrue = array([real(trace(stav@povm[i])) for i in range(l)])
    probtrue = probtrue/sum(probtrue)
    return probtrue

''' calculate the measurement matrix '''
def Cmat(pom,G):
    global dim
    l = shape(pom)[0]
    matC = zeros((shape(pom)[0],dim**2))
    for m in range(l):
        for n in range(dim**2):
            matC[m,n]=trace(pom[m]@G[n])
    return 1/dim*matC

''' Cholesky decomposition '''
def choFromRho(rho):
    noStates = shape(rho)[0]
    choVec = zeros((noStates,dim**2))
    for n in range(noStates):
        L = la.cholesky(rho[n])
        imL = imag(L)
        reL = real(L)
        k_i = 0
        for i in range(dim):
            for j in range(0,i+1):
                choVec[n,k_i] = reL[i,j]
                k_i+=1

        for i in range(1,dim):
            for j in range(i):
                choVec[n,k_i] = imL[i,j]
                k_i+=1
    return choVec

def rhoFromCho(choVec):
    rhoC = zeros((dim,dim),complex)
    k_i=0
    for i in range(dim):
        for j in range(0,i+1):
            rhoC[i,j] += choVec[k_i]
            k_i+=1

    for i in range(1,dim):
        for j in range(i):
            rhoC[i,j] += 1j*choVec[k_i]
            k_i+=1
    return dot(rhoC,rhoC.T.conj())/trace(dot(rhoC,rhoC.T.conj()))

def probDistN(pl):
    x = random.multinomial(1,pl)
    return dim**2+where(x == 1)[0][0]

l = cbrt(list(range(100000)))
pl = l/sum(l)

dims = [3,5,7,9]

version = 'v9'

''' training of the deep networks '''
for dim in dims:

	''''''
	# ~ dim = 9
	Q = herbasis(dim)
	GAll = gellmann(Q,dim)*sqrt(dim)
	G = GAll[1::]
	GAll[0] = identity(dim)
	
	'''nStates0 - number of probability distribution which will be sampled and then fed into the networks'''
	nStates = 600000
	nStates0 = 200000
	
	nPom = dim**2
	
	'''load pre-generated square root measurement with dim**2 outputs'''
	file=load('SRMpom'+str(dim**2)+'.npz')
	pom=file['arr_0']
	# ~ pom = srm(nPom)
	# ~ savez('SRMpom'+str(dim**2)',pom)
	# ~ print('nPom=',nPom)
	
	'''generate training and validation sets '''
	rhoList = array([ randomHS(dim) for n in range(nStates) ])
	rhoList0 = array([ randomHS(dim) for n in range(nStates0) ])
	
	probsTrue = array([probdists(rhoList[n],pom) for n in range(nStates)])
	probsTrue0 = array([probdists(rhoList0[n],pom) for n in range(nStates0)])
	
	rN = array([ probDistN(pl) for n in range(nStates0) ])
	probsExp = array([ random.multinomial(rN[n],probsTrue0[n])/rN[n] for n in range(nStates0) ])
	
	'''choose between learning the Bell parameters or the Cholsky representation'''
	# ~ sL = array([ bellsFromRho(rhoList[n],GAll) for n in range(nStates) ])/sqrt(dim-1)
	# ~ sL0 = array([ bellsFromRho(rhoList0[n],GAll) for n in range(nStates) ])/sqrt(dim-1)

	choV = choFromRho(rhoList)
	choV0 = choFromRho(rhoList0)
	
	
	probsExp2 = concatenate((probsTrue,probsExp))
	choV2 = concatenate((choV,choV0))
	# ~ sL2 = concatenate((sL,sL0))
	
	indxs = random.shuffle(arange(nStates+nStates0))
	probsExp2 = probsExp2[indxs][0]
	choV2 = choV2[indxs][0]
	# ~ sL2 = sL2[indxs][0]

	'''definition of our deep model'''
	def myNet():
		model = Sequential()
		
		model.add(Dense(200,
		input_shape=(shape(pom)[0],),
		activation='relu',
		kernel_initializer=keras.initializers.glorot_normal(seed=42)))
		
		model.add(Dense(180, activation='relu',kernel_initializer=keras.initializers.glorot_normal(seed=42)))
		model.add(Dense(180, activation='relu',kernel_initializer=keras.initializers.glorot_normal(seed=42)))
		model.add(Dense(160, activation='relu',kernel_initializer=keras.initializers.glorot_normal(seed=42)))
		model.add(Dense(160, activation='relu',kernel_initializer=keras.initializers.glorot_normal(seed=42)))
		model.add(Dense(160, activation='relu',kernel_initializer=keras.initializers.glorot_normal(seed=42)))
		model.add(Dense(160, activation='relu',kernel_initializer=keras.initializers.glorot_normal(seed=42)))
		model.add(Dense(100, activation='relu',kernel_initializer=keras.initializers.glorot_normal(seed=42)))
		
		model.add(Dense(dim**2, activation='tanh',kernel_initializer=keras.initializers.glorot_normal(seed=42)))
		
		model.compile(loss='mse', optimizer="Nadam", metrics=['mean_squared_error'])
		
		history = model.fit(x_train, y_train,
							batch_size=batch_size, 
							epochs=epochs,
							verbose=0,
							validation_data=(x_val, y_val)
							)
	
		return( model, history.history['val_mean_squared_error'][-1] )
		
	
	nStates2 = nStates+nStates0
	
	'''create inputs and true outputs for the deep network'''
	x_train = probsExp2[0:int(nStates2*0.8)]
	x_val = probsExp2[int(nStates2*0.8):int(nStates2)]
	
	# ~ y_train = sL2[0:int(nStates2*0.8)]
	# ~ y_val = sL2[int(nStates2*0.8):int(nStates2)]
	
	y_train = choV2[0:int(nStates2*0.8)]
	y_val = choV2[int(nStates2*0.8):int(nStates2)]
	
	
	'''uncomment to pre-train a few networks to then start training from the best possible inicialization'''
	# ~ batch_size = 200
	# ~ epochs = 50

	
	# ~ bestModel,bestERR = myNet()
	# ~ bestModel.save('bestModelCho'+str(dim**2)+version+'.h5')
	
	# ~ for n in range(5):
	    # ~ print(n)
	    # ~ currentModel,currentERR = myNet()
	    # ~ if currentERR<bestERR:
	        # ~ bestERR=currentERR
	        # ~ bestModel=currentModel
	        # ~ bestModel.save('bestModelCho'+str(dim**2)+version+'.h5')
	        # ~ print(bestERR)
	
	
	'''training process'''
	currentModel = load_model('bestModelCho'+str(dim**2)+version+'.h5')
	
	filepath='currentModelCho'+str(dim**2)+version+'.h5'
	
	checkpoint = ModelCheckpoint(filepath, monitor='val_mean_squared_error', verbose=0, save_best_only=True, mode='min')
	# ~ reduce_lr = ReduceLROnPlateau(monitor="val_mean_squared_error", factor=0.99, patience=50, verbose=0)
	earlystopping = EarlyStopping(monitor="val_mean_squared_error", mode="min", verbose=0, patience=100)
	callbacks_list = [checkpoint,earlystopping]
	
	history = currentModel.fit(x_train, y_train,
		batch_size=100,
		epochs=3000,
		verbose=1,
		validation_data=(x_val, y_val),
		callbacks = callbacks_list)
	
	currentModel.save('bestModelCho'+str(dim**2)+version+'.h5')
	
	mse = history.history['mean_squared_error']
	val_mse = history.history['val_mean_squared_error']
	# ~ savetxt('errorsDim'+str(dim)+'Cho'+version+'.txt',[mse,val_mse],fmt='%s')

































