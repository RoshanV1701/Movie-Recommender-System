import numpy as np 
import pandas as pd
from web.models import Myrating
import scipy.optimize 


def Myrecommend():
	def normalizeRatings(myY, myR):
    	# The mean is only counting movies that were rated
		Ymean = np.sum(myY,axis=1)/np.sum(myR,axis=1)
		Ymean = Ymean.reshape((Ymean.shape[0],1))
		return myY-Ymean, Ymean
	
	def flattenParams(myX, myTheta):
		return np.concatenate((myX.flatten(),myTheta.flatten()))
    
	def reshapeParams(flattened_XandTheta, mynm, mynu, mynf):
		assert flattened_XandTheta.shape[0] == int(mynm*mynf+mynu*mynf)
		reX = flattened_XandTheta[:int(mynm*mynf)].reshape((mynm,mynf))
		reTheta = flattened_XandTheta[int(mynm*mynf):].reshape((mynu,mynf))
		return reX, reTheta

	def cofiCostFunc(myparams, myY, myR, mynu, mynm, mynf, mylambda = 0.):
		myX, myTheta = reshapeParams(myparams, mynm, mynu, mynf)
		term1 = myX.dot(myTheta.T)
		term1 = np.multiply(term1,myR)
		cost = 0.5 * np.sum( np.square(term1-myY) )
    	# for regularization
		cost += (mylambda/2.) * np.sum(np.square(myTheta))
		cost += (mylambda/2.) * np.sum(np.square(myX))
		return cost

	def cofiGrad(myparams, myY, myR, mynu, mynm, mynf, mylambda = 0.):
		myX, myTheta = reshapeParams(myparams, mynm, mynu, mynf)
		term1 = myX.dot(myTheta.T)
		term1 = np.multiply(term1,myR)
		term1 -= myY
		Xgrad = term1.dot(myTheta)
		Thetagrad = term1.T.dot(myX)
    	# Adding Regularization
		Xgrad += mylambda * myX
		Thetagrad += mylambda * myTheta
		return flattenParams(Xgrad, Thetagrad)

	df=pd.DataFrame(list(Myrating.objects.all().values()))
	mynu=df.user_id.unique().shape[0]
	mynm=df.movie_id.unique().shape[0]
	mynf=10
	Y=np.zeros((mynm,mynu))
	for row in df.itertuples():
		Y[row[2]-1, row[4]-1] = row[3]
	R=np.zeros((mynm,mynu))
	for i in range(Y.shape[0]):
		for j in range(Y.shape[1]):
			if Y[i][j]!=0:
				R[i][j]=1

	Ynorm, Ymean = normalizeRatings(Y,R)
	X = np.random.rand(mynm,mynf)
	Theta = np.random.rand(mynu,mynf)
	myflat = flattenParams(X, Theta)
	mylambda = 12.2
	result = scipy.optimize.fmin_cg(cofiCostFunc,x0=myflat,fprime=cofiGrad,args=(Y,R,mynu,mynm,mynf,mylambda),maxiter=40,disp=True,full_output=True)
	resX, resTheta = reshapeParams(result[0], mynm, mynu, mynf)
	prediction_matrix = resX.dot(resTheta.T)
	return prediction_matrix,Ymean
	

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity

# Predefined list of all unique genres
preprocessed_genres = [
    'Animation', 'Action', 'Comedy', 'Biography', 'Crime', 'Drama',
    'Adventure', 'Fantasy', 'Mystery', 'Romance', 'Sci-Fi', 'Documentary',
    'Family', 'Adventure', 'Romance', 'Drama', 'Family', 'Crime', 'Drama',
    'Fantasy', 'Comedy', 'Comedy', 'Drama', 'Sci-Fi', 'Sci-Fi', 'Thriller',
    'War', 'Music', 'Mystery', 'Crime', 'Biography', 'Family', 'Horror',
    'Biography', 'Documentary', 'Romance', 'Horror', 'Comedy', 'Horror',
    'History', 'Mystery', 'Fantasy', 'Sport'
]


# Initialize and fit OneHotEncoder with the unique genres
encoder = OneHotEncoder()
encoder.fit(np.array(preprocessed_genres).reshape(-1, 1))

def split_genres(genre_string):
    return genre_string.split(' | ')

def calculate_movie_similarities(movie1_genre_string, movie2_genre_string):
    # Split genre strings into lists
    movie1_genres = split_genres(movie1_genre_string)
    movie2_genres = split_genres(movie2_genre_string)
    
    # Transform the genres of both movies into one-hot encoded vectors
    movie1_encoded = encoder.transform(np.array(movie1_genres).reshape(-1, 1))
    movie2_encoded = encoder.transform(np.array(movie2_genres).reshape(-1, 1))
    
    # Calculate cosine similarity between the two movies
    similarity_score = cosine_similarity(movie1_encoded, movie2_encoded)[0, 0]

    return similarity_score






