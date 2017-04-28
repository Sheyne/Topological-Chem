"csolid_hot.txt"
energies = np.array([sec.potential_energy for sec in sections])

print(energies[0])

X1 = groups[:,:,:3]/30
X2 = groups[:,:,3:]/10
Y = energies / energies[0]

X = (np.concatenate((X1, X2), axis=2))

from keras.models import Sequential
from keras.layers import Dense, Flatten

model = Sequential()
model.add(Flatten(input_shape=(172, 6)))
model.add(Dense(2000, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='mean_squared_error')

val_index = int(len(X) * 0.7)

model.fit(X[:val_index], Y[:val_index], shuffle=True, validation_split=0.0)

ypred = (model.predict(X[val_index:]) * energies[0]).reshape((-1,))
print(ypred)
ytest = Y[val_index:]

print(np.corrcoef(ytest, ypred))