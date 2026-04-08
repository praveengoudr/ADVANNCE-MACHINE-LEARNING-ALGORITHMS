from tensorflow.keras import layers, models

input_dim = 20

encoder = layers.Dense(10, activation="relu")
decoder = layers.Dense(input_dim, activation="sigmoid")

model = models.Sequential([encoder, decoder])
print("Autoencoder Ready")
