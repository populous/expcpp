import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from deap import base, creator, tools, algorithms
import random
import matplotlib.pyplot as plt

# 1D Heat Equation: ∂u/∂t = α ∂²u/∂x²

# Define the PINN Model
def create_pinn_model():
    model = models.Sequential([
        layers.InputLayer(input_shape=(2,)),  # Input is (x, t)
        layers.Dense(50, activation='tanh'),
        layers.Dense(50, activation='tanh'),
        layers.Dense(1)  # Output: temperature u(x,t)
    ])
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

# Define PDE residual (heat equation)
def heat_equation_residual(model, x, t, alpha):
    # Predictions
    u = model(np.hstack((x, t)))  # u(x, t)
    
    # Compute derivatives (simple finite difference)
    du_dt = np.gradient(u, t, axis=0)
    du_dx = np.gradient(u, x, axis=0)
    d2u_dx2 = np.gradient(du_dx, x, axis=0)
    
    # Heat equation residual
    residual = du_dt - alpha * d2u_dx2
    return residual

# Training the PINN
def train_pinn(model, X_train, y_train, alpha):
    for epoch in range(1000):
        with tf.GradientTape() as tape:
            # Predict the output
            u_pred = model(X_train)
            
            # Calculate the PDE residual
            residual = heat_equation_residual(model, X_train[:, 0], X_train[:, 1], alpha)
            
            # Loss function (Mean Squared Error)
            loss = tf.reduce_mean(tf.square(residual))
        
        grads = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return model

# Define the fitness function for Genetic Algorithm
def fitness_function(individual):
    alpha = individual[0]
    
    # Create PINN model
    model = create_pinn_model()
    
    # Generate training data (x, t)
    x = np.linspace(0, 1, 100)
    t = np.linspace(0, 1, 100)
    X_train = np.array(np.meshgrid(x, t)).T.reshape(-1, 2)
    
    # Train the model with the given alpha
    model = train_pinn(model, X_train, None, alpha)
    
    # Evaluate the fitness (mean squared error of residuals)
    residual = heat_equation_residual(model, X_train[:, 0], X_train[:, 1], alpha)
    mse = np.mean(np.square(residual))
    
    return mse,  # Return as a tuple

# Set up the Genetic Algorithm
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, 0.0, 1.0)  # Random float for alpha
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", fitness_function)

# Run the Genetic Algorithm
def run_ga():
    population = toolbox.population(n=10)
    algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=20, verbose=True)
    
    # Get best individual
    best_individual = tools.selBest(population, 1)[0]
    print("Best alpha:", best_individual[0])

# Execute the GA
run_ga()