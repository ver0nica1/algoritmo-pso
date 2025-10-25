# Importar división desde __future__ para compatibilidad Python 2/3
from __future__ import division
# Importar módulo random para generar números aleatorios
import random
# Importar módulo math para funciones matemáticas
import math

#--- COST FUNCTION ------------------------------------------------------------+
def func1(x):
    """Función objetivo original del código - Calcula la suma de cuadrados"""
    # Inicializar variable para acumular la suma total
    total = 0
    # Iterar sobre cada dimensión del vector x
    for i in range(len(x)):
        # Sumar el cuadrado de cada componente al total
        total += x[i]**2
    # Retornar el valor total calculado (fitness a minimizar)
    return total



#--- PARTICLE CLASS (ORIGINAL) ------------------------------------------------+
class Particle:
    def __init__(self, x0, num_dimensions):
        # Lista para almacenar la posición actual de la partícula
        self.position_i = []          # particle position
        # Lista para almacenar la velocidad actual de la partícula
        self.velocity_i = []          # particle velocity
        # Lista para almacenar la mejor posición personal encontrada
        self.pos_best_i = []          # best position individual
        # Variable para almacenar el mejor error personal (-1 indica no inicializado)
        self.err_best_i = -1          # best error individual
        # Variable para almacenar el error actual de la partícula
        self.err_i = -1               # error individual
        
        # Inicializar cada dimensión de la partícula
        for i in range(0, num_dimensions):
            # Generar velocidad inicial aleatoria entre -1 y 1
            self.velocity_i.append(random.uniform(-1, 1))
            # Establecer posición inicial basada en x0
            self.position_i.append(x0[i])
    
    def evaluate(self, costFunc):
        """Evaluar fitness actual - Calcula el error y actualiza el mejor personal"""
        # Calcular el error actual usando la función de costo
        self.err_i = costFunc(self.position_i)
        # Verificar si este es el mejor error personal encontrado hasta ahora
        if self.err_i < self.err_best_i or self.err_best_i == -1:
            # Guardar una copia de la posición actual como la mejor personal
            self.pos_best_i = list(self.position_i)
            # Actualizar el mejor error personal
            self.err_best_i = self.err_i
    
    def update_velocity(self, pos_best_g, num_dimensions):
        """Actualizar velocidad de la partícula usando la ecuación PSO"""
        # Peso de inercia - controla la influencia de la velocidad anterior
        w = 0.5       # constant inertia weight
        # Constante cognitiva - influencia del mejor personal
        c1 = 1        # cognative constant
        # Constante social - influencia del mejor global
        c2 = 2        # social constant
        
        # Actualizar velocidad en cada dimensión
        for i in range(0, num_dimensions):
            # Generar número aleatorio para componente cognitiva
            r1 = random.random()
            # Generar número aleatorio para componente social
            r2 = random.random()
            # Calcular componente cognitiva (atracción hacia mejor personal)
            vel_cognitive = c1 * r1 * (self.pos_best_i[i] - self.position_i[i])
            # Calcular componente social (atracción hacia mejor global)
            vel_social = c2 * r2 * (pos_best_g[i] - self.position_i[i])
            # Aplicar ecuación de actualización de velocidad PSO
            self.velocity_i[i] = w * self.velocity_i[i] + vel_cognitive + vel_social
    
    def update_position(self, bounds, num_dimensions):
        """Actualizar posición basada en la velocidad y aplicar límites"""
        # Actualizar posición en cada dimensión
        for i in range(0, num_dimensions):
            # Aplicar la velocidad a la posición actual
            self.position_i[i] = self.position_i[i] + self.velocity_i[i]
            
            # Verificar si excede el límite superior
            if self.position_i[i] > bounds[i][1]:
                # Ajustar al límite superior si lo excede
                self.position_i[i] = bounds[i][1]
            
            # Verificar si está por debajo del límite inferior
            if self.position_i[i] < bounds[i][0]:
                # Ajustar al límite inferior si está por debajo
                self.position_i[i] = bounds[i][0]

#--- PSO ENGINE ---------------------------------------------------------------+
class PSOEngine:
    """Motor del algoritmo PSO - Lógica pura sin interfaz"""
    
    def __init__(self, cost_function=func1, bounds=[(-10, 10), (-10, 10)]):
        # Función objetivo a optimizar (minimizar)
        self.cost_function = cost_function
        # Límites del espacio de búsqueda para cada dimensión
        self.bounds = bounds
        # Número de dimensiones del problema (basado en los límites)
        self.num_dimensions = len(bounds)
        
        # Lista para almacenar todas las partículas del enjambre
        self.swarm = []
        # Lista para almacenar la mejor posición global encontrada
        self.pos_best_g = []
        # Variable para almacenar el mejor error global (-1 indica no inicializado)
        self.err_best_g = -1
        # Contador de la iteración actual
        self.current_iteration = 0
        # Número máximo de iteraciones permitidas
        self.max_iterations = 30
        # Número de partículas en el enjambre
        self.num_particles = 15
        
    def initialize(self, initial_position, num_particles, max_iterations):
        """Inicializar el enjambre con parámetros específicos"""
        # Establecer número de partículas a crear
        self.num_particles = num_particles
        # Establecer número máximo de iteraciones
        self.max_iterations = max_iterations
        # Reinicializar lista de partículas
        self.swarm = []
        # Reinicializar mejor posición global
        self.pos_best_g = []
        # Reinicializar mejor error global
        self.err_best_g = -1
        # Reinicializar contador de iteraciones
        self.current_iteration = 0
        
        # Crear cada partícula del enjambre
        for i in range(num_particles):
            # Crear nueva partícula con posición inicial y agregar al enjambre
            self.swarm.append(Particle(initial_position, self.num_dimensions))
    
    def step(self):
        """Ejecutar un paso del algoritmo PSO"""
        # Verificar si se alcanzó el máximo de iteraciones
        if self.current_iteration >= self.max_iterations:
            # Retornar False para indicar que la optimización terminó
            return False  # Optimización completada
        
        # Evaluar todas las partículas del enjambre
        for j in range(len(self.swarm)):
            # Evaluar fitness de la partícula actual
            self.swarm[j].evaluate(self.cost_function)
            
            # Verificar si esta partícula encontró un nuevo mejor global
            if self.swarm[j].err_i < self.err_best_g or self.err_best_g == -1:
                # Guardar copia de la posición como nuevo mejor global
                self.pos_best_g = list(self.swarm[j].position_i)
                # Actualizar el mejor error global
                self.err_best_g = float(self.swarm[j].err_i)
        
        # Actualizar velocidades y posiciones de todas las partículas
        for j in range(len(self.swarm)):
            # Actualizar velocidad usando el mejor global
            self.swarm[j].update_velocity(self.pos_best_g, self.num_dimensions)
            # Actualizar posición basada en la nueva velocidad
            self.swarm[j].update_position(self.bounds, self.num_dimensions)
        
        # Incrementar contador de iteraciones
        self.current_iteration += 1
        # Retornar True para indicar que debe continuar
        return True  # Continuar
    
    def get_particles_positions(self):
        """Obtener posiciones actuales de todas las partículas (solo 2D)"""
        # Retornar lista de tuplas con coordenadas x,y de cada partícula
        return [(p.position_i[0], p.position_i[1]) for p in self.swarm]
    
    def get_particles_best_positions(self):
        """Obtener mejores posiciones personales de las partículas (solo 2D)"""
        # Retornar lista de tuplas con mejores coordenadas personales x,y
        return [(p.pos_best_i[0], p.pos_best_i[1]) if len(p.pos_best_i) >= 2 else None 
                for p in self.swarm]
    
    def get_global_best(self):
        """Obtener la mejor posición global encontrada"""
        # Retornar mejor posición global si existe, sino None
        return self.pos_best_g if self.pos_best_g else None
    
    def get_stats(self):
        """Obtener estadísticas actuales del algoritmo"""
        # Retornar diccionario con información del estado actual
        return {
            'iteration': self.current_iteration,      # Iteración actual
            'best_error': self.err_best_g,           # Mejor error encontrado
            'best_position': self.pos_best_g if self.pos_best_g else None  # Mejor posición
        }