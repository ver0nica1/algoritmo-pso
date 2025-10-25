"""
PSO - Interfaz gráfica con PyQt5
Archivo: pso_gui.py

DEPENDENCIAS:
pip install PyQt5 matplotlib numpy
"""

import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QSpinBox, QDoubleSpinBox,
                             QPushButton, QSlider, QGroupBox, QGridLayout,
                             QMessageBox)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# Importar la lógica del PSO
from logic import PSOEngine, func1


class PSOVisualizerWindow(QMainWindow):
    """Ventana principal de la aplicación PSO"""
    
    def __init__(self):
        super().__init__()
        self.pso_engine = PSOEngine(cost_function=func1)
        self.is_running = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.step_pso)
        
        self.init_ui()
        self.reset_pso()
        
    def init_ui(self):
        """Inicializar interfaz de usuario"""
        self.setWindowTitle('PSO - Particle Swarm Optimization Visualizer')
        self.setGeometry(100, 100, 1400, 800)
        
        # Widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout principal
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Panel de controles (izquierda)
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)
        
        # Panel de visualización (derecha)
        plot_panel = self.create_plot_panel()
        main_layout.addWidget(plot_panel, stretch=1)
        
        # Aplicar estilos
        self.apply_styles()
        
    def create_control_panel(self):
        """Crear panel de controles"""
        panel = QWidget()
        panel.setFixedWidth(350)
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # Título
        title = QLabel('🐦 Controles PSO')
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Grupo de parámetros
        params_group = QGroupBox('Parámetros')
        params_layout = QGridLayout()
        params_group.setLayout(params_layout)
        
        # Número de partículas
        params_layout.addWidget(QLabel('Número de Partículas:'), 0, 0)
        self.num_particles_spin = QSpinBox()
        self.num_particles_spin.setRange(5, 50)
        self.num_particles_spin.setValue(15)
        params_layout.addWidget(self.num_particles_spin, 0, 1)
        
        # Iteraciones máximas
        params_layout.addWidget(QLabel('Iteraciones Máximas:'), 1, 0)
        self.max_iter_spin = QSpinBox()
        self.max_iter_spin.setRange(10, 100)
        self.max_iter_spin.setValue(30)
        params_layout.addWidget(self.max_iter_spin, 1, 1)
        
        # Posición inicial X
        params_layout.addWidget(QLabel('Posición Inicial X:'), 2, 0)
        self.init_x_spin = QDoubleSpinBox()
        self.init_x_spin.setRange(-10, 10)
        self.init_x_spin.setSingleStep(0.5)
        self.init_x_spin.setValue(5.0)
        params_layout.addWidget(self.init_x_spin, 2, 1)
        
        # Posición inicial Y
        params_layout.addWidget(QLabel('Posición Inicial Y:'), 3, 0)
        self.init_y_spin = QDoubleSpinBox()
        self.init_y_spin.setRange(-10, 10)
        self.init_y_spin.setSingleStep(0.5)
        self.init_y_spin.setValue(5.0)
        params_layout.addWidget(self.init_y_spin, 3, 1)
        
        layout.addWidget(params_group)
        
        # Velocidad de animación
        speed_group = QGroupBox('Velocidad de Animación')
        speed_layout = QVBoxLayout()
        speed_group.setLayout(speed_layout)
        
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(50, 500)
        self.speed_slider.setValue(100)
        self.speed_slider.setTickPosition(QSlider.TicksBelow)
        self.speed_slider.setTickInterval(50)
        self.speed_slider.valueChanged.connect(self.update_speed_label)
        speed_layout.addWidget(self.speed_slider)
        
        self.speed_label = QLabel('100 ms')
        self.speed_label.setAlignment(Qt.AlignCenter)
        speed_layout.addWidget(self.speed_label)
        
        layout.addWidget(speed_group)
        
        # Botones
        button_layout = QHBoxLayout()
        
        self.start_button = QPushButton('▶ Iniciar')
        self.start_button.clicked.connect(self.start_pso)
        self.start_button.setMinimumHeight(50)
        button_layout.addWidget(self.start_button)
        
        self.reset_button = QPushButton('↻ Reiniciar')
        self.reset_button.clicked.connect(self.reset_pso)
        self.reset_button.setMinimumHeight(50)
        button_layout.addWidget(self.reset_button)
        
        layout.addLayout(button_layout)
        
        # Estadísticas
        stats_group = QGroupBox('📊 Estadísticas')
        stats_layout = QGridLayout()
        stats_group.setLayout(stats_layout)
        
        stats_layout.addWidget(QLabel('Iteración:'), 0, 0)
        self.iter_label = QLabel('0')
        self.iter_label.setStyleSheet('font-weight: bold; color: #667eea;')
        stats_layout.addWidget(self.iter_label, 0, 1)
        
        stats_layout.addWidget(QLabel('Mejor Error:'), 1, 0)
        self.error_label = QLabel('-')
        self.error_label.setStyleSheet('font-weight: bold; color: #667eea;')
        stats_layout.addWidget(self.error_label, 1, 1)
        
        stats_layout.addWidget(QLabel('Mejor X:'), 2, 0)
        self.pos_x_label = QLabel('-')
        self.pos_x_label.setStyleSheet('font-weight: bold; color: #667eea;')
        stats_layout.addWidget(self.pos_x_label, 2, 1)
        
        stats_layout.addWidget(QLabel('Mejor Y:'), 3, 0)
        self.pos_y_label = QLabel('-')
        self.pos_y_label.setStyleSheet('font-weight: bold; color: #667eea;')
        stats_layout.addWidget(self.pos_y_label, 3, 1)
        
        layout.addWidget(stats_group)
        layout.addStretch()
        
        return panel
    
    def create_plot_panel(self):
        """Crear panel de visualización"""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # Título del gráfico
        plot_title = QLabel('Visualización del Espacio de Búsqueda')
        plot_title_font = QFont()
        plot_title_font.setPointSize(14)
        plot_title_font.setBold(True)
        plot_title.setFont(plot_title_font)
        plot_title.setAlignment(Qt.AlignCenter)
        layout.addWidget(plot_title)
        
        # Canvas de matplotlib
        self.figure = Figure(figsize=(8, 8))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        layout.addWidget(self.canvas)
        
        # Leyenda
        legend_layout = QHBoxLayout()
        legend_layout.addStretch()
        
        legend_items = [
            ('🔵', 'Partículas', '#3498db'),
            ('🔴', 'Mejor Global', '#e74c3c'),
            ('🟢', 'Mejor Personal', '#2ecc71'),
            ('⭐', 'Óptimo (0,0)', '#f39c12')
        ]
        
        for emoji, text, color in legend_items:
            label = QLabel(f'{emoji} {text}')
            label.setStyleSheet(f'color: {color}; font-weight: bold; margin: 0 10px;')
            legend_layout.addWidget(label)
        
        legend_layout.addStretch()
        layout.addLayout(legend_layout)
        
        return panel
    
    def apply_styles(self):
        """Aplicar estilos CSS a la aplicación"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f8f9fa;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #667eea;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                background-color: #667eea;
                color: white;
                border: none;
                border-radius: 8px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #5568d3;
            }
            QPushButton:pressed {
                background-color: #4451b8;
            }
            QSpinBox, QDoubleSpinBox {
                padding: 5px;
                border: 2px solid #ddd;
                border-radius: 5px;
            }
            QSlider::groove:horizontal {
                height: 8px;
                background: #ddd;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #667eea;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
        """)
    
    def update_speed_label(self):
        """Actualizar etiqueta de velocidad"""
        self.speed_label.setText(f'{self.speed_slider.value()} ms')
    
    def init_plot(self):
        """Inicializar visualización del gráfico"""
        self.ax.clear()
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.ax.set_xlabel('X', fontsize=12, fontweight='bold')
        self.ax.set_ylabel('Y', fontsize=12, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        
        # Dibujar contorno de la función
        x = np.linspace(-10, 10, 100)
        y = np.linspace(-10, 10, 100)
        X, Y = np.meshgrid(x, y)
        Z = X**2 + Y**2
        
        self.ax.contourf(X, Y, Z, levels=20, cmap='Blues', alpha=0.4)
        self.ax.contour(X, Y, Z, levels=20, colors='black', alpha=0.2, linewidths=0.5)
        
        # Marcar el óptimo
        self.ax.plot(0, 0, 'r*', markersize=20, label='Óptimo (0,0)', zorder=5)
        
    def update_plot(self):
        """Actualizar visualización con las partículas"""
        self.init_plot()
        
        # Obtener posiciones de las partículas
        positions = self.pso_engine.get_particles_positions()
        best_positions = self.pso_engine.get_particles_best_positions()
        global_best = self.pso_engine.get_global_best()
        
        # Dibujar partículas y sus mejores posiciones
        for i, (pos, best_pos) in enumerate(zip(positions, best_positions)):
            x, y = pos
            
            # Línea hacia mejor posición personal
            if best_pos is not None:
                best_x, best_y = best_pos
                self.ax.plot([x, best_x], [y, best_y], 'g--', alpha=0.3, linewidth=1)
            
            # Partícula
            self.ax.plot(x, y, 'bo', markersize=8, markeredgecolor='darkblue', 
                        markeredgewidth=1.5, zorder=3)
        
        # Mejor posición global
        if global_best:
            self.ax.plot(global_best[0], global_best[1], 'ro', markersize=15, 
                        label='Mejor Global', markeredgecolor='darkred', 
                        markeredgewidth=2, zorder=4)
        
        self.ax.legend(loc='upper right')
        self.canvas.draw()
    
    def update_stats(self):
        """Actualizar estadísticas en la interfaz"""
        stats = self.pso_engine.get_stats()
        
        self.iter_label.setText(str(stats['iteration']))
        
        if stats['best_error'] != -1:
            self.error_label.setText(f"{stats['best_error']:.6f}")
        else:
            self.error_label.setText('-')
        
        if stats['best_position']:
            self.pos_x_label.setText(f"{stats['best_position'][0]:.4f}")
            self.pos_y_label.setText(f"{stats['best_position'][1]:.4f}")
        else:
            self.pos_x_label.setText('-')
            self.pos_y_label.setText('-')
    
    def reset_pso(self):
        """Reiniciar PSO"""
        self.is_running = False
        self.timer.stop()
        
        initial_x = self.init_x_spin.value()
        initial_y = self.init_y_spin.value()
        num_particles = self.num_particles_spin.value()
        max_iter = self.max_iter_spin.value()
        
        self.pso_engine.initialize([initial_x, initial_y], num_particles, max_iter)
        
        self.update_plot()
        self.update_stats()
        self.start_button.setText('▶ Iniciar')
    
    def start_pso(self):
        """Iniciar o pausar PSO"""
        if self.is_running:
            # Pausar
            self.is_running = False
            self.timer.stop()
            self.start_button.setText('▶ Continuar')
        else:
            # Iniciar/Continuar
            if self.pso_engine.current_iteration == 0:
                self.reset_pso()
            
            self.is_running = True
            self.timer.start(self.speed_slider.value())
            self.start_button.setText('⏸ Pausar')
    
    def step_pso(self):
        """Ejecutar un paso del algoritmo"""
        continue_running = self.pso_engine.step()
        
        self.update_plot()
        self.update_stats()
        
        if not continue_running:
            # Optimización completada
            self.is_running = False
            self.timer.stop()
            self.start_button.setText('▶ Iniciar')
            
            stats = self.pso_engine.get_stats()
            QMessageBox.information(
                self,
                'Optimización Completada',
                f"¡Optimización completada!\n\n"
                f"Mejor posición: [{stats['best_position'][0]:.4f}, {stats['best_position'][1]:.4f}]\n"
                f"Mejor error: {stats['best_error']:.6f}"
            )


def main():
    """Función principal"""
    app = QApplication(sys.argv)
    window = PSOVisualizerWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()