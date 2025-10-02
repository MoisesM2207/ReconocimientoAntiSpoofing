# ReconocimientoAntiSpoofing

Proyecto de **Reconocimiento Facial con Anti-Spoofing** para control de asistencia automática utilizando **Python**, **OpenCV**, **MediaPipe** y **Face Recognition**.

## 📌 Características principales
- Registro de estudiantes con nombre, grado y sección.
- Captura de rostros mediante parpadeo (liveness detection).
- Prevención contra intentos de suplantación (spoofing).
- Generación automática de asistencia en archivo **CSV (Excel)** con fecha, hora, ID único y datos del estudiante.
- Interfaz gráfica simple usando **Tkinter**.

## 🛠 Tecnologías utilizadas
- Python 3.8+
- OpenCV
- MediaPipe
- face-recognition
- Tkinter
- NumPy
- Pillow

## 📂 Estructura del proyecto
ReconocimientoAntiSpoofing/
│── main.py # Programa principal
│── requirements.txt # Dependencias del proyecto
│── README.md # Este archivo
│── .gitignore # Archivos a ignorar
│── DataBase/
│ ├── Usuarios/ # Datos de los estudiantes
│ ├── Caras/ # Imágenes de rostros registrados
│ └── asistencia.csv # Registro de asistencia
│── SetUp/
├── Paso0.png
├── Paso1.png
├── Paso2e.png
└── check.png

bash
Copiar código

## 🚀 Instalación y uso
1. Clonar el repositorio:
   ```bash
   git clone https://github.com/MoisesM2207/ReconocimientoAntiSpoofing.git
   cd ReconocimientoAntiSpoofing
Crear entorno virtual:

bash
Copiar código
python -m venv .venv
source .venv/Scripts/activate   # Windows
Instalar dependencias:

bash
Copiar código
pip install -r requirements.txt
Ejecutar el programa:

bash
Copiar código
python main.py