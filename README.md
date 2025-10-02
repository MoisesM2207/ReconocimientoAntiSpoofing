# ReconocimientoAntiSpoofing

Proyecto de **Reconocimiento Facial con Anti-Spoofing** para control de asistencia automática utilizando **Python**, **OpenCV**, **MediaPipe** y **Face Recognition**.

## Características principales
- Registro de estudiantes con nombre, grado y sección.
- Captura de rostros mediante parpadeo (liveness detection).
- Prevención contra intentos de suplantación (spoofing).
- Generación automática de asistencia en archivo **CSV (Excel)** con fecha, hora, ID único y datos del estudiante.
- Interfaz gráfica simple usando **Tkinter**.

## Tecnologías utilizadas
- Python 3.8+
- OpenCV
- MediaPipe
- face-recognition
- Tkinter
- NumPy
- Pillow

## Instalación y uso
1. Clonar el repositorio:
   ```bash
   git clone https://github.com/MoisesM2207/ReconocimientoAntiSpoofing.git
   cd ReconocimientoAntiSpoofing
Crear entorno virtual:
python -m venv .venv
source .venv/Scripts/activate   # Windows
Instalar dependencias:
pip install -r requirements.txt
Ejecutar el programa:
python main.py