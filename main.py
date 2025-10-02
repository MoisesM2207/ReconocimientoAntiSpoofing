# main.py
# -*- coding: utf-8 -*-

import os
import re
import csv
import uuid
import math
import time
import datetime as dt
import cv2
import numpy as np
import mediapipe as mp
import face_recognition as fr
from tkinter import Tk, Toplevel, Label, Entry, Button, StringVar, messagebox
from tkinter import ttk
from PIL import Image, ImageTk, Image

# =========================
# RUTAS (AJÚSTALAS)
# =========================
BASE = r"C:/Users/Moy/Desktop/ReconocimientoAntiSpoofing"
OUT_USERS = rf"{BASE}/DataBase/Usuarios"
OUT_FACES = rf"{BASE}/DataBase/Caras"
ATTEND_CSV = rf"{BASE}/DataBase/asistencia.csv"

IMG_STEP0 = rf"{BASE}/SetUp/Paso0.png"
IMG_STEP1 = rf"{BASE}/SetUp/Paso1.png"
IMG_STEP2 = rf"{BASE}/SetUp/Paso2e.png"   # “sonríe” o similar
IMG_CHECK = rf"{BASE}/SetUp/check.png"
BG_MAIN   = rf"{BASE}/SetUp/Registro de Estudiante.png"

os.makedirs(OUT_USERS, exist_ok=True)
os.makedirs(OUT_FACES, exist_ok=True)

# =========================
# HELPERS
# =========================
def load_cv2_or_none(path):
    try:
        if path and os.path.isfile(path):
            return cv2.imread(path)
    except Exception:
        pass
    return None

def safe_overlay(dst, src, top, left):
    if src is None:
        return
    H, W = dst.shape[:2]
    h, w = src.shape[:2]
    if top >= H or left >= W:
        return
    y1, y2 = max(0, top), min(H, top + h)
    x1, x2 = max(0, left), min(W, left + w)
    sy1, sy2 = 0, y2 - y1
    sx1, sx2 = 0, x2 - x1
    if sy2 <= 0 or sx2 <= 0:
        return
    dst[y1:y2, x1:x2] = src[sy1:sy2, sx1:sx2]

def eye_distance(landmarks, idx_a, idx_b, w, h):
    xa, ya = int(landmarks[idx_a].x * w), int(landmarks[idx_a].y * h)
    xb, yb = int(landmarks[idx_b].x * w), int(landmarks[idx_b].y * h)
    return math.hypot(xb - xa, yb - ya)

def today_str():
    return dt.date.today().isoformat()

def gen_short_id():
    # 8 caracteres hex: compacto y con colisiones extremadamente improbables
    return uuid.uuid4().hex[:8].upper()

# =========================
# APP
# =========================
class App:
    def __init__(self):
        # Estado
        self.state = "preview"  # preview | register_form | register_wait_blink
        self.pending_user = None  # (uid, nombre, grado, seccion)

        # Liveness
        self.blink_count = 0
        self.blink_lock = False

        # Offsets del bbox
        self.offset_x_pct = 20
        self.offset_y_pct = 40

        # Overlays
        self.step0 = load_cv2_or_none(IMG_STEP0)
        self.step1 = load_cv2_or_none(IMG_STEP1)
        self.step2 = load_cv2_or_none(IMG_STEP2)
        self.img_check = load_cv2_or_none(IMG_CHECK)

        # DB de caras
        self.known_encodings = []   # lista de encodings
        self.known_uids      = []   # mismo orden que encodings
        self.user_meta = {}         # uid -> (nombre, grado, seccion)
        self.face_tolerance = 0.45
        self.load_db()

        # CSV asistencia
        self.ensure_attendance_csv()

        # Cache de asistencias del día (por UID)
        self.marked_today = set()

        # Toast de “asistencia guardada”
        self.toast_until = 0
        self.toast_text  = ""

        # MediaPipe
        self.mp_draw   = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
        self.face_det  = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=1)

        # Tk
        self.root = Tk()
        self.root.title("Sistema de reconocimiento Facial")
        self.root.geometry("1280x720")
        if os.path.isfile(BG_MAIN):
            self.bg = ImageTk.PhotoImage(Image.open(BG_MAIN))
            Label(self.root, image=self.bg).place(x=0, y=0, relwidth=1, relheight=1)

        # Video label
        self.lbl = Label(self.root)
        self.lbl.place(x=0, y=0)

        Button(self.root, text="Registrar manualmente",
               command=self.open_register_form).place(x=1100, y=680)

        # Cámara (índice 1)
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not self.cap.isOpened():
            messagebox.showerror("Cámara", "No se pudo abrir la cámara en índice 1.\nSe intentará el índice 0.")
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Umbrales para abrir registro si no hay match
        self.no_face_frames = 0
        self.unrec_frames   = 0
        self.no_face_threshold = 120      # ~4 s
        self.unrec_threshold   = 120      # ~4 s con rostro no reconocido

        self.loop()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.mainloop()

    # -----------------------
    # Carga de DB
    # -----------------------
    def load_db(self):
        self.known_encodings = []
        self.known_uids      = []
        self.user_meta       = {}

        # Cargar metadata de usuarios (UID => nombre, grado, seccion)
        if os.path.isdir(OUT_USERS):
            for fn in os.listdir(OUT_USERS):
                if fn.lower().endswith(".txt"):
                    try:
                        with open(os.path.join(OUT_USERS, fn), "r", encoding="utf-8") as f:
                            txt = f.read().strip()
                        parts = [p for p in txt.split(",") if p != ""]
                        if len(parts) >= 4:
                            uid, nombre, grado, seccion = parts[0], parts[1], parts[2], parts[3]
                            self.user_meta[uid] = (nombre, grado, seccion)
                    except Exception:
                        pass

        # Cargar encodings desde Caras/<UID>.png
        if os.path.isdir(OUT_FACES):
            for fn in os.listdir(OUT_FACES):
                if fn.lower().endswith((".png", ".jpg", ".jpeg")):
                    uid = os.path.splitext(fn)[0]
                    path = os.path.join(OUT_FACES, fn)
                    try:
                        img = fr.load_image_file(path)
                        encs = fr.face_encodings(img)
                        if encs:
                            self.known_encodings.append(encs[0])
                            self.known_uids.append(uid)
                    except Exception:
                        pass

        print(f"DB cargada: {len(self.known_uids)} rostro(s); metadatos: {len(self.user_meta)}.")

    # -----------------------
    # CSV asistencia (con ;)
    # -----------------------
    def ensure_attendance_csv(self):
        if not os.path.isfile(ATTEND_CSV):
            with open(ATTEND_CSV, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f, delimiter=";")
                w.writerow(["fecha", "hora", "id", "nombre", "grado", "seccion"])

    def already_marked_today(self, uid):
        if uid in self.marked_today:
            return True
        today = today_str()
        try:
            with open(ATTEND_CSV, "r", encoding="utf-8") as f:
                r = csv.DictReader(f, delimiter=";")
                for row in r:
                    if row["fecha"] == today and row["id"] == uid:
                        self.marked_today.add(uid)
                        return True
        except Exception:
            pass
        return False

    def mark_attendance(self, uid):
        if self.already_marked_today(uid):
            return False
        nombre, grado, seccion = self.user_meta.get(uid, ("", "", ""))
        now = dt.datetime.now()
        with open(ATTEND_CSV, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow([now.date().isoformat(), now.strftime("%H:%M:%S"),
                        uid, nombre, grado, seccion])
        self.marked_today.add(uid)
        # Activa toast
        self.toast_text = f"✓ Asistencia guardada: {nombre}"
        self.toast_until = time.time() + 5  # 5 segundos
        return True

    # -----------------------
    # Registro
    # -----------------------
    def validate_name_char(self, S):
        return bool(re.fullmatch(r"[A-Za-zÁÉÍÓÚáéíóúÑñ ]*", S))

    def open_register_form(self):
        if self.state == "register_form":
            return
        self.state = "register_form"
        self.reg_win = Toplevel(self.root)
        self.reg_win.title("Registro de Estudiante")
        self.reg_win.geometry("520x300")

        self.nombre_var  = StringVar()
        self.grado_var   = StringVar()
        self.seccion_var = StringVar()
        self.id_var      = StringVar(value=gen_short_id())  # ID auto

        vcmd = (self.reg_win.register(self.validate_name_char), "%P")

        # ID (readonly)
        Label(self.reg_win, text="ID (auto):").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        Entry(self.reg_win, textvariable=self.id_var, width=20, state="readonly").grid(row=0, column=1, padx=10, sticky="w")

        Label(self.reg_win, text="Nombre completo:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        Entry(self.reg_win, textvariable=self.nombre_var, width=42,
              validate="key", validatecommand=vcmd).grid(row=1, column=1, padx=10)

        Label(self.reg_win, text="Grado:").grid(row=2, column=0, padx=10, pady=10, sticky="w")
        grados = ["1° Primaria","2° Primaria","3° Primaria","4° Primaria","5° Primaria","6° Primaria",
                  "1° Básico","2° Básico","3° Básico"]
        ttk.Combobox(self.reg_win, textvariable=self.grado_var, values=grados,
                     state="readonly", width=39).grid(row=2, column=1, padx=10)

        Label(self.reg_win, text="Sección:").grid(row=3, column=0, padx=10, pady=10, sticky="w")
        ttk.Combobox(self.reg_win, textvariable=self.seccion_var, values=["A","B","C","D","E"],
                     state="readonly", width=39).grid(row=3, column=1, padx=10)

        Button(self.reg_win, text="Continuar y capturar rostro",
               command=self.submit_register).grid(row=4, column=0, columnspan=2, pady=15)

        self.reg_win.protocol("WM_DELETE_WINDOW", self.cancel_register)

    def cancel_register(self):
        self.state = "preview"
        self.reg_win.destroy()

    def submit_register(self):
        nombre  = self.nombre_var.get().strip()
        grado   = self.grado_var.get().strip()
        seccion = self.seccion_var.get().strip()
        uid     = self.id_var.get().strip()

        if not nombre or not grado or not seccion:
            messagebox.showwarning("Registro", "Completa todos los campos.")
            return
        if not self.validate_name_char(nombre):
            messagebox.showwarning("Registro", "El nombre solo puede contener letras y espacios.")
            return

        # En teoría UID no colisiona; por si acaso, regenera si existe
        while os.path.isfile(os.path.join(OUT_USERS, f"{uid}.txt")) or os.path.isfile(os.path.join(OUT_FACES, f"{uid}.png")):
            uid = gen_short_id()
            self.id_var.set(uid)

        self.pending_user = (uid, nombre, grado, seccion)
        self.blink_count = 0
        self.blink_lock = False
        self.state = "register_wait_blink"
        self.reg_win.destroy()

    # -----------------------
    # Guardado de usuario
    # -----------------------
    def save_user(self, face_crop_bgr):
        uid, nombre, grado, seccion = self.pending_user

        # Guardar rostro
        face_path = os.path.join(OUT_FACES, f"{uid}.png")
        cv2.imwrite(face_path, face_crop_bgr)

        # Guardar metadata
        info_path = os.path.join(OUT_USERS, f"{uid}.txt")
        with open(info_path, "w", encoding="utf-8") as f:
            f.write(f"{uid},{nombre},{grado},{seccion},")

        # Añadir al DB en caliente
        try:
            img_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
            encs = fr.face_encodings(img_rgb)
            if encs:
                self.known_encodings.append(encs[0])
                self.known_uids.append(uid)
                self.user_meta[uid] = (nombre, grado, seccion)
        except Exception:
            pass

        messagebox.showinfo("Registro", f"Registro completado para {nombre} (ID {uid}).")
        self.pending_user = None
        self.state = "preview"
        self.blink_count = 0
        self.blink_lock = False

    # -----------------------
    # Reconocimiento en ROI → devuelve UID o None
    # -----------------------
    def recognize_in_roi(self, frame_bgr, rect):
        if rect is None:
            return None
        x, y, w, h = rect
        roi_bgr = frame_bgr[max(0, y):y+h, max(0, x):x+w]
        if roi_bgr.size == 0:
            return None

        small = cv2.resize(roi_bgr, (0, 0), fx=0.5, fy=0.5)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        locs = fr.face_locations(rgb, model="hog")
        if not locs:
            return None
        encs = fr.face_encodings(rgb, locs)
        if not encs:
            return None

        face_encoding = encs[0]
        if not self.known_encodings:
            return None

        dists = fr.face_distance(self.known_encodings, face_encoding)
        idx = int(np.argmin(dists))
        if dists[idx] <= self.face_tolerance:
            return self.known_uids[idx]
        return None

    # -----------------------
    # Bucle principal
    # -----------------------
    def loop(self):
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(10, self.loop)
            return

        frame_bgr = frame.copy()
        H, W = frame_bgr.shape[:2]

        # Detección
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        det = self.face_det.process(rgb)

        has_face = det.detections is not None and len(det.detections) > 0
        rect = None
        if has_face:
            face = det.detections[0]
            bbox = face.location_data.relative_bounding_box
            x = int(bbox.xmin * W)
            y = int(bbox.ymin * H)
            w = int(bbox.width * W)
            h = int(bbox.height * H)
            offx = int(self.offset_x_pct / 100.0 * w)
            offy = int(self.offset_y_pct  / 100.0 * h)
            x = max(0, x - offx // 2)
            y = max(0, y - offy // 2)
            w = min(W - x, w + offx)
            h = min(H - y, h + offy)
            rect = (x, y, w, h)

            # Malla blanca + parpadeo
            mesh = self.face_mesh.process(rgb)
            if mesh.multi_face_landmarks:
                lm = mesh.multi_face_landmarks[0].landmark
                self.mp_draw.draw_landmarks(
                    frame_bgr,
                    mesh.multi_face_landmarks[0],
                    mp.solutions.face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_styles.get_default_face_mesh_tesselation_style()
                )
                self.mp_draw.draw_landmarks(
                    image=frame_bgr,
                    landmark_list=mesh.multi_face_landmarks[0],
                    connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_draw.DrawingSpec(color=(255,255,255), thickness=1)
                )

                d_right = eye_distance(lm, 145, 159, W, H)
                d_left  = eye_distance(lm, 374, 386, W, H)
                closed_thresh = 10
                open_thresh   = 14
                if d_right <= closed_thresh and d_left <= closed_thresh and not self.blink_lock:
                    self.blink_count += 1
                    self.blink_lock = True
                elif d_right > open_thresh and d_left > open_thresh and self.blink_lock:
                    self.blink_lock = False

        # Rectángulo blanco
        if rect:
            x, y, w, h = rect
            cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (255, 255, 255), 2)

        # Reconocimiento
        label_to_show = None
        if has_face and self.state == "preview":
            uid = self.recognize_in_roi(frame_bgr, rect)
            if uid:
                nombre, grado, seccion = self.user_meta.get(uid, ("", "", ""))
                label_to_show = nombre or uid
                self.unrec_frames = 0
                self.no_face_frames = 0
                if self.mark_attendance(uid):
                    # recién marcada → ya se activó el toast
                    pass
            else:
                label_to_show = "DESCONOCIDO"
                self.unrec_frames += 1
                if self.unrec_frames > self.unrec_threshold:
                    self.open_register_form()
                    self.unrec_frames = 0
        else:
            if self.state == "preview":
                self.no_face_frames += 1
                if self.no_face_frames > self.no_face_threshold:
                    self.open_register_form()
                    self.no_face_frames = 0

        # Registro: overlays + parpadeos
        if self.state == "register_wait_blink":
            safe_overlay(frame_bgr, self.step0, 50, 50)      # arriba izquierda
            safe_overlay(frame_bgr, self.step1, 50, 1030)    # arriba derecha
            if self.step2 is not None:                       # abajo derecha, con margen 20
                h2, w2 = self.step2.shape[:2]
                safe_overlay(frame_bgr, self.step2, H - h2 - 20, W - w2 - 20)

            cv2.putText(frame_bgr, f"Parpadeos: {self.blink_count}",
                        (1070, 375), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            if self.img_check is not None:
                if self.blink_count >= 1:
                    safe_overlay(frame_bgr, self.img_check, 165, 1105)
                if self.blink_count >= 3:
                    safe_overlay(frame_bgr, self.img_check, 385, 1105)

            if self.blink_count >= 3 and rect and self.pending_user:
                x, y, w, h = rect
                crop = frame_bgr[max(0,y):min(y+h, H), max(0,x):min(x+w, W)].copy()
                if crop.size > 0:
                    self.save_user(crop)

        # Dibujar etiqueta (nombre o DESCONOCIDO)
        if label_to_show and rect:
            x, y, w, h = rect
            label = label_to_show
            (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            y0 = max(0, y - th - 10)
            cv2.rectangle(frame_bgr, (x, y0 - 5), (x + tw + 12, y0 + th + 10), (0, 0, 0), -1)
            cv2.putText(frame_bgr, label, (x + 6, y0 + th + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        # Toast “asistencia guardada”
        if time.time() < self.toast_until and self.toast_text:
            # caja semitransparente
            overlay = frame_bgr.copy()
            cv2.rectangle(overlay, (20, H-80), (520, H-20), (0, 128, 0), -1)
            frame_bgr = cv2.addWeighted(overlay, 0.4, frame_bgr, 0.6, 0)
            cv2.putText(frame_bgr, self.toast_text, (30, H-35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        # Mostrar
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
        self.lbl.configure(image=imgtk)
        self.lbl.image = imgtk

        self.root.after(10, self.loop)

    # -----------------------
    # Cierre
    # -----------------------
    def on_close(self):
        try:
            if self.cap and self.cap.isOpened():
                self.cap.release()
        except Exception:
            pass
        self.root.destroy()


if __name__ == "__main__":
    App()

