import cv2
import numpy as np
from detector_objetos import DetectorFondoHomogeneo

# Cargamos el detector del marcador ArUCo
parametros = cv2.aruco.DetectorParameters()

# Nota: Asegúrate de usar el diccionario correcto
diccionario = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)

# Cargamos el detector de objetos
detector = DetectorFondoHomogeneo()

# Intenta abrir la cámara con el índice correcto
cap = cv2.VideoCapture(0)  # Cambia el índice según sea necesario
cap.set(3, 640)
cap.set(4, 480)

# Verifica si la cámara está abierta
if not cap.isOpened():
    print("Error al abrir la cámara.")
    exit()

# Dimensiones reales del marcador en cm
dim_marker_cm = 10

# Accedemos al bucle principal
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detectamos el marcador ArUCo
    esquinas, ids, rechazados = cv2.aruco.detectMarkers(frame, diccionario, parameters=parametros)

    if len(esquinas) > 0:
        # Dibujamos los polígonos de los marcadores detectados
        cv2.aruco.drawDetectedMarkers(frame, esquinas, ids)
        
        # Convertimos las esquinas a enteros para dibujarlos
        esquinas_ent = np.int0(esquinas)

        # Calculamos el perímetro del primer marcador detectado
        perimetro_aruco = cv2.arcLength(esquinas_ent[0], True)

        # Calculamos la proporción en cm
        proporcion_cm = perimetro_aruco / (4 * dim_marker_cm)  # 4 lados del cuadrado

        # Detectamos los objetos en la imagen
        contornos = detector.deteccion_objetos(frame)

        for cont in contornos:
            rectangulo = cv2.minAreaRect(cont)
            (x, y), (an, al), angulo = rectangulo
            ancho = an / proporcion_cm
            alto = al / proporcion_cm

            # Dibujamos un círculo en el centro del rectángulo
            cv2.circle(frame, (int(x), int(y)), 5, (255, 255, 0), -1)

            # Dibujamos el rectángulo
            rect = cv2.boxPoints(rectangulo)
            rect = np.int0(rect)

            cv2.polylines(frame, [rect], True, (0, 255, 0), 2)

            # Mostramos la información en píxeles
            cv2.putText(frame, f"Ancho: {round(ancho, 1)} cm", (int(x), int(y - 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 0, 255), 2)
            cv2.putText(frame, f"Largo: {round(alto, 1)} cm", (int(x), int(y + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 0, 255), 2)
    else:
        print("No se detectaron marcadores.")

    cv2.imshow("Medicion de objetos", frame)

    # Salir del programa al presionar la tecla ESC
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

    