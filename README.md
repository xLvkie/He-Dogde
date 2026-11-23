# Esquiva la Silueta

Videojuego arcade interactivo en tiempo real controlado con tu propio cuerpo frente a la cámara web mediante **MediaPipe Selfie Segmentation**, **OpenCV**, **NumPy** y **Pygame (`pygame-ce`)**.

Tu silueta segmentada aparece en pantalla con estética neón futurista y debes moverte físicamente (inclinarte, desplazarte, agacharte) para esquivar obstáculos durante **5 minutos** y sobrevivir hasta el final.

---

## Requisitos del Sistema

- **Python 3.10+** (probado en **Python 3.14**)
- Cámara web funcional (o usar el parámetro `--demo` para probar sin cámara física)
- Conexión a internet la primera vez únicamente si se requiere descargar el modelo `.tflite` (ya incluido/auto-descargable en `models/selfie_segmenter_landscape.tflite`)

---

## Instalación y Configuración

El proyecto incluye soporte para entorno virtual `.venv` y dependencias fijadas en `requirements.txt` (usando `pygame-ce`, que expone exactamente el módulo `pygame` con soporte nativo para Python 3.14).

```bash
# 1. Crear entorno virtual (si aún no existe)
python3 -m venv .venv

# 2. Activar el entorno virtual
source .venv/bin/activate   # En Linux / macOS
# .venv\Scripts\activate    # En Windows

# 3. Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Cómo Ejecutar el Juego

### Modo Normal (con Cámara Web)
```bash
.venv/bin/python main.py
```

Si tienes varias cámaras conectadas, puedes seleccionar otra con `--camera`:
```bash
.venv/bin/python main.py --camera 1
```

### Modo Demo (sin Cámara Web)
Para probar todas las mecánicas (incluidos gestos de puño y escudo elástico) sin webcam física:
```bash
.venv/bin/python main.py --demo
```
En modo `--demo`, mueve el cuerpo con **flechas / WASD**, agáchate con **`C`** o **`Flecha Abajo`**, mueve tu mano virtual con el **ratón** y mantén **clic izquierdo** o **`F`** para cerrar el **puño**.

### Opciones Adicionales
- `--windowed` : Inicia en modo ventana redimensionable en lugar de pantalla completa nativa (por defecto inicia en pantalla completa).
- `--scale 1.0` : Factor de escala de la silueta neón (por defecto `1.0`, silueta completa inmersiva).
- `--duration 60` : Cambia la duración de la partida en segundos (por defecto `300.0`, equivalente a 5:00 minutos, escalando proporcionalmente las 4 fases).
- `--width 1280 --height 720` : Ajusta la resolución virtual del lienzo de juego (16:9 HD, escalado y centrado proporcionalmente en cualquier monitor).

---

## Controles

| Tecla / Acción | Función |
| :--- | :--- |
| `ESPACIO` / `ENTER` | Iniciar partida desde el menú (o reiniciar en pantalla final) |
| `F11` | Alternar entre pantalla completa nativa y modo ventana |
| `P` | Pausar / Reanudar la partida |
| `R` | Reiniciar partida rápidamente (en pausa o fin de juego) |
| `V` | Volver al menú principal (en pausa o fin de juego) |
| `B` | Alternar fondo entre cámara atenuada y fondo oscuro futurista |
| `M` | Silenciar / Activar efectos de sonido |
| `D` o `F3` | Mostrar/ocultar vista de diagnóstico de máscara y estado del Núcleo Vital (`F3` en `--demo`) |
| `ESC` | Pausar durante partida; salir desde menú, pausa o pantalla final |

---

## Mecánicas y Curva de 5 Minutos

- **Núcleo Vital en el Pecho (Estilo Bullet-Hell):** La silueta neón completa se dibuja mediante **MediaPipe Selfie Segmentation**, mientras **MediaPipe Pose** (`PoseLandmarker Lite`) detecta ambos hombros para situar un **Núcleo Vital** (radio `22 px`) exactamente en el centro del pecho (`mid_y + 15%` de la distancia entre hombros). Solo los impactos sobre ese núcleo quitan vidas, permitiendo esquivas milimétricas incluso cerca de la cámara.
- **Sistema de Respaldo en 3 Niveles:**
  1. **Pose + Suavizado Lerp (`0.35`):** Rastreo estable de hombros sin vibraciones.
  2. **Centroide de Máscara:** Si `Pose` pierde los hombros, el núcleo sigue suavemente el centro de masa superior-central de la silueta segmentada.
  3. **Memoria:** Si sales del encuadre o se tapa la cámara, el núcleo permanece en su última posición válida con el aviso `"BUSCANDO SEÑAL..."` sin detener el juego.
- **Vidas e Invulnerabilidad:** Empiezas con **5 vidas** y obtienes **1.5 segundos de escudo/invulnerabilidad** (parpadeo blanco/rojo en el núcleo) tras recibir un golpe.
- **Extremidades Reforzadas:** La máscara de `Selfie Segmentation` aplica dilatación ligera y suavizado gaussiano para no recortar brazos/manos rápidos, reforzada con un trazo neón esquelético de **MediaPipe Pose** sobre hombros, codos, muñecas, caderas, rodillas y tobillos.
- **Gestos de Mano y Puño Cerrado (`MediaPipe Hands`):** Rastrea ambas manos en tiempo real mostrando un cursor **cian abierto** o **dorado/magenta contraído** al cerrar el puño:
  - **Mecánica A · Atracción Magnética de Orbes:** Cierra el puño dentro de `130 px` de un orbe para engancharlo con un haz de energía y llévalo hasta tu corazón para absorberlo (`+250 pts` o `+1 Vida`). Si abres la mano antes, el orbe se libera.
  - **Mecánica B · Escudo Elástico del Núcleo Vital:** Cierra el puño sobre tu corazón y muévelo libremente **dentro de tu silueta** para esquivar impactos directos al pecho; al abrir la mano regresa elásticamente al centro del pecho.
- **Orbes Recolectables:**
  - **Orbes Dorados:** `+250` puntos extra al tocarlos o llevarlos a tu núcleo (`46 px` de radio de absorción).
  - **Orbes Verdes:** Recuperan `+1 vida` (o `+400` puntos si ya tienes el máximo).

### Fases de Dificultad
1. **Minuto 0:00 – 1:00 (`FASE 1 · CALENTAMIENTO`):** Meteoritos que caen desde arriba a velocidad moderada.
2. **Minuto 1:00 – 2:30 (`FASE 2 · LÁSERES`):** Se suman rayos láser horizontales y verticales con **1.5s de advertencia parpadeante** antes de activarse.
3. **Minuto 2:30 – 4:00 (`FASE 3 · HOLE IN THE WALL`):** Barreras que cubren casi toda la pantalla excepto una **zona segura verde** (*¡Agáchate!*, *¡A la izquierda!*, *¡A la derecha!*, *¡Al centro!*, etc.) con cuenta regresiva visual.
4. **Minuto 4:00 – 5:00 (`FASE 4 · CLÍMAX FINAL`):** Combinación rápida de meteoritos, láseres y zonas seguras hasta completar los **05:00** (Victoria).

---

## Estructura del Proyecto

- `main.py` — Punto de entrada, argumentos CLI y ciclo de vida de recursos.
- `config.py` — Constantes de resolución, umbrales de máscara, paleta neón y definición de fases.
- `segmentation.py` — Hilo de captura de webcam con ajuste 16:9 sin deformación, espejo horizontal (`cv2.flip`), MediaPipe Selfie Segmentation + Pose (`pose_landmarker_lite.task`) + Hands (`hand_landmarker.task`) en `480×270`, detección de puño cerrado (`detect_fist`), respaldo en 3 niveles y renderizado a `1280×720`.
- `obstacles.py` — Obstáculos (`Meteor`, `Laser`, `SafeZoneWall`), coleccionables (`Orb`) con colisión circular precisa contra el Núcleo Vital y efectos (`Particle`, `FloatingText`).
- `game.py` — Máquina de estados (`MENU`, `PLAYING`, `PAUSED`, `GAME_OVER`, `VICTORY`), lienzo virtual `1280×720` escalado centrado en pantalla completa, director de fases, colisiones al núcleo y HUD.
- `audio.py` — Sintetizador de efectos arcade en memoria con `numpy` y `pygame.mixer`.
- `requirements.txt` — Versiones exactas de dependencias.

---

## Solución de Problemas

- **No detecta bien mi silueta:** Aléjate un poco de la cámara para mostrar torso y hombros dentro del marco de calibración, evita contraluz fuerte detrás de ti y asegúrate de tener iluminación frontal uniforme.
- **Sin cámara web física o permiso denegado:** Usa `.venv/bin/python main.py --demo` para probar el juego completo sin webcam.
- **Error de audio en entornos sin tarjeta de sonido:** El módulo `audio.py` detecta automáticamente la falta de dispositivo de audio y continúa en modo silencioso sin cerrar el juego.
