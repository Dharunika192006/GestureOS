
import cv2
import mediapipe as mp
import pygame
import math
import numpy as np
import time

# ==================================================
# GESTUREOS - TOUCHLESS COMPUTER INTERFACE
# ==================================================

pygame.init()

WIDTH, HEIGHT = 1100, 720
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("GestureOS | Touchless Interface")
clock = pygame.time.Clock()

# Colors
BG = (8, 12, 25)
PANEL = (16, 25, 43)
PANEL_LIGHT = (25, 39, 60)
CYAN = (0, 235, 255)
PURPLE = (160, 100, 255)
WHITE = (235, 245, 255)
MUTED = (130, 153, 180)
GREEN = (0, 255, 155)
ORANGE = (255, 190, 70)
RED = (255, 90, 110)

# Fonts
font_small = pygame.font.SysFont("consolas", 15)
font_medium = pygame.font.SysFont("consolas", 19)
font_large = pygame.font.SysFont("consolas", 29, bold=True)
font_title = pygame.font.SysFont("consolas", 35, bold=True)

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not access webcam.")
    hands.close()
    pygame.quit()
    raise SystemExit(1)

# ==================================================
# GESTURE AND DASHBOARD SETTINGS
# ==================================================

PINCH_THRESHOLD = 0.05
PINCH_COOLDOWN = 0.6

pinching = False
previous_pinching = False
last_pinch_time = 0

hand_detected = False
system_active = False
gesture_name = "NONE"
cursor_position = None

lights_on = False
mode_index = 0
modes = ["FOCUS", "CREATIVE", "NORMAL"]

# Virtual buttons on the right side
light_button = pygame.Rect(725, 405, 150, 55)
mode_button = pygame.Rect(895, 405, 150, 55)

# Cursor mapping region: full dashboard
CURSOR_X, CURSOR_Y = 25, 105
CURSOR_W, CURSOR_H = 1050, 405

fps = 0
frame_count = 0
fps_start = time.time()

# ==================================================
# HELPER FUNCTIONS
# ==================================================

def draw_text(text, x, y, font, color=WHITE):
    surface = font.render(str(text), True, color)
    screen.blit(surface, (x, y))


def draw_panel(rect, color=PANEL, border=CYAN, radius=14):
    pygame.draw.rect(screen, color, rect, border_radius=radius)
    pygame.draw.rect(
        screen, border, rect, width=1, border_radius=radius
    )


def draw_background():
    screen.fill(BG)

    # Background grid
    for x in range(0, WIDTH, 40):
        pygame.draw.line(
            screen, (15, 25, 42), (x, 0), (x, HEIGHT)
        )

    for y in range(0, HEIGHT, 40):
        pygame.draw.line(
            screen, (15, 25, 42), (0, y), (WIDTH, y)
        )

    pygame.draw.line(
        screen, CYAN, (30, 82), (WIDTH - 30, 82), 2
    )


def is_open_palm(hand):
    p = hand.landmark

    fingers_open = (
        p[8].y < p[6].y and
        p[12].y < p[10].y and
        p[16].y < p[14].y and
        p[20].y < p[18].y
    )

    thumb_distance = math.hypot(
        p[4].x - p[5].x,
        p[4].y - p[5].y
    )

    return fingers_open and thumb_distance > 0.10


def draw_stat_card(rect, title, value, subtitle, accent):
    draw_panel(rect, border=accent)
    x, y, w, h = rect

    draw_text(title, x + 15, y + 14, font_small, MUTED)
    draw_text(value, x + 15, y + 40, font_large, accent)
    draw_text(subtitle, x + 15, y + 82, font_small, WHITE)


# ==================================================
# MAIN APPLICATION LOOP
# ==================================================

print("GestureOS started!")
print("Move your index fingertip to control the virtual cursor.")
print("Pinch thumb and index finger over a button to select.")
print("Press Q or ESC to quit.")

running = True

try:
    while running:

        # Handle keyboard and window events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False

        # Read webcam
        success, frame = cap.read()

        if not success:
            print("Could not read webcam frame.")
            break

        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # Reset per-frame tracking information
        hand_detected = False
        system_active = False
        gesture_name = "NONE"
        pinching = False
        cursor_position = None

        # ------------------------------------------
        # HAND TRACKING AND GESTURE RECOGNITION
        # ------------------------------------------

        if results.multi_hand_landmarks:

            hand_detected = True

            for hand_landmarks in results.multi_hand_landmarks:

                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                points = hand_landmarks.landmark

                # Map index fingertip to dashboard coordinates
                cursor_x = int(
                    CURSOR_X + points[8].x * CURSOR_W
                )
                cursor_y = int(
                    CURSOR_Y + points[8].y * CURSOR_H
                )

                # Keep cursor inside the dashboard
                cursor_x = max(0, min(WIDTH - 1, cursor_x))
                cursor_y = max(0, min(HEIGHT - 1, cursor_y))

                cursor_position = (cursor_x, cursor_y)

                # Thumb-to-index distance
                pinch_distance = math.hypot(
                    points[4].x - points[8].x,
                    points[4].y - points[8].y
                )

                pinching = pinch_distance < PINCH_THRESHOLD

                # Open palm activates the system
                if is_open_palm(hand_landmarks):
                    system_active = True
                    gesture_name = "OPEN PALM"

                elif pinching:
                    gesture_name = "PINCH"

                else:
                    gesture_name = "HAND DETECTED"

        # ------------------------------------------
        # PINCH-TO-SELECT
        # ------------------------------------------

        now = time.time()

        # Trigger only when a new pinch starts
        new_pinch = pinching and not previous_pinching

        if (
            new_pinch
            and cursor_position is not None
            and now - last_pinch_time >= PINCH_COOLDOWN
        ):
            cursor_x, cursor_y = cursor_position

            if light_button.collidepoint(cursor_x, cursor_y):
                lights_on = not lights_on
                last_pinch_time = now
                print(
                    "Virtual light:",
                    "ON" if lights_on else "OFF"
                )

            elif mode_button.collidepoint(cursor_x, cursor_y):
                mode_index = (mode_index + 1) % len(modes)
                last_pinch_time = now
                print("Interface mode:", modes[mode_index])

        previous_pinching = pinching

        # ------------------------------------------
        # CAMERA FRAME FOR DASHBOARD
        # ------------------------------------------

        camera_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        camera_rgb = cv2.resize(camera_rgb, (620, 349))

        camera_surface = pygame.surfarray.make_surface(
            np.transpose(camera_rgb, (1, 0, 2))
        )

        # FPS calculation
        frame_count += 1
        elapsed = time.time() - fps_start

        if elapsed >= 1:
            fps = frame_count / elapsed
            frame_count = 0
            fps_start = time.time()

        # ------------------------------------------
        # DRAW DASHBOARD
        # ------------------------------------------

        draw_background()

        # Header
        draw_text("GESTURE", 30, 25, font_title, WHITE)
        draw_text("OS", 185, 25, font_title, CYAN)

        draw_text(
            "TOUCHLESS CONTROL SYSTEM",
            250, 35, font_small, MUTED
        )

        draw_text("● LIVE", 900, 30, font_medium, GREEN)
        draw_text("CAMERA 01", 900, 55, font_small, MUTED)

        # Webcam panel
        draw_panel((25, 105, 660, 405), border=CYAN)

        draw_text(
            "LIVE HAND TRACKING",
            43, 118, font_medium, CYAN
        )

        screen.blit(camera_surface, (45, 153))

        pygame.draw.rect(
            screen, CYAN, (45, 153, 620, 349), 1
        )

        # System status panel
        draw_panel((705, 105, 370, 220), border=PURPLE)

        draw_text(
            "SYSTEM STATUS",
            727, 124, font_medium, MUTED
        )

        if system_active:
            status_text = "ACTIVATED"
            status_color = GREEN
            detail = "Open palm recognized"

        elif hand_detected:
            status_text = "STANDBY"
            status_color = ORANGE
            detail = "Hand detected"

        else:
            status_text = "NO HAND"
            status_color = MUTED
            detail = "Show your hand"

        pygame.draw.circle(
            screen, status_color, (743, 184), 7
        )

        draw_text(
            status_text, 762, 165, font_large, status_color
        )

        draw_text(detail, 728, 215, font_small, WHITE)

        pygame.draw.line(
            screen, (55, 75, 105),
            (728, 252), (1050, 252), 1
        )

        draw_text(
            "CURRENT GESTURE",
            728, 267, font_small, MUTED
        )

        draw_text(
            gesture_name, 728, 290, font_medium, CYAN
        )

        # ------------------------------------------
        # VIRTUAL CONTROL BUTTONS
        # ------------------------------------------

        draw_panel((705, 345, 370, 165), border=CYAN)

        draw_text(
            "VIRTUAL CONTROLS",
            727, 360, font_medium, CYAN
        )

        # Light button
        light_color = GREEN if lights_on else PANEL_LIGHT

        pygame.draw.rect(
            screen, light_color,
            light_button, border_radius=10
        )

        pygame.draw.rect(
            screen, CYAN,
            light_button, width=2, border_radius=10
        )

        light_label = "LIGHT: ON" if lights_on else "LIGHT: OFF"

        draw_text(
            light_label, 738, 423, font_small, WHITE
        )

        # Mode button
        pygame.draw.rect(
            screen, PANEL_LIGHT,
            mode_button, border_radius=10
        )

        pygame.draw.rect(
            screen, PURPLE,
            mode_button, width=2, border_radius=10
        )

        draw_text("MODE", 909, 411, font_small, MUTED)

        draw_text(
            modes[mode_index], 909, 431, font_small, WHITE
        )

        draw_text(
            "PINCH TO SELECT",
            727, 478, font_small, MUTED
        )

        # ------------------------------------------
        # STATISTICS
        # ------------------------------------------

        draw_stat_card(
            (25, 535, 250, 130),
            "HAND DETECTION",
            "YES" if hand_detected else "NO",
            "Webcam tracking",
            GREEN if hand_detected else MUTED
        )

        draw_stat_card(
            (295, 535, 250, 130),
            "ACTIVATION",
            "ON" if system_active else "OFF",
            "Open-palm control",
            GREEN if system_active else PURPLE
        )

        draw_stat_card(
            (565, 535, 250, 130),
            "PROCESSING",
            f"{fps:.0f} FPS",
            "Live camera frames",
            CYAN
        )

        draw_stat_card(
            (835, 535, 240, 130),
            "INPUT MODE",
            "GESTURE",
            "Touch-free interface",
            PURPLE
        )

        # Footer
        draw_text(
            "GESTUREOS / TOUCHLESS INTERFACE / Q: QUIT",
            30, 690, font_small, MUTED
        )

        # Draw virtual cursor last so it stays visible
        if cursor_position is not None:
            cursor_color = GREEN if pinching else CYAN

            pygame.draw.circle(
                screen, cursor_color, cursor_position, 13, 2
            )

            pygame.draw.circle(
                screen, WHITE, cursor_position, 3
            )

            if pinching:
                pygame.draw.circle(
                    screen, GREEN, cursor_position, 19, 1
                )

        pygame.display.flip()
        clock.tick(30)

finally:
    cap.release()
    hands.close()
    pygame.quit()
    print("GestureOS closed safely.")