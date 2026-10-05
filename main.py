
import cv2
import mediapipe as mp
import pygame
import math
import numpy as np
import time

# -------------------- INITIALIZATION --------------------

pygame.init()

WIDTH, HEIGHT = 1100, 720
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("GestureOS | Touchless Interface")

clock = pygame.time.Clock()

# Futuristic color palette
BG = (8, 12, 25)
PANEL = (16, 25, 43)
PANEL_LIGHT = (22, 35, 56)
CYAN = (0, 235, 255)
PURPLE = (150, 90, 255)
WHITE = (235, 245, 255)
MUTED = (130, 153, 180)
GREEN = (0, 255, 155)
ORANGE = (255, 190, 70)

font_small = pygame.font.SysFont("consolas", 15)
font_medium = pygame.font.SysFont("consolas", 19)
font_large = pygame.font.SysFont("consolas", 29, bold=True)
font_title = pygame.font.SysFont("consolas", 35, bold=True)

# -------------------- HAND TRACKING --------------------

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

# -------------------- HELPER FUNCTIONS --------------------

def draw_text(text, x, y, font, color=WHITE):
    image = font.render(text, True, color)
    screen.blit(image, (x, y))


def draw_panel(rect, color=PANEL, border=CYAN, radius=15):
    pygame.draw.rect(screen, color, rect, border_radius=radius)
    pygame.draw.rect(screen, border, rect, width=1, border_radius=radius)


def is_open_palm(hand):
    points = hand.landmark

    fingers_open = (
        points[8].y < points[6].y and
        points[12].y < points[10].y and
        points[16].y < points[14].y and
        points[20].y < points[18].y
    )

    thumb_distance = math.hypot(
        points[4].x - points[5].x,
        points[4].y - points[5].y
    )

    return fingers_open and thumb_distance > 0.10


def draw_background():
    screen.fill(BG)

    # Subtle futuristic grid
    for x in range(0, WIDTH, 40):
        pygame.draw.line(screen, (15, 25, 42), (x, 0), (x, HEIGHT))

    for y in range(0, HEIGHT, 40):
        pygame.draw.line(screen, (15, 25, 42), (0, y), (WIDTH, y))

    # Header accent
    pygame.draw.line(screen, CYAN, (30, 82), (WIDTH - 30, 82), 2)


def draw_stat_card(rect, title, value, subtitle, accent):
    draw_panel(rect, border=accent)
    x, y, w, h = rect

    draw_text(title, x + 16, y + 14, font_small, MUTED)
    draw_text(value, x + 16, y + 40, font_large, accent)
    draw_text(subtitle, x + 16, y + 80, font_small, WHITE)


# -------------------- MAIN LOOP --------------------

running = True
hand_detected = False
system_active = False
fps = 0
start_time = time.time()
frame_count = 0

print("GestureOS dashboard started!")
print("Show an open palm to activate.")
print("Press Q or close the window to quit.")

try:
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False

        success, frame = cap.read()

        if not success:
            print("Could not read webcam frame.")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        hand_detected = False
        system_active = False
        gesture_name = "NONE"

        if results.multi_hand_landmarks:
            hand_detected = True

            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                if is_open_palm(hand_landmarks):
                    system_active = True
                    gesture_name = "OPEN PALM"
                else:
                    gesture_name = "HAND DETECTED"

        # Webcam image for the dashboard
        camera_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        camera_rgb = cv2.resize(camera_rgb, (620, 349))
        camera_surface = pygame.surfarray.make_surface(
            np.transpose(camera_rgb, (1, 0, 2))
        )

        # Update FPS
        frame_count += 1
        elapsed = time.time() - start_time

        if elapsed >= 1:
            fps = frame_count / elapsed
            frame_count = 0
            start_time = time.time()

        # -------------------- DRAW DASHBOARD --------------------

        draw_background()

        # Header
        draw_text("GESTURE", 30, 25, font_title, WHITE)
        draw_text("OS", 185, 25, font_title, CYAN)
        draw_text("TOUCHLESS CONTROL SYSTEM", 250, 35, font_small, MUTED)

        draw_text("● LIVE", 900, 30, font_medium, GREEN)
        draw_text("CAMERA 01", 900, 55, font_small, MUTED)

        # Left: Camera panel
        draw_panel((25, 105, 660, 405), border=CYAN)
        draw_text("LIVE HAND TRACKING", 43, 118, font_medium, CYAN)

        screen.blit(camera_surface, (45, 153))

        # Camera image border
        pygame.draw.rect(
            screen, CYAN, (45, 153, 620, 349), width=1
        )

        # Right: Activation panel
        draw_panel((705, 105, 370, 220), border=PURPLE)
        draw_text("SYSTEM STATUS", 727, 124, font_medium, MUTED)

        if system_active:
            status_text = "ACTIVATED"
            status_color = GREEN
            detail = "Open palm recognized"
        elif hand_detected:
            status_text = "STANDBY"
            status_color = ORANGE
            detail = "Show an open palm"
        else:
            status_text = "NO HAND"
            status_color = MUTED
            detail = "Place hand in camera view"

        # Status indicator
        pygame.draw.circle(screen, status_color, (743, 184), 7)
        draw_text(status_text, 762, 165, font_large, status_color)
        draw_text(detail, 728, 215, font_small, WHITE)

        pygame.draw.line(
            screen, (55, 75, 105), (728, 252), (1050, 252), 1
        )

        draw_text("CURRENT GESTURE", 728, 267, font_small, MUTED)
        draw_text(gesture_name, 728, 290, font_medium, CYAN)

        # Bottom: Statistics cards
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
            "GESTUREOS  /  HAND INTERFACE  /  Q: QUIT",
            30, 690, font_small, MUTED
        )

        pygame.display.flip()
        clock.tick(30)

finally:
    cap.release()
    hands.close()
    pygame.quit()