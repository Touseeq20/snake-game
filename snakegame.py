
import cv2
import numpy as np
import random
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont
import time

# ------------------- Hand Tracking Setup -------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# ------------------- Snake Game Variables -------------------
snake_points = []
snake_length = []
total_length = 0
allowed_length = 150
food = None
current_food_emoji = None
current_food_img = None
score = 0
game_over = False

# ------------------- Game Configuration -------------------
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
SNAKE_SIZE = 20
FOOD_SIZE = 60  # emoji image size

# ------------------- Emoji Food Setup -------------------
emojis = ['🍎', '🍌', '🍒', '🍇', '🍉']

def emoji_to_image(emoji_char, size=FOOD_SIZE):
    img_pil = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img_pil)
    try:
        # Change font path if needed, e.g. for macOS or Linux
        font = ImageFont.truetype("seguiemj.ttf", size - 10)
    except:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), emoji_char, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((size - w) / 2, (size - h) / 2), emoji_char, font=font, embedded_color=True, fill=(255, 255, 255, 255))
    img_cv = np.array(img_pil)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGBA2BGRA)
    return img_cv

def generate_food():
    global current_food_emoji, current_food_img
    current_food_emoji = random.choice(emojis)
    current_food_img = emoji_to_image(current_food_emoji)
    # Generate food within bounds with margin
    x = random.randint(100, FRAME_WIDTH - 100)
    y = random.randint(100, FRAME_HEIGHT - 100)
    return [x, y]

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def reset_game():
    global snake_points, snake_length, total_length, allowed_length, food, score, game_over
    snake_points = []
    snake_length = []
    total_length = 0
    allowed_length = 150
    food = generate_food()
    score = 0
    game_over = False

def overlay_image_alpha(img, img_overlay, pos):
    x, y = pos
    h, w = img_overlay.shape[:2]
    if x < 0 or y < 0 or x + w > img.shape[1] or y + h > img.shape[0]:
        return img
    overlay_img = img_overlay[..., :3]
    mask = img_overlay[..., 3:] / 255.0
    img_crop = img[y:y+h, x:x+w]
    img[y:y+h, x:x+w] = (1.0 - mask) * img_crop + mask * overlay_img
    return img.astype(np.uint8)

# --------- New function to get snake color based on score ---------
def get_snake_color(score):
    if score < 5:
        return (0, 255, 0)      # Green
    elif score < 10:
        return (255, 0, 0)      # Blue (BGR)
    elif score < 15:
        return (255, 255, 255)  # White
    elif score < 20:
        return (0, 255, 255)    # Yellow-ish (Cyan)
    else:
        return (0, 165, 255)    # Orange

# ------------------- Main Game Loop -------------------

cap = cv2.VideoCapture(0)
cap.set(3, FRAME_WIDTH)
cap.set(4, FRAME_HEIGHT)

reset_game()

prev_time = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        index_finger_tip = hand_landmarks.landmark[8]

        cx = int(index_finger_tip.x * FRAME_WIDTH)
        cy = int(index_finger_tip.y * FRAME_HEIGHT)
        current_head = [cx, cy]

        if not game_over:
            # To smooth out sudden jumps, add only if movement > threshold (e.g. 5 px)
            if not snake_points or distance(current_head, snake_points[-1]) > 5:
                snake_points.append(current_head)

                if len(snake_points) > 1:
                    dist = distance(snake_points[-1], snake_points[-2])
                    snake_length.append(dist)
                    total_length += dist

                # Keep snake length within allowed_length
                while total_length > allowed_length and len(snake_length) > 0:
                    total_length -= snake_length[0]
                    snake_length.pop(0)
                    snake_points.pop(0)

                # Self-collision detection (if snake is long enough)
                if len(snake_points) > 30:
                    for i in range(len(snake_points) - 20):
                        if distance(current_head, snake_points[i]) < 15:
                            game_over = True
                            break

                # Food collision
                if distance(current_head, food) < FOOD_SIZE / 2:
                    score += 1
                    allowed_length += 30
                    food = generate_food()

    # Draw snake
    if not game_over:
        snake_color = get_snake_color(score)
        for i in range(1, len(snake_points)):
            cv2.line(frame, tuple(snake_points[i - 1]), tuple(snake_points[i]), snake_color, SNAKE_SIZE)

        # Draw emoji food
        if food and current_food_img is not None:
            x = food[0] - FOOD_SIZE // 2
            y = food[1] - FOOD_SIZE // 2
            frame = overlay_image_alpha(frame, current_food_img, (x, y))

        # Score display with shadow
        cv2.putText(frame, f'Score: {score}', (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 5)
        cv2.putText(frame, f'Score: {score}', (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, snake_color, 3)

    else:
        cv2.putText(frame, 'Game Over', (450, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 7)
        cv2.putText(frame, f'Final Score: {score}', (470, 380),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
        cv2.putText(frame, 'Press R to Restart', (470, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)

    # Draw hand landmarks
    if result.multi_hand_landmarks:
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # FPS counter
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time
    cv2.putText(frame, f'FPS: {int(fps)}', (FRAME_WIDTH - 150, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("Hand Gesture Snake Game", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('r'):
        reset_game()

cap.release()
cv2.destroyAllWindows()
