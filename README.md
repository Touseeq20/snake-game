# 🐍 Hand Gesture Snake Game 🎮

A modern **Snake Game** controlled entirely by **hand gestures** using [MediaPipe](https://developers.google.com/mediapipe) and [OpenCV](https://opencv.org/).  
Move your **index finger** to control the snake, eat emoji-food 🍎🍌🍒, and watch your score grow!  

---

## 🚀 Features
- ✋ **Hand tracking** with MediaPipe (index finger = snake head).  
- 🍉 **Emoji food** with random spawning.  
- 🌈 Snake **color changes** based on score.  
- 💀 **Self-collision detection** and Game Over screen.  
- 🎯 Real-time FPS counter.  
- 🔄 Press **R** to restart, **Q** to quit.  

---

## 📸 Demo
(Add a GIF or screenshot of gameplay here)  

---

## ⚙️ Requirements
Install dependencies before running:
```bash
pip install opencv-python mediapipe pillow numpy
```

---

## ▶️ How to Run
Clone the repository and run the script:
```bash
git clone https://github.com/Touseeq20/Hand-Gesture-Snake-Game.git
cd Hand-Gesture-Snake-Game
python snake_game.py
```

---

## 🎨 Controls
- **Index finger** → Controls snake movement  
- **Eat food** → Increases score and snake length  
- **R** → Restart the game  
- **Q** → Quit the game  

---

## 📂 Project Structure
```
Hand-Gesture-Snake-Game/
│── snake_game.py      # Main game script
│── README.md          # Documentation
│── assets/            # (Optional) Screenshots / Demo GIFs
```

---

## 🧠 Tech Stack
- **Python**
- **OpenCV**
- **MediaPipe**
- **PIL (Pillow)**
- **NumPy**

---

## 🏆 Score Colors
| Score Range | Snake Color |
|-------------|-------------|
| 0–4         | 🟢 Green |
| 5–9         | 🔵 Blue |
| 10–14       | ⚪ White |
| 15–19       | 🟡 Cyan |
| 20+         | 🟠 Orange |

---

## 👨‍💻 Author
**Muhammad Touseeq**  
📧 mtouseeq20@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/muhammad-touseeq-ai) | [GitHub](https://github.com/Touseeq20)

---

⭐ If you like this project, don’t forget to **star the repo**!
