import cv2
import os
import train as t
# Укажите имя пользователя
user_name = input("Введите ваше имя: ")

# Укажите путь к папке, где будут храниться изображения
data_path = "data/"+user_name
if os.path.exists('user_model.pkl'):
    os.remove('user_model.pkl')
else:
    pass


# Создаем папку для сохранения изображений, если она не существует
if not os.path.exists(data_path):
    os.makedirs(data_path)

# Инициализируем камеру
cap = cv2.VideoCapture(0)

# Задаем параметры сохраняемых изображений
img_width, img_height = 640, 480
face_width, face_height = 200, 200

# Задаем параметры каскада Хаара
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Счетчик сохраненных изображений
count = 0

while True:
    # Считываем изображение с камеры
    ret, frame = cap.read()

    # Переворачиваем изображение, чтобы зеркально отобразить его
    frame = cv2.flip(frame, 1)

    # Конвертируем изображение в чб
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Находим лица на изображении
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Обрезаем изображение по лицу
        face = gray[y:y + h, x:x + w]

        # Изменяем размер изображения
        face = cv2.resize(face, (face_width, face_height))

        # Задаем имя и путь файла
        file_name = f"{count}.jpg"
        file_path = os.path.join(data_path, file_name)

        # Сохраняем изображение
        cv2.imwrite(file_path, face)

        # Увеличиваем счетчик
        count += 1

        # Если сохранены все 50 изображений, завершаем программу
        if count >= 10:
            cap.release()
            cv2.destroyAllWindows()
            exit()
    
        # Отрисовываем прямоугольник вокруг лица
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Отображаем изображение с прямоугольником
        cv2.imshow("frame", frame)

    # Ожидаем нажатия клавиши для выхода
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()
    t.train(user_name)


