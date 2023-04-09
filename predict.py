import os
import shutil
import face_recognition
import cv2
import pickle

# Загружаем список дескрипторов из файла
with open('user_model.pkl', 'rb') as handle:
    data = pickle.load(handle)
    known_face_descriptors = data["known_face_descriptors"]
    name = data["name"]

if os.path.exists('data/'+name):
    shutil.rmtree('data/'+name)
else: pass
# Запускаем видеопоток с камеры
video_capture = cv2.VideoCapture(0)

while True:
    # Получаем кадр с камеры
    ret, frame = video_capture.read()

    # Находим все лица на кадре
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Проходимся по всем найденным лицам
    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Сравниваем дескриптор лица с известными дескрипторами
        matches = face_recognition.compare_faces(known_face_descriptors, face_encoding)
        # Находим наиболее похожий дескриптор
        face_distances = face_recognition.face_distance(known_face_descriptors, face_encoding)
        best_match_index = int(face_distances.argmin())
        # Проверяем, что индекс не выходит за пределы списка matches

        # Если нашли соответствие, выводим имя пользователя
        if matches[best_match_index]:
            print("User: " + name)
        else:
            print("Unknown user")


    cv2.imshow('Video', frame)

    # Для остановки приложения нажмите клавишу 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
